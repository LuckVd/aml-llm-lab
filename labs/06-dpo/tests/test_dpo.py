"""Lab 06 测试：实现 dpo.py 直到全部通过（离线 tiny Qwen2）。"""

import importlib.util
import math
import os
import random
import string

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
pytest.importorskip("peft")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB05_DIR = os.path.join(os.path.dirname(LAB_DIR), "05-sft")
SOLUTION_PATH = os.path.join(LAB_DIR, "solution", "dpo_solution.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


USE_SOLUTION = os.environ.get("AML_LAB_USE_SOLUTION") == "1"


def _pick(lab_dir, student, solution):
    return os.path.join(lab_dir, solution) if USE_SOLUTION else os.path.join(lab_dir, student)


lab05 = _load("lab05_for_dpo_test", _pick(LAB05_DIR, "sft.py", "solution/sft_solution.py"))
m = _load("lab06_for_test", _pick(LAB_DIR, "dpo.py", "solution/dpo_solution.py"))

CORPUS = (
    lab05.PROMPT_TEMPLATE.format(instruction="")
    + lab05.STOP
    + string.printable
    + "你好巴黎计算：。？＋－×÷＝·"
)

PAIRS = [
    {"instruction": "计算：1+1", "chosen": "1+1=2", "rejected": "1+1=3"},
    {"instruction": "计算：2+2", "chosen": "2+2=4", "rejected": "2+2=5"},
]


@pytest.fixture(scope="module")
def tok():
    return lab05.CharTokenizer(CORPUS)


def tiny_qwen(vocab_size, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=96,
    )
    return Qwen2ForCausalLM(cfg)


# ---------- sequence_logps ----------


def test_sequence_logps_matches_manual(tok):
    model = tiny_qwen(len(tok.itos), seed=1)
    batch = lab05.collate([lab05.build_example(tok, PAIRS[0]["instruction"], PAIRS[0]["chosen"])],
                          tok.pad_token_id)
    got = m.sequence_logps(model, **batch)
    # 手工：逐 token log_softmax 取目标
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    tgt = batch["labels"]
    ref = torch.zeros(1)
    for pos in range(1, tgt.shape[1]):
        if tgt[0, pos].item() != -100:
            ref[0] += F.log_softmax(logits[0, pos - 1], dim=-1)[tgt[0, pos]]
    assert torch.allclose(got, ref, atol=1e-5)


def test_sequence_logps_prompt_only_is_zero(tok):
    """全部 -100 的序列 logp 应为 0（没有可加的项）。"""
    model = tiny_qwen(len(tok.itos), seed=1)
    batch = lab05.collate([lab05.build_example(tok, "hi", "ok")], tok.pad_token_id)
    batch["labels"].fill_(-100)
    got = m.sequence_logps(model, **batch)
    assert torch.allclose(got, torch.zeros(1), atol=1e-6)


# ---------- dpo_loss ----------


def test_dpo_loss_at_reference_is_ln2():
    z = torch.zeros(4)
    loss, r_c, r_r, acc = m.dpo_loss(z, z, z, z, beta=0.1)
    assert torch.allclose(loss, torch.tensor(math.log(2)), atol=1e-6)
    assert torch.allclose(r_c, z) and torch.allclose(r_r, z)
    assert acc.item() == 0.0  # margin=0 不算 > 0


def test_dpo_loss_formula():
    pi_c = torch.tensor([1.0, 2.0])
    pi_r = torch.tensor([0.0, -1.0])
    ref_c = torch.tensor([0.5, 0.0])
    ref_r = torch.tensor([0.0, 0.5])
    loss, r_c, r_r, acc = m.dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.2)
    margin = 0.2 * ((pi_c - ref_c) - (pi_r - ref_r))
    expected = -F.logsigmoid(margin).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.allclose(r_c, 0.2 * (pi_c - ref_c))
    assert acc.item() == 1.0  # margin 全为正


def test_dpo_loss_gradient_directions():
    pi_c = torch.zeros(1, requires_grad=True)
    pi_r = torch.zeros(1, requires_grad=True)
    z = torch.zeros(1)
    loss, *_ = m.dpo_loss(pi_c, pi_r, z, z, beta=0.1)
    loss.backward()
    assert pi_c.grad < 0  # 抬高 chosen 的 logp -> loss 降
    assert pi_r.grad > 0  # 压低 rejected 的 logp -> loss 降


# ---------- 偏好 batch 与训练 ----------


def test_build_preference_batch_shapes(tok):
    batch = m.build_preference_batch(tok, PAIRS)
    for side in ("chosen", "rejected"):
        b = batch[side]
        assert b["input_ids"].shape[0] == 2
        assert (b["labels"] != -100).any()  # 答案可见
        # 两条样本的答案部分确实不同
    c_ans = batch["chosen"]["labels"][0][batch["chosen"]["labels"][0] != -100]
    r_ans = batch["rejected"]["labels"][0][batch["rejected"]["labels"][0] != -100]
    assert not torch.equal(c_ans, r_ans)


def test_train_dpo_grows_margin_and_keeps_ref_frozen(tok):
    torch.manual_seed(2)
    policy = lab05.add_lora(tiny_qwen(len(tok.itos), seed=3), r=8, alpha=16, dropout=0.0)
    ref = tiny_qwen(len(tok.itos), seed=3)  # π_ref：与底座同初始化的裸模型
    ref_w = ref.model.layers[0].self_attn.q_proj.weight.clone()
    random.seed(0)
    torch.manual_seed(0)
    h = m.train_dpo(policy, ref, tok, PAIRS * 3, steps=40, batch_size=2, beta=0.3,
                    lr=5e-3, log=lambda *a: None)
    first_margin = sum(h["margin"][:5]) / 5
    last_margin = sum(h["margin"][-5:]) / 5
    assert last_margin > first_margin + 0.5, f"margin 应显著拉开: {first_margin:.3f} -> {last_margin:.3f}"
    assert h["acc"][-1] >= h["acc"][0]
    assert torch.equal(ref_w, ref.model.layers[0].self_attn.q_proj.weight), "π_ref 必须冻结"
