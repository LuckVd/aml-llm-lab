"""Lab 05 测试：实现 sft.py 直到全部通过（离线：tiny 随机 Qwen2，不下载模型）。"""

import importlib.util
import os
import string

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTION_PATH = os.path.join(LAB_DIR, "solution", "sft_solution.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STUDENT_PATH = (
    SOLUTION_PATH
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1"
    else os.path.join(LAB_DIR, "sft.py")
)
m = _load("student_sft", STUDENT_PATH)

# 测试语料：模板本身 + 打印字符 + 测试里用到的中文字符
CORPUS = (
    m.PROMPT_TEMPLATE.format(instruction="")
    + m.STOP
    + string.printable
    + "你好巴黎计算：。？＋－×÷＝·"
)


@pytest.fixture(scope="module")
def tok():
    return m.CharTokenizer(CORPUS)


def tiny_qwen(vocab_size, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=96,
    )
    return Qwen2ForCausalLM(cfg)


# ---------- 样本构造 ----------


def test_build_example_masks_prompt(tok):
    ex = m.build_example(tok, "计算：1+1", "1+1=2")
    prompt_ids = tok(m.PROMPT_TEMPLATE.format(instruction="计算：1+1"), add_special_tokens=False)["input_ids"]
    assert ex["input_ids"][: len(prompt_ids)] == prompt_ids
    assert ex["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    # 答案部分可见且以 STOP 结尾
    answer_ids = tok("1+1=2" + m.STOP, add_special_tokens=False)["input_ids"]
    assert ex["labels"][-len(answer_ids):] == answer_ids
    assert ex["input_ids"][-len(answer_ids):] == answer_ids


def test_build_example_truncates_left(tok):
    ex = m.build_example(tok, "a" * 400, "ok", max_len=64)
    assert len(ex["input_ids"]) == 64 == len(ex["labels"])
    assert ex["labels"][-1] != -100  # 答案尾巴保住了


def test_collate_pads(tok):
    ex1 = m.build_example(tok, "hi", "ok")
    ex2 = m.build_example(tok, "hello", "yes")
    batch = m.collate([ex1, ex2], pad_id=tok.pad_token_id)
    t = max(len(ex1["input_ids"]), len(ex2["input_ids"]))
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape == (2, t)
    for i, ex in enumerate((ex1, ex2)):
        n = len(ex["input_ids"])
        assert batch["attention_mask"][i, :n].all() and not batch["attention_mask"][i, n:].any()
        assert (batch["labels"][i, n:] == -100).all()
        assert (batch["input_ids"][i, n:] == tok.pad_token_id).all()


# ---------- 损失 ----------


def test_sft_loss_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.tensor([
        [-100, -100, 3, 4, -100, 5],
        [-100, 2, 7, -100, -100, 1],
    ])
    loss = m.sft_loss(logits, labels)
    # 手工：位置 i 的 logits 预测 labels[i+1]
    pairs = [(0, 1, 3), (0, 2, 4), (0, 4, 5), (1, 0, 2), (1, 1, 7), (1, 4, 1)]
    import torch.nn.functional as F

    per = [F.cross_entropy(logits[b, t : t + 1], labels[b, t + 1 : t + 2]) for b, t, _ in pairs]
    expected = torch.stack(per).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_sft_loss_ignores_prompt_gradient():
    """prompt 位置的 logits 不应产生梯度。"""
    logits = torch.randn(1, 4, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, 2]])
    loss = m.sft_loss(logits, labels)
    loss.backward()
    # 只有位置 2（预测 labels[3]）拿到梯度
    assert logits.grad[0, 2].abs().sum() > 0
    assert logits.grad[0, 0].abs().sum() == 0 and logits.grad[0, 1].abs().sum() == 0


# ---------- LoRA ----------


def test_lora_freezes_base_and_trains_lora(tok):
    model = tiny_qwen(len(tok.itos), seed=1)
    model = m.add_lora(model, r=4, alpha=8, dropout=0.0)
    tr, tot = m.lora_stats(model)
    assert 0 < tr < 0.1 * tot, f"LoRA 可训练参数应 <10%，实际 {tr}/{tot}"
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable_names and all("lora" in n for n in trainable_names)


def test_train_sft_reduces_loss_and_keeps_base_frozen(tok):
    """随机底座 + LoRA 学得慢（LoRA 依赖预训练底座，见 demo 的对照实验），
    这里只验证接线正确：loss 下降且底座参数纹丝不动。"""
    model = tiny_qwen(len(tok.itos), seed=2)
    torch.manual_seed(3)  # 固定 LoRA 初始化的 RNG，避免用例间顺序依赖
    model = m.add_lora(model, r=16, alpha=32, dropout=0.0)
    base_w = model.base_model.model.model.layers[0].self_attn.q_proj.weight.clone()
    examples = [{"instruction": "计算：1+1", "answer": "1+1=2"},
                {"instruction": "计算：2+2", "answer": "2+2=4"}]
    import random

    random.seed(0)  # train_sft 内部用 random.sample 抽 batch，固定用例间顺序
    torch.manual_seed(0)
    losses = m.train_sft(model, tok, examples, steps=200, batch_size=2, lr=2e-2, log=lambda *a: None)
    assert losses[-1] < losses[0] - 0.2, f"loss 应明显下降: {losses[0]:.3f} -> {losses[-1]:.3f}"
    after = model.base_model.model.model.layers[0].self_attn.q_proj.weight
    assert torch.equal(base_w, after), "底座权重必须纹丝不动"


def test_generate_answer_greedy_deterministic(tok):
    model = m.add_lora(tiny_qwen(len(tok.itos), seed=3), r=4, alpha=8, dropout=0.0)
    a1 = m.generate_answer(model, tok, "计算：1+1", max_new_tokens=8, temperature=0.0)
    a2 = m.generate_answer(model, tok, "计算：1+1", max_new_tokens=8, temperature=0.7)
    a3 = m.generate_answer(model, tok, "计算：1+1", max_new_tokens=8, temperature=0.0)
    assert isinstance(a1, str)
    assert a1 == a3  # 贪心确定
    assert m.STOP not in a1  # STOP 及之后被截掉
