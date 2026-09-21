"""Lab 07 测试：实现 grpo.py 直到全部通过。"""

import importlib.util
import math
import os

import pytest
import torch
import torch.nn.functional as F

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB02_DIR = os.path.join(os.path.dirname(LAB_DIR), "02-transformer")
SOLUTION_PATH = os.path.join(LAB_DIR, "solution", "grpo_solution.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


USE_SOLUTION = os.environ.get("AML_LAB_USE_SOLUTION") == "1"


def _pick(lab_dir, student, solution):
    return os.path.join(lab_dir, solution) if USE_SOLUTION else os.path.join(lab_dir, student)


lab02 = _load("lab02_for_grpo_test", _pick(LAB02_DIR, "model.py", "solution/model_solution.py"))
m = _load("lab07_for_test", _pick(LAB_DIR, "grpo.py", "solution/grpo_solution.py"))


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return lab02.TransformerLM(vocab_size=len(m.CHARSET), d_model=48, n_layers=2, n_heads=3,
                               d_ff=96, context_length=24)


# ---------- 任务与判定器 ----------


def test_make_problems_levels():
    p1 = m.make_problems(1, n=32, seed=0)
    assert len(p1) == 32
    assert all(len(q["prompt"]) == 4 for q in p1)  # "3+4="
    assert all(len(q["answer"]) == 1 for q in p1)
    for q in p1:
        a, b = q["prompt"][0], q["prompt"][2]
        assert q["answer"] == str(int(a) + int(b))
    p2 = m.make_problems(2, n=8, seed=0)
    assert all(len(q["prompt"]) == 6 for q in p2)  # "07+35="
    assert all(len(q["answer"]) == 3 for q in p2)


def test_reward():
    assert m.reward("7", "7") == 1.0
    assert m.reward(" 7 ", "7") == 1.0
    assert m.reward("8", "7") == 0.0
    assert m.reward("", "7") == 0.0


# ---------- 组相对优势 ----------


def test_group_advantages_basics():
    r = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 对应 adv: +0.87, -0.87, +0.87, -0.87
    a = m.group_advantages(r)
    assert abs(a.mean().item()) < 1e-6
    assert a[0] > 0 and a[1] < 0 and a[2] > 0 and a[3] < 0
    # 全对/全错 -> 零信号
    assert torch.equal(m.group_advantages(torch.ones(8)), torch.zeros(8))
    assert torch.equal(m.group_advantages(torch.zeros(8)), torch.zeros(8))


# ---------- rollout 与 logps ----------


def test_rollout_shapes_and_logps_valid():
    model = tiny_model(1)
    prompt = torch.tensor([m.encode("3+4=")], dtype=torch.long)
    actions, logps = m.rollout(model, prompt, group_size=6, answer_len=1, temperature=1.0)
    assert actions.shape == (6, 1) and logps.shape == (6, 1)
    assert (logps <= 0).all()  # log 概率非正
    assert actions.min() >= 0 and actions.max() < len(m.CHARSET)


def test_token_logps_matches_rollout():
    """µ=1 时 new==old：同权重下重算的 logps 应与 rollout 记录一致。"""
    torch.manual_seed(0)
    model = tiny_model(2)
    prompt = torch.tensor([m.encode("2+3=")], dtype=torch.long)
    actions, old_lp = m.rollout(model, prompt, group_size=4, answer_len=1, temperature=1.0)
    new_lp = m.token_logps(model, prompt, actions)
    assert torch.allclose(new_lp, old_lp, atol=1e-5)
    # 且有梯度
    assert new_lp.requires_grad


# ---------- grpo_loss ----------


def test_grpo_loss_zero_advantage_and_reference():
    z = torch.zeros(4, 3)
    a = torch.zeros(4, 3)
    loss, kl, ratio = m.grpo_loss(z, z, z, a, beta=0.1)
    assert loss.item() == 0.0 and kl == 0.0 and abs(ratio - 1.0) < 1e-6


def test_grpo_loss_formula_and_clip():
    new = torch.zeros(2, 2, requires_grad=True)
    old = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    ref = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    adv = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    loss, kl, _ = m.grpo_loss(new, old, ref, adv, beta=0.5)
    # ratio=1: policy = -mean(adv) = 0；kl = mean(new - ref) = 1.0
    assert abs(loss.item() - 0.5 * 1.0) < 1e-6
    assert abs(kl - 1.0) < 1e-6

    # clip 分支：ratio 冲出上界且 adv>0 -> 用 clamp 后的 ratio（clamp 的是 ratio 本身，[0.8, 1.2]）
    new2 = torch.full((1, 1), 5.0)  # ratio = e^5
    old2 = torch.zeros(1, 1)
    loss2, _, ratio2 = m.grpo_loss(new2, old2, old2, torch.ones(1, 1), beta=0.0, clip_eps=0.2)
    assert abs(loss2.item() - (-1.2)) < 1e-6  # -clamp(e^5, 0.8, 1.2) * 1
    assert ratio2 > 100  # 真实 ratio 确实爆了


# ---------- 端到端：reward 从随机水平起飞 ----------


def test_grpo_learns_level1():
    """端到端：毛坯底座（三成正确率的数字汤）+ GRPO，reward 应明显上升。

    经验值：β=0.05 + lr=1e-3 是稳定区（β→0 会熵塌缩，lr→3e-3 会漂移，见 book 第 4 节）。
    """
    torch.manual_seed(0)
    model = m.pretrain_soup(steps=150, seed=0, log=lambda *a: None)
    ref = lab02.TransformerLM(vocab_size=len(m.CHARSET), d_model=128, n_layers=2, n_heads=4,
                              d_ff=384, context_length=24)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    problems = m.make_problems(level=1, n=64, seed=1)
    h = m.train_grpo(model, ref, problems, steps=250, prompts_per_step=8,
                     group_size=8, lr=1e-3, beta=0.05, log=lambda *a: None)
    first10 = sum(h["reward"][:10]) / 10
    last10 = sum(h["reward"][-10:]) / 10
    assert last10 > first10 + 0.1, f"reward 应上升: {first10:.2f} -> {last10:.2f}"
    acc = m.evaluate_accuracy(model, problems[:32])
    assert acc > 0.25, f"贪心准确率应明显高于随机，实际 {acc:.0%}"
