"""Lab 02 测试：实现 model.py 直到全部通过。"""

import importlib.util
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTION_PATH = os.path.join(LAB_DIR, "solution", "model_solution.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 出题人自检：AML_LAB_USE_SOLUTION=1 时直接对参考答案跑测试（make verify）。
STUDENT_PATH = (
    SOLUTION_PATH
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1"
    else os.path.join(LAB_DIR, "model.py")
)
m = _load("student_model", STUDENT_PATH)

# 20M 基座的标准配置（configs/default.toml）
BASE_CONFIG = dict(
    vocab_size=10000, d_model=512, n_layers=4, n_heads=16, d_ff=1344, context_length=256
)

torch.manual_seed(0)


# ---------- 基础组件 ----------


def test_silu():
    x = torch.linspace(-6, 6, 101)
    assert torch.allclose(m.silu(x), F.silu(x), atol=1e-6)
    assert m.silu(torch.tensor(0.0)).item() == 0.0


def test_rmsnorm():
    x = torch.randn(5, 3, 512)
    w = torch.randn(512)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    assert torch.allclose(m.rmsnorm(x, w), ref, atol=1e-5)
    # weight 全 1 时输出 RMS ≈ 1
    out = m.rmsnorm(x, torch.ones(512))
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(5, 3), atol=1e-3)


# ---------- RoPE ----------


def test_rope_preserves_norm():
    x = torch.randn(2, 3, 16, 32)
    cos, sin = m.precompute_rope_freqs(32, 16)
    out = m.apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)


def test_rope_position_zero_unrotated():
    x = torch.randn(2, 3, 16, 32)
    cos, sin = m.precompute_rope_freqs(32, 16)
    out = m.apply_rope(x, cos, sin)
    assert torch.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6)  # cos0=1, sin0=0


def test_rope_relative_property():
    """内积只依赖相对位置——RoPE 的灵魂。"""
    T, hd = 24, 32
    cos, sin = m.precompute_rope_freqs(hd, T)
    q0 = torch.randn(hd)
    k0 = torch.randn(hd)
    q = q0.view(1, 1, 1, hd).expand(1, 1, T, hd).clone()
    k = k0.view(1, 1, 1, hd).expand(1, 1, T, hd).clone()
    qr = m.apply_rope(q, cos, sin)[0, 0]
    kr = m.apply_rope(k, cos, sin)[0, 0]
    dots = qr @ kr.T  # (T, T)
    for delta in (1, 3, 7):
        ref = dots[0, delta]
        for i in range(T - delta):
            assert torch.allclose(dots[i, i + delta], ref, atol=1e-4), (delta, i)


# ---------- 注意力 ----------


def test_attention_matches_torch():
    q, k, v = (torch.randn(2, 4, 12, 32) for _ in range(3))
    out = m.causal_self_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_attention_is_causal():
    """未来变了，过去不许变。"""
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 2, 10, 16) for _ in range(3))
    out1 = m.causal_self_attention(q, k, v)
    v2 = v.clone()
    v2[:, :, -1] += 100.0  # 扰动最后一个位置（未来）
    k2 = k.clone()
    k2[:, :, -1] += 100.0
    out2 = m.causal_self_attention(q, k2, v2)
    assert torch.allclose(out1[:, :, :-1], out2[:, :, :-1], atol=1e-4)
    assert not torch.allclose(out1[:, :, -1], out2[:, :, -1], atol=1e-2)


def test_swiglu():
    g, u = torch.randn(4, 8), torch.randn(4, 8)
    assert torch.allclose(m.swiglu(g, u), F.silu(g) * u, atol=1e-6)


# ---------- 初始化 / 整机 ----------


def test_init_weights():
    torch.manual_seed(2)
    model = m.TransformerLM(vocab_size=97, d_model=64, n_layers=4, n_heads=4, d_ff=176)
    names = dict(model.named_parameters())
    assert torch.all(names["final_norm.weight"] == 1)
    for i in range(4):
        wo_std = names[f"blocks.{i}.attn.wo.weight"].std().item()
        wq_std = names[f"blocks.{i}.attn.wq.weight"].std().item()
        expect_small = 0.02 / math.sqrt(2 * 4)
        assert abs(wo_std - expect_small) / expect_small < 0.15
        assert abs(wq_std - 0.02) / 0.02 < 0.15


def test_model_shapes_and_param_count():
    model = m.TransformerLM(**BASE_CONFIG, do_init=False)
    n = sum(p.numel() for p in model.parameters())
    assert n == 22_696_448, f"20M 基座参数量应为 22,696,448，实际 {n}"
    idx = torch.randint(0, BASE_CONFIG["vocab_size"], (2, 16))
    logits = model(idx)
    assert logits.shape == (2, 16, BASE_CONFIG["vocab_size"])
    # 全网络无 bias
    assert all(name.endswith(".weight") for name, _ in model.named_parameters())


# ---------- 训练三件套 ----------


def test_cross_entropy():
    torch.manual_seed(3)
    logits = torch.randn(64, 100)
    targets = torch.randint(0, 100, (64,))
    assert torch.allclose(m.cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-5)


def test_adamw_matches_torch():
    torch.manual_seed(4)
    p0 = torch.randn(50)
    grads = [torch.randn(50) * 0.5 for _ in range(30)]

    # torch 侧
    p_torch = p0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([p_torch], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    for g in grads:
        opt.zero_grad()
        p_torch.grad = g.clone()
        opt.step()

    # 手写侧
    p, mm, vv = p0.clone(), torch.zeros(50), torch.zeros(50)
    for t, g in enumerate(grads, start=1):
        p, mm, vv = m.adamw_step(p, g, mm, vv, t=t, lr=1e-2, beta1=0.9, beta2=0.95,
                                  eps=1e-8, weight_decay=0.01)
    assert torch.allclose(p, p_torch.detach(), atol=1e-6)


def test_lr_schedule():
    base, total, warmup = 6e-4, 1000, 100
    assert m.lr_at(0, base, total, warmup) == 0.0
    assert abs(m.lr_at(warmup, base, total, warmup) - base) < 1e-12
    assert abs(m.lr_at(total, base, total, warmup) - 0.1 * base) < 1e-9
    assert abs(m.lr_at(total + 999, base, total, warmup) - 0.1 * base) < 1e-9
    lrs = [m.lr_at(s, base, total, warmup) for s in range(warmup, total + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))  # warmup 后单调不增


# ---------- 集成：小模型真的能学 ----------


def test_tiny_model_learns():
    torch.manual_seed(5)
    model = m.TransformerLM(vocab_size=64, d_model=64, n_layers=2, n_heads=4,
                            d_ff=176, context_length=32)
    # 数据：长度 8 的随机 pattern 无限循环（可完全预测）
    pattern = torch.randint(0, 64, (8,))
    seq = pattern.repeat(64)

    def batch(bs=32):
        i = torch.randint(0, 56, (bs,))
        x = torch.stack([seq[i:i+32] for i in i.tolist()])
        return x[:, :-1], x[:, 1:]

    params = list(model.parameters())
    ms = [torch.zeros_like(p) for p in params]
    vs = [torch.zeros_like(p) for p in params]
    first = last = None
    for step in range(1, 81):
        x, y = batch()
        logits = model(x)
        loss = m.cross_entropy(logits.reshape(-1, 64), y.reshape(-1))
        grads = torch.autograd.grad(loss, params)
        lr = m.lr_at(step, 3e-3, 80, 10)
        for i, (p, g) in enumerate(zip(params, grads)):
            new_p, ms[i], vs[i] = m.adamw_step(p, g, ms[i], vs[i], t=step, lr=lr)
            with torch.no_grad():
                p.copy_(new_p)
        if step == 1:
            first = loss.item()
        last = loss.item()
    assert first - last > 1.5, f"80 步应至少降 1.5 nat（实际 {first:.3f} -> {last:.3f}）"
