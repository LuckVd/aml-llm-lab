"""Lab 04 测试：实现 scaling.py 直到全部通过。"""

import importlib.util
import os

import numpy as np
import pytest

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTION_PATH = os.path.join(LAB_DIR, "solution", "scaling_solution.py")
PRESETS = os.path.abspath(os.path.join(LAB_DIR, "..", "..", "presets"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STUDENT_PATH = (
    SOLUTION_PATH
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1"
    else os.path.join(LAB_DIR, "scaling.py")
)
m = _load("student_scaling", STUDENT_PATH)


@pytest.fixture(scope="module")
def runs():
    n, d, loss = m.load_runs(os.path.join(PRESETS, "scaling_runs.csv"))
    assert len(n) == 56
    return n, d, loss


@pytest.fixture(scope="module")
def fit(runs):
    n, d, loss = runs
    return m.fit_scaling_law(n, d, loss)


# ---------- 基础公式 ----------


def test_flops():
    assert m.flops(1e6, 1e9) == 6e15
    assert m.flops(22.7e6, 1.6e6) == pytest.approx(6 * 22.7e6 * 1.6e6)


def test_chinchilla_loss_matches_reference():
    p = m.REFERENCE
    got = m.chinchilla_loss(np.array([1e6, 1e8]), np.array([1e8, 1e10]), p)
    exp = np.array([
        p["E"] + p["A"] / 1e6**p["alpha"] + p["B"] / 1e8**p["beta"],
        p["E"] + p["A"] / 1e8**p["alpha"] + p["B"] / 1e10**p["beta"],
    ])
    assert np.allclose(got, exp)


def test_loss_monotone_in_n_and_d():
    p = m.REFERENCE
    base = m.chinchilla_loss(1e7, 1e9, p)
    assert m.chinchilla_loss(1e8, 1e9, p) < base  # 模型更大 -> loss 更低
    assert m.chinchilla_loss(1e7, 1e10, p) < base  # 数据更多 -> loss 更低


# ---------- 拟合 ----------


def test_fit_recovers_generating_params(fit):
    """从 2% 噪声的合成数据里把生成参数找回来（±10% / ±0.03）。"""
    r = m.REFERENCE
    assert abs(fit["A"] - r["A"]) / r["A"] < 0.10
    assert abs(fit["B"] - r["B"]) / r["B"] < 0.10
    assert abs(fit["alpha"] - r["alpha"]) < 0.03
    assert abs(fit["beta"] - r["beta"]) < 0.03


def test_fit_predicts_held_out_points(fit, runs):
    n, d, loss = runs
    pred = m.chinchilla_loss(n, d, fit)
    # 拟合曲线对数据点的预测误差中位数应与 2% 噪声同量级
    assert np.median(np.abs(pred - loss)) < 0.08


# ---------- 计算最优 ----------


def test_compute_optimal_tokens_per_param(fit):
    ratios = []
    for budget in (1e17, 1e19, 1e21):
        best = m.compute_optimal(budget, fit)
        assert 10 < best["tokens_per_param"] < 60  # Chinchilla 区间（随预算 20->50）
        assert m.flops(best["n_params"], best["n_tokens"]) <= budget * 1.001
        ratios.append(best["tokens_per_param"])
    assert ratios[0] < ratios[1] < ratios[2]  # 预算越大，每参数配的 token 越多


def test_compute_optimal_is_minimum(fit):
    """网格附近随意扰动都应更差。"""
    best = m.compute_optimal(1e19, fit)
    for k in (0.3, 3.0):
        n2 = best["n_params"] * k
        d2 = 1e19 / (6 * n2)
        assert m.chinchilla_loss(n2, d2, fit) > best["loss"]


def test_mfu():
    # 20M 模型 1.4M tokens/s（A100）：MFU = 6*22.7e6*1.4e6/312e12 ≈ 0.61
    assert 0.5 < m.mfu(22.7e6, 1.4e6) < 0.7
    # lab03 的 CPU 档：20M 模型约 1000 tokens/s，多核 CPU 峰值按 500 GFLOPS 估
    assert 0 < m.mfu(22.7e6, 1000, peak_flops=500e9) < 1
