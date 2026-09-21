"""
Lab 04: Scaling Law 拟合（你要实现的文件）

数据在 presets/scaling_runs.csv（合成，见 presets/README.md）。
本 lab 不训练任何模型：拟合 + 外推 + 找计算最优配置。

全部 TODO 完成并让 tests/test_scaling.py 全绿后，运行 demo_scaling.py。
原理讲解见 ../book/04-scaling.md。
"""

import numpy as np
import torch

# presets/generate_presets.py 里用的"文献量级"参数（作业就是把它拟合回来）
REFERENCE = {"E": 1.70, "A": 406.4, "alpha": 0.34, "B": 410.7, "beta": 0.28}


def flops(n_params: float, n_tokens: float) -> float:
    """训练 FLOPs 的标准近似：前向 2ND + 反向 2 倍前向 = 6ND。"""
    # TODO: 一行
    raise NotImplementedError


def chinchilla_loss(n: np.ndarray, d: np.ndarray, params: dict) -> np.ndarray:
    """L(N, D) = E + A/N^alpha + B/D^beta。n/d 可以是标量或数组。"""
    # TODO: 一行（按公式实现，支持 numpy 广播）
    raise NotImplementedError


def load_runs(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读 presets/scaling_runs.csv，返回 (n_params, n_tokens, val_loss) 三个数组。"""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return np.asarray(data["n_params"], dtype=np.float64), \
        np.asarray(data["n_tokens"], dtype=np.float64), \
        np.asarray(data["val_loss"], dtype=np.float64)


def fit_scaling_law(
    n: np.ndarray, d: np.ndarray, loss: np.ndarray,
    E: float = 1.70, iters: int = 8000, lr: float = 0.05, seed: int = 0,
) -> dict:
    """用梯度下降拟合 (A, alpha, B, beta)，E 当给定常数。

    做法（book 第 2 节）：
      1. 可训练张量：logA、logB、alpha、beta（torch, requires_grad=True）
      2. 预测 L̂ = E + exp(logA)/N^alpha + exp(logB)/D^beta（N、D 转成 torch 张量）
      3. 损失 = mean((log L̂ - log L)^2)  ← 对数域！原始 MSE 会被高 loss 点主导
      4. Adam(lr) 迭代 iters 步（默认 8000，秒级）
      5. 返回 {"E": E, "A": float, "alpha": float, "B": float, "beta": float}
    """
    # TODO: 十二行左右
    raise NotImplementedError


def compute_optimal(budget_flops: float, params: dict) -> dict:
    """给定算力预算 C，在约束 6ND = C 下最小化 L。

    做法：N 在对数网格（如 1e5 .. 1e10，500 个点）上扫，
    D = C / (6N)，算 L，取最小。返回：
      {"n_params": N*, "n_tokens": D*, "loss": L*, "tokens_per_param": D*/N*}
    """
    # TODO: 七八行
    raise NotImplementedError


def mfu(n_params: float, tokens_per_sec: float, peak_flops: float = 312e12) -> float:
    """模型产生 tokens 的 FLOPs 利用率：6N·tokens/s / peak。"""
    # TODO: 一行
    raise NotImplementedError
