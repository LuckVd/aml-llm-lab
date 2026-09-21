"""Lab 04 参考答案：Scaling Law 拟合。"""

import numpy as np
import torch

REFERENCE = {"E": 1.70, "A": 406.4, "alpha": 0.34, "B": 410.7, "beta": 0.28}


def flops(n_params: float, n_tokens: float) -> float:
    return 6.0 * n_params * n_tokens


def chinchilla_loss(n, d, params: dict):
    return params["E"] + params["A"] / np.power(n, params["alpha"]) + params["B"] / np.power(d, params["beta"])


def load_runs(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return np.asarray(data["n_params"], dtype=np.float64), \
        np.asarray(data["n_tokens"], dtype=np.float64), \
        np.asarray(data["val_loss"], dtype=np.float64)


def fit_scaling_law(
    n: np.ndarray, d: np.ndarray, loss: np.ndarray,
    E: float = 1.70, iters: int = 8000, lr: float = 0.05, seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    n_t = torch.as_tensor(n, dtype=torch.float64)
    d_t = torch.as_tensor(d, dtype=torch.float64)
    y_t = torch.as_tensor(loss, dtype=torch.float64)

    log_a = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # A 初值 1
    log_b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    alpha = torch.full((1,), 0.3, dtype=torch.float64, requires_grad=True)
    beta = torch.full((1,), 0.3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_a, log_b, alpha, beta], lr=lr)

    for _ in range(iters):
        pred = E + torch.exp(log_a) / n_t**alpha + torch.exp(log_b) / d_t**beta
        # 对数域 MSE：原始 MSE 会被高 loss 点（小模型）主导，拟合参数系统性跑偏
        mse = (torch.log(pred) - torch.log(y_t)).pow(2).mean()
        opt.zero_grad()
        mse.backward()
        opt.step()

    return {
        "E": E,
        "A": float(log_a.detach().exp()),
        "alpha": float(alpha.detach()),
        "B": float(log_b.detach().exp()),
        "beta": float(beta.detach()),
    }


def compute_optimal(budget_flops: float, params: dict) -> dict:
    ns = np.logspace(5, 10, 500)
    ds = budget_flops / (6.0 * ns)
    losses = chinchilla_loss(ns, ds, params)
    i = int(np.argmin(losses))
    return {
        "n_params": float(ns[i]),
        "n_tokens": float(ds[i]),
        "loss": float(losses[i]),
        "tokens_per_param": float(ds[i] / ns[i]),
    }


def mfu(n_params: float, tokens_per_sec: float, peak_flops: float = 312e12) -> float:
    return 6.0 * n_params * tokens_per_sec / peak_flops
