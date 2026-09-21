"""生成 lab04 用的预置实验数据（预计算，非真实训练）。

诚实声明：本课程没有预算真的跑 56 个不同规模的预训练。
两个 CSV 都是用公开发表的函数形式合成的：

  scaling_runs.csv      L(N, D) = E + A/N^alpha + B/D^beta，
                        参数取 Hoffmann et al. 2022（Chinchilla）的拟合量级，
                        加 2% 对数正态噪声（seed=0，可复现）。
  systems_benchmark.csv 屋顶模型：tokens/s = peak_FLOPS x MFU / (6N)，
                        MFU 随 batch 增大而提高（小 batch 算子喂不满 GPU）。

用法: uv run python presets/generate_presets.py
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Chinchilla 量级的"真值"（生成用；lab04 里学员要把它拟合回来）
E, A, ALPHA, B, BETA = 1.70, 406.4, 0.34, 410.7, 0.28


def scaling_runs() -> list[dict]:
    rng = np.random.default_rng(0)
    n_grid = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9])
    d_grid = np.array([1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10])
    rows = []
    rid = 0
    for n in n_grid:
        for d in d_grid:
            loss = E + A / n**ALPHA + B / d**BETA
            loss *= float(np.exp(rng.normal(0, 0.02)))  # 2% 对数正态噪声
            rows.append({
                "run_id": f"run{rid:03d}",
                "n_params": int(n),
                "n_tokens": int(d),
                "train_flops": int(6 * n * d),
                "val_loss": round(loss, 4),
            })
            rid += 1
    return rows


def systems_benchmark() -> list[dict]:
    rng = np.random.default_rng(1)
    peak = 312e12  # A100-80G bf16 峰值
    rows = []
    for n, name in [(7e6, "7M"), (22.7e6, "20M"), (1.4e8, "140M"), (7e9, "7B")]:
        for batch in (8, 32, 128, 512):
            seq = 256
            # MFU：batch 越大算子越喂得满（log2 增），模型越大同 batch 下 MFU 略高
            mfu = 0.05 + 0.075 * np.log2(batch) + 0.05 * np.log10(n / 1e6)
            mfu = float(np.clip(mfu, 0.05, 0.62))
            mfu *= float(np.exp(rng.normal(0, 0.02)))
            tps = peak * mfu / (6 * n)  # tokens/s
            rows.append({
                "model": name,
                "n_params": int(n),
                "batch_size": batch,
                "seq_len": seq,
                "peak_flops": int(peak),
                "mfu": round(mfu, 4),
                "tokens_per_sec": int(tps),
            })
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{os.path.relpath(path)}: {len(rows)} 行")


if __name__ == "__main__":
    write_csv(os.path.join(HERE, "scaling_runs.csv"), scaling_runs())
    write_csv(os.path.join(HERE, "systems_benchmark.csv"), systems_benchmark())
