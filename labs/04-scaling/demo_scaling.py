"""跑通 lab04 后的演示：拟合曲线、计算最优前沿、定位 lab03 的点。

用法: uv run python labs/04-scaling/demo_scaling.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
    from solution.scaling_solution import (  # noqa: E402
        REFERENCE as REF, chinchilla_loss, compute_optimal, fit_scaling_law, flops, load_runs, mfu,
    )
else:
    from scaling import (  # noqa: E402
        REFERENCE as REF, chinchilla_loss, compute_optimal, fit_scaling_law, flops, load_runs, mfu,
    )

PRESETS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "presets"))


def main() -> None:
    n, d, loss = load_runs(os.path.join(PRESETS, "scaling_runs.csv"))

    fit = fit_scaling_law(n, d, loss)
    print("拟合结果 vs 生成真值（Hoffmann et al. 2022 量级）：")
    print(f"  {'':10} {'拟合':>10} {'真值':>10}")
    for k in ("E", "A", "alpha", "B", "beta"):
        print(f"  {k:10} {fit[k]:>10.4f} {REF[k]:>10.4f}")

    pred = chinchilla_loss(n, d, fit)
    print(f"  中位拟合误差: {np.median(np.abs(pred - loss)):.4f} nat\n")

    print("计算最优前沿（6ND = C）：")
    print(f"  {'预算 (FLOPs)':>14} {'最优 N':>12} {'最优 D':>14} {'D/N':>7} {'最优 loss':>9}")
    lab03_c = flops(22.7e6, 1.6e6)
    for c in (lab03_c, 1e17, 1e19, 1e21):
        best = compute_optimal(c, fit)
        tag = "  <-- lab03 的预算" if c == lab03_c else ""
        print(f"  {c:>14.2e} {best['n_params']:>12.3e} {best['n_tokens']:>14.3e}"
              f" {best['tokens_per_param']:>7.1f} {best['loss']:>9.3f}{tag}")

    # lab03 的实际位置 vs 同预算最优
    lab03_loss = float(chinchilla_loss(22.7e6, 1.6e6, fit))
    best = compute_optimal(lab03_c, fit)
    print(f"\nlab03 点位: N=22.7M, D=1.6M  (D/N={1.6e6/22.7e6:.2f}，计算最优是 {best['tokens_per_param']:.0f})")
    print(f"  该点预测 loss {lab03_loss:.3f} vs 同预算最优 {best['loss']:.3f}"
          f"——浪费 {lab03_loss - best['loss']:.2f} nat（D 严重不足的教学代价）")

    print("\nASCII 图：最优 loss vs 预算（幂律下包络）")
    cs = np.logspace(13.2, 21, 40)
    stars = [compute_optimal(float(c), fit)["loss"] for c in cs]
    lo, hi = min(stars), max(stars)
    for c, s in zip(cs[::4], stars[::4]):
        bar = "#" * max(1, round((s - lo) / (hi - lo) * 44))
        print(f"  C={c:>10.2e}  L*={s:.3f}  {bar}")

    bench = np.genfromtxt(os.path.join(PRESETS, "systems_benchmark.csv"),
                          delimiter=",", names=True, dtype=None, encoding="utf-8")
    print("\n系统侧 MFU（屋顶模型合成数据，A100-312TF）：")
    for row in bench:
        calc = mfu(float(row["n_params"]), float(row["tokens_per_sec"]), float(row["peak_flops"]))
        match = "OK" if abs(calc - row["mfu"]) < 0.02 else "MISMATCH"
        print(f"  {row['model']:>5} batch={int(row['batch_size']):>4}  tokens/s={int(row['tokens_per_sec']):>9,}  MFU={calc:.2f} [{match}]")

    print("\n思考题：")
    print("  1. 预算从 1e17 到 1e19 放大 100 倍，最优 N 放大了多少倍？（幂指数 = log(放大)/2）")
    print("  2. lab03 若把 1.6MB 切片换成全量 TinyStories（~2B token），D/N 变成多少？还缺多少？")
    print("  3. 按 100 GFLOPS 峰值算，你 lab03 实测 ~3 万 tokens/s 的 CPU MFU 是多少？")


if __name__ == "__main__":
    main()
