# presets/：预置实验数据

> **诚实标注：这里没有一行数据来自真实训练。** 课程没有预算跑 56 个规模的预训练，
> 数据全部由 `generate_presets.py`（seed 固定，可复现）按下述函数形式合成。

| 文件 | 内容 | 生成方式 |
|---|---|---|
| `scaling_runs.csv` | 56 个 (参数量, 训练 token 数, 最终 val loss) | Chinchilla 损失面 `L(N,D) = E + A/N^α + B/D^β`，参数取 Hoffmann et al. 2022 拟合量级 + 2% 对数正态噪声 |
| `systems_benchmark.csv` | 4 个模型 × 4 个 batch 的 tokens/s 与 MFU | 屋顶模型：`tokens/s = peak·MFU/6N`（A100-312TFLOPS），MFU 随 batch/模型大小增大 |

lab04 的作业本质：把这些"别人家实验"的数字**拟合回来**、外推预测——
这正是 scaling law 的实际工作方式（大头时间在跑实验，点睛之笔在拟合与外推）。

想看真数据对照：Hoffmann et al. 2022 *Training Compute-Optimal LLMs* 的 Figure 3 就是同款曲线。
