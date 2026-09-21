# burst/：可选租卡升级 runbook（非必需）

整门课在 CPU 上都能完成（这是设计目标）。如果你想在几个小时内跑完"过夜"档的实验，
这里是租卡的最短路径。**不提供自动下单脚本**（各平台 API 变动频繁），只给清单。

## 租什么

| 平台 | 参考价位（2026） | 适合 |
|---|---|---|
| AutoDL / 恒源云（国内） | 4090 ~¥1.5/h | lab03/05/06 全流程 |
| vast.ai（海外） | 4090 ~$0.35/h | 同上，需外币支付 |
| Colab 免费档 | T4 | lab03 缩短版（够用） |

一张 24GB 消费卡对本课程的所有实验都严重过剩——20M 模型 + batch 32×256 在 4090 上
约 3k tokens/s（对齐 `presets/systems_benchmark.csv` 的量级），lab03 两小时变两分钟。

## 环境（租到机器后）

```bash
# GPU 版 torch（把 pyproject 的 cpu 源换成 cu124 即可）
uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install numpy regex pytest transformers peft
# 数据与模型
uv run python data/download.py tinystories
uv run python data/download.py qwen05b
```

## 实验对齐注意

- GPU 上跑 lab03：`configs/default.toml` 超参**不变**（课程纪律），只是快
- lab05/06 用 `--steps` 拉满（CPU 默认值是保守档）：SFT 2000 步、DPO 500 步在 4090 上各 <1h
- 结果请回填到各 lab README 的"实测"栏，并标注硬件——CPU 与 GPU 的对比本身就是作业
