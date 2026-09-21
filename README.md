# AML-LLM-Lab

0 成本、CPU 优先的 LLM 全链路学习脚手架。
作业结构对标 Stanford CS336（*Language Modeling from Scratch*），内容覆盖清华唐杰《高级机器学习》课程的五大作业方向，整体缩微到消费级 CPU 上可完成。

## 双基座

| 基座 | 来源 | 承担 |
|---|---|---|
| 20M（4 层 / d_model 512 / 16 头，实际 22,696,448 参数） | 你在 lab03 亲手从零训练 | 预训练、scaling law、GRPO/RLVR |
| Qwen2.5-0.5B | HuggingFace 下载 | SFT、DPO、Agent |

## 学习路线（8 个 lab，全部就绪）

| Lab | 主题 | 基座 | 预计耗时 | 测试 |
|---|---|---|---|---|
| 01 | BPE Tokenizer | — | 1–2 天 | 10 |
| 02 | Transformer 组件 | — | 2–3 天 | 14 |
| 03 | 预训练 20M | 自训 | 代码半天 + 过夜（~2h CPU） | 8 |
| 04 | Scaling Law（预置实验数据拟合） | presets | 半天 | 8 |
| 05 | SFT（LoRA） | Qwen 0.5B | 一晚 | 8 |
| 06 | DPO | Qwen 0.5B | 一个长周末 | 7 |
| 07 | GRPO / RLVR（学算术） | 小模型（可接 20M） | 一晚 | 8 |
| 08 | Agent + self-judge | Qwen 0.5B | 2–3 天 | 12 |

## 快速开始

```bash
make setup          # uv sync（清华 PyPI 镜像 + pytorch-cpu 源）
make test           # 学员视角：全部测试（没做完 TODO 的 lab 会红，属预期）
make verify         # 出题人视角：对参考答案跑全部测试 + 全部演示，应当永远全绿
make lab1           # 跑 lab01 测试 + 演示（lab2..lab8 同理）
```

lab05/06/08 需要 `uv sync --group posttrain`（transformers + peft）；真模型权重与大数据先跑 `uv run python data/download.py`（hf-mirror 镜像，见脚本头部）。

## 目录

- `book/` 原理讲解（可视化 → 公式人话 → 最小实现），01–08 每 lab 一章
- `labs/` 动手作业（TODO 驱动 + pytest + `solution/` 参考答案 + demo）
- `data/sample/` 预置语料（TinyStories 切片 + SFT/DPO 指令样例，离线可跑）
- `data/download.py` 大语料/模型权重下载（hf-mirror 镜像）
- `presets/` 预计算实验结果（**合成数据**，scaling 曲线与系统 benchmark，生成脚本可复现，详见其 README）
- `configs/default.toml` 唯一标准训练配置（模型/数据/超参）
- `burst/` 可选租卡升级 runbook（非必需）

## 纪律

- 每个 lab：先做 TODO，pytest 绿了再看 `solution/`
- 所有训练只有一个标准配置，不设快速档；调试用极小切片手工验证（tests/demo 里就是这么干的）
- 凡是预置数据替代真实实验的地方，README 明确标注（目前：lab04 的 presets 全部合成；lab05-08 的测试与 demo 用离线小模型/剧本替身，真模型训练需下载权重）
- lab 之间的依赖按编号递进：03 用 01+02 的成果，06 用 05 的 adapter，07 可选接 03 的 checkpoint
