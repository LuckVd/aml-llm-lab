# AML-LLM-Lab

0 成本、CPU 优先的 LLM 全链路学习脚手架。
作业结构对标 Stanford CS336（*Language Modeling from Scratch*），内容覆盖清华唐杰《高级机器学习》课程的五大作业方向，整体缩微到消费级 CPU 上可完成。

## 双基座

| 基座 | 来源 | 承担 |
|---|---|---|
| 20M（4 层 / d_model 512 / 16 头） | 你在 lab03 亲手从零训练 | 预训练、scaling law、GRPO/RLVR |
| Qwen2.5-0.5B | HuggingFace 下载 | SFT、DPO、Agent |

## 学习路线（8 个 lab）

| Lab | 主题 | 基座 | 预计耗时 |
|---|---|---|---|
| 01 | BPE Tokenizer | — | 1–2 天 |
| 02 | Transformer 组件 | — | 2–3 天 |
| 03 | 预训练 20M | 自训 | 过夜（~2h CPU） |
| 04 | Scaling Law（预置实验数据拟合） | presets | 半天 |
| 05 | SFT（LoRA） | Qwen 0.5B | 一晚 |
| 06 | DPO | Qwen 0.5B | 一个长周末 |
| 07 | GRPO / RLVR（学算术） | 自训 20M | 一晚 |
| 08 | Agent + self-judge | Qwen 0.5B | 2–3 天 |

## 快速开始

```bash
make setup          # uv sync（清华 PyPI 镜像）
make test           # 全部测试（当前只有 lab01）
make lab1           # 跑 lab01 测试 + 演示
```

## 目录

- `book/` 原理讲解（可视化 → 公式人话 → 最小实现）
- `labs/` 动手作业（TODO 驱动 + pytest + `solution/` 参考答案）
- `data/sample/` 预置语料（TinyStories 切片，与 CS336 同源，离线可跑）
- `data/download.py` 大语料/模型权重下载（hf-mirror 镜像）
- `presets/` 预计算实验结果（scaling 曲线、多卡 benchmark 等）
- `burst/` 可选租卡升级脚本（非必需）

## 纪律

- 每个 lab：先做 TODO，pytest 绿了再看 `solution/`
- 所有训练只有一个标准配置，不设快速档；调试用极小切片手工验证
- 凡是预置数据替代真实实验的地方，README 会明确标注
