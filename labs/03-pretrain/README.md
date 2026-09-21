# Lab 03: 预训练 20M

> 对应 CS336 Assignment 1 Part 3 / 唐杰课程作业③主线
> 预计耗时：代码半天 + 标准训练过夜（~2h CPU）　|　资源：纯 CPU，<4GB 内存
> 前置：lab01、lab02 全绿（本 lab 直接复用你的 tokenizer 和 TransformerLM）

## 你要做什么

把 lab01 的 tokenizer + lab02 的模型装成一条完整预训练流水线（`train.py` 的 TODO）：

1. `tokenize_corpus()`：全文 → int32 id 数组（按 `<|endoftext|>` 切文档，doc 之间补 eos），带 `.npy` 缓存
2. `get_batch()`：随机窗口取 `(x, y)`，y 与 x 错开一位
3. `evaluate()`：无梯度估 val loss
4. `save_checkpoint()` / `load_checkpoint()`
5. `sample()`：temperature + top-k 自回归采样
6. `train()`：完整训练循环（AdamW + warmup/cosine + grad clip + 定期 eval/checkpoint）

## 文件

| 文件 | 说明 |
|---|---|
| `train.py` | **你要写的文件**，所有 `TODO` 都在这里 |
| `tests/test_pretrain.py` | 测试（用极小切片，分钟级跑完） |
| `solution/train_solution.py` | 参考答案。**先自己写，卡住了再看** |
| `demo_pretrain.py` | 小模型短训练演示：loss 下降 + 采样文本 |
| `../../book/03-pretrain.md` | 原理讲解（先读这个） |

## 通过标准

```bash
make lab3                          # 测试 + 小模型演示
uv run python labs/03-pretrain/train.py   # 标准配置（configs/default.toml），过夜
```

1. pytest 全绿
2. demo 里能看到：loss 从 ~ln(V) 一路下降、采样的文本从乱码变成词 soup
3. 标准训练完成后 `labs/03-pretrain/cache/final.pt` 可加载，采样出"像 TinyStories 的句子"（20M 只训 16M token，能通顺已属超预期，看到 the/said/little 高频词成句即算成功）

## 标注：与真实实验的差异

- 默认在**内置 1.6MB 切片**（2000 个故事）上训练：离线可跑，但 20M 模型对这点数据必然过拟合——train/val 分叉本身就是教学点
- 想要真实 loss：`uv run python data/download.py tinystories` 后把 `configs/default.toml` 的 `[data].train` 改成全量文件，**超参不变**（"只有一个标准配置"的纪律）

## 思考题（demo 末尾会用到）

- 2000 步里 train/val 从第几步开始分叉？
- 采样文本里最高频的 5 个 token 是什么？和语料词频 Top5 对得上吗？
- 同一个 checkpoint，T=0.7 / 1.0 / 1.3 各采一段，哪个"最像英语"？
