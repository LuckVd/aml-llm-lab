# Lab 01: BPE Tokenizer

> 对应 CS336 Assignment 1 Part 2 / 唐杰课程作业①第一站
> 预计耗时：1–2 天　|　资源：纯 CPU，<1GB 内存　|　依赖：仅 `regex`

## 你要做什么

从零实现 **byte-level BPE**（GPT-2 同款）：

1. `train_bpe()`：在语料上训练一个 BPE tokenizer（学出词表 + 合并规则）
2. `Tokenizer` 类：加载词表，完成文本 ↔ token ID 双向转换

## 文件

| 文件 | 说明 |
|---|---|
| `bpe.py` | **你要写的文件**，所有 `TODO` 都在这里 |
| `tests/test_bpe.py` | 测试。全绿 = 完成 |
| `solution/bpe_solution.py` | 参考答案。**先自己写，卡住了再看** |
| `demo_tokenizer.py` | 跑通后的演示：看你实现的 tokenizer 怎么切分语料 |
| `../book/01-tokenizer.md` | 原理讲解（先读这个） |

## 通过标准

```bash
make lab1
```

1. pytest 全绿
2. demo 输出中能看到：高频词被合并成单个 token、文本能无损还原

## 思考题（demo 末尾会用到）

- 为什么基于 byte 起步就永远不会遇到 OOV？
- 为什么合并要在"预切分"的小片段内部做，而不是全文任意位置？
- vocab_size 调大/调小，分别损失什么？
