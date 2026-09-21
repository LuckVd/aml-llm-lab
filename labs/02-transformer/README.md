# Lab 02: Transformer 组件（从零到 20M 基座）

> 对应 CS336 Assignment 1 Part 1 / 唐杰课程作业②前置
> 预计耗时：2–3 天　|　资源：纯 CPU，<2GB 内存　|　依赖：`torch`（cpu 版）

## 你要做什么

把 lab03 预训练要用的 **20M decoder-only Transformer**（`configs/default.toml`）逐组件手写出来。
torch 只当"张量计算器"用（matmul、softmax 可以用），**归一化 / RoPE / 因果注意力 / AdamW / 调度器全部自己写**：

1. 基础组件：`silu`、`rmsnorm`
2. 位置编码：`precompute_rope_freqs` + `apply_rope`
3. `causal_self_attention`（多头拆装在 `MultiHeadSelfAttention.forward` 里）
4. `swiglu` / `SwiGLUMLP`
5. 整机 `TransformerBlock` / `TransformerLM` + `init_weights_`
6. 训练件：`cross_entropy`、`adamw_step`、`lr_at`

## 文件

| 文件 | 说明 |
|---|---|
| `model.py` | **你要写的文件**，所有 `TODO` 都在这里 |
| `tests/test_model.py` | 测试。全绿 = 完成 |
| `solution/model_solution.py` | 参考答案。**先自己写，卡住了再看** |
| `demo_transformer.py` | 跑通后的演示：参数量表 + 一条真的会降的 loss 曲线 |
| `../../book/02-transformer.md` | 原理讲解（先读这个） |

## 通过标准

```bash
make lab2
```

1. pytest 全绿（RoPE 相对性、注意力因果性、AdamW 对齐 `torch.optim.AdamW` 等都有硬校验）
2. demo 里能看到：20M 基座的参数分解表（总数应恰好 22,696,448），以及一个周期任务上 80 步内 loss 明显下降

## 思考题（demo 末尾会用到）

- 为什么 wo/w2 的初始化要除以 `sqrt(2·n_layers)`，别的矩阵不用？
- attention 里的 scale 若删掉，scores 的方差大约变成多少倍？
- 你的 `adamw_step` 和 `torch.optim.AdamW` 轨迹能对齐到小数点后几位？差在哪一步？
