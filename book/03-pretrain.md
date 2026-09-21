# Lab 03 原理：预训练 20M

> 读顺序建议：本页 → train.py 的 TODO → 卡住再回来查。
> lab01 的 tokenizer + lab02 的 TransformerLM 在这里合体。标准配置见 `configs/default.toml`。

## 1. 预训练在学什么：下一个 token 预测

整门课最反直觉的事实：**把"猜下一个词"做好，推理能力会自己长出来**。
训练数据免费无穷无尽（任何文本天然自带标签：第 i+1 个 token 就是第 i 个的答案），这就是为什么预训练能 scale。

交叉熵损失在每个位置上算 `−log P(下一个 | 前面)`。初始 loss ≈ `ln(vocab_size)` ≈ 9.2（1 万词表均匀猜）；训完大概到 2–3。别小看这个数字：loss 从 3 降到 2，等于每个位置的不确定性从 e³≈20 个候选砍到 e²≈7 个。

## 2. 数据管道：一次性 tokenize，之后全是整数

```
text --(lab01 BPE)--> ids: int32 ndarray --(随机窗口)--> (x, y) batch
```

三个工程点：

- **tokenize 只做一次**，结果缓存成 `.npy`。1.6MB 文本编码约半分钟，2000 步训练读它 2000 次——必须缓存。
- **x 和 y 是同一段 ids 错开一位**：`x = ids[o : o+L]`，`y = ids[o+1 : o+1+L]`。模型在 x 的每个位置预测 y 的对应位，标签不用单独造。
- **随机窗口采样**：每个 batch 在全量 ids 里随机挑 `batch_size` 个起点。故事之间用 `<|endoftext|>` 隔开，模型靠这个 token 学会"一段讲完了"。

## 3. 训练循环解剖

一步 = 六件事：

```
1. get_batch        随机窗口
2. forward          logits = model(x)
3. loss             lab02 的 cross_entropy(logits, y)
4. backward         loss.backward()（autograd 只在这两步出场）
5. clip             梯度范数超过 grad_clip(1.0) 就整体缩回——防爆梯度
6. step             AdamW(lr=lr_at(step))——warmup + cosine 都在调度里
```

优化器直接用 `torch.optim.AdamW`（lab02 已手写过一遍，真跑 2 小时的活交给官方优化实现），`betas=(0.9, 0.95)`、`weight_decay=0.01`。

**train loss vs val loss**：训练集上的 loss 反映"学没学"，验证集上的反映"泛化"。两者开始分叉 = 开始背书（过拟合）。本 lab 的 1.6MB 切片对 20M 模型太小，**必然**过拟合——看着 loss 曲线分叉，这正是作业要你亲眼见到的东西。

## 4. 采样：把概率分布变成文本

```
logits /= temperature          # T<1 更保守（尖峰），T>1 更放飞（平滑）
top_k 之外的位置置 -inf        # 砍掉长尾，防止偶尔蹦出离谱 token
softmax -> multinomial 采样    # argmax（贪心）容易陷入复读循环
```

直观标定：T=0.7 说明书、T=1.0 正常、T=1.3 发酒疯。采样在 `torch.no_grad()` 里做——推理不需要梯度。

## 5. CPU 上的现实账

一步的计算量 ≈ `6 · N · D`（N 参数量，D 每步 token 数）= 6 × 22.7M × 32 × 256 ≈ **1.1 TFLOPs**。
笔记本 CPU 实际吞吐几十到一两百 GFLOPS，所以一步几秒到几十秒，2000 步 = 过夜。这就是"20M 刚好"的原因：再大一个数量级，CPU 就彻底跑不动了（也正好引出 lab04 的 scaling law——怎么在大预算下选模型大小）。

## 6. 带着这些问题去写 TODO

1. `y` 若不与 `x` 错开一位（直接用同一段），模型会学到什么"废招"？
2. 随机窗口为什么比"顺序滑窗"更好？（提示：2000 步顺序滑窗只覆盖语料前 X%）
3. loss 曲线上 train/val 分叉后继续训练，val loss 会怎样？你的 2000 步里分叉发生了吗？
4. 采样时 `temperature → 0` 等价于什么？为什么 demos 里不用它？
5. 把 `<|endoftext|>` 从语料里删掉再训，采样的故事会长成什么样？

写完 pytest 全绿后跑 `demo_pretrain.py`（小模型短训练，看 loss 下降和采样文本），然后 `uv run python labs/03-pretrain/train.py` 过夜跑标准配置。
