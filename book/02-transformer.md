# Lab 02 原理：从零写一个 Transformer

> 读顺序建议：本页 → model.py 的 TODO → 卡住再回来查。
> 目标模型就是 lab03 要预训练的 20M 基座：4 层 / d_model 512 / 16 头 / SwiGLU / RoPE（见 `configs/default.toml`）。

## 1. 整体解剖：decoder-only LM 在算什么

一个 token 序列进去，每个位置输出"下一个 token 的概率分布"：

```
idx (B,T) → embedding → N × [TransformerBlock] → final norm → lm_head → logits (B,T,V)
```

`TransformerBlock` 内部是两个"残差 + 归一化"结构（pre-norm，现代标准做法）：

```
x = x + Attention(norm(x))     # token 之间交换信息
x = x + MLP(norm(x))           # 每个 token 独立地"思考"
```

pre-norm 的意思是：归一化放在子层**里面**的入口，而不是残差路径上。
这样梯度可以沿着 `x = x + ...` 无衰减地直通底层，深网络才训得动。

## 2. RMSNorm：LayerNorm 的减法

LayerNorm 要减均值再除标准差；实践发现**减均值那一步可以省**，只除 RMS（均方根）：

```
rmsnorm(x) = x / sqrt(mean(x²) + eps) * weight
```

少一次统计、快一点，效果不掉——Llama/Qwen 全家都在用。`weight` 是可学习的逐维缩放，初始化为全 1。

## 3. RoPE：用"旋转"编码位置

注意力本身是位置无关的（把序列打乱，attention 输出跟着乱序），必须注入位置信息。
RoPE 的思路：把每个头的向量看成 `head_dim/2` 个二维平面，把第 `m` 个位置的向量在每个平面上旋转 `m·θ_i` 角（不同平面转速 θ_i 不同，从 1 到 1/theta 几何递减）。

魔法在于：**旋转是正交变换**。`q_m · k_n = (R_m q)·(R_n k) = q·(R_{n-m} k)`——内积只依赖**相对位置 n−m**。位置信息进了注意力，又天然具备平移不变性。

实现用"rotate-half"约定（Qwen 同款）：把向量劈成两半 `x1|x2`，

```
out = [x1·cos − x2·sin ,  x1·sin + x2·cos]      # cos/sin 形状 (T, head_dim/2)
```

注意：**只给 q、k 旋转，v 不转**（v 是被加权求和的"内容"，不是"寻址"的钥匙）。

## 4. 因果自注意力

每个 token 只能看过去。三步：

```
scores = q @ k^T / sqrt(head_dim)      # (B,H,T,T)
scores 屏蔽 j > i 的位置（置 -inf，softmax 后为 0）
out = softmax(scores) @ v
```

除以 `sqrt(head_dim)` 是防止点积随维度增大而方差爆炸（d=512 时点积方差 ~512，softmax 会退化成 one-hot）。
多头 = 把 d_model 切成 H 份各自做注意力再拼回来——H 组不同的"关注模式"并行。

## 5. SwiGLU 前馈层

普通 FFN 是 `W2(activation(W1 x))`。SwiGLU 加了一条"门控"支路：

```
out = W2( silu(W1 x) ⊙ (W3 x) ),   silu(x) = x · sigmoid(x)
```

⊙ 是逐元素乘：W3 支路提供"内容"，silu(W1 x) 支路决定"放行多少"。
参数从 2 个矩阵变 3 个，所以 d_ff 取 8/3·d_model ≈ 1344 保持总参数量持平（1344 = 4/3 × 512 × 2 的舍入）。

## 6. 初始化：0.02 和那个 sqrt(2L)

所有 Linear/Embedding 权重 ~ N(0, 0.02²)。例外：**每个 block 的两个出口投影**（注意力的 `wo`、MLP 的 `w2`）用 `0.02 / sqrt(2·n_layers)`——残差是逐层累加的，出口方差若不随深度收缩，叠加 L 层后激活方差会 ~L 倍增长，训练不稳。

## 7. 训练三件套（lab03 直接复用）

**cross-entropy**：logits 先做 log-sum-exp 稳定化再减目标 logit：
`loss = mean(logsumexp(logits) − logits[target])`。写错这里，训练会"莫名"出 NaN。

**AdamW**：一阶/二阶动量 + 偏差修正 + **解耦**权重衰减：

```
m ← β1·m + (1−β1)·g ;  v ← β2·v + (1−β2)·g²
p ← p·(1−lr·λ) − lr · m̂/(√v̂ + eps)
```

和 Adam 的区别：衰减直接作用在参数上，不经过动量（Adam 把 λ·p 塞进梯度会让自适应步长把衰减也"除走"）。

**lr schedule**：linear warmup（前 warmup 步从 0 线性升到 base）→ cosine 衰减到 min_lr_ratio·base。冷启动直接上大 lr，AdamW 二阶动量还没攒起来，步长会爆炸。

## 8. 带着这些问题去写 TODO

1. `apply_rope` 之后向量模长变了吗？为什么这很重要（提示：内积=模长×cos 夹角）。
2. 屏蔽用 `-inf` 而不是 0，为什么？（softmax 前还是后的区别）
3. v 为什么不套 RoPE？试着给 v 也转一下，猜猜 loss 会怎样。
4. `adamw_step` 里 `t` 从 1 还是 0 开始数，对偏差修正意味着什么？
5. 数一遍 20M 基座的确切参数量（词表 10000：两 个 5.12M 的 embedding 矩阵 + 4 层 × 3.11M），和 demo 输出对一下。

写完 pytest 全绿后跑 `demo_transformer.py`，看参数量表和一条真的会下降的 loss 曲线。
