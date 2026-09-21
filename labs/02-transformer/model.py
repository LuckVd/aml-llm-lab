"""
Lab 02: 从零实现 Transformer（你要实现的文件）

torch 在本 lab 里只是"张量计算器"：matmul / softmax / 基本算子可用，
归一化、RoPE、因果注意力、AdamW、lr 调度全部自己写。

全部 TODO 完成并让 tests/test_model.py 全绿后，运行 demo_transformer.py。
卡住了再看 solution/model_solution.py。原理讲解见 ../book/02-transformer.md。
"""

import math

import torch
import torch.nn as nn


# ========== 1. 基础组件 ==========


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU（swish）激活：x * sigmoid(x)。"""
    # TODO: 一行
    raise NotImplementedError


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm：x / sqrt(mean(x^2) + eps) * weight。

    x 形状 (..., d_model)，weight 形状 (d_model,) 逐维缩放。
    在最后一个维度上做统计。
    """
    # TODO: 两行——先求 rms 归一化，再乘 weight
    raise NotImplementedError


class RMSNorm(nn.Module):
    """把 rmsnorm 包成模块（weight 可学习，初始为全 1）。"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)


# ========== 2. 位置编码：RoPE ==========


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos/sin 表。

    返回 (cos, sin)，形状都是 (max_seq_len, head_dim // 2)。
    第 i 个"转速"满足 inv_freq[i] = 1 / theta^(2i/head_dim)（i = 0..head_dim/2-1），
    位置 m 的第 i 维旋转角 = m * inv_freq[i]。
    """
    # TODO:
    #   1. inv_freq = 1 / theta ** (arange(0, head_dim, 2) / head_dim)，float 类型
    #   2. 位置序列 m = arange(max_seq_len)
    #   3. 外积 freqs[m, i] = m * inv_freq[i]（提示：torch.outer）
    #   4. 返回 (freqs.cos(), freqs.sin())
    raise NotImplementedError


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """给 q 或 k 套 RoPE（rotate-half 约定，Qwen 同款）。

    x 形状 (B, n_heads, T, head_dim)；cos/sin 形状 (T, head_dim // 2)。
    把 x 劈成两半 x1|x2（各 head_dim/2 宽），输出：
        [x1*cos - x2*sin ,  x1*sin + x2*cos]
    注意 cos/sin 需要广播到 (1, 1, T, head_dim//2)。
    """
    # TODO: chunk 成两半 → 旋转 → cat 回去（4 行以内）
    raise NotImplementedError


# ========== 3. 因果自注意力 ==========


def causal_self_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """单头版因果注意力（多头在外面拆装）。

    q/k/v 形状 (B, n_heads, T, head_dim)，返回同形状。
    步骤：
      1. scores = q @ k^T / sqrt(head_dim)        -> (B, H, T, T)
      2. 屏蔽未来：scores[..., i, j] (j > i) 置 -inf
         （提示：torch.full 上三角，或 scores.masked_fill(mask, float("-inf"))，
           mask 可用 torch.tril(torch.ones(T, T, dtype=torch.bool)) 取反）
      3. softmax（最后一个维度）
      4. 加权求和 @ v
    """
    # TODO: 四步，提示见 docstring
    raise NotImplementedError


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力：投影 → 拆头 → q/k 套 RoPE → 因果注意力 → 并头 → 出投影。"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)。cos/sin: (T, head_dim//2)。"""
        B, T, _ = x.shape
        # TODO:
        #   1. q/k/v = 各自投影，view(B, T, H, head_dim).transpose(1, 2) -> (B, H, T, hd)
        #   2. q, k 套 apply_rope（v 不套！）
        #   3. causal_self_attention
        #   4. transpose(1, 2).reshape(B, T, d_model) 并头，过 wo 返回
        raise NotImplementedError


# ========== 4. 前馈层：SwiGLU ==========


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """门控：silu(gate) * up。gate/up 是同一输入经两个不同矩阵的投影。"""
    # TODO: 一行
    raise NotImplementedError


class SwiGLUMLP(nn.Module):
    """W2( silu(W1 x) ⊙ (W3 x) )，三个矩阵都无 bias。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate 支路
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up 支路
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # 出投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


# ========== 5. Block 与整机 ==========


class TransformerBlock(nn.Module):
    """pre-norm 残差结构（已写好，不用改）：

        x = x + Attn(norm(x))
        x = x + MLP(norm(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TransformerLM(nn.Module):
    """decoder-only 语言模型，与 configs/default.toml 对应：

        idx (B,T) -> tok_emb -> blocks -> final_norm -> lm_head -> logits (B,T,V)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 16,
        d_ff: int = 1344,
        context_length: int = 256,
        rope_theta: float = 10000.0,
        do_init: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        cos, sin = precompute_rope_freqs(d_model // n_heads, context_length, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if do_init:
            self.reset_parameters()

    def reset_parameters(self):
        init_weights_(self)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # TODO: embedding -> 逐 block（cos/sin 按 T 截断：self.rope_cos[:T]）-> final_norm -> lm_head
        raise NotImplementedError


def init_weights_(model: nn.Module, std: float = 0.02) -> None:
    """GPT-2 风格初始化（就地修改）：

    - 名字以 norm.weight 结尾（RMSNorm）：置全 1
    - 名字以 wo.weight / w2.weight 结尾（每个 block 的两个出口投影）：N(0, (std/sqrt(2*n_layers))^2)
      （n_layers 从 model.blocks 数出来；若没有 blocks 属性则退化为 std）
    - 其余权重（wq/wk/wv/w1/w3/tok_emb/lm_head）：N(0, std^2)

    提示：遍历 model.named_parameters()，用 endswith 分流后 .normal_(0, ...) / .fill_(1.0)。
    注意就地修改 requires_grad 的参数要包在 torch.no_grad() 里。
    """
    # TODO
    raise NotImplementedError


# ========== 6. 训练三件套（lab03 直接复用） ==========


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """标量交叉熵（数值稳定版），与 F.cross_entropy(logits, targets) 完全一致。

    logits (N, V)，targets (N,) 的 int64。实现必须走 logsumexp，不能先 exp。
    返回标量：mean over N。
    """
    # TODO: logZ = logsumexp(logits, dim=-1)；loss = mean(logZ - logits[arange(N), targets])
    #       （logsumexp 请手写：x.max() + log(sum(exp(x - max)))，别直接调 torch.logsumexp）
    raise NotImplementedError


def adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    *,
    t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AdamW 单参数单步更新（纯函数，返回新值，不就地修改）。

    与 torch.optim.AdamW 的单步完全对齐：
      m' = β1 m + (1-β1) g ;  v' = β2 v + (1-β2) g²
      m̂ = m'/(1-β1^t)     ;  v̂ = v'/(1-β2^t)        （t 从 1 开始数）
      p' = p·(1-lr·λ) - lr · m̂/(√v̂ + eps)
    """
    # TODO: 按公式六行
    raise NotImplementedError


def lr_at(
    step: int,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """linear warmup + cosine decay。

    - step < warmup_steps:  base_lr * step / warmup_steps（step=0 时为 0）
    - warmup <= step <= total: 从 base 余弦衰减到 min_lr_ratio * base_lr，
      进度 t = (step - warmup) / max(1, total - warmup)，
      lr = min + (base - min) * (1 + cos(pi t)) / 2
    - step > total: 维持在 min
    """
    # TODO: 三段，各一到两行
    raise NotImplementedError
