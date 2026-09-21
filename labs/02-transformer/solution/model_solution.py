"""Lab 02 参考答案：从零实现的 Transformer 组件。"""

import math

import torch
import torch.nn as nn


# ========== 1. 基础组件 ==========


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms * weight


class RMSNorm(nn.Module):
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
    inv_freq = 1.0 / theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    )
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T = x.shape[2]
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ========== 3. 因果自注意力 ==========


def causal_self_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    B, H, T, hd = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(hd)  # (B, H, T, T)
    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


class MultiHeadSelfAttention(nn.Module):
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
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = causal_self_attention(q, k, v)  # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


# ========== 4. 前馈层：SwiGLU ==========


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return silu(gate) * up


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


# ========== 5. Block 与整机 ==========


class TransformerBlock(nn.Module):
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
        T = idx.shape[1]
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos[:T], self.rope_sin[:T])
        return self.lm_head(self.final_norm(x))


def init_weights_(model: nn.Module, std: float = 0.02) -> None:
    n_layers = len(model.blocks) if hasattr(model, "blocks") else 1
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.fill_(1.0)
            elif name.endswith(("wo.weight", "w2.weight")):
                p.normal_(0.0, std / math.sqrt(2 * n_layers))
            else:
                p.normal_(0.0, std)


# ========== 6. 训练三件套 ==========


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    x_max = logits.max(dim=-1, keepdim=True).values
    log_z = x_max.squeeze(-1) + (logits - x_max).exp().sum(dim=-1).log()
    picked = logits[torch.arange(logits.shape[0], device=logits.device), targets]
    return (log_z - picked).mean()


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
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    param = param * (1 - lr * weight_decay) - lr * m_hat / (v_hat.sqrt() + eps)
    return param, m, v


def lr_at(
    step: int,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    min_lr = base_lr * min_lr_ratio
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(t, 1.0)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * t)) / 2
