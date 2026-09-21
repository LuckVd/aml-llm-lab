"""跑通 lab02 后的演示：20M 基座参数分解 + 一条真的会降的 loss 曲线。

用法: uv run python labs/02-transformer/demo_transformer.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# AML_LAB_USE_SOLUTION=1（make verify）时演示参考答案，否则演示你自己的实现
if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
    from solution.model_solution import TransformerLM, cross_entropy, lr_at, adamw_step  # noqa: E402
else:
    from model import TransformerLM, cross_entropy, lr_at, adamw_step  # noqa: E402

BASE_CONFIG = dict(
    vocab_size=10000, d_model=512, n_layers=4, n_heads=16, d_ff=1344, context_length=256
)


def param_table(model: torch.nn.Module) -> None:
    groups = {"embedding": 0.0, "attention": 0.0, "mlp": 0.0, "norm": 0.0, "lm_head": 0.0}
    for name, p in model.named_parameters():
        if name == "tok_emb.weight":
            key = "embedding"
        elif ".attn." in name:
            key = "attention"
        elif ".mlp." in name:
            key = "mlp"
        elif name.endswith("norm.weight"):
            key = "norm"
        else:
            key = "lm_head"
        groups[key] += p.numel()
    total = sum(groups.values())
    print(f"20M 基座（{BASE_CONFIG}）参数分解：")
    for k, v in groups.items():
        bar = "#" * int(v / total * 40)
        print(f"  {k:10} {v:>12,}  {v/total:5.1%} {bar}")
    print(f"  {'total':10} {int(total):>12,}\n")


def learn_a_pattern() -> None:
    torch.manual_seed(0)
    model = TransformerLM(vocab_size=64, d_model=64, n_layers=2, n_heads=4,
                          d_ff=176, context_length=32)
    pattern = torch.randint(0, 64, (8,))
    seq = pattern.repeat(64)

    def batch(bs=32):
        starts = torch.randint(0, 56, (bs,))
        x = torch.stack([seq[s:s+32] for s in starts.tolist()])
        return x[:, :-1], x[:, 1:]

    params = list(model.parameters())
    ms = [torch.zeros_like(p) for p in params]
    vs = [torch.zeros_like(p) for p in params]
    print("在周期为 8 的随机序列上训练 80 步（loss 应从 ~ln64≈4.16 快速下降）：")
    t0 = time.time()
    losses = []
    for step in range(1, 81):
        x, y = batch()
        loss = cross_entropy(model(x).reshape(-1, 64), y.reshape(-1))
        grads = torch.autograd.grad(loss, params)
        lr = lr_at(step, 3e-3, 80, 10)
        for i, (p, g) in enumerate(zip(params, grads)):
            new_p, ms[i], vs[i] = adamw_step(p, g, ms[i], vs[i], t=step, lr=lr)
            with torch.no_grad():
                p.copy_(new_p)
        losses.append(loss.item())
        if step % 10 == 0 or step == 1:
            print(f"  step {step:>3}  loss {loss.item():.3f}  lr {lr:.2e}")
    print(f"  训练耗时 {time.time()-t0:.1f}s\n")

    # 用训好的模型滚动预测：看它是否学会了"下一个一定是 pattern 里的那个"
    model.eval()
    with torch.no_grad():
        ctx = seq[:16].unsqueeze(0)
        for _ in range(12):
            nxt = model(ctx[:, -16:])[0, -1].argmax().view(1)
            ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
        pred = ctx[0, 16:].tolist()
        gold = seq[16:28].tolist()
    ok = sum(a == b for a, b in zip(pred, gold))
    print(f"滚动预测 12 步命中 {ok}/12（pattern = {pattern.tolist()}）")
    print("  预测:", pred)
    print("  真值:", gold)


def lr_curve() -> None:
    base, total, warmup = 6e-4, 1000, 100
    pts = [lr_at(s, base, total, warmup) for s in range(0, total + 1, 10)]
    print("\nlr 调度（linear warmup 100 步 → cosine 衰减到 10%）：")
    lo, hi = 0.0, base
    for s, lr in zip(range(0, total + 1, 10), pts):
        n = int((lr - lo) / (hi - lo) * 50)
        print(f"  step {s:>4}  {lr:.2e}  {'*' * max(n, 0)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    param_table(TransformerLM(**BASE_CONFIG))
    learn_a_pattern()
    lr_curve()
    print("\n思考题：")
    print("  1. wo/w2 初始化除以 sqrt(2L)——把 demo 里的总 std 改成统一 0.02 再训，loss 还稳吗？")
    print("  2. 中文 corpus 上 20M 模型的 loss 大概在什么量级？lab03 见分晓。")
