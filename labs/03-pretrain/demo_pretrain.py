"""跑通 lab03 后的演示：小模型在内置语料上短训练，看 loss 下降与采样文本。

标准 20M 训练不在这里（那是 `uv run python labs/03-pretrain/train.py`，读
configs/default.toml，过夜）。本 demo 用 ~1M 参数的小模型 + 150 步，
验证整条流水线是通的。

用法: uv run python labs/03-pretrain/demo_pretrain.py
"""

import os
import sys
import time
from collections import Counter

import numpy as np
import torch

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(LAB_DIR, "..", ".."))
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "labs", "01-tokenizer"))
sys.path.insert(0, os.path.join(REPO_ROOT, "labs", "02-transformer"))

USE_SOLUTION = os.environ.get("AML_LAB_USE_SOLUTION") == "1"
if USE_SOLUTION:
    from solution.train_solution import (  # noqa: E402
        _lab02, ensure_tokenizer, sample, tokenize_corpus, train,
    )
    lm = _lab02()
else:
    from train import ensure_tokenizer, sample, tokenize_corpus, train  # noqa: E402
    import model as lm  # noqa: E402  （lab02 的，labs/02-transformer 已在 sys.path）

DATA = os.path.join(REPO_ROOT, "data", "sample", "tinystories_sample.txt")
CACHE = os.path.join(LAB_DIR, "cache")
SPECIAL = "<|endoftext|>"

# demo 专用小配置（不是标准训练配置，标准配置见 configs/default.toml）
DEMO = dict(d_model=128, n_layers=2, n_heads=4, d_ff=352, context_length=128,
            steps=150, batch_size=8, seq_len=128, lr=1.2e-3, vocab=1500)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(CACHE, exist_ok=True)

    t0 = time.time()
    tok = ensure_tokenizer(DATA, DEMO["vocab"], [SPECIAL], 1_600_000,
                           cache_path=os.path.join(CACHE, "demo.tok.pkl"))
    eos_id = tok.byte_to_id[SPECIAL.encode()]
    print(f"tokenizer: 词表 {len(tok.vocab)}（缓存于 cache/demo.tok.pkl）")

    ids = tokenize_corpus(DATA, tok, eos_id, cache_path=os.path.join(CACHE, "demo.ids.npy"))
    print(f"语料: {len(ids):,} tokens  {len(tok.vocab)} 词表")
    n_train = int(len(ids) * 0.95)
    train_ids, val_ids = ids[:n_train], ids[n_train:]

    model = lm.TransformerLM(
        vocab_size=len(tok.vocab), d_model=DEMO["d_model"], n_layers=DEMO["n_layers"],
        n_heads=DEMO["n_heads"], d_ff=DEMO["d_ff"], context_length=DEMO["context_length"],
    )
    print(f"demo 模型: {sum(p.numel() for p in model.parameters()):,} 参数，"
          f"训练 {DEMO['steps']} 步 × {DEMO['batch_size']}×{DEMO['seq_len']} tokens/步\n")

    t1 = time.time()
    history = train(
        model, train_ids, val_ids,
        steps=DEMO["steps"], batch_size=DEMO["batch_size"], seq_len=DEMO["seq_len"],
        lr=DEMO["lr"], warmup_steps=15, min_lr_ratio=0.1,
        eval_every=25, eval_iters=10, ckpt_dir=None,
    )
    dt = time.time() - t1
    tps = DEMO["steps"] * DEMO["batch_size"] * DEMO["seq_len"] / dt
    print(f"\n训练 {DEMO['steps']} 步耗时 {dt:.0f}s（{tps:,.0f} tokens/s）\n")
    print("loss 曲线（train / val）:")
    for (s, tr), (_, va) in zip(history["train"], history["val"]):
        bar = "#" * max(1, int(tr * 8))
        print(f"  step {s:>4}  train {tr:.3f}  val {va:.3f}  {bar}")

    print("\n采样（T=1.0, top_k=10）：")
    all_pieces = []
    for prompt in ("Once upon a time", "The little girl", "."):  # "." 充当"文档开头"
        new = sample(model, tok, prompt, max_new_tokens=40, temperature=1.0, top_k=10, eos_id=eos_id)
        text = prompt + tok.decode(new)
        print(f"  [{prompt}] {text!r}")
        all_pieces.extend(tok.vocab[i] for i in new)
    top = Counter(p.decode("utf-8", "replace") for p in all_pieces if len(p) > 1).most_common(5)
    print(f"\n采样输出里最高频的多字节 token: {top}")

    per_step_std = 6 * 22_696_448 * 32 * 256  # 20M 标准配置每步 FLOPs
    print(f"\n换算：20M 标准配置一步 ≈ {per_step_std/1e12:.2f} TFLOPs；"
          f"本机吞吐按上面 tokens/s 折算，2000 步约需过夜。")
    print("标准训练：uv run python labs/03-pretrain/train.py\n")

    print("思考题：")
    print("  1. demo 的 train/val 从哪一步开始分叉？为什么这个语料注定过拟合？")
    print("  2. 采样里最高频 token 与 TinyStories 词频 Top5（the/said/and/a/Lily）对得上吗？")
    print("  3. 把 T 改成 0.7 / 1.3 各采一次，文本风格差在哪？")


if __name__ == "__main__":
    main()
