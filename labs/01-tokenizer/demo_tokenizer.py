"""跑通 lab01 后的演示：看你训练的 tokenizer 如何切分文本。

用法: uv run python labs/01-tokenizer/demo_tokenizer.py
"""

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpe import Tokenizer, train_bpe  # noqa: E402

LAB = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(LAB, "..", "..", "data", "sample", "tinystories_sample.txt")
SPECIALS = ["<|endoftext|>"]


def main() -> None:
    t0 = time.time()
    vocab, merges = train_bpe(DATA, vocab_size=1000, special_tokens=SPECIALS)
    print(f"训练完成：词表 {len(vocab)}，{len(merges)} 条合并规则，耗时 {time.time() - t0:.1f}s\n")

    print("前 10 条合并规则（最高频的字节对先合并）：")
    for a, b in merges[:10]:
        print(f"  {a!r:12} + {b!r:12} -> {(a + b)!r}")

    print("\n词表里最长的 5 个 token：")
    longest = sorted((v for v in vocab.values() if len(v) > 1), key=len, reverse=True)[:5]
    for tok in longest:
        print(f"  {tok!r}  (len={len(tok)})")

    tok = Tokenizer(vocab, merges, SPECIALS)
    samples = [
        "Once upon a time, there was a little girl named Lucy.",
        "The beautiful butterfly flew over the shiny pond.",
        "你好，世界！BPE 也能无损表示中文。",
    ]
    print("\n切分演示（| 分隔 token 边界）：")
    for s in samples:
        ids = tok.encode(s)
        pieces = [tok.vocab[i] for i in ids]
        n_multi = sum(1 for p in pieces if len(p) > 1)
        shown = "".join(
            (p.decode("utf-8", errors="replace") if len(p) > 1 else "·" + p.decode("latin1"))
            + "|"
            for p in pieces
        )
        print(f"\n  原文: {s}")
        print(f"  {len(s.encode('utf-8'))} bytes -> {len(ids)} tokens（其中 {n_multi} 个多字节 token）")
        print(f"  切分: {shown}")

    print("\n思考题：")
    print("  1. 对比三句的压缩率——英文 vs 中文为什么差这么多？")
    print("  2. 把 vocab_size 改成 500 / 3000 重跑，切分方式怎么变？")


if __name__ == "__main__":
    main()
