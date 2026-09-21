"""跑通 lab05 后的演示（离线）：对照实验——随机底座 vs 预训练底座，LoRA 各学一遍。

结论预告：LoRA 在随机底座上几乎学不动；先做 150 步玩具预训练（lab03 的缩影），
同样的 LoRA SFT 立刻能逐字背出指令答案。**LoRA 的前提是底座有像样的表征。**

真模型流程相同：下载 Qwen 后 `uv run --group posttrain python labs/05-sft/sft.py`。

用法: uv run --group posttrain python labs/05-sft/demo_sft.py
"""

import os
import random
import sys

import torch
import torch.nn.functional as F

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
    from solution.sft_solution import (  # noqa: E402
        PROMPT_TEMPLATE, STOP, CharTokenizer, add_lora, build_example, generate_answer,
        lora_stats, train_sft,
    )
else:
    from sft import (  # noqa: E402
        PROMPT_TEMPLATE, STOP, CharTokenizer, add_lora, build_example, generate_answer,
        lora_stats, train_sft,
    )

# 4 条短指令（离线演示用）
DATA = [
    {"instruction": "a+b?", "answer": "a+b=2"},
    {"instruction": "c-d?", "answer": "c-d=5"},
    {"instruction": "e*f?", "answer": "e*f=6"},
    {"instruction": "g/h?", "answer": "g/h=7"},
]


def make_model(vocab, seed):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=128, max_position_embeddings=96,
    )
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(cfg)


def pretrain_tiny(model, ids: torch.Tensor, steps: int = 150, lr: float = 1e-3) -> list[float]:
    """lab03 的缩影：全参数普通 LM 训练（非 TODO，对照实验用）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(1, steps + 1):
        starts = torch.randint(0, len(ids) - 49, (8,))
        x = torch.stack([ids[s:s + 48] for s in starts.tolist()])
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    corpus = "".join(
        PROMPT_TEMPLATE.format(instruction=d["instruction"]) + d["answer"] + STOP for d in DATA
    ) * 20 + "abcdefghijklmnopqrstuvwxyz0123456789" * 4  # 干扰字符：抬高"瞎猜"的词表熵
    tok = CharTokenizer(corpus)
    ids = torch.tensor(tok(corpus)["input_ids"])
    print(f"玩具语料: {len(ids):,} tokens，词表 {len(tok.itos)}，4 条指令\n")

    # ---- A. 随机底座 ----
    model_a = add_lora(make_model(len(tok.itos), seed=1), r=8, alpha=16, dropout=0.0)
    tr, tot = lora_stats(model_a)
    print(f"[A 随机底座] 总参数 {tot:,}，LoRA 可训练 {tr:,}（{tr/tot:.1%}）")
    torch.manual_seed(0)
    losses_a = train_sft(model_a, tok, DATA, steps=100, batch_size=4, lr=5e-3, log=lambda *a: None)
    ans_a = generate_answer(model_a, tok, "a+b?", max_new_tokens=12, temperature=0.0)
    print(f"  LoRA SFT 100 步: loss {losses_a[0]:.3f} -> {losses_a[-1]:.3f}（下降有限）")
    print(f"  生成 'a+b?' -> {ans_a!r}（乱码）\n")

    # ---- B. 先预训练底座，再同样的 LoRA ----
    base_b = make_model(len(tok.itos), seed=1)
    pre = pretrain_tiny(base_b, ids, steps=150)
    print(f"[B 预训练底座] 150 步玩具预训练: loss {pre[0]:.3f} -> {pre[-1]:.3f}")
    model_b = add_lora(base_b, r=8, alpha=16, dropout=0.0)
    torch.manual_seed(0)
    losses_b = train_sft(model_b, tok, DATA, steps=100, batch_size=4, lr=5e-3, log=lambda *a: None)
    print(f"  同样的 LoRA SFT 100 步: loss {losses_b[0]:.3f} -> {losses_b[-1]:.3f}")
    for d in DATA + [{"instruction": "x+y?"}]:
        tag = "（训练内）" if d in DATA else "（没见过）"
        ans = generate_answer(model_b, tok, d["instruction"], max_new_tokens=12, temperature=0.0)
        print(f"  Q: {d['instruction']} {tag}  A: {ans!r}")

    ex = build_example(tok, DATA[0]["instruction"], DATA[0]["answer"])
    valid = sum(1 for x in ex["labels"] if x != -100)
    print(f"\n遮罩统计：一条样本 {len(ex['labels'])} 个 token，只有 {valid} 个参与损失"
          f"（{valid/len(ex['labels']):.0%}）——SFT 的有效信号天生稀疏")
    print("\n真训练（下载后执行）：")
    print("  uv run python data/download.py qwen05b && uv run python data/download.py alpaca")
    print("  uv run --group posttrain python labs/05-sft/sft.py --data data/raw/alpaca_data.json --steps 2000")
    print("\n思考题：")
    print("  1. A 组为什么学不动？把'LoRA 省算力'和'LoRA 需要好底座'两件事连起来说一遍。")
    print("  2. 'x+y?' 没见过，B 组答了什么？这算泛化吗？")
    print("  3. 遮罩 prompt 后有效 token 只占 ~20%，梯度信号为什么被稀释 5 倍？")


if __name__ == "__main__":
    main()
