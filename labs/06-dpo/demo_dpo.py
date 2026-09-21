"""跑通 lab06 后的演示（离线）：把"五五开"的底座掰成偏好正确的模型。

底座先在**歧义语料**上预训练（同一指令后面跟 chosen/rejected 各半），
DPO 之后每对偏好 p(chosen) > p(rejected) 全部翻转，采样行为跟着翻转。
真模型流程相同（dpo.py main，走 LoRA adapter）。

用法: uv run --group posttrain python labs/06-dpo/demo_dpo.py
"""

import copy
import os
import random
import sys

import torch
import torch.nn.functional as F

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
    from solution.dpo_solution import build_preference_batch, sequence_logps, sft, train_dpo  # noqa: E402
else:
    from dpo import build_preference_batch, sequence_logps, sft, train_dpo  # noqa: E402

# 4 条指令；底座预训练语料里 chosen/rejected 各出现一半（歧义只落在最后一个数字上）
DATA = [
    {"instruction": "aa?", "chosen": "aa=1", "rejected": "aa=9"},
    {"instruction": "bb?", "chosen": "bb=2", "rejected": "bb=8"},
    {"instruction": "cc?", "chosen": "cc=3", "rejected": "cc=7"},
    {"instruction": "dd?", "chosen": "dd=4", "rejected": "dd=6"},
]


def make_model(vocab, seed):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=128, max_position_embeddings=96,
    )
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(cfg)


def pretrain(model, ids, steps=300, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss = None
    for _ in range(steps):
        starts = torch.randint(0, len(ids) - 49, (8,))
        x = torch.stack([ids[s:s + 48] for s in starts.tolist()])
        loss = F.cross_entropy(model(input_ids=x).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


@torch.no_grad()
def divergence_probs(model, tok, instruction: str, chosen: str, rejected: str):
    """在 chosen/rejected 分歧的那一位上（本题是最后一个数字），
    返回 (P(chosen 那个字), P(rejected 那个字))。prompt 前缀取两者公共部分。"""
    prefix = os.path.commonprefix([chosen, rejected])  # 如 "aa="
    prompt = sft.PROMPT_TEMPLATE.format(instruction=instruction) + prefix
    ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]], dtype=torch.long)
    probs = torch.softmax(model(input_ids=ids).logits[0, -1], dim=-1)
    return float(probs[tok.stoi[chosen[-1]]]), float(probs[tok.stoi[rejected[-1]]])


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    corpus = "".join(
        sft.PROMPT_TEMPLATE.format(instruction=d["instruction"]) + d[side] + sft.STOP
        for d in DATA for side in ("chosen", "rejected") for _ in range(12)
    )
    tok = sft.CharTokenizer(corpus)
    ids = torch.tensor(tok(corpus)["input_ids"])
    print(f"歧义语料: {len(ids):,} tokens，词表 {len(tok.itos)}（每条指令后面跟对/错答案各半）\n")

    base = make_model(len(tok.itos), seed=1)
    final = pretrain(base, ids)
    print(f"底座预训练 300 步: 最终 LM loss {final:.3f}——指令学会了，最后一个数字仍是'五五开'")

    ref = copy.deepcopy(base)  # π_ref：必须在训练 π 之前深拷贝冻结
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy = copy.deepcopy(base)  # 玩具模型直接全参训练；真模型（main）走 LoRA
    print("DPO 训练（β=0.3，lr=5e-5，100 步）：起步 loss 恰为 ln2≈0.693（π=π_ref）\n")
    random.seed(0)
    torch.manual_seed(0)
    h = train_dpo(policy, ref, tok, DATA, steps=100, batch_size=4, beta=0.3, lr=5e-5)

    print("\n偏好概率搬家（每对序列 log-prob，π 相对 π_ref 的变化）：")
    batch = build_preference_batch(tok, DATA)
    with torch.no_grad():
        pc = sequence_logps(policy, **batch["chosen"]).tolist()
        pr = sequence_logps(policy, **batch["rejected"]).tolist()
        rc = sequence_logps(ref, **batch["chosen"]).tolist()
        rr = sequence_logps(ref, **batch["rejected"]).tolist()
    print(f"  {'指':4} {'logp_ref(c)':>11} {'logp_ref(r)':>11}   ->  {'logp(c)':>8} {'logp(r)':>8}  c>r?")
    flipped = 0
    for i, d in enumerate(DATA):
        ok = pc[i] > pr[i]
        flipped += ok
        print(f"  {d['instruction']:4} {rc[i]:>11.2f} {rr[i]:>11.2f}   ->  {pc[i]:>8.2f} {pr[i]:>8.2f}  {'✓' if ok else '✗'}")
    print(f"  概率排序翻转: {flipped}/{len(DATA)}")

    print("\n行为验证（分歧位上两个候选 token 的概率，DPO 前 -> 后）：")
    for d in DATA:
        c0, r0 = divergence_probs(ref, tok, d["instruction"], d["chosen"], d["rejected"])
        c1, r1 = divergence_probs(policy, tok, d["instruction"], d["chosen"], d["rejected"])
        mark = "翻转" if (c0 <= r0 and c1 > r1) or (c0 > r0 and c1 > r1 and c1 / max(r1, 1e-9) > c0 / max(r0, 1e-9)) else ""
        print(f"  {d['instruction']:4} {d['chosen'][-1]}: {c0:.2f} -> {c1:.2f}   "
              f"{d['rejected'][-1]}: {r0:.2f} -> {r1:.2f}  {mark}")

    print(f"\n注意 logπ(chosen) 的绝对值: 初 {h['chosen_logp'][0]:.2f} -> 末 {h['chosen_logp'][-1]:.2f}——chosen 自己也降了！")
    print("但 rejected 降得更多（见上表），相对排序才是 DPO 优化的一切。")
    print("\n思考题：")
    print("  1. 上表里 chosen 的 logp 为什么也下降了？这就是'概率质量蒸发'——真模型上它对应什么病症？")
    print("  2. 把 β 改成 0.03 / 1.0 重跑，margin 增长速度和 chosen logp 的跌幅各怎么变？")
    print("  3. 若底座只预训练过 rejected（没见过 chosen），DPO 还能翻转吗？（改语料试一试）")


if __name__ == "__main__":
    main()
