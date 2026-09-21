"""Lab 06 参考答案：DPO 偏好优化。"""

import argparse
import importlib.util
import os
import random

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
# 本文件在 solution/ 子目录里，路径基准上跳一层到 lab 根
LAB_DIR = os.path.dirname(_HERE) if os.path.basename(_HERE) == "solution" else _HERE


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lab05():
    d = os.path.join(os.path.dirname(LAB_DIR), "05-sft")
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
        return _load_module("lab05_for_dpo", os.path.join(d, "solution", "sft_solution.py"))
    return _load_module("lab05_for_dpo", os.path.join(d, "sft.py"))


sft = _lab05()


def sequence_logps(model, input_ids, attention_mask, labels) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)          # 预测 labels[:, 1:]
    tgt = labels[:, 1:]
    mask = tgt != -100
    safe_tgt = tgt.masked_fill(~mask, 0)
    token_logps = logprobs.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def dpo_loss(pi_chosen_logps, pi_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (pi_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (pi_rejected_logps - ref_rejected_logps)
    margin = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(margin).mean()
    acc = (margin > 0).float().mean()
    return loss, chosen_rewards, rejected_rewards, acc


def build_preference_batch(tokenizer, pairs: list[dict]) -> dict:
    chosen = [sft.build_example(tokenizer, p["instruction"], p["chosen"]) for p in pairs]
    rejected = [sft.build_example(tokenizer, p["instruction"], p["rejected"]) for p in pairs]
    pad = tokenizer.pad_token_id
    return {"chosen": sft.collate(chosen, pad), "rejected": sft.collate(rejected, pad)}


def train_dpo(
    model,
    ref_model,
    tokenizer,
    pairs: list[dict],
    *,
    steps: int = 100,
    batch_size: int = 2,
    beta: float = 0.1,
    lr: float = 5e-4,
    grad_clip: float = 1.0,
    log=print,
) -> dict:
    model.train()
    ref_model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    history = {"loss": [], "margin": [], "acc": [], "chosen_logp": []}

    for step in range(1, steps + 1):
        batch_pairs = random.sample(pairs, min(batch_size, len(pairs)))
        batch = build_preference_batch(tokenizer, batch_pairs)

        pi_c = sequence_logps(model, **batch["chosen"])
        pi_r = sequence_logps(model, **batch["rejected"])
        with torch.no_grad():
            ref_c = sequence_logps(ref_model, **batch["chosen"])
            ref_r = sequence_logps(ref_model, **batch["rejected"])

        loss, r_c, r_r, acc = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        history["loss"].append(loss.item())
        history["margin"].append(float((r_c - r_r).detach().mean()))
        history["acc"].append(float(acc))
        history["chosen_logp"].append(float(pi_c.detach().mean()))
        if step % max(1, steps // 10) == 0 or step == 1:
            log(f"  step {step:>4}  loss {loss.item():.4f}  margin {history['margin'][-1]:+.3f}"
                f"  acc {acc:.2f}  logpi(c) {history['chosen_logp'][-1]:.2f}")
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen2.5-0.5b")
    ap.add_argument("--adapter", default="labs/05-sft/cache/adapter", help="lab05 训好的 SFT adapter")
    ap.add_argument("--data", default="data/sample/dpo_sample.jsonl")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save", default="labs/06-dpo/cache/adapter")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(f"模型目录不存在: {args.model}\n先运行: uv run python data/download.py qwen05b")
    if not os.path.isdir(args.adapter):
        raise SystemExit(f"adapter 不存在: {args.adapter}\n先完成 lab05 的真模型训练")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    ref = PeftModel.from_pretrained(base, args.adapter)  # π_ref = SFT 模型，冻结
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32), args.adapter
    )
    tr, tot = sft.lora_stats(policy)
    print(f"[lab06] π 可训练 {tr:,}/{tot:,}（{tr/tot:.1%}），β={args.beta}")

    pairs = sft.load_jsonl(os.path.join(repo, args.data))
    print(f"[lab06] 偏好数据 {len(pairs)} 对")
    history = train_dpo(policy, ref, tokenizer, pairs, steps=args.steps,
                        beta=args.beta, lr=args.lr)
    last = {k: v[-1] for k, v in history.items()}
    print(f"[lab06] 最终: loss {last['loss']:.4f}  margin {last['margin']:.3f}  acc {last['acc']:.2f}")

    os.makedirs(args.save, exist_ok=True)
    policy.save_pretrained(args.save)
    print(f"[lab06] adapter 已保存: {args.save}")
    for q in ("帮我写一句春节祝福。", "水的化学式是什么？"):
        print(f"  Q: {q}\n  A: {sft.generate_answer(policy, tokenizer, q, max_new_tokens=48)}")


if __name__ == "__main__":
    main()
