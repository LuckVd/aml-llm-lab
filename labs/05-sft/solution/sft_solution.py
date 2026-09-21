"""Lab 05 参考答案：SFT + LoRA。"""

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F

PROMPT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
STOP = "<|im_end|>"


class CharTokenizer:
    def __init__(self, corpus: str):
        chars = ["<pad>"] + sorted(set(corpus))
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [self.stoi[c] for c in text]}

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_example(tokenizer, instruction: str, answer: str, max_len: int = 512) -> dict:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + STOP, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(ids) > max_len:  # 左截断，保住答案尾巴
        ids, labels = ids[-max_len:], labels[-max_len:]
    return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    t = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attn, labels = [], [], []
    for ex in batch:
        n = len(ex["input_ids"])
        pad = t - n
        input_ids.append(ex["input_ids"] + [pad_id] * pad)
        attn.append([1] * n + [0] * pad)
        labels.append(ex["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    valid = shift_labels != -100
    return F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction="mean")


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules=None):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def lora_stats(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_sft(
    model,
    tokenizer,
    examples: list[dict],
    *,
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 1e-4,
    warmup_ratio: float = 0.03,
    grad_clip: float = 1.0,
    log=print,
) -> list[float]:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    warmup = max(1, int(steps * warmup_ratio))
    losses = []
    for step in range(1, steps + 1):
        batch_raw = [build_example(tokenizer, e["instruction"], e["answer"])
                     for e in random.sample(examples, min(batch_size, len(examples)))]
        batch = collate(batch_raw, tokenizer.pad_token_id)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = sft_loss(logits, batch["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.param_groups[0]["lr"] = lr * min(1.0, step / warmup)
        opt.step()
        losses.append(loss.item())
        if step % max(1, steps // 10) == 0 or step == 1:
            log(f"  step {step:>4}  loss {loss.item():.4f}")
    return losses


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 20,
) -> str:
    model.eval()
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    enc = tokenizer(prompt, add_special_tokens=False)
    ids = torch.tensor([enc["input_ids"]], dtype=torch.long)
    do_sample = temperature > 0
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        kwargs.update(temperature=temperature, top_k=top_k)
    out = model.generate(input_ids=ids, **kwargs)
    text = tokenizer.decode(out[0][ids.shape[1]:].tolist())
    stop = text.find(STOP)
    return text[:stop] if stop != -1 else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen2.5-0.5b", help="本地 Qwen 路径（先跑 data/download.py qwen05b）")
    ap.add_argument("--data", default="data/sample/sft_sample.jsonl")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save", default="labs/05-sft/cache/adapter")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(f"模型目录不存在: {args.model}\n先运行: uv run python data/download.py qwen05b")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    torch.manual_seed(0)
    random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model = add_lora(model, r=16, alpha=32)
    tr, tot = lora_stats(model)
    print(f"[lab05] LoRA 挂载完成：可训练 {tr:,}/{tot:,} ({tr/tot:.1%})")

    raw = load_jsonl(os.path.join(repo, args.data))
    examples = [(d.get("instruction") or d["input"], d["output"] if "output" in d else d["answer"])
                for d in raw]
    pairs = [{"instruction": i, "answer": a} for i, a in examples]
    print(f"[lab05] 数据 {len(pairs)} 条（真训练建议换 alpaca：data/download.py alpaca）")
    losses = train_sft(model, tokenizer, pairs, steps=args.steps,
                       batch_size=args.batch_size, lr=args.lr)
    print(f"[lab05] 最终 loss {losses[-1]:.4f}")

    os.makedirs(args.save, exist_ok=True)
    model.save_pretrained(args.save)
    print(f"[lab05] adapter 已保存: {args.save}")
    for q in ("水的化学式是什么？", "计算：3 + 4 = ?", "帮我写一句生日祝福。"):
        print(f"  Q: {q}\n  A: {generate_answer(model, tokenizer, q, max_new_tokens=48)}")


if __name__ == "__main__":
    main()
