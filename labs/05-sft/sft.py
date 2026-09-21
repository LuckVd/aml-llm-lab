"""
Lab 05: SFT + LoRA（你要实现的文件）

基座：Qwen2.5-0.5B（真训练）；测试与 demo：随机初始化的 tiny Qwen2（离线）。
对话格式用 Qwen 的 ChatML 变体（<|im_start|>...<|im_end|>）。

全部 TODO 完成并让 tests/test_sft.py 全绿后，运行 demo_sft.py（离线），
下载 Qwen 后跑本文件做真训练。原理讲解见 ../book/05-sft.md。
"""

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F

# Qwen2.5 的对话模板（简化版：单轮、无 system）。special tokens 在真 tokenizer 里
# 是特殊 id；离线 CharTokenizer 里就是普通字符串，逻辑完全一致。
PROMPT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
STOP = "<|im_end|>"


class CharTokenizer:
    """离线调试用的字符级 tokenizer（已写好，不是 TODO）。

    只实现 SFT 流水线需要的最小接口，接口形状与 HF tokenizer 对齐：
      tok(text, add_special_tokens=False) -> {"input_ids": [...]}
      tok.decode(ids) -> str
      tok.pad_token_id
    """

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


# ========== 1. 样本构造：prompt 遮罩 ==========


def build_example(tokenizer, instruction: str, answer: str, max_len: int = 512) -> dict:
    """一条 SFT 样本：input_ids = prompt + answer_ids，labels 只在 answer 部分可见。

    返回 {"input_ids": [...], "labels": [...]}（list[int]，长度相同）：
      - prompt 部分的 labels 全部为 -100（cross-entropy 忽略）
      - answer 部分 = tokenizer(answer + STOP) 的 id，labels 与 input_ids 相同
      - 超过 max_len 时从**左侧**截断（保住答案尾巴）
    """
    # TODO:
    #   1. prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    #   2. prompt_ids / answer_ids = tokenizer(..., add_special_tokens=False)["input_ids"]
    #   3. ids = prompt_ids + answer_ids；labels = [-100]*len(prompt_ids) + answer_ids
    #   4. 若 len(ids) > max_len：只保留每个序列最后 max_len 个
    raise NotImplementedError


def collate(batch: list[dict], pad_id: int) -> dict:
    """变长样本 -> 整齐 batch。

    返回 {"input_ids": (B,T) int64, "attention_mask": (B,T) int64, "labels": (B,T) int64}，
    T 取 batch 内最长。pad 位置：input_ids=pad_id、attention_mask=0、labels=-100。
    """
    # TODO: 六行左右（不需要 torch.nn.utils.rnn.pad_sequence，手写循环更清楚）
    raise NotImplementedError


# ========== 2. 损失 ==========


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits (B, T, V)，labels (B, T)（含 -100）。返回标量损失。

    步骤：logits[:, :-1] 预测 labels[:, 1:]；把非 -100 的位置挑出来；
    用 F.cross_entropy(..., reduction="mean") 只在有效位置上平均。
    """
    # TODO: 四行（错位 -> 展平挑有效 -> cross_entropy）
    raise NotImplementedError


# ========== 3. LoRA ==========


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules=None):
    """给 (HF) 模型挂 LoRA，返回 peft 包装后的模型。

    target_modules 默认 ["q_proj", "k_proj", "v_proj", "o_proj"]。
    用 peft.LoraConfig + peft.get_peft_model；bias 训不训选 "none"。
    """
    # TODO: 四行
    raise NotImplementedError


def lora_stats(model) -> tuple[int, int]:
    """返回 (可训练参数量, 总参数量)。"""
    # TODO: 两行
    raise NotImplementedError


# ========== 4. 训练循环 ==========


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
    """mini 训练循环（lab03 的套路搬到 LoRA 上）。

    每步：随机抽 batch_size 条 -> build_example -> collate(pad_id=tokenizer.pad_token_id)
    -> forward 取 logits -> sft_loss -> backward -> clip -> AdamW.step（只挂 requires_grad 的参数）。
    每 steps//10 步 log 一次，返回每步 loss 的 list。
    """
    # TODO: 二十行左右；优化器只收可训练参数
    raise NotImplementedError


# ========== 5. 生成 ==========


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 20,
) -> str:
    """按对话模板采样一个回答。

    编码 PROMPT_TEMPLATE（真 HF tokenizer 用 add_special_tokens=False），
    model.generate(...) 采样，解码时**去掉 prompt 前缀**，截掉 STOP 及其后内容。
    temperature<=0 用贪心（do_sample=False）。
    """
    # TODO: 八行左右
    raise NotImplementedError


# ========== 6. 真模型训练入口（已写好，不用改） ==========


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
