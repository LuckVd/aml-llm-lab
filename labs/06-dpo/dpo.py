"""
Lab 06: DPO 偏好优化（你要实现的文件）

复用 lab05 的 build_example / collate / add_lora / CharTokenizer（本文件顶部自动加载）。
π = 待训模型（LoRA 可训练），π_ref = 冻结的参考模型。

全部 TODO 完成并让 tests/test_dpo.py 全绿后，运行 demo_dpo.py。
原理讲解见 ../book/06-dpo.md。
"""

import argparse
import importlib.util
import os

import torch
import torch.nn.functional as F

LAB_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lab05():
    """lab05 的 sft 模块（make verify 时用参考答案）。"""
    d = os.path.join(os.path.dirname(LAB_DIR), "05-sft")
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
        return _load_module("lab05_for_dpo", os.path.join(d, "solution", "sft_solution.py"))
    return _load_module("lab05_for_dpo", os.path.join(d, "sft.py"))


sft = _lab05()  # build_example / collate / add_lora / CharTokenizer / load_jsonl ...


# ========== 1. 序列 log-prob ==========


def sequence_logps(model, input_ids, attention_mask, labels) -> torch.Tensor:
    """batch 内每条序列"答案部分"的 log p(token) 之和。返回 (B,) 张量。

    步骤：forward 取 logits -> log_softmax -> 错位对齐 -> 用 gather 取出
    labels 中非 -100 位置的 log-prob -> 按序列求和。
    """
    # TODO: 五行左右
    raise NotImplementedError


# ========== 2. DPO 损失 ==========


def dpo_loss(
    pi_chosen_logps: torch.Tensor,
    pi_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (loss, chosen_rewards, rejected_rewards, accuracy)。

    隐式奖励 r = beta * (logpi - logpi_ref)；margin = r_c - r_r；
    loss = -logsigmoid(margin) 的均值；accuracy = (margin > 0) 的比例。
    """
    # TODO: 六行
    raise NotImplementedError


# ========== 3. 偏好 batch ==========


def build_preference_batch(tokenizer, pairs: list[dict]) -> dict:
    """pairs: [{"instruction", "chosen", "rejected"}, ...]
    -> {"chosen": collate_batch, "rejected": collate_batch}
    每条用 lab05 的 build_example(tokenizer, instruction, chosen/rejected)。
    """
    # TODO: 五行
    raise NotImplementedError


# ========== 4. 训练循环 ==========


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
    """DPO 训练循环。返回 {"loss": [...], "margin": [...], "acc": [...], "chosen_logp": [...]}。

    每步：
      1. 随机抽 batch_size 条偏好对 -> build_preference_batch
      2. pi 的 chosen/rejected logps（带梯度）；ref 的同样两组（no_grad 且 eval）
      3. dpo_loss -> backward（只更新 pi）-> clip -> step
      4. 记录 loss、margin 均值、acc 均值、pi chosen logp 均值
    """
    # TODO: 二十五行左右
    raise NotImplementedError


# ========== 5. 真模型训练入口（已写好，不用改） ==========


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
