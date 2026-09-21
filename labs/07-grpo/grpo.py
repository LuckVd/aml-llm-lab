"""
Lab 07: GRPO / RLVR（你要实现的文件）

任务：算术。没有监督答案，reward = 答案是否精确正确（可机器验证）。
模型：lab02 的 TransformerLM（本文件顶部自动加载），字符级词表。

流程：数字汤预训练（pretrain_soup，已写好）-> 你实现的 rollout/loss ->
train_grpo 循环 -> reward 从随机水平起飞。

原理讲解见 ../book/07-grpo.md。
"""

import argparse
import importlib.util
import os

import numpy as np
import torch
import torch.nn.functional as F

LAB_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lab02():
    d = os.path.join(os.path.dirname(LAB_DIR), "02-transformer")
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
        return _load_module("lab02_for_grpo", os.path.join(d, "solution", "model_solution.py"))
    return _load_module("lab02_for_grpo", os.path.join(d, "model.py"))


lm = _lab02()  # TransformerLM / cross_entropy

CHARSET = "0123456789+= "  # 字符级词表（id 即下标）


def encode(text: str) -> list[int]:
    table = {c: i for i, c in enumerate(CHARSET)}
    return [table[c] for c in text]


def decode(ids) -> str:
    return "".join(CHARSET[i] for i in ids)


# ========== 1. 任务与判定器 ==========


def make_problems(level: int = 1, n: int = 64, seed: int = 0) -> list[dict]:
    """生成 n 道题。level 1: 0..4 + 0..4，prompt 'a+b='，answer 一位。
    level 2: 0..99 + 0..99，prompt 零填充 '07+35='，answer 补零到三位。
    返回 [{"prompt": "3+4=", "answer": "7"}, ...]"""
    rng = np.random.default_rng(seed)
    hi = 5 if level == 1 else 100
    width = 1 if level == 1 else 2
    ans_width = 1 if level == 1 else 3
    problems = []
    for a, b in zip(rng.integers(0, hi, n), rng.integers(0, hi, n)):
        prompt = f"{a:0{width}d}+{b:0{width}d}="
        problems.append({"prompt": prompt, "answer": f"{a + b:0{ans_width}d}"})
    return problems


def reward(answer_text: str, expected: str) -> float:
    """验证器：精确匹配得 1.0，否则 0.0。strip 空白。"""
    # TODO: 两行
    raise NotImplementedError


# ========== 2. GRPO 核心 ==========


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """A_i = (r_i - mean) / (std + 1e-4)。全对/全错组应返回全 0。"""
    # TODO: 三行（记得处理 std=0 的组）
    raise NotImplementedError


@torch.no_grad()
def rollout(model, prompt_ids: torch.Tensor, group_size: int, answer_len: int,
            temperature: float = 1.0):
    """同一道题（prompt_ids (1, T) 重复成 (G, T)）采样 G 个答案。

    每步：取最后一个位置的 logits / temperature -> log_softmax ->
    按该分布采样 next token，记录其 log-prob，append 到序列。
    返回 (actions (G, answer_len) int64, logps (G, answer_len) float)。
    """
    # TODO: 循环 answer_len 次，约十行
    raise NotImplementedError


def token_logps(model, prompt_ids: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """重算每个 action token 的 logπ（带梯度）。返回 (G, answer_len)。

    提示：把 prompt 和 actions 拼起来 forward 一次；预测第 t 个 action 的
    logits 在位置 T_prompt + t - 1；log_softmax 后 gather。
    """
    # TODO: 五行
    raise NotImplementedError


def grpo_loss(new_logps, old_logps, ref_logps, advantages, beta: float = 0.05,
              clip_eps: float = 0.2):
    """GRPO 目标（advantages 形状与 logps 相同，逐 token 广播）。

    ratio = exp(new - old)；policy 项 = -mean(min(ratio*A, clamp(ratio,1±eps)*A))；
    KL 项（k1）= mean(new - ref)；返回 (loss, kl, ratio_mean)。
    """
    # TODO: 五行
    raise NotImplementedError


def train_grpo(
    model,
    ref_model,
    problems: list[dict],
    *,
    steps: int = 200,
    prompts_per_step: int = 8,
    group_size: int = 8,
    beta: float = 0.05,
    lr: float = 1e-3,
    temperature: float = 1.0,
    log=print,
) -> dict:
    """经验参数：β=0.05 + lr=1e-3 是稳的（β 太小会熵塌缩、lr 太大漂移，见 book 第 4 节）。"""
    """外层循环。每步：抽 prompts_per_step 道题，每题一组 rollout，
    累计各组的 grpo_loss 后一次反向（µ=1：ratio 恒 1，clip 不激活，
    等价于带组基线的 REINFORCE）。

    返回 {"reward": [...每步平均 reward...], "kl": [...], "entropy": [...]}。
    """
    # TODO: 二十五行左右。每道题：
    #   x = tensor([encode(p["prompt"])]) 重复 G 份
    #   answer_len = len(p["answer"])
    #   actions, old_lp = rollout(...)
    #   r = tensor([reward(decode(a.tolist()), p["answer"]) for a in actions])
    #   A = group_advantages(r) -> 广播到 (G, answer_len)
    #   ref_lp = token_logps(ref_model, x, actions)（no_grad，先 eval ref）
    #   new_lp = token_logps(model, x, actions)
    #   累加 grpo_loss；记录统计
    raise NotImplementedError


# ========== 3. 底座预训练（已写好，不用改） ==========


def pretrain_soup(steps: int = 300, seed: int = 0, correct_rate: float = 0.3, log=print):
    """在'数字汤'上预训练小模型：格式总是正确，答案只有 correct_rate 的比例是对的。

    这是对 R1-Zero 的诚实微缩：base 不是从零学起——它带着三分正确的"毛坯加法先验"，
    RLVR 的工作是把这个火花放大到可靠（RL 放大既有能力，而不是无中生有；
    完全随机的答案会让 GRPO 无从 bootstrap，自己试一试就知道了）。
    语料一半一位数格式（level 1 用）、一半两位数格式（level 2 用），分两个等长半批。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = len(CHARSET)
    model = lm.TransformerLM(vocab_size=vocab, d_model=128, n_layers=2, n_heads=4,
                             d_ff=384, context_length=24)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def half_batch(fmt: int):
        seqs = []
        for _ in range(16):
            if fmt == 1:  # "3+4=7 "
                a, b = rng.integers(0, 5, 2)
                ans = a + b if rng.random() < correct_rate else rng.integers(0, 10)
                seqs.append(encode(f"{a}+{b}={int(ans)} "))
            else:  # "07+35=042 "
                a, b = rng.integers(0, 100, 2)
                ans = a + b if rng.random() < correct_rate else rng.integers(0, 1000)
                seqs.append(encode(f"{a:02d}+{b:02d}={int(ans):03d} "))
        x = torch.tensor([s[:-1] for s in seqs], dtype=torch.long)
        y = torch.tensor([s[1:] for s in seqs], dtype=torch.long)
        return lm.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))

    for step in range(1, steps + 1):
        loss = half_batch(1) + half_batch(2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            log(f"  [soup] step {step} loss {loss.item():.3f}")
    return model


def evaluate_accuracy(model, problems, temperature=0.0) -> float:
    """贪心（temperature=0）或采样的答题准确率，逐题前向。"""
    model.eval()
    hits = 0
    with torch.no_grad():
        for p in problems:
            x = torch.tensor([encode(p["prompt"])], dtype=torch.long)
            ids = x
            for _ in range(len(p["answer"])):
                logits = model(ids)[0, -1]
                if temperature > 0:
                    logits = logits / temperature
                    nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
                else:
                    nxt = int(logits.argmax())
                ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
            hits += reward(decode(ids[0, x.shape[1]:].tolist()), p["answer"]) > 0
    model.train()
    return hits / len(problems)


# ========== 4. 训练入口（已写好，不用改） ==========


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()

    torch.manual_seed(0)
    print("[lab07] 预训练数字汤底座（格式正确，三成答案对——毛坯先验）")
    model = pretrain_soup()
    ref = lm.TransformerLM(vocab_size=len(CHARSET), d_model=128, n_layers=2, n_heads=4,
                           d_ff=384, context_length=24)
    ref.load_state_dict(model.state_dict())
    ref.eval()

    problems = make_problems(level=1, n=256, seed=1)
    print(f"[lab07] level1 起始：贪心准确率 {evaluate_accuracy(model, problems[:50]):.1%}")
    h = train_grpo(model, ref, problems, steps=args.steps, log=print)
    acc1 = evaluate_accuracy(model, problems[:50])
    print(f"[lab07] level1 最终贪心准确率 {acc1:.1%}（采样 reward 末段 {sum(h['reward'][-10:])/10:.2f}）")

    os.makedirs(os.path.join(LAB_DIR, "cache"), exist_ok=True)
    path = os.path.join(LAB_DIR, "cache", "grpo_model.pt")
    torch.save(model.state_dict(), path)
    print(f"[lab07] 模型已保存: {path}")


if __name__ == "__main__":
    main()
