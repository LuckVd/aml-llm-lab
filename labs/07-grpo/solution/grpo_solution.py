"""Lab 07 参考答案：GRPO / RLVR 学算术。"""

import argparse
import importlib.util
import os

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(_HERE) if os.path.basename(_HERE) == "solution" else _HERE


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


lm = _lab02()

CHARSET = "0123456789+= "


def encode(text: str) -> list[int]:
    table = {c: i for i, c in enumerate(CHARSET)}
    return [table[c] for c in text]


def decode(ids) -> str:
    return "".join(CHARSET[i] for i in ids)


def make_problems(level: int = 1, n: int = 64, seed: int = 0) -> list[dict]:
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
    return 1.0 if answer_text.strip() == expected.strip() else 0.0


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    std = rewards.std()
    if std < 1e-6:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + 1e-4)


@torch.no_grad()
def rollout(model, prompt_ids: torch.Tensor, group_size: int, answer_len: int,
            temperature: float = 1.0):
    ids = prompt_ids.repeat(group_size, 1)  # (G, T)
    actions, logps = [], []
    for _ in range(answer_len):
        logits = model(ids)[:, -1, :] / temperature
        logp = F.log_softmax(logits, dim=-1)
        nxt = torch.multinomial(torch.exp(logp), 1)  # (G, 1)
        actions.append(nxt)
        logps.append(logp.gather(-1, nxt))
        ids = torch.cat([ids, nxt], dim=1)
    return torch.cat(actions, dim=1), torch.cat(logps, dim=1)


def token_logps(model, prompt_ids: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    t = prompt_ids.shape[1]
    ids = torch.cat([prompt_ids.repeat(actions.shape[0], 1), actions], dim=1)
    logits = model(ids)
    logp = F.log_softmax(logits[:, t - 1:-1, :], dim=-1)  # 预测每个 action 的位置
    return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def grpo_loss(new_logps, old_logps, ref_logps, advantages, beta: float = 0.05,
              clip_eps: float = 0.2):
    ratio = torch.exp(new_logps - old_logps)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    kl = (new_logps - ref_logps).mean()
    loss = policy_loss + beta * kl
    return loss, float(kl), float(ratio.mean())


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
    """经验参数：β=0.05 + lr=1e-3 是稳的（β 太小会熵塌缩、lr 太大漂移）。"""
    model.train()
    ref_model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    history = {"reward": [], "kl": [], "entropy": []}
    rng = np.random.default_rng(0)

    for step in range(1, steps + 1):
        batch = [problems[i] for i in rng.integers(0, len(problems), prompts_per_step)]
        opt.zero_grad(set_to_none=True)
        step_rewards, step_kl, step_ent = [], 0.0, 0.0
        for p in batch:
            prompt = torch.tensor([encode(p["prompt"])], dtype=torch.long)
            answer_len = len(p["answer"])
            actions, old_lp = rollout(model, prompt, group_size, answer_len, temperature)
            r = torch.tensor([reward(decode(a.tolist()), p["answer"]) for a in actions])
            adv = group_advantages(r)
            adv_tokens = adv.unsqueeze(1).expand_as(actions)

            with torch.no_grad():
                ref_lp = token_logps(ref_model, prompt, actions)
            new_lp = token_logps(model, prompt, actions)
            loss, kl, _ = grpo_loss(new_lp, old_lp, ref_lp, adv_tokens, beta=beta)
            (loss / len(batch)).backward()

            step_rewards.extend(r.tolist())
            step_kl += kl
            with torch.no_grad():
                ent = -(torch.exp(new_lp) * new_lp).mean()  # 粗略观测
                step_ent += float(ent)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history["reward"].append(float(np.mean(step_rewards)))
        history["kl"].append(step_kl / len(batch))
        history["entropy"].append(step_ent / len(batch))
        if step % max(1, steps // 10) == 0 or step == 1:
            log(f"  step {step:>4}  reward {history['reward'][-1]:.2f}  kl {history['kl'][-1]:+.3f}")
    return history


def pretrain_soup(steps: int = 300, seed: int = 0, correct_rate: float = 0.3, log=print):
    """'数字汤'底座：格式正确，答案三成正确（毛坯先验，RL 负责放大）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = len(CHARSET)
    model = lm.TransformerLM(vocab_size=vocab, d_model=128, n_layers=2, n_heads=4,
                             d_ff=384, context_length=24)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def half_batch(fmt: int):
        seqs = []
        for _ in range(16):
            if fmt == 1:
                a, b = rng.integers(0, 5, 2)
                ans = a + b if rng.random() < correct_rate else rng.integers(0, 10)
                seqs.append(encode(f"{a}+{b}={int(ans)} "))
            else:
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
