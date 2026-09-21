"""跑通 lab07 后的演示：RLVR 放大"毛坯加法先验"。

流程：数字汤底座（三成答案正确）-> GRPO level 1（一位数加法）->
采样 reward / 贪心准确率曲线 -> before/after 采样对比 -> level 2 稀疏奖励探针。

用法: uv run python labs/07-grpo/demo_grpo.py
"""

import copy
import os
import random
import sys

import numpy as np
import torch

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
    from solution.grpo_solution import (  # noqa: E402
        decode, encode, evaluate_accuracy, make_problems, pretrain_soup, rollout, train_grpo,
    )
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("lm_demo", os.path.join(LAB_DIR, "..", "02-transformer", "solution", "model_solution.py"))
    lm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(lm)
else:
    from grpo import (  # noqa: E402
        decode, encode, evaluate_accuracy, lm, make_problems, pretrain_soup, train_grpo,
    )


def answer(model, prompt: str, n_chars: int) -> str:
    """贪心连推 n_chars 个字符。"""
    ids = torch.tensor([encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        for _ in range(n_chars):
            nxt = int(model(ids)[0, -1].argmax())
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return decode(ids[0, len(ids[0]) - n_chars:].tolist())


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    print("[1] 数字汤底座：格式正确，三成答案正确（毛坯先验）")
    model = pretrain_soup(steps=300, seed=0)
    ref = lm.TransformerLM(vocab_size=len("0123456789+= "), d_model=128, n_layers=2,
                           n_heads=4, d_ff=384, context_length=24)
    ref.load_state_dict(model.state_dict())
    ref.eval()
    problems = make_problems(level=1, n=64, seed=1)
    greedy0 = evaluate_accuracy(model, problems[:32])
    sample_r0 = []
    with torch.no_grad():
        for p in problems[:32]:
            x = torch.tensor([encode(p["prompt"])], dtype=torch.long)
            acts, _ = rollout(model, x, 8, len(p["answer"]), temperature=1.0)
            sample_r0.append(np.mean([decode(a.tolist()) == p["answer"] for a in acts]))
    print(f"    采样命中率 {np.mean(sample_r0):.0%}（有火花），贪心准确率 {greedy0:.0%}")
    print("    样例（贪心）:", {p["prompt"]: answer(model, p["prompt"], 1) for p in problems[:4]})

    print("\n[2] GRPO 训练 level 1（β=0.05 锚住 π_ref，防熵塌缩）")
    h = train_grpo(model, ref, problems, steps=800, prompts_per_step=8, group_size=8,
                   lr=1e-3, beta=0.05, temperature=1.0)
    greedy1 = evaluate_accuracy(model, problems[:32])
    print(f"\n    采样 reward: 初段 {np.mean(h['reward'][:20]):.2f} -> 末段 {np.mean(h['reward'][-20:]):.2f}；"
          f"贪心准确率 {greedy0:.0%} -> {greedy1:.0%}")

    print("\n    reward 曲线（每 100 步均值）：")
    w = 100
    for i in range(800 // w):
        seg = np.mean(h["reward"][i * w:(i + 1) * w])
        print(f"    {i * w:>4}-{(i + 1) * w:<4} {'#' * int(seg * 40):<40} {seg:.2f}")

    print("\n[3] before / after 采样对比（贪心）：")
    model0 = copy.deepcopy(ref)
    for p in problems[:8]:
        a0, a1 = answer(model0, p["prompt"], 1), answer(model, p["prompt"], 1)
        ok = "✓" if a1 == p["answer"] else "✗"
        print(f"    {p['prompt']:<5} 真值 {p['answer']}  训练前 {a0}  训练后 {a1} {ok}")

    print("\n[4] level 2 探针（两位数加法，答案 3 位）：随机命中率 1/1000")
    p2 = make_problems(level=2, n=64, seed=2)
    print(f"    直接考 level 2: 准确率 {evaluate_accuracy(model, p2[:32]):.0%}")
    h2 = train_grpo(model, ref, p2, steps=300, prompts_per_step=8, group_size=8,
                    lr=1e-3, beta=0.05, temperature=1.0, log=lambda *a: None)
    print(f"    再训 300 步: 采样 reward 末段 {np.mean(h2['reward'][-20:]):.2f}，"
          f"准确率 {evaluate_accuracy(model, p2[:32]):.0%}——卡死（一组 8 个几乎必全错，零信号）")

    print("\n思考题：")
    print("  1. pretrain_soup 的 correct_rate 改成 0.0 重跑，reward 还起飞吗？这说明了 RL 的什么前提？")
    print("  2. β 改成 0.0 或 lr 改成 3e-3 重跑，观察 reward 曲线中途躺平——发生了什么？")
    print("  3. 给 level 2 设计课程：先只训'和 < 10'的两位数题（答案 00x），能启动吗？")


if __name__ == "__main__":
    main()
