# Lab 06 原理：DPO——不用 RL 的偏好对齐

> 读顺序建议：本页 → dpo.py 的 TODO → 卡住再回来查。
> 基座同 lab05（Qwen2.5-0.5B + 你训好的 SFT adapter）；测试与 demo 离线可跑。

## 1. SFT 之后还差什么？

SFT 教会"怎么回答"，但每个问题只见过一个标准答案。真实需求常常是**两难**：
- "教我做炸弹" —— 拒答（harmless）比详细（helpful）重要
- 两个都语法正确的回答 —— 要选更清楚、更诚实的那个

偏好优化拿到的是**比较信号**："A 比 B 好"，不需要 A 是满分答案。标注容易、天花板高。

## 2. 从 RLHF 到 DPO：一步代数变换

经典 RLHF 三阶段：训奖励模型 → 用 PPO 按 reward 优化策略 → 加 KL 惩罚别跑偏。
DPO 的发现：整个优化问题有**闭式解**，奖励可以直接用策略和参考策略的比值表示：

```
r(x, y) = β · log( π(y|x) / π_ref(y|x) ) + 常数
```

代回"最大化奖励 − KL"的目标，得到一个纯监督损失：

```
L_DPO = − E[ log σ( β · ( logπ(chosen)/π_ref(chosen) − logπ(rejected)/π_ref(rejected) ) ) ]
```

人话：**提高 chosen 的相对概率、压低 rejected 的相对概率，sigmoid 软化后取负对数**。
没有奖励模型、没有采样循环、没有 value network——一个 batch 一次反传，和 SFT 一样便宜。

## 3. 三个关键量

- **隐式奖励** `r = β·(logπ − logπ_ref)`：策略相对参考模型把这条路走"多远"了。DPO 里没显式的奖励模型，但每一步梯度都等价于在调这个隐式奖励。
- **β（温度）**：小 β → 激进（敢大幅偏离参考模型，容易说话越来越极端）；大 β → 保守（贴着参考模型，学得慢但稳）。0.05–0.3 是常见区间。
- **margin 与准确率**：`Δ = r_chosen − r_rejected`，训练要看的不是 loss 本身而是 Δ 的移动和 acc = 1[Δ>0] 的比例。acc 到 95%+ 说明偏好学进去了。

注意一个阴险的失败模式：DPO 只要求**相对**概率。把 rejected 的概率压到 0 也能让 loss 下降——
代价是整体语言质量崩坏（概率质量被抽走，采样出来的句子开始不像人话）。所以 β 和步数都要克制，
并且要盯住 chosen 的**绝对** logπ 是否也在下降（两者同降 = 警报）。

## 4. 实现要点

- π_ref 冻结（`torch.no_grad()` + `model.eval()`），通常就是 SFT 结束时那份权重
- 一次前向四个 logp：π(chosen)、π(rejected)、π_ref(chosen)、π_ref(rejected)
- 每个序列的 logp = 答案部分各 token log-prob 之和（prompt 照 lab05 遮 -100）
- 用 LoRA 有个天然优势：参考模型 = 冻结底座 + 空 adapter（B=0 时 π=π_ref），起步 loss 恰好 ln2≈0.693

## 5. 带着这些问题去写 TODO

1. π = π_ref 时 L_DPO = ln 2，为什么是这个数？（提示：σ(0) = ?）
2. β 从 0.1 改成 0.01，梯度变大还是变小？训练曲线会怎么变？
3. 若 chosen 和 rejected 恰好相等（标注矛盾），损失和梯度是什么？
4. 训练中 r_chosen 上升但 logπ(chosen) 绝对值下降——发生了什么？该怎么做？（提示：mode collapse / 概率质量蒸发）
5. DPO 之后模型一定会变得更"好"吗？它优化的是谁的偏好？（提示：标注者）

写完 pytest 全绿后跑 `demo_dpo.py`：看 margin 从 0 拉开、acc 冲到 1，以及概率质量从 rejected 搬到 chosen。
