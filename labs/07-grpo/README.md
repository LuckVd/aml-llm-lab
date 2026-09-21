# Lab 07: GRPO / RLVR——教模型做算术

> 对应 CS336 之外的对齐线 / 唐杰课程作业⑦（RLHF 方向）
> 预计耗时：代码 1 天 + 训练一晚（CPU 也轻松）　|　依赖：仅 torch（lab02 的模型）
> 任务：不给任何标准答案，只给一个"算得对不对"的验证器

## 你要做什么

在 `grpo.py` 里实现（模型用 lab02 的 `TransformerLM`，字符级词表）：

1. `make_problems()`：两级算术题（`3+4=` / `12+35=`，答案定宽）
2. `reward()`：精确匹配判定器
3. `group_advantages()`：组相对优势 A = (r − mean)/std
4. `rollout()`：采样 G 个答案并记录逐 token logπ（无梯度）
5. `token_logps()`：给定 (prompt, actions) 重算逐 token logπ（带梯度）
6. `grpo_loss()`：clip + KL 的 GRPO 目标
7. `train_grpo()`：外层循环（每题单独成组，天然免 padding）

## 文件

| 文件 | 说明 |
|---|---|
| `grpo.py` | **你要写的文件**，所有 `TODO` 都在这里（含预训练"数字汤"底座的 `pretrain_soup`，已写好） |
| `tests/test_grpo.py` | 测试（含端到端：reward 从随机水平起飞） |
| `solution/grpo_solution.py` | 参考答案。**先自己写，卡住了再看** |
| `demo_grpo.py` | 两级课程的完整训练 + 采样对比 |
| `../../book/07-grpo.md` | 原理讲解（先读这个） |

## 通过标准

```bash
make lab7
```

1. pytest 全绿（含：全对/全错组零梯度、clip 分支、端到端 reward 上升）
2. demo：采样 reward 从 ~20% 爬到 65%+、贪心准确率 0% → 60%+，before/after 采样对比；
   level 2 探针停在 0%——稀疏奖励的真实困境（这本身就是作业的一部分）

## 标注：与"真 20M 基座"的关系

README 路线表里 lab07 标的是"自训 20M"。本 lab 默认用**小字符模型**（lab02 架构的缩小版 +
内置"数字汤"预训练），保证离线可跑、一晚可成。`grpo.py --checkpoint labs/03-pretrain/cache/final.pt`
可对接你 lab03 的 20M（要求 lab03 用全量语料训练过，数字 token 才可靠）——作为选做。

## 思考题（demo 末尾会用到）

- 把 `pretrain_soup` 的 correct_rate 改成 0.0（纯随机答案）重跑——为什么 GRPO 卡死在随机水平？
- β 改成 0 / 0.5，lr 改成 3e-3，各会看到什么病？（熵塌缩 / 学不动 / 漂移）
- reward 若只匹配首位数字，"奖励黑客"会怎么长？
- level 2 卡在 0%：设计一个课程让它动起来（提示：先训"和 < 10"的两位数题，答案形如 00x）
