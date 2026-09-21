# Lab 06: DPO——用偏好数据对齐 Qwen 0.5B

> 对应 CS336 之外的后训练线 / 唐杰课程作业⑥
> 预计耗时：代码 1 天 + 真训练一个长周末　|　依赖：`transformers` + `peft`（同 lab05）
> 测试与演示**离线可跑**；真训练 = Qwen + lab05 的 adapter + 偏好数据

## 你要做什么

在 `dpo.py` 里实现：

1. `sequence_logps()`：一条 batch 的**答案部分**逐 token log-prob 之和（prompt 遮 -100）
2. `dpo_loss()`：DPO 损失 + 隐式奖励 + margin 准确率
3. `build_preference_batch()`：偏好对 → chosen/rejected 两个 collate batch（复用 lab05）
4. `train_dpo()`：π 与 π_ref 各两次前向、只训 π、记录 reward/acc 曲线

## 文件

| 文件 | 说明 |
|---|---|
| `dpo.py` | **你要写的文件**，所有 `TODO` 都在这里 |
| `tests/test_dpo.py` | 测试（离线 tiny Qwen2） |
| `solution/dpo_solution.py` | 参考答案。**先自己写，卡住了再看** |
| `demo_dpo.py` | 离线演示：把"一半对一半错"的底座掰成全对的 |
| `../../book/06-dpo.md` | 原理讲解（先读这个） |
| `../../data/sample/dpo_sample.jsonl` | 内置 24 条偏好对（真训练建议换更大的开源偏好集） |

## 通过标准

```bash
make lab6
# 真训练（需先完成 lab05 的 adapter）：
uv run --group posttrain python labs/06-dpo/dpo.py --adapter labs/05-sft/cache/adapter
```

1. pytest 全绿（含：π=π_ref 时 loss=ln2、梯度方向、margin 随训练增大）
2. demo：margin 从 0 拉开、acc→1、贪心生成从"一半对"变"全对"
3. 真训练后：同一问题，模型回答更倾向 chosen 风格；整体语言不崩（人审）

## 思考题（demo 末尾会用到）

- β=0.02 / 0.1 / 0.5 各训一遍，margin 曲线形状有何不同？
- demo 里 logπ(chosen) 绝对值在训练中降了吗？这意味着什么？
- 如果偏好数据里有 10% 标反了，DPO 会怎样？SFT 会怎样？（哪个更鲁棒？）
