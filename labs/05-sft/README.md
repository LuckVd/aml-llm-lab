# Lab 05: SFT（LoRA）——把 Qwen 0.5B 变成助手

> 对应 CS336 之外的后训练线 / 唐杰课程作业⑤
> 预计耗时：代码半天 + 真模型训练一晚（CPU 也可跑完）　|　依赖：`transformers` + `peft`（`uv sync --group posttrain`）
> 测试与演示**离线可跑**（用随机初始化的 tiny Qwen2），不下载模型

## 你要做什么

在 `sft.py` 里实现：

1. `build_example()`：指令对 → (input_ids, labels)，**prompt 遮 -100**
2. `collate()`：变长 batch → pad / attention_mask / labels
3. `sft_loss()`：错位 + 忽略 -100 的交叉熵（自己算，不用模型内置 loss）
4. `add_lora()` / `lora_stats()`：peft 挂 LoRA、数可训练参数
5. `train_sft()`：训练循环（lab03 套路，只训 LoRA 参数）
6. `generate_answer()`：按对话模板采样

## 文件

| 文件 | 说明 |
|---|---|
| `sft.py` | **你要写的文件**，所有 `TODO` 都在这里（含离线调试用的 `CharTokenizer`，已写好） |
| `tests/test_sft.py` | 测试。全绿 = 完成（离线，不需下载 Qwen） |
| `solution/sft_solution.py` | 参考答案。**先自己写，卡住了再看** |
| `demo_sft.py` | 离线演示：tiny Qwen2 + LoRA 背下 4 条指令，看 loss 与生成变化 |
| `../../book/05-sft.md` | 原理讲解（先读这个） |
| `../../data/sample/sft_sample.jsonl` | 内置 60 条指令对（离线 smoke 用；真训练用 alpaca） |

## 通过标准

```bash
make lab5                                        # 测试 + 离线 demo（无需下载）
uv run python data/download.py qwen05b           # 真 base 模型 (~1GB)
uv run python data/download.py alpaca            # 真指令数据 (~30MB)
uv run --group posttrain python labs/05-sft/sft.py   # 真训练（CPU 一晚）
```

1. pytest 全绿
2. 离线 demo：LoRA 挂上后可训练参数 <5%，训练后模型能逐字背出见过的指令答案
3. 真训练后：模型以 `<|im_end|>` 收尾、不复读问题、能答对 `sft_sample.jsonl` 里的简单事实题

## 思考题（demo 末尾会用到）

- 遮罩 prompt 后，有效训练 token 占比是多少？
- 离线 demo 里模型"背下"4 条指令算学会了吗？和"泛化"差在哪？
- 真训练后故意过训练 3 倍轮数，回答会出现什么变化？
