# Lab 08: Agent + self-judge——给 Qwen 装上手脚和裁判

> 对应 CS336 之外的应用线 / 唐杰课程作业⑤的 Agent 方向
> 预计耗时：2–3 天　|　核心依赖：**纯 Python**（接真模型时需 `uv sync --group posttrain` + Qwen 权重）
> 测试与演示离线可跑（"剧本模型"替身），真模型是最后一步

## 你要做什么

在 `agent.py` 里实现一个工具调用 agent 与自评裁判：

1. 工具：`calc()`（ast 白名单安全算术）、`count_words()`
2. 协议：`build_prompt()`（系统提示 + 工具文档 + 历史）
3. 解析：`extract_json()`（鲁棒 JSON 提取）、`parse_action()`
4. 循环：`run_agent()`（最多 N 轮：调工具 / 给答案 / 格式兜底）
5. 裁判：`rule_judge()`（规则）、`self_judge()`（模型 YES/NO）

## 文件

| 文件 | 说明 |
|---|---|
| `agent.py` | **你要写的文件**，所有 `TODO` 都在这里 |
| `tests/test_agent.py` | 测试（剧本模型替身，离线） |
| `solution/agent_solution.py` | 参考答案。**先自己写，卡住了再看** |
| `demo_agent.py` | 三方对比：裸猜 vs 用工具 vs self-judge 把关 |
| `../../book/08-agent.md` | 原理讲解（先读这个） |

## 通过标准

```bash
make lab8
# 真模型（可选收尾）：
uv run python data/download.py qwen05b
uv run --group posttrain python labs/08-agent/agent.py
```

1. pytest 全绿（含：安全白名单拒绝危险表达式、坏 JSON 兜底、死循环熔断、裁判解析）
2. demo：同一套题，裸猜模型 1-2 分、工具模型 5-6 分、self-judge 能抓住裸猜的错
3. 真模型：能吐出合法 JSON 动作、调对 calc、给出与工具观察一致的答案（0.5B 模型 JSON 遵从率有限，
   观察失败模式本身就是作业——把 build_prompt 调到它能稳定输出的格式）

## 思考题（demo 末尾会用到）

- 剧本模型是"完美模型"，真 Qwen 0.5B 会怎么不完美？协议上你预留了多少容错？
- self-judge 对"自信的错误"和"犹豫的正确"分别怎么判？怎么设计提示词缓解？
- best-of-n=5 + 自评最高，和 best-of-n=5 + 多数投票，在你这套题上哪个更好？为什么？
