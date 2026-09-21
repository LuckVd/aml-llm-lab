"""跑通 lab08 后的演示（离线）：三方对比——裸猜 vs 工具 vs self-judge 把关。

真模型路径：`uv run --group posttrain python labs/08-agent/agent.py`（先下载 Qwen）。
剧本模型是"理想化"替身；真 0.5B 模型的 JSON 遵从率有限——把 prompt 调到它能稳定
输出的格式，正是 lab08 的实战部分。

用法: uv run python labs/08-agent/demo_agent.py
"""

import os
import sys

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
    from solution.agent_solution import get_tools, rule_judge, run_agent, self_judge  # noqa: E402
else:
    from agent import get_tools, rule_judge, run_agent, self_judge  # noqa: E402

QUESTIONS = [
    ("计算 (17 + 5) * 3 是多少？", "66"),
    ("计算 2 + 3 * 4 是多少？", "14"),
    ("'the quick brown fox jumps over the lazy dog' 有几个单词？", "9"),
    ("计算 12345 * 67 是多少？", "827115"),
    ("计算 2 的 16 次方是多少？", "65536"),
]


def _question(prompt: str) -> str:
    """从完整提示词里抽出问题行（提示词其他部分含工具描述，不能拿来匹配）。"""
    return prompt.split("问题: ", 1)[1].split("\n", 1)[0]


def no_tools_model(prompt: str) -> str:
    """裸猜模型：不调工具直接抢答。心算乘方/大数乘法会错，而且错得自信。"""
    q = _question(prompt)
    if "单词" in q:
        return '{"answer": "7"}'          # 数错
    if "2 的 16 次方" in q:
        return '{"answer": "32768"}'      # 错（真值 65536）
    if "12345 * 67" in q:
        return '{"answer": "837115"}'     # 错
    if "2 + 3 * 4" in q:
        return '{"answer": "20"}'         # 优先级错（真值 14）
    return '{"answer": "66"}'             # 这题碰巧对


def tool_model(prompt: str) -> str:
    """工具模型：第一轮调工具，第二轮照着观察回答。"""
    if "观察" in prompt:
        obs = prompt.split("观察: ")[-1].split("\n")[0]
        return f'{{"answer": "{obs}"}}'
    q = _question(prompt)
    if "单词" in q:
        return "```json\n" + '{"tool": "count_words", "args": {"text": "the quick brown fox jumps over the lazy dog"}}' + "\n```"
    if "2 的 16 次方" in q:
        return '我想想…… {"tool": "calc", "args": {"expr": "2**16"}}'
    if "12345" in q:
        return '{"tool": "calc", "args": {"expr": "12345*67"}}'
    expr = "(17+5)*3" if "17" in q else "2+3*4"
    return '{"tool": "calc", "args": {"expr": "%s"}}' % expr


def honest_judge(prompt: str) -> str:
    """剧本裁判：检查答案是否与观察一致（理想化的诚实裁判）。"""
    import re

    ans = re.search(r"最终答案: (.+)", prompt).group(1).strip()
    obs_block = prompt.split("工具观察与模型输出:")[-1]
    obs = re.findall(r"\[tool\] (.+)", obs_block)
    if obs and obs[-1].strip() == ans:
        return "答案与工具观察一致。\nYES"
    if not obs and ans in ("32768", "837115", "20", "7"):
        return "没有任何工具观察支撑，且数值可疑。\nNO"
    return "无观察可比对，保守判否。\nNO"


def show(result) -> None:
    for h in result["history"]:
        tag = {"model": "模型", "tool": "观察"}.get(h["role"], h["role"])
        print(f"      [{tag}] {h['content'][:90]}")


def main() -> None:
    tools = get_tools()
    print("=" * 70)
    print("A. 裸猜模型（不调工具）")
    score_a = 0.0
    caught = 0
    for q, expected in QUESTIONS:
        r = run_agent(no_tools_model, q, tools)
        s = rule_judge(q, str(r["answer"]), expected)
        score_a += s
        verdict, _ = self_judge(honest_judge, q, r)
        caught += (verdict is False and s == 0.0)
        mark = "✓" if s else "✗"
        print(f"   {q} -> {r['answer']} {mark}（self-judge: {'抓错' if not verdict else '放行'}）")
    print(f"   得分 {score_a:.0f}/{len(QUESTIONS)}，self-judge 抓出 {caught} 个错误答案\n")

    print("=" * 70)
    print("B. 工具模型（calc / count_words，观察里带 fenced JSON、散文 JSON 各种真实格式）")
    score_b = 0.0
    for q, expected in QUESTIONS[:3]:
        r = run_agent(tool_model, q, tools)
        s = rule_judge(q, str(r["answer"]), expected)
        score_b += s
        mark = "✓" if s else "✗"
        print(f"   {q} -> {r['answer']} {mark}")
        show(r)
    print(f"\n   得分（前 3 题）{score_b:.0f}/3 —— 裸猜模型同样 3 题只对 1 题\n")

    print("=" * 70)
    print("C. 结论：工具把'嘴'接到'手'上，self-judge 给'自信的错误'兜底。")
    print("   真模型实战：uv run python data/download.py qwen05b")
    print("             uv run --group posttrain python labs/08-agent/agent.py")
    print("\n思考题：")
    print("  1. 剧本裁判是'完美裁判'。真模型的裁判会犯什么错（长答案偏好/自信语气）？")
    print("  2. no_tools_model 每题都秒答（1 轮），tool_model 要 2 轮——什么时候值得多花一轮？")
    print("  3. 把 honest_judge 的'保守判否'改成'宽松判是'，对 A 组得分各有什么影响？")


if __name__ == "__main__":
    main()
