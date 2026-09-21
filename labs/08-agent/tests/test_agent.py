"""Lab 08 测试：实现 agent.py 直到全部通过（剧本模型替身，离线）。"""

import importlib.util
import os

import pytest

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTION_PATH = os.path.join(LAB_DIR, "solution", "agent_solution.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STUDENT_PATH = (
    SOLUTION_PATH
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1"
    else os.path.join(LAB_DIR, "agent.py")
)
m = _load("student_agent", STUDENT_PATH)


# ---------- 工具 ----------


def test_calc_basic_and_precedence():
    assert m.calc("2+3*4") == 14
    assert m.calc("(17+5)*3") == 66
    assert m.calc("10/4") == 2.5
    assert m.calc("7//2") == 3
    assert m.calc("2**10") == 1024
    assert m.calc("-3+5") == 2


def test_calc_safety():
    out = m.calc("__import__('os').system('echo hacked')")
    assert isinstance(out, str) and out.startswith("Error")
    out2 = m.calc("open('/etc/passwd')")
    assert isinstance(out2, str) and out2.startswith("Error")
    out3 = m.calc("[1,2,3][0]")
    assert isinstance(out3, str) and out3.startswith("Error")
    out4 = m.calc("hello")
    assert isinstance(out4, str) and out4.startswith("Error")
    out5 = m.calc("1/0")
    assert isinstance(out5, str) and out5.startswith("Error")  # 报错不抛异常


def test_count_words():
    assert m.count_words("the quick brown fox jumps") == 5
    assert m.count_words("") == 0
    assert m.count_words("  a  b ") == 2


# ---------- 协议与解析 ----------


def test_build_prompt_contains_tools_and_question():
    tools = m.get_tools()
    p = m.build_prompt("1+1=?", [{"role": "model", "content": '{"tool":"calc","args":{"expr":"1+1"}}'},
                                 {"role": "tool", "content": "2"}], tools)
    assert "calc" in p and "count_words" in p
    assert "1+1=?" in p
    assert "2" in p


def test_extract_json_variants():
    assert m.extract_json('{"answer": "42"}') == {"answer": "42"}
    assert m.extract_json('```json\n{"answer": "42"}\n```') == {"answer": "42"}
    assert m.extract_json('我认为 {"answer": "42"} 就是这样') == {"answer": "42"}
    assert m.extract_json("完全不是 JSON") is None
    assert m.extract_json("[1, 2, 3]") is None  # 不是对象


def test_parse_action():
    assert m.parse_action('{"tool": "calc", "args": {"expr": "1+1"}}') == ("tool", "calc", {"expr": "1+1"})
    assert m.parse_action('{"answer": 66}') == ("answer", 66)
    assert m.parse_action("垃圾输出") is None
    # args 不是 dict -> 按 {} 处理
    kind = m.parse_action('{"tool": "calc", "args": "1+1"}')
    assert kind[0] == "tool" and kind[2] == {}


# ---------- Agent 循环（剧本模型） ----------


def scripted(outputs: list[str]):
    """按剧本逐轮吐预置输出的假模型。"""
    calls = {"i": 0, "prompts": []}

    def model_call(prompt: str) -> str:
        calls["prompts"].append(prompt)
        out = outputs[min(calls["i"], len(outputs) - 1)]
        calls["i"] += 1
        return out

    return model_call, calls


def test_agent_uses_tool_then_answers():
    call, meta = scripted(['{"tool": "calc", "args": {"expr": "(17+5)*3"}}',
                           '{"answer": "66"}'])
    result = m.run_agent(call, "计算 (17+5)*3", m.get_tools())
    assert result["status"] == "answered"
    assert result["turns"] == 2
    assert m.rule_judge("", str(result["answer"]), "66") == 1.0
    kinds = [h["role"] for h in result["history"]]
    assert kinds == ["model", "tool", "model"]
    assert "66" in result["history"][1]["content"]  # 工具观察进历史


def test_agent_recovers_from_bad_format():
    call, _ = scripted(["我不会输出 JSON",
                        '{"tool": "unknown_tool", "args": {}}',
                        '{"answer": "ok"}'])
    result = m.run_agent(call, "随便什么问题", m.get_tools())
    assert result["status"] == "answered"
    assert result["answer"] == "ok"
    assert result["turns"] == 3
    contents = " || ".join(h["content"] for h in result["history"])
    assert "格式错误" in contents and "未知工具" in contents


def test_agent_max_turns_fuse():
    call, _ = scripted(["我就是在聊天，不输出 JSON"])
    result = m.run_agent(call, "问题", m.get_tools(), max_turns=3)
    assert result["status"] == "max_turns"
    assert result["turns"] == 3
    assert result["answer"] is None


# ---------- 裁判 ----------


def test_rule_judge_normalize():
    assert m.rule_judge("", "66", "66") == 1.0
    assert m.rule_judge("", " 66. ", "66") == 1.0
    assert m.rule_judge("", "66。", "66") == 1.0
    assert m.rule_judge("", "67", "66") == 0.0
    assert m.rule_judge("", None, "66") == 0.0


def test_self_judge_parsing():
    traj = {"answer": "66", "history": [{"role": "tool", "content": "66"}]}
    yes, raw = m.self_judge(scripted(["推理一下……最终答案与工具观察一致。\nYES"])[0], "算 (17+5)*3", traj)
    assert yes is True and "YES" in raw
    no, _ = m.self_judge(scripted(["NO"])[0], "算 (17+5)*3", traj)
    assert no is False
    garbage, _ = m.self_judge(scripted(["我不知道"])[0], "算 (17+5)*3", traj)
    assert garbage is False  # 解析失败保守判 False


def test_mini_benchmark():
    """裸猜 vs 用工具：同一套题，工具模型得分应显著更高。"""
    qs = [("计算 (17+5)*3 是多少？", "66"),
          ("计算 2+3*4 是多少？", "14"),
          ("'the quick brown fox jumps' 有几个单词？", "5")]

    def no_tools_model(prompt: str) -> str:
        # 裸猜：直接抢答（心算第 2 题会错——运算符优先级）。注意只看问题行，
        # 提示词其他部分含工具描述，拿来匹配会串台。
        q = prompt.split("问题: ", 1)[1].split("\n", 1)[0]
        if "单词" in q:
            return '{"answer": "3"}'
        if "2+3*4" in q:
            return '{"answer": "20"}'
        return '{"answer": "66"}'

    def tool_model(prompt: str) -> str:
        if "观察" not in prompt:  # 第一轮：调工具
            q = prompt.split("问题: ", 1)[1].split("\n", 1)[0]
            if "单词" in q:
                return '{"tool": "count_words", "args": {"text": "the quick brown fox jumps"}}'
            expr = "(17+5)*3" if "17" in q else "2+3*4"
            return '{"tool": "calc", "args": {"expr": "%s"}}' % expr
        obs = prompt.split("观察: ")[-1].split("\n")[0]  # 从提示词里读工具结果
        return f'{{"answer": "{obs.split(chr(10))[0]}"}}'

    score = lambda call: sum(m.rule_judge(q, str(m.run_agent(call, q, m.get_tools())["answer"]), e)
                             for q, e in qs)
    assert score(tool_model) == 3.0
    assert score(no_tools_model) < score(tool_model)
