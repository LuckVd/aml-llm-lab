"""Lab 08 参考答案：Agent + self-judge。"""

import ast
import json
import operator
import re

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

SYSTEM_TEMPLATE = """你是一个会用工具的助手。可用工具：
{tool_docs}

每一轮你必须输出**恰好一个 JSON**（可以包在 ```json 代码块里）：
- 调用工具：{{"tool": "工具名", "args": {{...}}}}
- 给出最终答案：{{"answer": "..."}}（已获得足够信息时）
不要输出 JSON 以外的任何内容。"""


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    raise ValueError(f"illegal expression element: {type(node).__name__}")


def calc(expr: str):
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        return int(result) if isinstance(result, float) and result.is_integer() else result
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:  # 语法错、危险节点等，一律字符串兜底
        return f"Error: {e}"


def count_words(text: str) -> int:
    return len(text.split())


def get_tools() -> dict:
    return {
        "calc": {"fn": calc, "desc": "算术计算器。args: {\"expr\": \"2+3*4\"} -> 14"},
        "count_words": {"fn": count_words, "desc": "数单词数。args: {\"text\": \"hello world\"} -> 2"},
    }


def build_prompt(question: str, history: list[dict], tools: dict) -> str:
    tool_docs = "\n".join(f"- {name}: {t['desc']}" for name, t in tools.items())
    parts = [SYSTEM_TEMPLATE.format(tool_docs=tool_docs), f"问题: {question}"]
    for h in history:
        if h["role"] == "model":
            parts.append(f"你上一轮的输出: {h['content']}")
        else:
            parts.append(f"观察: {h['content']}")
    parts.append("你的下一轮输出（一个 JSON）:")
    return "\n".join(parts)


def extract_json(text: str):
    try:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else None
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def parse_action(text: str):
    obj = extract_json(text)
    if obj is None:
        return None
    if "answer" in obj:
        return ("answer", obj["answer"])
    if isinstance(obj.get("tool"), str):
        args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
        return ("tool", obj["tool"], args)
    return None


def run_agent(model_call, question: str, tools: dict = None, max_turns: int = 6) -> dict:
    tools = tools or get_tools()
    history: list[dict] = []
    answer = None
    status = "max_turns"
    turns = 0
    for turn in range(1, max_turns + 1):
        turns = turn
        prompt = build_prompt(question, history, tools)
        out = model_call(prompt)
        history.append({"role": "model", "content": out})
        action = parse_action(out)
        if action is None:
            history.append({"role": "tool", "content": '格式错误：请输出 {"tool":..., "args":{...}} 或 {"answer": "..."}'})
            continue
        if action[0] == "answer":
            answer = action[1]
            status = "answered"
            break
        _, name, args = action
        if name not in tools:
            history.append({"role": "tool", "content": f"Error: 未知工具 {name!r}，可用: {list(tools)}"})
            continue
        try:
            result = tools[name]["fn"](**args)
        except TypeError as e:
            result = f"Error: 参数不匹配 {e}"
        history.append({"role": "tool", "content": str(result)})
    return {"answer": answer, "turns": turns, "status": status, "history": history}


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip(".。!！?？ ")


def rule_judge(question: str, answer: str, expected: str) -> float:
    if answer is None:
        return 0.0
    return 1.0 if _normalize(str(answer)) == _normalize(str(expected)) else 0.0


def self_judge(model_call, question: str, trajectory: dict) -> tuple[bool, str]:
    obs = "\n".join(f"[{h['role']}] {h['content']}" for h in trajectory["history"])
    prompt = (
        "你是严格的裁判。根据问题和工具观察，判断下面的最终答案是否正确。\n"
        f"问题: {question}\n工具观察与模型输出:\n{obs}\n"
        f"最终答案: {trajectory['answer']}\n"
        "先简要推理，最后一行只写 YES 或 NO。"
    )
    raw = model_call(prompt)
    if re.search(r"\bYES\b", raw, re.IGNORECASE) and not re.search(r"\bNO\b", raw, re.IGNORECASE):
        return True, raw
    if re.search(r"\bNO\b", raw, re.IGNORECASE) and not re.search(r"\bYES\b", raw, re.IGNORECASE):
        return False, raw
    if re.search(r"\bYES\b", raw, re.IGNORECASE):
        return True, raw
    return False, raw  # 解析失败保守判 False


def make_qwen_call(model_dir: str = "models/qwen2.5-0.5b"):
    import os

    if not os.path.isdir(model_dir):
        raise SystemExit(f"模型目录不存在: {model_dir}\n先运行: uv run python data/download.py qwen05b")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    @torch.no_grad()
    def call(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    return call


def main() -> None:
    call = make_qwen_call()
    tools = get_tools()
    questions = [
        ("计算 (17 + 5) * 3 是多少？", "66"),
        ("'the quick brown fox jumps' 有几个单词？", "5"),
        ("98765 除以 7 等于多少（整数部分）？", "14109"),
    ]
    for q, expected in questions:
        result = run_agent(call, q, tools)
        score = rule_judge(q, str(result["answer"]), expected)
        verdict, raw = self_judge(call, q, result)
        print(f"Q: {q}\n  答案: {result['answer']}（规则判 {score:.0f}，自评 {verdict}，{result['turns']} 轮）")
        for h in result["history"]:
            print(f"    [{h['role']}] {h['content'][:100]}")


if __name__ == "__main__":
    main()
