"""
Lab 08: Agent + self-judge（你要实现的文件）

核心全部是纯 Python：工具、协议、解析、循环、裁判。
模型被抽象成 model_call: str -> str——测试用"剧本模型"替身，main() 接真 Qwen。

全部 TODO 完成并让 tests/test_agent.py 全绿后，运行 demo_agent.py。
原理讲解见 ../book/08-agent.md。
"""

import ast
import json
import operator
import re


# ========== 1. 工具 ==========

# 白名单：只放行这些 ast 节点和运算符
_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calc(expr: str):
    """安全算术求值。只接受 + - * / // % ** 和数字（含小数、负号）。

    返回 float/int 结果；任何非法输入（函数调用、下标、变量名、语法错误、
    除零……）都返回 "Error: <原因>" 字符串——绝不抛异常、绝不让 eval 碰到代码。
    提示：ast.parse(expr, mode="eval") 后递归校验节点类型再求值。
    """
    # TODO: 递归求值函数 _eval(node)（数字/一元/二元三种节点），入口 try/except
    raise NotImplementedError


def count_words(text: str) -> int:
    """数英文单词个数（按空白切分）。"""
    # TODO: 一行
    raise NotImplementedError


def get_tools() -> dict:
    """工具注册表：名字 -> {"fn": 可调用, "desc": 描述}。desc 会进提示词。"""
    return {
        "calc": {"fn": calc, "desc": "算术计算器。args: {\"expr\": \"2+3*4\"} -> 14"},
        "count_words": {"fn": count_words, "desc": "数单词数。args: {\"text\": \"hello world\"} -> 2"},
    }


# ========== 2. 协议 ==========

SYSTEM_TEMPLATE = """你是一个会用工具的助手。可用工具：
{tool_docs}

每一轮你必须输出**恰好一个 JSON**（可以包在 ```json 代码块里）：
- 调用工具：{{"tool": "工具名", "args": {{...}}}}
- 给出最终答案：{{"answer": "..."}}（已获得足够信息时）
不要输出 JSON 以外的任何内容。"""


def build_prompt(question: str, history: list[dict], tools: dict) -> str:
    """拼出下一轮给模型的完整提示词。

    history 是 [{"role": "model"|"tool", "content": str}, ...]：
    model 侧放模型上一轮的原始输出，tool 侧放工具执行结果（或格式错误提示）。
    返回：SYSTEM_TEMPLATE 填好工具文档 + "问题: ..." + 逐条历史。
    """
    # TODO: tool_docs = 每个工具一行 "名字: desc"；然后逐条拼 history
    raise NotImplementedError


# ========== 3. 解析 ==========


def extract_json(text: str):
    """从模型输出里鲁棒地提取一个 JSON 对象。

    依次尝试：
      1. ```json ... ``` 代码块
      2. 第一个 '{' 到最后一个 '}' 的子串
      3. 直接 json.loads 整段
    成功返回 dict；失败返回 None（不许抛异常）。
    """
    # TODO: 三段式 try，约十行
    raise NotImplementedError


def parse_action(text: str):
    """模型输出 -> 动作。

    返回 ("tool", 工具名, args_dict) / ("answer", 值) / None（无法解析）。
    规则：dict 有 "answer" 键 -> answer；有 "tool" 且工具名合法 -> tool；否则 None。
    args 不是 dict 时按 {} 处理。
    """
    # TODO: 六行
    raise NotImplementedError


# ========== 4. Agent 循环 ==========


def run_agent(model_call, question: str, tools: dict = None, max_turns: int = 6) -> dict:
    """跑一个 agent。model_call: (prompt: str) -> str。

    返回 {
      "answer": 最终答案（没答出来为 None）,
      "turns": 实际轮数,
      "status": "answered" | "max_turns" ,
      "history": [{"role","content"}, ...]  # 含模型输出与工具观察
    }
    每轮：build_prompt -> model_call -> parse_action
      - tool: 执行（工具报错也走字符串），观察进 history，继续
      - answer: 结束
      - None: history.append("格式错误提示")，继续
    """
    # TODO: 二十行左右
    raise NotImplementedError


# ========== 5. 裁判 ==========


def _normalize(text: str) -> str:
    """归一化：去首尾空白、转小写、去掉末尾标点（. 。 ! ！ ？ ?）。"""
    return text.strip().lower().rstrip(".。!！?？ ")


def rule_judge(question: str, answer: str, expected: str) -> float:
    """规则裁判：归一化后精确匹配 -> 1.0 / 0.0。answer 为 None 直接 0。"""
    # TODO: 三行
    raise NotImplementedError


def self_judge(model_call, question: str, trajectory: dict) -> tuple[bool, str]:
    """模型裁判：把完整轨迹给裁判模型，让它先推理再回答 YES / NO。

    返回 (verdict: bool, raw: 裁判的原始输出)。
    解析规则：输出里找 YES/NO（大小写不敏感）；都找不到时**保守判 False**。
    """
    # TODO: 拼裁判提示词（问题 + 答案 + 工具观察）-> model_call -> 解析，十行左右
    raise NotImplementedError


# ========== 6. 真模型入口（已写好，不用改） ==========


def make_qwen_call(model_dir: str = "models/qwen2.5-0.5b"):
    """把 Qwen 0.5B 包成 model_call: str -> str。"""
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
