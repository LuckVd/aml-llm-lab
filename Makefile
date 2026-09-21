.PHONY: setup test verify lab1 lab2 lab3 lab4 lab5 lab6 lab7 lab8

setup:
	uv sync

# 学员视角：测你自己的实现（没写完 TODO 的 lab 会红，属预期）
test:
	uv run pytest -q

# 出题人/自检视角：对参考答案跑全部测试 + 全部演示，应当永远全绿
verify:
	AML_LAB_USE_SOLUTION=1 uv run pytest -q
	AML_LAB_USE_SOLUTION=1 uv run python labs/01-tokenizer/demo_tokenizer.py
	AML_LAB_USE_SOLUTION=1 uv run python labs/02-transformer/demo_transformer.py
	AML_LAB_USE_SOLUTION=1 uv run python labs/03-pretrain/demo_pretrain.py
	AML_LAB_USE_SOLUTION=1 uv run python labs/04-scaling/demo_scaling.py
	AML_LAB_USE_SOLUTION=1 uv run python labs/05-sft/demo_sft.py
	AML_LAB_USE_SOLUTION=1 uv run python labs/06-dpo/demo_dpo.py
	AML_LAB_USE_SOLUTION=1 uv run python labs/07-grpo/demo_grpo.py
	AML_LAB_USE_SOLUTION=1 uv run python labs/08-agent/demo_agent.py

lab1:
	uv run pytest labs/01-tokenizer/tests -q
	uv run python labs/01-tokenizer/demo_tokenizer.py

lab2:
	uv run pytest labs/02-transformer/tests -q
	uv run python labs/02-transformer/demo_transformer.py

lab3:
	uv run pytest labs/03-pretrain/tests -q
	uv run python labs/03-pretrain/demo_pretrain.py

lab4:
	uv run pytest labs/04-scaling/tests -q
	uv run python labs/04-scaling/demo_scaling.py

lab5:
	uv run --group posttrain pytest labs/05-sft/tests -q
	uv run --group posttrain python labs/05-sft/demo_sft.py

lab6:
	uv run --group posttrain pytest labs/06-dpo/tests -q
	uv run --group posttrain python labs/06-dpo/demo_dpo.py

lab7:
	uv run pytest labs/07-grpo/tests -q
	uv run python labs/07-grpo/demo_grpo.py

lab8:
	uv run --group posttrain pytest labs/08-agent/tests -q
	uv run --group posttrain python labs/08-agent/demo_agent.py
