.PHONY: setup test lab1

setup:
	uv sync

test:
	uv run pytest -q

lab1:
	uv run pytest labs/01-tokenizer/tests -q
	uv run python labs/01-tokenizer/demo_tokenizer.py
