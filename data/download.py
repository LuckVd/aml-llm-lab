"""下载大语料与模型权重（可选：data/sample 已内置切片，不下载也能跑 lab01-03）。

默认走 hf-mirror.com（国内直连）。用法：
    uv run python data/download.py tinystories   # TinyStories 全量 valid+train (~2GB)
    uv run python data/download.py qwen05b       # Qwen2.5-0.5B 权重 (~1GB)
    uv run python data/download.py all
"""

import os
import subprocess
import sys

MIRROR = "https://hf-mirror.com"

TARGETS = {
    "tinystories": [
        "datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
    ],
    "qwen05b": [
        "Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors",
        "Qwen/Qwen2.5-0.5B/resolve/main/config.json",
        "Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json",
        "Qwen/Qwen2.5-0.5B/resolve/main/tokenizer_config.json",
        "Qwen/Qwen2.5-0.5B/resolve/main/vocab.json",
        "Qwen/Qwen2.5-0.5B/resolve/main/merges.txt",
        "Qwen/Qwen2.5-0.5B/resolve/main/generation_config.json",
    ],
}

DEST = {
    "tinystories": "data/raw/",
    "qwen05b": "models/qwen2.5-0.5b/",
}


def fetch(repo_file: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    url = f"{MIRROR}/{repo_file}"
    out = os.path.join(dest_dir, os.path.basename(repo_file))
    if os.path.exists(out):
        print(f"已存在，跳过: {out}")
        return
    print(f"下载 {url} -> {out}")
    subprocess.check_call(["curl", "-L", "--retry", "3", "-o", out, url])


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = TARGETS if which == "all" else [which]
    for name in names:
        for f in TARGETS[name]:
            fetch(f, DEST[name])


if __name__ == "__main__":
    main()
