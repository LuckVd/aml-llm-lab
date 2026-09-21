"""
Lab 03: 预训练 20M（你要实现的文件）

把 lab01 的 tokenizer 和 lab02 的 TransformerLM 装成完整预训练流水线：
    语料 --tokenize_corpus--> ids --get_batch--> (x,y) --train--> checkpoint --sample--> 文本

全部 TODO 完成并让 tests/test_pretrain.py 全绿后，运行 demo_pretrain.py，
再 `uv run python labs/03-pretrain/train.py` 过夜跑标准配置（configs/default.toml）。
原理讲解见 ../book/03-pretrain.md。
"""

import importlib.util
import os
import pickle
import time

import numpy as np
import torch

try:  # python >= 3.11 自带，3.10 用 tomli（已在依赖里）
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(LAB_DIR, "..", ".."))
CACHE_DIR = os.path.join(LAB_DIR, "cache")
_LAB02_DIR = os.path.join(os.path.dirname(LAB_DIR), "02-transformer")
_LAB01_DIR = os.path.join(os.path.dirname(LAB_DIR), "01-tokenizer")


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lab02():
    """lab02 的模型与训练件。make verify（AML_LAB_USE_SOLUTION=1）时用参考答案。"""
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
        return _load_module("lab02_for_pretrain", os.path.join(_LAB02_DIR, "solution", "model_solution.py"))
    return _load_module("lab02_for_pretrain", os.path.join(_LAB02_DIR, "model.py"))


def _lab01():
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1":
        return _load_module("lab01_for_pretrain", os.path.join(_LAB01_DIR, "solution", "bpe_solution.py"))
    return _load_module("lab01_for_pretrain", os.path.join(_LAB01_DIR, "bpe.py"))


# ========== 1. 数据管道 ==========


def tokenize_corpus(
    text_path: str,
    tokenizer,
    eos_id: int,
    cache_path: str = None,
    force: bool = False,
) -> np.ndarray:
    """全文 -> 一维 int32 id 数组，带 .npy 缓存。

    做法：按 eos 字符串（如 "<|endoftext|>"）把全文切成文档，
    每个文档 encode 后**追加一个 eos_id**，全部拼接。
    cache_path 给定且文件已存在（且未 force）时直接 np.load。
    """
    # TODO:
    #   1. 缓存命中：np.load 返回（注意 dtype 应为 int32）
    #   2. 读文件，按 eos 的字符串形式 split（eos 的文本 = tokenizer 里它的 bytes 解码）
    #   3. 非空文档逐个 tokenizer.encode，末尾补 eos_id，拼接成一个大 list
    #   4. np.asarray(..., dtype=np.int32)；若给了 cache_path 则 np.save 后返回
    raise NotImplementedError


def get_batch(
    ids: np.ndarray, batch_size: int, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机窗口采样。返回 (x, y)，都是 (batch_size, seq_len) 的 int64 张量。

    第 i 个样本：随机起点 o_i（上界 len(ids) - seq_len - 1），
    x_i = ids[o_i : o_i + seq_len]，y_i = ids[o_i + 1 : o_i + 1 + seq_len]。
    """
    # TODO: np.random.randint 采样 batch_size 个起点 → 切片堆叠 → torch.from_numpy().long()
    raise NotImplementedError


# ========== 2. 评估 ==========


def evaluate(model, ids: np.ndarray, batch_size: int, seq_len: int, iters: int = 20) -> float:
    """在随机窗口上估计平均 loss（无梯度）。返回 python float。"""
    # TODO: model.eval() + torch.no_grad()，取 iters 个 batch 用 lab02 的 cross_entropy
    #       求均值，最后把模型放回 model.train() 并返回 float(loss)
    raise NotImplementedError


# ========== 3. checkpoint ==========


def save_checkpoint(path: str, model, step: int, config: dict = None) -> None:
    """torch.save 一个 dict: {"step", "config", "model": state_dict}。"""
    # TODO: 三行
    raise NotImplementedError


def load_checkpoint(path: str, model) -> dict:
    """加载 state_dict 到 model（就地），返回整个 dict。"""
    # TODO: 两行
    raise NotImplementedError


# ========== 4. 采样 ==========


@torch.no_grad()
def sample(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 10,
    eos_id: int = None,
) -> list[int]:
    """自回归采样。返回新生成的 id 列表（不含 prompt、不含结尾 eos）。

    循环 max_new_tokens 次：
      1. 取 ids 末尾 model.context_length 个作为上下文
      2. logits = model(ctx)[0, -1]
      3. temperature <= 0：argmax（贪心）；否则 logits /= temperature 后做 top_k 过滤
         （保留 top_k 个，其余置 -inf），softmax 后 torch.multinomial 采一个
      4. 采到 eos_id 就停（不 append）；否则 append
    prompt 为空串时，用 eos_id 当"文档开头"起头（有 bos 的作用）。
    """
    # TODO: 按上述四步写循环
    raise NotImplementedError


# ========== 5. 训练循环 ==========


def train(
    model,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.1,
    weight_decay: float = 0.01,
    beta2: float = 0.95,
    grad_clip: float = 1.0,
    eval_every: int = 100,
    eval_iters: int = 20,
    ckpt_dir: str = None,
    log=print,
) -> dict:
    """完整训练循环（lab02 的 lr_at / cross_entropy 在这里复用）。

    每步：get_batch -> forward -> cross_entropy -> backward ->
          clip_grad_norm_(grad_clip) -> 调度 lr -> opt.step()
    优化器用 torch.optim.AdamW(betas=(0.9, beta2), weight_decay=weight_decay)。

    每 eval_every 步：log 当前 train loss + evaluate(val)；
    返回 history = {"train": [(step, loss), ...], "val": [(step, loss), ...]}。
    """
    # TODO: 组装上述循环（约 30 行）。提示：
    #   - 优化器只建一次；lr 每步 opt.param_groups[0]["lr"] = lm.lr_at(step, ...)
    #   - 训完（或中途 eval 时）把模型放回 model.train()，evaluate 内部自己管 eval()
    #   - ckpt_dir 给定时每 eval_every 步往里存 step_XXXXXX.pt（os.makedirs(..., exist_ok=True)）
    raise NotImplementedError


# ========== 6. 标准训练入口（已写好，不用改） ==========


def ensure_tokenizer(
    corpus_path: str,
    vocab_size: int,
    special_tokens: list[str],
    train_bytes: int,
    cache_path: str = None,
) -> object:
    """取训练好的 lab01 tokenizer（有缓存读缓存，没有就在语料切片上训一个）。"""
    bpe = _lab01()
    if cache_path and os.path.exists(cache_path) and os.environ.get("AML_LAB_FORCE") != "1":
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        return bpe.Tokenizer(blob["vocab"], blob["merges"], special_tokens)

    with open(corpus_path, "rb") as f:
        raw = f.read(train_bytes).decode("utf-8", errors="ignore")
    slice_path = (cache_path or corpus_path) + ".toksrc"
    with open(slice_path, "w", encoding="utf-8") as f:
        f.write(raw)
    vocab, merges = bpe.train_bpe(slice_path, vocab_size=vocab_size, special_tokens=special_tokens)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"vocab": vocab, "merges": merges}, f)
    return bpe.Tokenizer(vocab, merges, special_tokens)


def main() -> None:
    with open(os.path.join(REPO_ROOT, "configs", "default.toml"), "rb") as f:
        cfg = tomllib.load(f)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]

    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    train_path = os.path.join(REPO_ROOT, dcfg["train"])
    val_path = os.path.join(REPO_ROOT, dcfg["val"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(dcfg["train"]))[0]

    log = print
    log(f"[lab03] 语料: {dcfg['train']}")
    tokenizer = ensure_tokenizer(
        train_path, mcfg["vocab_size"], dcfg["special_tokens"],
        dcfg["tokenizer_train_bytes"], cache_path=os.path.join(CACHE_DIR, f"{stem}.tok.pkl"),
    )
    special = dcfg["special_tokens"][0]
    eos_id = tokenizer.byte_to_id[special.encode("utf-8")]
    log(f"[lab03] 词表 {len(tokenizer.vocab)}，eos id = {eos_id}")

    train_ids = tokenize_corpus(train_path, tokenizer, eos_id,
                                cache_path=os.path.join(CACHE_DIR, f"{stem}.ids.npy"))
    val_ids = tokenize_corpus(val_path, tokenizer, eos_id,
                              cache_path=os.path.join(CACHE_DIR, f"{stem}.val.ids.npy"))
    log(f"[lab03] tokens: train {len(train_ids):,} / val {len(val_ids):,}")

    lm = _lab02()
    model = lm.TransformerLM(
        vocab_size=len(tokenizer.vocab),
        d_model=mcfg["d_model"], n_layers=mcfg["num_layers"], n_heads=mcfg["num_heads"],
        d_ff=mcfg["d_ff"], context_length=mcfg["context_length"], rope_theta=mcfg["rope_theta"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[lab03] 模型 {n_params:,} 参数，开始训练 {tcfg['steps']} 步（{tcfg['batch_size']}×{tcfg['seq_len']} tokens/步）")

    t0 = time.time()
    history = train(
        model, train_ids, val_ids,
        steps=tcfg["steps"], batch_size=tcfg["batch_size"], seq_len=tcfg["seq_len"],
        lr=tcfg["lr"], warmup_steps=tcfg["warmup_steps"], min_lr_ratio=tcfg["min_lr_ratio"],
        weight_decay=tcfg["weight_decay"], beta2=tcfg["beta2"], grad_clip=tcfg["grad_clip"],
        eval_every=tcfg["eval_every"], eval_iters=tcfg["eval_iters"], ckpt_dir=CACHE_DIR, log=log,
    )
    log(f"[lab03] 训练完成，耗时 {(time.time()-t0)/60:.1f} 分钟")

    final_path = os.path.join(CACHE_DIR, "final.pt")
    save_checkpoint(final_path, model, step=tcfg["steps"], config=dict(mcfg))
    log(f"[lab03] 最终 checkpoint: {final_path}")
    for text in ("Once upon a time", "The little girl"):
        new_ids = sample(model, tokenizer, text, max_new_tokens=60, temperature=1.0, top_k=10, eos_id=eos_id)
        log(f"[lab03] 样例 [{text}]: {text + tokenizer.decode(new_ids)!r}")


if __name__ == "__main__":
    main()
