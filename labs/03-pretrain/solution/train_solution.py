"""Lab 03 参考答案：完整预训练流水线。结构与 train.py 一一对应。"""

import importlib.util
import os
import pickle
import time

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_HERE = os.path.dirname(os.path.abspath(__file__))
# 本文件在 solution/ 子目录里，路径基准上跳一层到 lab 根
LAB_DIR = os.path.dirname(_HERE) if os.path.basename(_HERE) == "solution" else _HERE
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
    if cache_path and os.path.exists(cache_path) and not force:
        return np.load(cache_path)

    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    eos_text = tokenizer.vocab[eos_id].decode("utf-8")
    all_ids: list[int] = []
    for doc in text.split(eos_text):
        if doc:
            all_ids.extend(tokenizer.encode(doc))
            all_ids.append(eos_id)
    ids = np.asarray(all_ids, dtype=np.int32)
    if cache_path:
        np.save(cache_path, ids)
    return ids


def get_batch(
    ids: np.ndarray, batch_size: int, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    upper = len(ids) - seq_len - 1
    offsets = np.random.randint(0, upper, size=batch_size)
    x = np.stack([ids[o : o + seq_len] for o in offsets])
    y = np.stack([ids[o + 1 : o + 1 + seq_len] for o in offsets])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()


# ========== 2. 评估 ==========


def evaluate(model, ids: np.ndarray, batch_size: int, seq_len: int, iters: int = 20) -> float:
    lm = _lab02()
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch(ids, batch_size, seq_len)
            logits = model(x)
            losses.append(lm.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return float(np.mean(losses))


# ========== 3. checkpoint ==========


def save_checkpoint(path: str, model, step: int, config: dict = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"step": step, "config": config, "model": model.state_dict()}, path)


def load_checkpoint(path: str, model) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    return blob


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
    model.eval()
    ids = tokenizer.encode(prompt) if prompt else ([eos_id] if eos_id is not None else [])
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        ctx = torch.tensor([ids[-model.context_length:]], dtype=torch.long)
        logits = model(ctx)[0, -1]
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k and top_k < logits.shape[-1]:
                kth = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        if eos_id is not None and nxt == eos_id:
            break
        ids.append(nxt)
        new_ids.append(nxt)
    return new_ids


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
    lm = _lab02()
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, beta2), weight_decay=weight_decay
    )
    history = {"train": [], "val": []}
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    for step in range(1, steps + 1):
        x, y = get_batch(train_ids, batch_size, seq_len)
        logits = model(x)
        loss = lm.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.param_groups[0]["lr"] = lm.lr_at(step, lr, steps, warmup_steps, min_lr_ratio)
        opt.step()

        if step % eval_every == 0 or step == steps:
            val_loss = evaluate(model, val_ids, batch_size, seq_len, iters=eval_iters)
            history["train"].append((step, loss.item()))
            history["val"].append((step, val_loss))
            log(f"  step {step:>6}  train {loss.item():.4f}  val {val_loss:.4f}")
            if ckpt_dir:
                save_checkpoint(os.path.join(ckpt_dir, f"step_{step:06d}.pt"), model, step)
    return history


# ========== 6. 标准训练入口（与 train.py 相同） ==========


def ensure_tokenizer(
    corpus_path: str,
    vocab_size: int,
    special_tokens: list[str],
    train_bytes: int,
    cache_path: str = None,
) -> object:
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
