"""Lab 03 测试：实现 train.py 直到全部通过（用极小切片，分钟级）。"""

import importlib.util
import math
import os
import sys

import numpy as np
import pytest
import torch

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(LAB_DIR, "..", ".."))

LAB01_DIR = os.path.join(REPO_ROOT, "labs", "01-tokenizer")
LAB02_DIR = os.path.join(REPO_ROOT, "labs", "02-transformer")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


USE_SOLUTION = os.environ.get("AML_LAB_USE_SOLUTION") == "1"


def _pick(lab_dir, student, solution):
    return os.path.join(lab_dir, solution) if USE_SOLUTION else os.path.join(lab_dir, student)


lab01 = _load("lab01_for_test", _pick(LAB01_DIR, "bpe.py", "solution/bpe_solution.py"))
lab02 = _load("lab02_for_test", _pick(LAB02_DIR, "model.py", "solution/model_solution.py"))
m = _load("lab03_for_test", _pick(LAB_DIR, "train.py", "solution/train_solution.py"))

SPECIAL = "<|endoftext|>"


@pytest.fixture(scope="module")
def tiny_world(tmp_path_factory):
    """极小语料 + 极小 tokenizer + 极小模型。"""
    tmp = tmp_path_factory.mktemp("lab03")
    stories = [
        "Tom has a red ball. He likes the ball.",
        "The cat sat on the mat and slept.",
        "Anna and Ben 中文 测试 went to the park.",
        "A dog ran fast. The dog was happy.",
    ]
    corpus = f"{SPECIAL}".join(stories) + SPECIAL
    corpus_path = tmp / "tiny.txt"
    corpus_path.write_text(corpus, encoding="utf-8")

    tok = lab01.Tokenizer(
        *lab01.train_bpe(str(corpus_path), vocab_size=280, special_tokens=[SPECIAL]),
        special_tokens=[SPECIAL],
    )
    eos_id = tok.byte_to_id[SPECIAL.encode()]
    return tmp, str(corpus_path), corpus, tok, eos_id


def tiny_model(vocab_size, seed=0):
    torch.manual_seed(seed)
    return lab02.TransformerLM(
        vocab_size=vocab_size, d_model=48, n_layers=2, n_heads=3, d_ff=96, context_length=64
    )


# ---------- 数据管道 ----------


def test_tokenize_corpus_roundtrip_and_eos(tiny_world):
    tmp, corpus_path, corpus, tok, eos_id = tiny_world
    ids = m.tokenize_corpus(corpus_path, tok, eos_id)
    assert isinstance(ids, np.ndarray) and ids.dtype == np.int32 and ids.ndim == 1
    # 4 个文档 -> 4 个 eos
    assert (ids == eos_id).sum() == 4
    # 无损往返
    assert tok.decode(ids.tolist()) == corpus


def test_tokenize_corpus_cache(tiny_world):
    tmp, corpus_path, corpus, tok, eos_id = tiny_world
    cache = str(tmp / "tiny.ids.npy")
    ids1 = m.tokenize_corpus(corpus_path, tok, eos_id, cache_path=cache)
    assert os.path.exists(cache)
    # 语料删掉也能从缓存读（证明真的走了缓存）
    os.rename(corpus_path, str(tmp / "hidden.txt"))
    ids2 = m.tokenize_corpus(corpus_path, tok, eos_id, cache_path=cache)
    os.rename(str(tmp / "hidden.txt"), corpus_path)
    assert np.array_equal(ids1, ids2)


def test_get_batch_shifted(tiny_world):
    tmp, corpus_path, corpus, tok, eos_id = tiny_world
    ids = m.tokenize_corpus(corpus_path, tok, eos_id)
    np.random.seed(0)
    x, y = m.get_batch(ids, batch_size=8, seq_len=16)
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == torch.long and y.dtype == torch.long
    # y 是 x 错开一位：同一批窗口内 x[:,1:] == y[:,:-1]
    assert torch.equal(x[:, 1:], y[:, :-1])


# ---------- 评估 / checkpoint ----------


def test_evaluate_untrained_close_to_ln_v(tiny_world):
    # 用极小语料重复拼接出足够长的 ids
    tmp, corpus_path, corpus, tok, eos_id = tiny_world
    base = m.tokenize_corpus(corpus_path, tok, eos_id)
    ids = np.tile(base, 40)
    model = tiny_model(len(tok.vocab))
    loss = m.evaluate(model, ids, batch_size=4, seq_len=32, iters=5)
    assert abs(loss - math.log(len(tok.vocab))) < 0.5  # 未训练 ≈ ln(V)


def test_checkpoint_roundtrip(tiny_world):
    tmp, _, _, tok, _ = tiny_world
    model_a = tiny_model(len(tok.vocab), seed=1)
    model_b = tiny_model(len(tok.vocab), seed=2)
    path = str(tmp / "ckpt.pt")
    m.save_checkpoint(path, model_a, step=7, config={"d_model": 48})
    blob = m.load_checkpoint(path, model_b)
    assert blob["step"] == 7
    idx = torch.randint(0, len(tok.vocab), (2, 8))
    with torch.no_grad():
        assert torch.allclose(model_a(idx), model_b(idx), atol=1e-6)


# ---------- 采样 ----------


def test_sample_greedy_matches_manual(tiny_world):
    tmp, corpus_path, corpus, tok, eos_id = tiny_world
    base = m.tokenize_corpus(corpus_path, tok, eos_id)
    model = tiny_model(len(tok.vocab), seed=3)
    prompt = "The cat"
    got = m.sample(model, tok, prompt, max_new_tokens=12, temperature=1.0, top_k=1, eos_id=eos_id)
    # 手工贪心滚动
    ids = tok.encode(prompt)
    ref = []
    with torch.no_grad():
        for _ in range(12):
            ctx = torch.tensor([ids[-model.context_length:]], dtype=torch.long)
            nxt = int(model(ctx)[0, -1].argmax())
            if nxt == eos_id:
                break
            ids.append(nxt)
            ref.append(nxt)
    assert got == ref


def test_sample_lengths_and_prompt_prefix(tiny_world):
    tmp, corpus_path, corpus, tok, eos_id = tiny_world
    model = tiny_model(len(tok.vocab), seed=4)
    out = m.sample(model, tok, "Tom has", max_new_tokens=20, temperature=1.0, top_k=5, eos_id=eos_id)
    assert 0 <= len(out) <= 20
    assert all(isinstance(i, int) for i in out)
    full = m.sample(model, tok, "", max_new_tokens=10, temperature=1.0, top_k=5, eos_id=eos_id)
    assert len(full) <= 10  # 空 prompt 也能起头


# ---------- 训练循环 ----------


def test_train_loop_reduces_loss(tiny_world):
    tmp, corpus_path, corpus, tok, eos_id = tiny_world
    base = m.tokenize_corpus(corpus_path, tok, eos_id)
    train_ids = np.tile(base, 30)
    val_ids = np.tile(base, 10)
    model = tiny_model(len(tok.vocab), seed=5)
    torch.manual_seed(0)
    np.random.seed(0)
    history = m.train(
        model, train_ids, val_ids,
        steps=12, batch_size=8, seq_len=32, lr=2e-3, warmup_steps=2,
        eval_every=4, eval_iters=2, ckpt_dir=str(tmp / "ckpts"), log=lambda *a: None,
    )
    train = [l for _, l in history["train"]]
    assert len(history["train"]) == 3 and len(history["val"]) == 3
    assert train[-1] < train[0] - 0.3, f"12 步应明显下降: {train}"
    assert (tmp / "ckpts" / "step_000004.pt").exists()
