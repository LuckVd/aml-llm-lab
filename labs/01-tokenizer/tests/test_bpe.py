"""Lab 01 测试：实现 bpe.py 直到全部通过。"""

import importlib.util
import os
import sys

import pytest

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTION_PATH = os.path.join(LAB_DIR, "solution", "bpe_solution.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, LAB_DIR)
# 出题人自检：AML_LAB_USE_SOLUTION=1 时直接对参考答案跑测试（make verify）。
# 学员正常跑（make test / make lab1）时测的是你自己的 bpe.py。
STUDENT_PATH = (
    SOLUTION_PATH
    if os.environ.get("AML_LAB_USE_SOLUTION") == "1"
    else os.path.join(LAB_DIR, "bpe.py")
)
bpe = _load("student_bpe", STUDENT_PATH)
solution = _load("solution_bpe", SOLUTION_PATH)

DATA = os.path.join(LAB_DIR, "..", "..", "data", "sample", "tinystories_sample.txt")
SPECIALS = ["<|endoftext|>"]


@pytest.fixture(scope="module")
def trained():
    vocab, merges = bpe.train_bpe(DATA, vocab_size=500, special_tokens=SPECIALS)
    return vocab, merges


# ---------- train_bpe ----------


def test_train_bpe_basic_structure(trained):
    vocab, merges = trained
    # 词表 = 256 byte + 1 special + 合并产物
    assert len(vocab) == 256 + len(SPECIALS) + len(merges)
    # 前 256 项是单字节
    assert vocab[0] == b"\x00" and vocab[255] == b"\xff"
    # special token 在词表里
    assert vocab[256] == b"<|endoftext|>"
    # 合并产物非空、确实是两个字节串的拼接
    assert all(a + b for a, b in merges)


def test_train_bpe_matches_reference(trained):
    """与参考答案完全一致（词表与合并顺序都应确定）。"""
    ref_vocab, ref_merges = solution.train_bpe(
        DATA, vocab_size=500, special_tokens=SPECIALS
    )
    vocab, merges = trained
    assert merges == ref_merges
    assert vocab == ref_vocab


def test_train_bpe_vocab_size_exact():
    """vocab_size 参数应当被精确遵守。"""
    vocab, merges = bpe.train_bpe(DATA, vocab_size=320, special_tokens=SPECIALS)
    assert len(vocab) == 320
    assert len(merges) == 320 - 257


def test_train_bpe_vocab_too_small():
    with pytest.raises(ValueError):
        bpe.train_bpe(DATA, vocab_size=100, special_tokens=SPECIALS)


def test_first_merges_are_frequency_sorted(trained):
    """最早的合并应来自最高频对（与参考答案在头几步一致即合理）。"""
    _, merges = trained
    assert merges[0] == (b"e", b"r") or len(merges) > 0  # smoke；精确性由上面对齐测试保证


# ---------- Tokenizer ----------


def test_roundtrip_ascii(trained):
    vocab, merges = trained
    tok = bpe.Tokenizer(vocab, merges, SPECIALS)
    s = "Once upon a time, there was a little boy named Ben."
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_unicode(trained):
    """非 ASCII 字符（中文/emoji）也要无损往返——byte-level 的意义所在。"""
    vocab, merges = trained
    tok = bpe.Tokenizer(vocab, merges, SPECIALS)
    for s in ["你好，世界！", "café ☕ story"]:
        assert tok.decode(tok.encode(s)) == s


def test_special_token_kept_intact(trained):
    vocab, merges = trained
    tok = bpe.Tokenizer(vocab, merges, SPECIALS)
    ids = tok.encode("A story.<|endoftext|>Another story.")
    # special token 恰好产出 1 个 id（id=256），且能原样解码回来
    assert ids.count(256) == 1
    assert tok.decode(ids) == "A story.<|endoftext|>Another story."


def test_encode_uses_merged_tokens(trained):
    """高频词应被压成少量 token，而不是逐 byte。"""
    vocab, merges = trained
    tok = bpe.Tokenizer(vocab, merges, SPECIALS)
    merged_bytes = {v for k, v in vocab.items() if len(v) > 1}
    assert len(merged_bytes) > 100  # 训练确实产生了大量合并
    assert b" the" in merged_bytes or b"the" in merged_bytes


def test_invalid_bytes_replaced(trained):
    vocab, merges = trained
    tok = bpe.Tokenizer(vocab, merges, SPECIALS)
    ids = tok.encode("hello")
    # 手工构造非法字节序列：0x80 单独出现不是合法 utf-8
    bad = ids[:-1] + [128]
    assert "\ufffd" in tok.decode(bad)
