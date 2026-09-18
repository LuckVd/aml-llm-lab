"""参考答案：byte-level BPE（先自己写 TODO，卡住再对照）。"""

from collections import Counter

import regex as re

PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str) -> list[str]:
    return re.findall(PRETOKENIZE_PATTERN, text)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    text = open(input_path, encoding="utf-8").read()

    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size 太小，装不下 256 个 byte 和 special tokens")

    # 步骤 0：词表 = 256 单字节 + special tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # 步骤 1：预切分 + 统计唯一片段频率
    word_counts: Counter[tuple[bytes, ...]] = Counter(
        tuple(bytes([b]) for b in w.encode("utf-8")) for w in pretokenize(text)
    )

    num_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []

    def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        """在单个片段内从左到右、不重叠地合并 pair。"""
        merged = pair[0] + pair[1]
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    for _ in range(num_merges):
        # a) 统计相邻对（按片段频率加权）
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for word, freq in word_counts.items():
            if len(word) < 2:
                continue
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += freq
        if not pair_counts:
            break

        # b) 频率最高；并列取字典序最大（确定性，GPT-2 惯例）
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))

        # c) 记录合并，扩展词表
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        # d) 在所有片段中执行合并（只更新含该对的片段）
        new_counts: Counter[tuple[bytes, ...]] = Counter()
        for word, freq in word_counts.items():
            if len(word) >= 2 and any(
                word[i] == best[0] and word[i + 1] == best[1] for i in range(len(word) - 1)
            ):
                new_counts[merge_word(word, best)] += freq
            else:
                new_counts[word] += freq
        word_counts = new_counts

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        if self.special_tokens:
            self.special_re = re.compile(
                "|".join(f"({re.escape(t)})" for t in self.special_tokens)
            )

    def _encode_piece(self, piece: bytes) -> list[int]:
        word = [bytes([b]) for b in piece]
        while len(word) > 1:
            # 找 rank 最小的相邻对
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(
                (p for p in pairs if p in self.merge_ranks),
                key=lambda p: self.merge_ranks[p],
                default=None,
            )
            if best is None:
                break
            merged = best[0] + best[1]
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        return [self.byte_to_id[b] for b in word]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self.special_tokens:
            parts = self.special_re.split(text)
        else:
            parts = [text]
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.byte_to_id[part.encode("utf-8")])
                continue
            for piece in pretokenize(part):
                ids.extend(self._encode_piece(piece.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
