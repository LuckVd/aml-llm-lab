"""
Lab 01: byte-level BPE tokenizer（你要实现的文件）

实现两个东西：
  1. train_bpe()        —— 在语料上训练 BPE，得到词表 + 合并规则
  2. Tokenizer 类       —— 用训练好的词表做 encode / decode

全部 TODO 完成并让 tests/test_bpe.py 全绿后，运行 demo_tokenizer.py 看效果。
卡住了再看 solution/bpe_solution.py。

原理讲解见 ../book/01-tokenizer.md，先读它。
"""

from collections import Counter

# GPT-2 的预切分正则：把文本切成"词级"小片段，BPE 合并只在片段内部发生。
# 建议直接使用，不必修改（想深究每一段的含义，见 book/01-tokenizer.md）。
import regex as re

PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str) -> list[str]:
    """把文本切成预切分片段（BPE 合并的边界）。"""
    return re.findall(PRETOKENIZE_PATTERN, text)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """在文本文件上训练 byte-level BPE。

    参数:
        input_path: 训练语料路径（utf-8 文本）
        vocab_size: 目标词表总大小（含 256 个 byte、special tokens、全部合并产物）
        special_tokens: 需要加入词表的特殊 token，如 ["<|endoftext|>"]

    返回:
        vocab:  dict[int, bytes]，token id -> 字节串
        merges: list[tuple[bytes, bytes]]，按合并顺序排列的合并规则
    """
    special_tokens = special_tokens or []
    text = open(input_path, encoding="utf-8").read()

    # ---- 步骤 0：初始化词表 ----------------------------------------
    # TODO: vocab 的前 256 项是全部单字节 b"\x00" .. b"\xff"，
    #       然后追加 special_tokens（编码为 utf-8 字节）。
    #       校验 vocab_size >= 256 + len(special_tokens)，否则 raise ValueError。
    vocab: dict[int, bytes] = {}
    raise NotImplementedError

    # ---- 步骤 1：预切分并统计片段频率 --------------------------------
    # TODO: 用 pretokenize(text) 切片，用 Counter 统计每个片段出现次数。
    #       片段转成 tuple[bytes]（每个元素是单字节），方便后续合并操作。
    #       提示：只需要统计"唯一片段"的频率，语料大时这是性能关键。
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    raise NotImplementedError

    # ---- 步骤 2：循环合并 -------------------------------------------
    # TODO: 重复以下过程，直到词表达到 vocab_size：
    #   a) 统计所有相邻字节对的频率（按片段频率加权，不是按对出现次数！）
    #      （提示：统计时跳过长度为 1 的片段——没有"相邻对"可言）
    #   b) 选出频率最高的对；并列时取"字典序最大"的字节对（GPT-2 惯例，保证确定性）
    #      （若已无对可合并，提前结束）
    #   c) 把该对的并接字节追加进 vocab，记录合并规则到 merges
    #   d) 在所有包含该对的片段中执行合并（相邻、从左到右、不重叠）
    merges: list[tuple[bytes, bytes]] = []
    raise NotImplementedError

    return vocab, merges


class Tokenizer:
    """加载训练好的 BPE，做文本 <-> token id 的双向转换。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        # id 查找表：字节串 -> id（vocab 反转）
        self.byte_to_id = {v: k for k, v in vocab.items()}
        # 合并规则 -> 优先级（序号小的先合并）
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        if self.special_tokens:
            self.special_re = re.compile(
                "|".join(re.escape(t) for t in self.special_tokens)
            )

    def _encode_piece(self, piece: bytes) -> list[int]:
        """对一个不含 special token 的字节片段做 BPE 编码。

        算法：先拆成单字节序列，然后反复找出"优先级最高（rank 最小）的相邻对"
        合并之，直到没有任何对在 merge_ranks 中为止。
        最后把每个字节串查表转成 id。
        """
        # TODO 实现（提示：每轮扫描找 rank 最小的对即可，不必追求堆优化）
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """文本 -> token id 序列。special token 必须作为整体编号，不参与 BPE。"""
        # TODO:
        #   1. 若有 special_tokens，用 self.special_re.split(text) 把文本切开
        #      （切出的元素要么是普通文本，要么恰好是某个 special token）
        #   2. 普通文本：pretokenize -> 每个片段 utf-8 编码 -> _encode_piece
        #   3. special token：直接查 self.byte_to_id
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """token id 序列 -> 文本。非法 utf-8 字节用 U+FFFD 替换。"""
        # TODO: 拼接所有 id 对应的字节串，bytes.decode(errors="replace")
        raise NotImplementedError
