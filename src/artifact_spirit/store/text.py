"""文本归一化与内容寻址指纹（D-21 / INV-14）。

本模块**纯计算**，无 I/O。

## 为什么需要 CJK 归一化

FTS5 内置的 ``unicode61`` 分词器按"非字母数字"切分，而中日韩文字本身属于字母类，
于是一整句中文会变成**一个超长 token**——"深色主题"这类词根本检索不到。
``trigram`` 分词器可解，但要求查询 ≥ 3 字符，**2 字中文词（"偏好""花生"）依然无解**。

器灵面向中文使用者，2 字词是主流而非边缘情形，因此采用：

    入库时：在每个 CJK 字符两侧插入空格 → 每个字成为独立 token
    查询时：把连续的单字 CJK token 组成**短语**（"深 色 主 题"）

这样"深色主题"是短语匹配（精确且可索引），"RAGFlow"保持原样（英文单词不受影响），
中英混排亦正确。代价是 FTS 表自存一份归一化文本（体积约为正文的 1 倍），
但它是**派生索引**而非真相源——可随时由 ``memories`` 重建（INV-1）。
"""

from __future__ import annotations

from ..common import collapse_ws, content_hash_of, is_cjk, normalize_content

__all__ = [
    "content_hash_of",
    "fts_match_query",
    "normalize_content",
    "normalize_for_fts",
]

# FTS5 各列权重（content, abstract, subject, object）—— 正文最重要
FTS_COLUMN_WEIGHTS: tuple[float, ...] = (10.0, 5.0, 3.0, 3.0)


def normalize_for_fts(text: str | None) -> str:
    """把文本归一化为"CJK 逐字 + 其余原样"的形式，供 FTS5 索引与检索。"""
    if not text:
        return ""
    chunks: list[str] = []
    for ch in text:
        if is_cjk(ord(ch)):
            chunks.append(" ")
            chunks.append(ch)
            chunks.append(" ")
        else:
            chunks.append(ch)
    return collapse_ws("".join(chunks))


def _quote(term: str) -> str:
    """把 token 包成 FTS5 短语并转义内部双引号。"""
    return '"' + term.replace('"', '""') + '"'


CJK_PHRASE_MAX = 2
"""CJK 连续串切分的**最大字长**。

为什么不是"整串当一个短语"：中文查询往往是一整句话（"用户喜欢什么颜色"），
把整句当短语等于要求**逐字完全子串匹配**，召回率会低到不可用。

为什么是 2 而不是 1：单字（unigram）召回率高但噪声大（"的""是"满天飞），
而**二元组（bigram）是 CJK 检索的通行折中**——它足够精确，且对 2 字词
（"偏好""花生"）是天然的精确匹配。索引侧仍是逐字 token，所以 bigram 短语
必定能命中。
"""


def _cjk_units(run: str) -> list[str]:
    """把一段连续 CJK 切成短语单元。

    返回的每个单元是**空格分隔的单字序列**——因为索引侧把每个 CJK 字
    做成了独立 token，短语里也必须逐个分开才算 token 序列
    （``"深色"`` 是一个 token，``"深 色"`` 才是两个连续 token）。
    """
    if len(run) <= CJK_PHRASE_MAX:
        return [" ".join(run)]
    return [
        " ".join(run[i : i + CJK_PHRASE_MAX]) for i in range(len(run) - CJK_PHRASE_MAX + 1)
    ]


def fts_match_query(text: str | None) -> str | None:
    """把自然语言查询翻译为 FTS5 MATCH 表达式。

    规则：
    - 连续 CJK 串切成 **bigram**（见 :data:`CJK_PHRASE_MAX`），各自成短语
    - 其余 token 各自成为一个短语（英文单词、数字串原样）
    - 组间用 ``OR`` 连接（提高召回，排序交给 BM25）

    无法构造查询时返回 ``None``（调用方应跳过关键词路）。
    """
    normalized = normalize_for_fts(text)
    if not normalized:
        return None

    groups: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            groups.extend(_cjk_units("".join(buffer)))
            buffer.clear()

    for token in normalized.split():
        if len(token) == 1 and is_cjk(ord(token)):
            buffer.append(token)
            continue
        flush()
        groups.append(token)

    flush()
    return " OR ".join(_quote(g) for g in groups) or None


# --------------------------------------------------------------------------- #
# 内容寻址（D-21 / INV-14）
# --------------------------------------------------------------------------- #
# 实现在包根 ``common.py``——它是**领域概念**（"这段内容是不是同一个事实"），
# 被 AL2 的去重与 AL3 的导入幂等同时需要。放在任一层里都会让另一层越界引用。
# 此处直接转发，使 ``from artifact_spirit.store.text import content_hash_of`` 保持可用。

