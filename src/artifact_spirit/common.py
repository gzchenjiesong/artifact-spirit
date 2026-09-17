"""跨层共享的**纯工具**（无 I/O、无状态、无第三方依赖）。

放在包根而非某一层，是因为它被 store / core / model / runtime 共同需要；
架构规则 R1–R8 约束的是**层与层**的依赖方向，本模块不构成层。

约定：**全项目时间格式统一为 ISO8601 字符串**（ENC-000 §3.1）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

__all__ = [
    "collapse_ws",
    "content_hash_of",
    "estimate_tokens",
    "first_sentence",
    "is_cjk",
    "normalize_content",
    "now_iso",
    "summarize",
    "truncate_to_tokens",
]

# CJK 及全角字符区段（用于分词与 token 估算）
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x2E80, 0x2EFF),  # CJK 部首补充
    (0x3000, 0x303F),  # CJK 符号与标点
    (0x3040, 0x30FF),  # 日文假名
    (0x3400, 0x4DBF),  # CJK 扩展 A
    (0x4E00, 0x9FFF),  # CJK 统一表意
    (0xF900, 0xFAFF),  # CJK 兼容表意
    (0xAC00, 0xD7AF),  # 谚文
    (0xFF00, 0xFFEF),  # 全角形式
)


def now_iso(timespec: str = "seconds") -> str:
    """带本地时区偏移的 ISO8601 时间戳。

    Args:
        timespec: 精度。默认秒级（全项目统一约定）；
            工作记忆的 ``last_touched`` 用 ``"microseconds"``——同一秒内的多次触碰
            必须能正确排序，秒级精度会让"最近触碰"退化为插入顺序。
    """
    return datetime.now(UTC).astimezone().isoformat(timespec=timespec)


def is_cjk(codepoint: int) -> bool:
    """判断码点是否属于 CJK / 全角区段。"""
    for low, high in _CJK_RANGES:
        if low <= codepoint <= high:
            return True
    return False


def collapse_ws(text: str) -> str:
    """折叠所有空白为单空格并去首尾空白。"""
    return " ".join(text.split())


def estimate_tokens(text: str | None) -> int:
    """估算 token 数：CJK 按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token。

    只用于**预算裁剪**，不需要精确——真实分词器属于 AL4 的职责。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if is_cjk(ord(ch)))
    other = len(text) - cjk
    return int(cjk + other / 4) + 1


def truncate_to_tokens(text: str, budget: int) -> str:
    """按 token 预算截断文本，返回**预算内的最长前缀**。

    ``budget <= 0`` 或"连一个字符都装不下"时返回空串。

    裁剪用二分：:func:`estimate_tokens` 对前缀单调不减（CJK 数与其余字符数
    都非降），所以二分得到的边界就是最长可行前缀——不是"先按比例估、再逐字回退"。

    **本函数必须是全函数（对任何输入都终止）**：旧实现逐字回退，
    步长 ``max(1, len(result) - 1)`` 在只剩 1 个字符时切片结果等于自身，
    而单字（汉字）估算恒为 2 —— 于是 ``budget == 1`` 会**原地死循环**。
    受害路径是宿主每轮都调的 ``system_prompt_block``：``token_budget=60`` 时
    内部派生出的核心记忆预算正好是 1，宿主进程被永久挂死（DES-REV-009 P0-1）。
    """
    if budget <= 0:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    if estimate_tokens(text[:1]) > budget:
        return ""  # 预算容不下任何字符，没有可行前缀
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def first_sentence(text: str, *, limit: int = 60) -> str:
    """取首句（L0 缺失时的退化策略，LLD-AL2 M9）。"""
    if not text:
        return ""
    stripped = text.strip()
    for sep in ("。", "！", "？", "\n", ". ", "! ", "? "):
        idx = stripped.find(sep)
        if 0 <= idx < limit * 2:
            candidate = stripped[: idx + len(sep)].strip()
            if candidate:
                return candidate
    return stripped[:limit].strip()


def summarize(text: str, *, budget: int = 24) -> str:
    """生成 L0 摘要：首句 + **token 预算截断**。

    这里的关键不是"怎么摘要得漂亮"，而是**摘要必须比正文短**。
    首句常常就是全文（用户一句话说不完的偏好，往往就写在一句话里），
    所以必须有第二道保险：截断到预算。
    """
    if not text:
        return ""
    candidate = first_sentence(text, limit=max(budget, 20))
    if estimate_tokens(candidate) <= budget:
        return candidate
    clipped = truncate_to_tokens(candidate, budget)
    return clipped.rstrip("，,。.、；; ") + "…"


# --------------------------------------------------------------------------- #
# 内容寻址（D-21 / INV-14）
# --------------------------------------------------------------------------- #
#
# 放在包根而非 store/ 或 core/：它是**领域概念**（"这段内容是不是同一个事实"），
# 被 AL2 的去重与 AL3 的导入幂等同时需要。放在任一层里都会让另一层越界引用。


def normalize_content(text: str | None) -> str:
    """内容归一化：去首尾、折叠空白、拉丁字母小写。"""
    if not text:
        return ""
    return collapse_ws(text).casefold()


MEMORY_TYPES: tuple[str, ...] = (
    "fact",
    "preference",
    "event",
    "entity",
    "skill",
    "identity",
    "soul",
    "intent",
)
"""记忆类型取值域（**领域常量**）。

放在包根的原因与 :func:`content_hash_of` 相同：它被 AL2 的层服务、AL2 的提取契约
（schema 与提示词同源，C12）、以及 AL3 的写入校验同时需要。

**位置本身是设计约束**：一旦它住在 ``core/`` 里，``extract/`` 就必须反向依赖
``core`` 包——而 ``core/__init__`` 会加载 facade，facade 又依赖 extractor，
于是"先 import extract 的调用方"会撞上循环导入。领域常量放在无人依赖的包根，
这条环从结构上就不存在。
"""


def content_hash_of(
    *,
    content: str,
    subject: str | None = None,
    predicate: str | None = None,
    object_: str | None = None,
    scope: dict | None = None,
) -> str:
    """计算内容寻址指纹（D-21 / INV-14）。

    三元组齐备时按 ``{subject, predicate, object, scope}`` 计算（``sort_keys=True``），
    保证"同一事实、不同措辞"得到**同一指纹**；三元组缺失时退化为 ``normalize(content)``。

    指纹是**导入幂等与去重的锚点**，不是主键——``memories.content_hash`` 上是
    **非唯一索引**：同一事实在不同情境（scope）下允许共存。
    """
    if subject and predicate and object_:
        payload = {
            "subject": normalize_content(subject),
            "predicate": normalize_content(predicate),
            "object": normalize_content(object_),
            "scope": scope or {},
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    else:
        raw = normalize_content(content)

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"sha256:{digest}"
