"""AL2 的提取子系统：schema / 提取器 / 去重器。

三者都属于"把一段文本变成可落库记忆"的链路，与 ``core/`` 并列而非从属——
它们**不含认知机制**（巩固、衰减、扩散），只做形式转换。
"""

from __future__ import annotations

from .dedup import Decision, DedupDecision, Deduplicator
from .extractor import Extractor, ExtractResult
from .schema import (
    DEDUP_SCHEMA,
    EXTRACTION_SCHEMA,
    SYSTEM_PROMPT,
    parse_memories,
    validation_errors,
)

__all__ = [
    "DEDUP_SCHEMA",
    "EXTRACTION_SCHEMA",
    "SYSTEM_PROMPT",
    "Decision",
    "DedupDecision",
    "Deduplicator",
    "ExtractResult",
    "Extractor",
    "parse_memories",
    "validation_errors",
]
