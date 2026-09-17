"""层服务契约（LLD-AL2 §2.3）。

六个层服务共享同一接口，但**各自有特有的语义**：

| 服务 | 特有语义 |
|---|---|
| `sensory` | 只做显著性过滤，**不产出落库候选**（噪声不落盘） |
| `working` | 组块聚类 + 容量控制（4±1）+ 意图槽提取 |
| `episodic` | 会话 → 事件序列，带时空上下文 |
| `semantic` | 去情境化事实 / 偏好 / 实体 |
| `procedural` | 技能与可复用轨迹（**MVP 仅占位**，C11） |
| `core_memory` | identity / soul 装载与更新（高门槛） |
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ...common import MEMORY_TYPES
from ...store.base import MemoryBackend
from ..base import Clock, RecallQuery, Scored, TurnContext

__all__ = ["MEMORY_TYPES", "BaseLayerService", "LayerService", "derive_chunk_key"]


_CHUNK_KEY_RE = re.compile(r"[A-Z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,6}")


def derive_chunk_key(text: str, *, fallback: str = "misc") -> str:
    """从文本推出"话题标识"——同话题的输入会累积到同一个组块。

    零 LLM：取首个专有名词或前几个汉字。够用且免费。
    """
    match = _CHUNK_KEY_RE.search(text or "")
    if match:
        return match.group(0).casefold()
    cleaned = " ".join((text or "").split())[:8]
    return cleaned.casefold() or fallback


class LayerService(Protocol):
    """层服务协议。"""

    layer: str

    def recall(self, q: RecallQuery) -> list[Scored]: ...

    def candidates(self, ctx: TurnContext) -> list[dict]: ...


@dataclass(slots=True)
class BaseLayerService:
    """共享实现的基类（不是 ABC——器灵用 Protocol 而非继承表达契约）。

    默认的 :meth:`recall` 过滤本层；各层的特有语义体现在 :meth:`candidates`。
    """

    backend: MemoryBackend
    clock: Clock
    layer: str = "semantic"

    def recall(self, q: RecallQuery) -> list[Scored]:
        """只做"层过滤"——真正的多因子打分在 :mod:`artifact_spirit.core.recall`。"""
        records = self.backend.query(layer=self.layer, status=None, limit=q.top_k * 4)
        return [Scored(record=r, raw={}, score=0.0) for r in records]

    def candidates(self, ctx: TurnContext) -> list[dict]:  # pragma: no cover - 抽象
        raise NotImplementedError
