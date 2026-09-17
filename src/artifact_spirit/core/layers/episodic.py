"""情景记忆（LLD-AL2 §5 M2 / M9）。

会话 → 事件序列，带**时空上下文**。

情景记忆回答的是"**当时具体是这么说的**"——它是 L2 展开的默认来源，
也是巩固提升到语义记忆的原料。
"""

from __future__ import annotations

from dataclasses import dataclass

from ...common import first_sentence
from ..base import RecallQuery, Scored, TurnContext
from .base import BaseLayerService

__all__ = ["EpisodicLayer"]


@dataclass(slots=True)
class EpisodicLayer(BaseLayerService):
    """情景记忆层。"""

    layer: str = "episodic"

    def candidates(self, ctx: TurnContext) -> list[dict]:
        """一轮对话本身不直接产出情景记忆——情景记忆由**会话结束时的巩固**产生。

        实时把每一轮都写成 episode 会让记忆表被大量琐碎片段淹没，
        且失去"事件"应有的整体性。这里返回空是刻意的。
        """
        return []

    def recent(self, limit: int = 10) -> list[Scored]:
        """最近的情景记忆（按 `id` 倒序——ULID 时间有序）。"""
        records = self.backend.query(layer="episodic", status=None, limit=limit)
        return [Scored(record=r, raw={}, score=0.0) for r in sorted(records, key=lambda r: r.id, reverse=True)]

    def recall(self, q: RecallQuery) -> list[Scored]:
        return self.recent(limit=q.top_k)


def episode_abstract(content: str) -> str:
    """情景记忆的 L0 摘要。"""
    return first_sentence(content)
