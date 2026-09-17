"""语义记忆（LLD-AL2 §5 M2）。

**去情境化**的事实 / 偏好 / 实体。

与情景记忆的分界：
- 情景记忆回答"**当时发生了什么**"（带时间地点，不可复用）
- 语义记忆回答"**一般来说是什么**"（去掉情境，可复用）

从情景到语义的提升由 :class:`~artifact_spirit.core.consolidation.Consolidator` 完成，
本层只负责组织与召回。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base import RecallQuery, Scored, TurnContext
from .base import BaseLayerService

__all__ = ["SemanticLayer"]

SEMANTIC_TYPES = ("fact", "preference", "entity")


@dataclass(slots=True)
class SemanticLayer(BaseLayerService):
    """语义记忆层。"""

    layer: str = "semantic"

    def candidates(self, ctx: TurnContext) -> list[dict]:
        """语义记忆由**提取器**从一轮对话中产出（见 extract/extractor），
        而不是由层服务自己生成——层服务只定义"什么样的内容属于这一层"。
        """
        return [
            {"layer": self.layer, "type": "fact"},
            {"layer": self.layer, "type": "preference"},
        ]

    def recall(self, q: RecallQuery) -> list[Scored]:
        records = self.backend.query(layer="semantic", status=None, limit=q.top_k * 4)
        return [Scored(record=r, raw={}, score=0.0) for r in records]
