"""程序性记忆（LLD-AL2 §5 M2 · **MVP 仅占位**）。

技能与可复用轨迹。

> **C11 纪律**：MVP **只建表**，不实现固化逻辑。
> 提前实现它属于范围蔓延——而且"某模式复用 ≥K 次就固化为技能"这件事，
> 在没有真实使用数据之前无从调参。

本模块因此**刻意保持为空壳**：它声明了层的存在与归属，
但 ``candidates()`` 恒返回空，也不产出任何写意图。这条纪律由测试断言
（见 ``tests/test_core.py::test_procedural_layer_is_placeholder``）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base import RecallQuery, Scored, TurnContext
from .base import BaseLayerService

__all__ = ["SOLIDIFY_THRESHOLD_PLACEHOLDER", "ProceduralLayer"]

SOLIDIFY_THRESHOLD_PLACEHOLDER = 3
"""固化阈值 K 的**占位值**——待有真实数据后校准（LLD-AL2 §10 未决项 3）。"""


@dataclass(slots=True)
class ProceduralLayer(BaseLayerService):
    """程序性记忆层（占位）。"""

    layer: str = "procedural"

    def candidates(self, ctx: TurnContext) -> list[dict]:
        """MVP 不产出候选（C11）。"""
        return []

    def recall(self, q: RecallQuery) -> list[Scored]:
        records = self.backend.query(layer="procedural", status=None, limit=q.top_k)
        return [Scored(record=r, raw={}, score=0.0) for r in records]
