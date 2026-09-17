"""核心记忆（LLD-AL2 §5 M2 · INV-11）。

``identity`` / ``soul`` 的装载与更新——**人格的载体**。

## 为什么它有"高门槛"

其他四层记错了，损失是"一条记忆不准"；核心记忆记错了，**污染的是人格**。
所以这里的门比别处高得多：

- 只有 ``layer='core'`` 的记录算数
- 更新必须过 :data:`CORE_UPDATE_THRESHOLD` 的置信度门槛
- **M3+ 必须经交叉验证**（INV-11）——MVP 用高门槛替代，但接口已按此设计
- 核心记忆**永不参与**衰减降级与优化删除（见 consolidation 的 `NON_OPTIMIZABLE_LAYERS`）
"""

from __future__ import annotations

from dataclasses import dataclass

from ...common import truncate_to_tokens
from ...store.base import AuditEvent
from ..base import RecallQuery, Scored, TurnContext, WriteIntent
from .base import BaseLayerService

__all__ = ["CORE_TYPES", "CORE_UPDATE_THRESHOLD", "CoreMemoryLayer"]

CORE_TYPES = ("identity", "soul")

CORE_UPDATE_THRESHOLD = 0.9
"""核心记忆更新的置信度门槛（INV-11）。

比其他层高得多：这里错了污染的是人格，不是一条记忆。
"""


@dataclass(slots=True)
class CoreMemoryLayer(BaseLayerService):
    """核心记忆层。"""

    layer: str = "core"

    # ------------------------------------------------------------------ #

    def load(self) -> list:
        """装载全部核心记忆（``system_prompt_block`` 的数据源）。"""
        return self.backend.query(layer="core", status=None, limit=100)

    def identity(self) -> str | None:
        """取身份陈述（``type='identity'``）。"""
        for record in self.load():
            if record.type == "identity":
                return record.content
        return None

    def soul(self) -> str | None:
        """取器灵本体陈述（``type='soul'``）。"""
        for record in self.load():
            if record.type == "soul":
                return record.content
        return None

    def prompt_block(self, *, token_budget: int = 400) -> str:
        """核心记忆摘要（注入系统提示的部分）。

        **必须有 token 预算并截断**（C11）——核心记忆也不能无限增长，
        否则"注入人格"会挤掉"注入任务"。
        """
        parts: list[str] = []
        identity = self.identity()
        soul = self.soul()
        if identity:
            parts.append(f"【我是谁】{identity}")
        if soul:
            parts.append(f"【我的底色】{soul}")
        if not parts:
            return ""
        return truncate_to_tokens("\n".join(parts), token_budget)

    # ------------------------------------------------------------------ #

    def candidates(self, ctx: TurnContext) -> list[dict]:
        """核心记忆的候选**由提取器标记**（``type`` 落在 `CORE_TYPES`），
        经 :meth:`accept` 的高门槛审核后才允许落库。
        """
        return []

    def accept(self, confidence: float) -> bool:
        """高门槛准入（INV-11）。"""
        return confidence >= CORE_UPDATE_THRESHOLD

    def propose(self, record, *, reason: str) -> list[WriteIntent]:
        """提出一次核心记忆写入（**自动升格**）。**不过门槛则返回空**。

        升格是"把一条记忆写进人格"——**所有写入里后果最重的一次**，
        因此它必须留审计：不然"它什么时候变成我的底色的"永远查不出来（INV-11 / INV-12）。
        """
        if not self.accept(record.confidence):
            return []
        return [
            WriteIntent(
                op="put",
                record=record,
                embed_text=record.abstract or record.content,
                actor="consolidator",
                reason=reason,
                audit=AuditEvent(
                    op="promote",
                    actor="consolidator",
                    target_kind="memory",
                    target_id=record.id,
                    after={
                        "layer": "core",
                        "type": record.type,
                        "confidence": record.confidence,
                    },
                    reason=reason,
                ),
            )
        ]

    def recall(self, q: RecallQuery) -> list[Scored]:
        return [Scored(record=r, raw={}, score=1.0) for r in self.load()]
