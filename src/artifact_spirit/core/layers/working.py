"""工作记忆（LLD-AL2 §5 M6 / M7）。

容量 **5 个组块**（Baddeley 4±1 的上界）。

## 一处需要说明的设计判断：**"淘汰"是退出注意力，不是删除**

字面上"超容量淘汰最冷组块"容易被读成"丢掉它"。但那是错的，理由有二：

1. LLD 自己写明"**被淘汰的组块不是丢弃——它会随会话结束一并固化进情景记忆**"。
   若真删了，这句话就无法成立。
2. 与 ``dormant`` 的语义保持一致——**淡出的是注意力，不是信息**（用户原话）。
   同一个项目里对"淡出"用两套含义，是最容易埋下误解的地方。

因此本实现中：

- 全部组块都留在 ``working_memory``（会话级，且**不是真相源**）
- **参与召回**的只有按 ``(act_count, last_touched)`` 排序的前 5 个
- :meth:`evict` 返回"被挤出注意力的组块"，供 `reflect` 展示

这让"4±1"回到了它本来的含义——**注意力的容量**，而不是存储的容量。

> 依据：容量限制作用于**组块数**而非信息量（约 4±1），故上限取上界 5——
> 见 ``docs/design/10-神经科学依据与机制映射.md`` §4.6（DES-RES-003）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ...common import first_sentence
from ..base import RecallQuery, Scored, TurnContext, WriteIntent
from .base import BaseLayerService, derive_chunk_key

__all__ = ["WORKING_CAPACITY", "WorkingLayer"]

WORKING_CAPACITY = 5
"""组块容量上限——Baddeley 4±1 的上界。"""

INTENT_MARKERS = ("提醒", "待办", "计划", "下周", "明天", "稍后", "别忘了", "记得", "remind", "todo", "later")


@dataclass(slots=True)
class WorkingLayer(BaseLayerService):
    """组块聚类 + 容量控制 + 意图槽。"""

    layer: str = "working"
    capacity: int = WORKING_CAPACITY

    # ------------------------------------------------------------------ #

    def ingest(self, ctx: TurnContext, salience: float = 0.0) -> list[WriteIntent]:
        """把一轮输入写进工作记忆（按 ``chunk_key`` 聚类）。

        产出 ``WriteIntent`` 而非直接写库——AL2 不落库（见 :mod:`..base`）。
        """
        chunk_key = derive_chunk_key(ctx.user or ctx.text)
        intents = [
            WriteIntent(
                op="wm_put",
                session_id=ctx.session_id,
                chunk_key=chunk_key,
                content=first_sentence(ctx.text, limit=200) or ctx.text,
                salience=salience,
                ts=ctx.ts,
                actor="system",
            )
        ]

        # 意图槽：前瞻记忆（M7）。**不受组块容量限制**——待办是另一个维度的事。
        for marker in INTENT_MARKERS:
            if marker in ctx.text:
                intents.append(
                    WriteIntent(
                        op="wm_put",
                        session_id=ctx.session_id,
                        chunk_key="__intent__",
                        content=first_sentence(ctx.text, limit=200),
                        salience=max(salience, 0.8),
                        ts=ctx.ts,
                        actor="system",
                    )
                )
                break
        return intents

    def active_chunks(self, session_id: str, *, capacity: int | None = None) -> list:
        """**参与注意力**的组块（容量内的那些）。

        排序：先看激活次数（``act_count``），同次数再看最近触碰时间。
        最冷的那几个自动退出注意力——但不离开存储。
        """
        limit = capacity or self.capacity
        chunks = self.backend.wm_list(session_id, limit=200)
        ranked = sorted(
            chunks, key=lambda c: (c.act_count, c.last_touched), reverse=True
        )
        return ranked[:limit]

    def evict(self, session_id: str, *, capacity: int | None = None) -> list[str]:
        """被挤出注意力的组块 id（**它们仍在库中**，会话结束仍会被固化）。"""
        limit = capacity or self.capacity
        active = {c.id for c in self.active_chunks(session_id, capacity=limit)}
        return [c.id for c in self.backend.wm_list(session_id, limit=200) if c.id not in active]

    def candidates(self, ctx: TurnContext) -> list[dict]:
        """工作记忆**不产出落库候选**——它是会话内的暂存区。"""
        return []

    def recall(self, q: RecallQuery) -> list[Scored]:
        """会话内召回：本次对话"刚才说过什么"。"""
        if not q.session_id:
            return []
        return [
            Scored(
                record=_chunk_as_record(chunk, q.session_id),
                raw={},
                score=0.0,
            )
            for chunk in self.active_chunks(q.session_id)
        ]


def _chunk_as_record(chunk, session_id: str):
    """把工作记忆组块适配成 ``MemoryRecord``（跨层统一形态的代价）。

    它**不是**记忆表里的记录，因此 `layer` 标为 ``episodic``（会话结束就会变成情景记忆），
    `type` 标为 ``event``——这是最诚实的映射，而不是新造一个层。
    """
    from ...store.base import MemoryRecord

    return MemoryRecord(
        id=chunk.id,
        layer="episodic",
        type="event",
        content=chunk.content,
        abstract=first_sentence(chunk.content),
        scope={"type": "session", "id": session_id},
        salience=chunk.salience,
        source_session=session_id,
        created_at=chunk.created_at,
        updated_at=chunk.last_touched,
    )
