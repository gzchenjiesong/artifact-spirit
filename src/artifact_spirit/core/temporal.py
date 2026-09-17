"""双时态（T-AL2-16 · M3）。

## 三件事必须分清

| 概念 | 字段 | 回答什么 |
|---|---|---|
| **真实世界的有效区间** | `valid_from` / `valid_to` | "这件事**什么时候是真的**？" |
| **器灵记录变化的时刻** | `created_at` / `updated_at` | "我们**什么时候知道/改了**它？" |
| **被谁取代** | `superseded_by` | "它**为什么**失效了？" |

把前两者混起来是最常见的一类错误：改个错别字会让 `updated_at` 变，
但那**不代表事实变了**——`valid_from` 一动，"这条记忆在某时刻是否成立"的答案就全乱了。

## "失效" ≠ "删除"（D-17 / INV-7）

`INVALIDATE` 只写 `valid_to` + `superseded_by`——**记录仍在库里、历史仍可查**。
物理删除只有两个入口：用户显式 `forget`，以及优化任务对"图结构不可达"的判定（D-23）。

> 这条区分是整个时态能力的立足点：如果"被取代"等于"被删除"，
> 那用户问"我上个月填的地址是什么"就永远答不出来——而这正是记忆系统比搜索框值钱的地方。
"""

from __future__ import annotations

from ..store.base import MemoryBackend, MemoryRecord, NotFoundError
from .base import AuditEvent, Clock, WriteIntent

__all__ = ["TemporalService"]


class TemporalService:
    """时态判定与失效标记。**只产出写意图，不落库**（R5）。"""

    def __init__(self, *, backend: MemoryBackend, clock: Clock) -> None:
        self.backend = backend
        self.clock = clock

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #

    def asof(self, ref: str, ts: str) -> MemoryRecord | None:
        """取 ``ts`` 时刻**有效**的那一版。

        **返回 ``None`` 就是"那时没有有效版本"**——不抛错，也不回退到当前值。
        调用方（`spirit_asof`）必须能区分"那时不存在"与"看错了时间"：
        把前者静默成后者，等于把一个可回答的问题变成一句谎话。

        时态链的遍历在存储层（`MemoryBackend.asof`）——那里判定落在 SQL 里，
        而不是把表捞进内存比时间。
        """
        return self.backend.asof(ref, ts)

    # ------------------------------------------------------------------ #
    # 失效（INVALIDATE）
    # ------------------------------------------------------------------ #

    def invalidate(
        self,
        mem_id: str,
        *,
        superseded_by: str,
        reason: str,
        valid_to: str | None = None,
        actor: str = "user",
    ) -> list[WriteIntent]:
        """把 ``mem_id`` 标记为**失效**——不是删除。

        产出**两条**意图，且必须一起应用：

        1. ``update``：写 ``valid_to`` + ``superseded_by`` —— 回答"**被谁取代**"（走主键，快）；
        2. ``link(rel_type='supersedes')``：新 → 旧 —— 回答"**它取代了谁**"（图查询用）。

        只写一个，另一个方向的问题就永远答不出来。而"这条记忆的来龙去脉"
        正是 V2 的卖点——**一半的追溯链等于没有追溯链**。

        Raises:
            NotFoundError: 目标记忆不存在（不静默产生一条指向空气的意图）。
            ValueError: 自指、或 ``valid_to`` 不晚于 ``valid_from``。
        """
        if mem_id == superseded_by:
            raise ValueError(
                "superseded_by 不能指向自己——时态链成环后 as-of 永远查不出结果，"
                "而且那只是一个 None、不报错"
            )

        current = self.backend.get(mem_id)
        if current is None:
            raise NotFoundError(f"记忆不存在：{mem_id}")

        ts = valid_to or self.clock()
        if current.valid_from and ts <= current.valid_from:
            raise ValueError(
                f"valid_to（{ts}）必须晚于 valid_from（{current.valid_from}）——"
                "否则这条记录的「有效期」是个空区间（或负数），任何时刻都查不到它"
            )

        return [
            WriteIntent(
                op="update",
                mem_id=mem_id,
                patch={"valid_to": ts, "superseded_by": superseded_by},
                actor=actor,
                reason=reason,
                audit=AuditEvent(
                    op="invalidate",
                    actor=actor,
                    target_kind="memory",
                    target_id=mem_id,
                    before={
                        "valid_to": current.valid_to,
                        "superseded_by": current.superseded_by,
                    },
                    after={"valid_to": ts, "superseded_by": superseded_by},
                    reason=reason,
                ),
            ),
            WriteIntent(
                op="link",
                a_kind="memory",
                a_id=superseded_by,
                b_kind="memory",
                b_id=mem_id,
                rel_type="supersedes",
                weight=1.0,
                actor=actor,
                reason=reason,
            ),
        ]
