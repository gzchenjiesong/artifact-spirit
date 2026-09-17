"""衰减**排序**与状态治理（LLD-AL2 §5 M3 · D-16 / D-17 / D-22 / D-23）。

## 本模块的红线

**这里不存在任何删除或降级调用。**

用户的原话：*"某条记忆不应该被时间或者频率衡量而删除"*（D-23）。
因此本模块只做一件事——**重算排序信号 ``strength``**，并把它作为**可见的候选清单**
报告出来。状态迁移（`active → dormant`）与物理删除由 :mod:`consolidation` 的
优化路径依据**图结构不可达**判定，与本模块无关。

这条红线是**可断言的**：`tests` 里有一条源码扫描，确认本文件中不出现
``set_status`` / ``forget`` / ``hard_delete``。

## 为什么移除"衰减触发删除"

按访问频率衰减会**系统性优先淘汰最不可替代的记忆**：

| 记忆 | 调用频率 | 一旦缺失 |
|---|---|---|
| 身份证号 / 血型 / 过敏史 | 极低 | **不可接受** |
| 紧急联系人 | 极低 | **不可接受** |

这个设计与**记忆研究的结论相悖**——记忆研究里"重要性"与"频率"是两个独立维度，
高重要性的记忆衰减极慢。纯按频率衰减，既违背该结论，也不适合 Agent。

**形状依据**：衰减只作排序信号，取"近期下降快、远期长尾"的混合形状
（见 ``docs/design/10-神经科学依据与机制映射.md`` §4.3，DES-RES-003）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..store.base import AuditEvent, MemoryBackend, MemoryRecord
from .base import Clock, DecayReport, WriteIntent
from .recall import DecayParams, strength_at

__all__ = ["LOW_STRENGTH_QUANTILE", "STRENGTH_WRITE_EPSILON", "Decayer"]

STRENGTH_WRITE_EPSILON = 0.01
"""强度变化小于该值就不写库——避免每次维护都产生成百上千次无意义写入。"""

LOW_STRENGTH_QUANTILE = 0.25
"""报告里"低强度候选"的取法：排序后最低的 25%（**仅供参考，不触发任何动作**）。"""


@dataclass(slots=True)
class Decayer:
    """重算排序信号。

    Args:
        backend: 存储协议（只读 + 产出写意图，不直接落库）。
        clock: 注入的时钟（C2）。
        params: Wixted 衰减参数。
        limit: 单次扫描上限（避免维护任务长时间占住 writer）。
    """

    backend: MemoryBackend
    clock: Clock
    params: DecayParams = field(default_factory=DecayParams)
    limit: int = 5000
    write_epsilon: float = STRENGTH_WRITE_EPSILON

    def run(self, *, now: str | None = None, dry_run: bool = True) -> DecayReport:
        """重算 ``strength``。

        **不改变任何 ``status``**——``DecayReport.downgrades`` 因此恒为空，
        这是刻意的：状态治理属于 :class:`~artifact_spirit.core.consolidation.Optimizer`。
        """
        current = now or self.clock()
        records = self.backend.query(status=None, limit=self.limit)
        report = DecayReport(total=len(records), dry_run=dry_run)

        for record in records:
            value = strength_at(record, now=current, params=self.params)
            if abs(value - record.strength) < self.write_epsilon:
                continue
            report.recomputed += 1
            if dry_run:
                continue
            report.intents.append(
                WriteIntent(
                    op="update",
                    mem_id=record.id,
                    patch={"strength": round(value, 6)},
                    actor="decay",
                    audit=AuditEvent(
                        op="update",
                        actor="decay",
                        target_kind="memory",
                        target_id=record.id,
                        before={"strength": record.strength},
                        after={"strength": round(value, 6)},
                        reason="衰减排序信号重算",
                    ),
                )
            )

        report.low_strength = self._low_strength(records, current)
        # 显式留空：状态迁移不归本模块管（D-16 / D-23）
        report.downgrades = []
        return report

    def _low_strength(
        self, records: list[MemoryRecord], now: str
    ) -> list[tuple[str, float]]:
        scored = sorted(
            ((r.id, strength_at(r, now=now, params=self.params)) for r in records),
            key=lambda pair: pair[1],
        )
        take = max(1, int(len(scored) * LOW_STRENGTH_QUANTILE)) if scored else 0
        return [(mid, round(value, 6)) for mid, value in scored[:take]]


def describe_decay(params: DecayParams) -> str:
    """人类可读的参数描述（`reflect` 用）。"""
    return (
        f"Wixted 混合衰减：快衰权重 {params.w}、时间常数 {params.tau_fast} 天、"
        f"幂律指数 {params.beta}；**仅用于排序**，不触发任何删除或降级"
    )
