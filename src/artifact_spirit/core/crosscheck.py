"""交叉验证（T-AL2-18 · M3）：对"疑似互相矛盾"的记忆做一次 LLM 仲裁。

## 为什么不在写入时做

写入是热路径，而 LLM 调用是秒级的——放进去直接违反 **INV-4**（绝不阻断宿主）。

正确的位置是 **maintenance 周期任务**（默认 30 min 一次）：那时已经攒够了一批
"看起来互相矛盾"的候选，**一次调用裁决一批**。这也是 INV-11 说的"高门槛低频"。

## 限频体现在哪三处

即使异步，也不能"有冲突就调"——冲突是常态（同一属性的历史版本全在库里）：

1. **只挑真冲突**：同一 `subject` + `predicate`、`object` 不同、且**都还 active**；
2. **已裁决的跳过**：已经有 `superseded_by` 或 `supersedes` 边的对不再问一遍；
3. **每轮有上限**：`limit`，且只取相邻版本（同属性多版本时全配对比对是 O(n²)，
   而真正待裁决的永远是"最近这两条"）。

## 降级：**失败一律 KEEP_BOTH**

LLM 不可用、输出非法、两次解析不出来——一律回到"两条都留"。
理由是**代价不对称**：多留一条只是多占一点空间，而误判 `INVALIDATE`
会让一条真实记忆**静默地从召回里消失**，且用户无从察觉。

## 确定性兜底

`llm is None` 时不做任何裁决（只报告冲突），不假装"我看过了"。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from ..extract.dedup import merge_texts
from ..model.base import LLMError, LLMProvider, ProviderUnavailableError
from ..store.base import MemoryBackend, MemoryRecord
from .base import AuditEvent, Clock, CrossCheckReport, WriteIntent
from .temporal import TemporalService

__all__ = ["INVALIDATE", "KEEP_BOTH", "MERGE", "ConflictPair", "CrossChecker"]

KEEP_BOTH = "KEEP_BOTH"
INVALIDATE = "INVALIDATE"
MERGE = "MERGE"
_DECISIONS = (KEEP_BOTH, INVALIDATE, MERGE)

_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(_DECISIONS)},
        "reason": {"type": "string"},
    },
    "required": ["decision"],
}


def _norm(value: str | None) -> str:
    return (value or "").strip().casefold()


@dataclass(frozen=True, slots=True)
class ConflictPair:
    """一对疑似矛盾的记忆：同一属性、取值不同。"""

    newer: MemoryRecord
    older: MemoryRecord

    @property
    def attribute(self) -> str:
        return f"{self.older.subject}.{self.older.predicate}"

    @property
    def values(self) -> str:
        return f"{self.older.object!r} → {self.newer.object!r}"


@dataclass(slots=True)
class CrossChecker:
    """交叉验证。**只产出写意图，不落库**（R5）。"""

    backend: MemoryBackend
    temporal: TemporalService
    llm: LLMProvider | None = None
    clock: Clock = None  # type: ignore[assignment]
    limit: int = 10
    scan_limit: int = 500

    def __post_init__(self) -> None:
        if self.clock is None:
            from ..common import now_iso

            self.clock = now_iso

    # ------------------------------------------------------------------ #

    def find_conflicts(self, *, limit: int | None = None) -> list[ConflictPair]:
        """挑出**真冲突**：同属性、取值不同、都 active、且尚未裁决过。"""
        cap = limit if limit is not None else self.limit
        by_attr: dict[tuple[str, str], list[MemoryRecord]] = {}
        for rec in self.backend.query(status="active", limit=self.scan_limit):
            if rec.subject and rec.predicate:
                by_attr.setdefault((_norm(rec.subject), _norm(rec.predicate)), []).append(rec)

        pairs: list[ConflictPair] = []
        for group in by_attr.values():
            if len(group) < 2:
                continue
            # 按时间排序后只看**相邻**两条：全配对比对是 O(n²)，
            # 而真正待裁决的永远是"最近这两条"（更早的已经被更近的那条取代过）。
            ordered = sorted(group, key=lambda r: (r.valid_from or r.created_at or "", r.id))
            for older, newer in pairwise(ordered):
                if _norm(older.object) == _norm(newer.object):
                    continue
                if self._already_settled(older, newer):
                    continue
                pairs.append(ConflictPair(newer=newer, older=older))
                if len(pairs) >= cap:
                    return pairs
        return pairs

    def _already_settled(self, older: MemoryRecord, newer: MemoryRecord) -> bool:
        """这对是否已经裁决过——`superseded_by` 与 `supersedes` 边**任一**成立即算。

        查两处而不是一处：写入侧承诺"两者一起写"，但历史数据里可能只有一边
        （M3 之前写下的记录）。**判据宁可宽松，也不要重复问模型**——
        重复问的代价是钱和延迟，而漏判的代价只是下一轮再问一次。
        """
        if older.superseded_by == newer.id:
            return True
        return any(
            edge.get("rel_type") == "supersedes"
            and edge.get("src_id") == newer.id
            and edge.get("dst_id") == older.id
            for edge in self.backend.outbound_edges([newer.id])
        )

    # ------------------------------------------------------------------ #

    def arbitrate(self, pair: ConflictPair) -> tuple[str, str]:
        """让模型裁决。**任何失败都回到 `KEEP_BOTH`**，并说明原因。

        返回 ``(decision, reason)``。
        """
        if self.llm is None:
            return KEEP_BOTH, "未配置 LLM——不做裁决（不假装看过了）"

        try:
            payload = self.llm.complete_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "两条关于同一属性的记忆取值冲突。判断它们的关系，输出 JSON："
                            '{"decision": "KEEP_BOTH|INVALIDATE|MERGE", "reason": "…"}。'
                            "INVALIDATE = 旧的那条已被新事实取代（如换了地址）；"
                            "MERGE = 二者互补、应合成一条；"
                            "KEEP_BOTH = 确实是两件事，都该留。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"属性：{pair.attribute}\n"
                            f"旧（{pair.older.valid_from or pair.older.created_at}）："
                            f"{pair.older.content}\n"
                            f"新（{pair.newer.valid_from or pair.newer.created_at}）："
                            f"{pair.newer.content}"
                        ),
                    },
                ],
                schema=_SCHEMA,
            )
        except (LLMError, ProviderUnavailableError) as exc:
            return KEEP_BOTH, f"仲裁失败（{type(exc).__name__}）——保守保留两条"

        decision = str(payload.get("decision", KEEP_BOTH)).upper()
        if decision not in _DECISIONS:
            return KEEP_BOTH, f"模型返回了未知裁决 {decision!r}——保守保留两条"
        return decision, str(payload.get("reason") or "")

    # ------------------------------------------------------------------ #

    def run(self, *, now: str | None = None) -> CrossCheckReport:
        """跑一轮：找冲突 → 仲裁 → 产写意图。**不落库**。"""
        report = CrossCheckReport()
        pairs = self.find_conflicts()
        report.conflicts = len(pairs)
        if not pairs:
            return report
        if self.llm is None:
            report.skipped = "未配置 LLM"
            return report

        ts = now or self.clock()
        for pair in pairs:
            decision, reason = self.arbitrate(pair)
            if decision == KEEP_BOTH:
                report.kept_both += 1
                continue
            report.arbitrated += 1
            report.intents.extend(self._apply(pair, decision, reason, ts))
        return report

    def _apply(
        self, pair: ConflictPair, decision: str, reason: str, ts: str
    ) -> list[WriteIntent]:
        """把裁决翻成写意图。**两条裁决都以"旧条失效"收尾**——只是内容怎么留不同。"""
        intents: list[WriteIntent] = []
        if decision == MERGE:
            merged = merge_texts(pair.newer.content, pair.older.content)
            intents.append(
                WriteIntent(
                    op="update",
                    mem_id=pair.newer.id,
                    patch={"content": merged},
                    actor="consolidator",
                    reason=reason or "交叉验证：合并互补的两条",
                    embed_text=merged,
                    audit=AuditEvent(
                        op="update",
                        actor="consolidator",
                        target_kind="memory",
                        target_id=pair.newer.id,
                        before={"content": pair.newer.content},
                        after={"content": merged},
                        reason=reason or "交叉验证：合并互补的两条",
                    ),
                )
            )

        valid_to = pair.newer.valid_from or ts
        if pair.older.valid_from and valid_to <= pair.older.valid_from:
            # 时序倒挂（如两条的 valid_from 相同）→ 用当前时间兜底，
            # 否则 `invalidate` 会因为"空区间"拒绝执行，而这一轮的裁决就白做了。
            valid_to = ts
        intents.extend(
            self.temporal.invalidate(
                pair.older.id,
                superseded_by=pair.newer.id,
                reason=reason or f"交叉验证：{pair.attribute} {pair.values}",
                valid_to=valid_to,
                actor="consolidator",
            )
        )
        return intents
