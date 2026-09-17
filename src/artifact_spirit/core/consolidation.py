"""巩固流水线（LLD-AL2 §5 M2）与记忆优化（D-23）。

## 巩固：单向递进

| 迁移 | 触发条件 | 动作 |
|---|---|---|
| 工作 → 情景 | 会话结束 / 上下文压缩前 | 整段会话固化为 episode |
| 情景 → 语义 | 同一实体 / 主题在 **≥ N 个不同会话**出现 | 提炼去情境化知识 + ``derived_from`` 关联 |
| 语义 → 程序 | 某模式成功复用 ≥ K 次 | MVP **仅建表占位**（C11，不提前实现） |

硬约束：**迁移单向**（不出现逆向搬移）、**提升产生新记录**（保留来源链）、
**离线后台执行**（不在在线热路径）、**幂等**（同一 session 重复巩固不产生重复记忆，C6）。

> 依据：情景记忆随时间推移摆脱情境依赖、转为语义知识（系统巩固），重放多发生在
> 离线阶段——因此"**单向递进 + 离线执行**"是形状选择，不是实现便利
> （见 ``docs/design/10-神经科学依据与机制映射.md`` §4.2，DES-RES-003）。

## 记忆优化：删除收敛到这里的理由

用户给出的判据很明确：**"某条记忆不应该被时间或者频率衡量而删除"**，
但"错误记忆 / 多条记忆冲突"若不处理，问题会一直存在。

因此把删除收敛到优化任务里，触发条件是**语义性**的：

| 触发 | 判据 |
|---|---|
| **错误记忆** | 被更高置信度的记忆直接取代（``superseded_by`` 非空） |
| **不可达记忆** | **图结构上的孤立**：无入边 + 无出边 + 无实体关联 + 不在任何 L1 概览中 |

> **"不可达"是最重要的判据**：它不是"很久没用"，而是"**从任何路径都到不了它**"。
> 这才是真正没有认知价值的记忆。

安全网：删除走 :meth:`Reviewer.forget` 语义（**保留快照、可恢复**），
且**默认不自动执行**——必须先看到候选集。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..common import content_hash_of, summarize
from ..store.base import AuditEvent, MemoryBackend, MemoryRecord
from .base import Clock, ConsolidationReport, IdGen, OptimizationReport, WriteIntent
from .recall import score_importance

__all__ = [
    "LOW_IMPORTANCE_THRESHOLD",
    "MAX_EPISODE_CHUNKS",
    "PROMOTABLE_TYPES",
    "PROMOTION_SESSION_THRESHOLD",
    "Consolidator",
    "Optimizer",
]

PROMOTION_SESSION_THRESHOLD = 2
"""同一实体在多少个**不同会话**出现后提升为语义记忆（LLD-AL2 §10 未决项 2 的建议值）。"""

EPISODE_ABSTRACT_BUDGET = 32
"""episode 的 L0 摘要预算（token）。

与 `spirit_remember` 同理：**摘要必须比正文短**，否则 L0 就等于全文，
分级加载（V1）失去意义。
"""

MAX_EPISODE_CHUNKS = 50
"""单个 episode 最多吸收多少个组块。"""

PROMOTABLE_TYPES = frozenset({"fact", "preference", "entity"})
"""可被提升的语义类型。``skill`` 属程序性记忆，MVP 不做（C11）。"""

NON_OPTIMIZABLE_LAYERS = frozenset({"core", "procedural"})
"""优化任务**永不触碰**的层：核心记忆是人格，程序性记忆是技能，误删代价不可接受。"""

LOW_IMPORTANCE_THRESHOLD = 0.5
"""降级（active → dormant）的重要度门槛。

注意：判据是 **重要度 + 图结构**，**不含时间、不含访问频率**（D-23）。
"""


@dataclass(slots=True)
class Consolidator:
    """工作 → 情景 → 语义。"""

    backend: MemoryBackend
    clock: Clock
    id_gen: IdGen
    promotion_threshold: int = PROMOTION_SESSION_THRESHOLD
    max_chunks: int = MAX_EPISODE_CHUNKS

    def consolidate(self, *, session_id: str) -> ConsolidationReport:
        """把一个会话固化为情景记忆，并尝试提升为语义记忆。

        **幂等**：已经固化过的会话直接返回 ``skipped``，不产生重复记忆（C6）。
        """
        report = ConsolidationReport(session_id=session_id)

        chunks = self.backend.wm_list(session_id, limit=self.max_chunks)
        if not chunks:
            report.skipped = "empty_working_memory"
            return report

        existing = self.backend.find_by_source_session(session_id, layer="episodic")
        if existing:
            report.skipped = "already_consolidated"
            report.episode_id = existing[0].id
            return report

        now = self.clock()
        content = "\n".join(chunk.content for chunk in chunks if chunk.content)
        episode = MemoryRecord(
            id=self.id_gen("episodic"),
            layer="episodic",
            type="event",
            content=content,
            abstract=summarize(content, budget=EPISODE_ABSTRACT_BUDGET),
            scope={"type": "session", "id": session_id},
            confidence=0.8,
            salience=round(sum(c.salience for c in chunks) / max(1, len(chunks)), 4),
            source_session=session_id,
            source_turn=len(chunks),
            created_at=now,
            updated_at=now,
        )
        report.episode_id = episode.id
        report.intents.append(
            WriteIntent(
                op="put",
                record=episode,
                embed_text=episode.abstract or episode.content,
                actor="consolidator",
                audit=AuditEvent(
                    op="consolidate",
                    actor="consolidator",
                    target_kind="memory",
                    target_id=episode.id,
                    after={"chunks": len(chunks)},
                    reason="会话结束固化",
                    session_id=session_id,
                ),
            )
        )

        report.intents.extend(self._promote(chunks, episode, now))
        report.promoted = [i.record.id for i in report.intents if i.op == "put" and i.record and i.record.id != episode.id]
        return report

    def _promote(self, chunks: list, episode: MemoryRecord, now: str) -> list[WriteIntent]:
        """情景 → 语义：跨 ≥N 个会话反复出现的话题才沉淀为"知识"。"""
        intents: list[WriteIntent] = []
        seen: set[str] = set()

        for chunk in chunks:
            for entity in self.backend.entity_find(chunk.content):
                key = entity.name.casefold()
                if key in seen:
                    continue
                seen.add(key)

                sessions = self._sessions_mentioning(entity.name, episode)
                if len(sessions) < self.promotion_threshold:
                    continue

                # 提升的**锚点是实体，不是本次会话的措辞**——否则每个会话都会
                # 生成一条"新的知识"，幂等性荡然无存。
                content = f"{entity.name}：在 {len(sessions)} 个不同会话中被反复提及"
                digest = content_hash_of(
                    content=content,
                    subject=entity.name,
                    predicate="recurring_topic",
                    object_="multi_session",
                    scope={"type": "global"},
                )
                if self.backend.find_by_hash(digest):
                    continue  # 已经提升过 → 幂等

                promoted = MemoryRecord(
                    id=self.id_gen("semantic"),
                    layer="semantic",
                    type="fact",
                    content=content,
                    abstract=f"{entity.name} 是一个反复出现的主题",
                    subject=entity.name,
                    predicate="recurring_topic",
                    object="multi_session",
                    scope={"type": "global"},
                    confidence=0.75,
                    salience=chunk.salience,
                    source_session=episode.source_session,
                    created_at=now,
                    updated_at=now,
                    content_hash=digest,
                )
                intents.append(
                    WriteIntent(
                        op="put",
                        record=promoted,
                        embed_text=promoted.abstract or promoted.content,
                        actor="consolidator",
                        audit=AuditEvent(
                            op="consolidate",
                            actor="consolidator",
                            target_kind="memory",
                            target_id=promoted.id,
                            after={"promoted_from": episode.id, "sessions": len(sessions)},
                            reason=f"实体 {entity.name} 跨 {len(sessions)} 个会话复现",
                        ),
                    )
                )
                # 保留来源链——迁移可追溯，不做"凭空出现"的知识
                for source_id in sessions:
                    intents.append(
                        WriteIntent(
                            op="link",
                            a_id=promoted.id,
                            a_kind="memory",
                            b_kind="memory",
                            b_id=source_id,
                            rel_type="derived_from",
                            weight=1.0,
                            actor="consolidator",
                        )
                    )
        return intents

    def _sessions_mentioning(self, entity_name: str, episode: MemoryRecord) -> list[str]:
        """该实体出现在哪些不同会话里（含本次）。"""
        sessions: set[str] = set()
        if episode.source_session:
            sessions.add(episode.source_session)
        lowered = entity_name.casefold()
        for record in self.backend.query(layer="episodic", status=None, limit=500):
            if record.source_session and lowered in (record.content or "").casefold():
                sessions.add(record.source_session)
        return sorted(sessions)


@dataclass(slots=True)
class Optimizer:
    """记忆优化：治理性删除与注意力降级（D-23）。

    **默认什么都不做**：要么 ``dry_run=True``，要么显式 ``autonomous=True``，
    要么提供 ``confirm`` 回调。真实删除是**不可逆动作**（合规删除尤甚），
    不该由一次疏忽触发。
    """

    backend: MemoryBackend
    clock: Clock
    scan_limit: int = 5000
    low_importance_threshold: float = LOW_IMPORTANCE_THRESHOLD
    confirm: Callable[[list[str]], bool] | None = None
    """默认的确认回调（可被 :meth:`run` 的同名参数覆盖）。"""

    def run(
        self,
        *,
        dry_run: bool = True,
        autonomous: bool = False,
        confirm: Callable[[list[str]], bool] | None = None,
    ) -> OptimizationReport:
        records = [
            r
            for r in self.backend.query(status=None, limit=self.scan_limit)
            if r.layer not in NON_OPTIMIZABLE_LAYERS and r.status == "active"
        ]
        report = OptimizationReport(scanned=len(records), dry_run=dry_run)

        relations = self.backend.all_relations()
        outbound: dict[str, int] = {}
        inbound: dict[str, int] = {}
        for edge in relations:
            if edge["src_kind"] == "memory":
                outbound[edge["src_id"]] = outbound.get(edge["src_id"], 0) + 1
            if edge["dst_kind"] == "memory":
                inbound[edge["dst_id"]] = inbound.get(edge["dst_id"], 0) + 1

        covered = self._overview_covered()
        entities = {e.name.casefold() for e in self.backend.entity_list()}
        committed_sessions = self._committed_sessions()
        if not entities:
            report.notes.append(
                "实体表为空——无法依据'实体关联'判定可达性，"
                "本次不可达判定结果偏保守（可能漏报）"
            )

        for record in records:
            # **未定形**的记忆不参与治理：会话还没结束，谈"孤立"为时过早。
            # 注意这不是时间判据，是**生命周期判据**（会话是否已 committed）。
            if not self._is_settled(record, committed_sessions):
                continue
            if self._has_entity_association(record, entities):
                continue
            if record.id in covered:
                continue

            has_in = inbound.get(record.id, 0) > 0
            has_out = outbound.get(record.id, 0) > 0

            if has_in or has_out:
                # 还挂在图上，只是没人引用 → 降级（可逆），不删除
                importance = score_importance(record)
                if not has_in and importance < self.low_importance_threshold:
                    report.downgrades.append(record.id)
                continue

            # 图结构完全孤立 → 不可达
            report.unreachable.append(record.id)

        report.stale_overviews = [
            ov.id for ov in self.backend.overview_stale_list(limit=self.scan_limit)
        ]

        if dry_run:
            return report

        candidates = report.unreachable + report.downgrades
        approver = confirm or self.confirm
        if candidates:
            if approver is not None:
                if not approver(candidates):
                    report.notes.append("已请求确认但未获批准 → 不做任何处置")
                    return report
            elif not autonomous:
                report.notes.append(
                    "删除与降级需要授权：请传 autonomous=True（系统自主）"
                    "或提供 confirm 回调（用户确认）。本次只报告，未处置。"
                )
                return report

        for mem_id in report.unreachable:
            report.intents.append(
                WriteIntent(
                    op="forget",
                    mem_id=mem_id,
                    reason="不可达记忆：图结构孤立（无关联、无实体、不在任何概览中）",
                    actor="optimizer",
                    purge_snapshot=False,  # 保留快照 → 可 restore（D-22 安全网）
                    audit=AuditEvent(
                        op="forget",
                        actor="optimizer",
                        target_kind="memory",
                        target_id=mem_id,
                        reason="不可达记忆（图结构孤立）",
                        after={"criterion": "unreachable"},
                    ),
                )
            )

        for mem_id in report.downgrades:
            report.intents.append(
                WriteIntent(
                    op="set_status",
                    mem_id=mem_id,
                    status="dormant",
                    reason="未被任何记忆引用且重要度低：降为只保留 L0 参与召回",
                    actor="optimizer",
                    audit=AuditEvent(
                        op="set_status",
                        actor="optimizer",
                        target_kind="memory",
                        target_id=mem_id,
                        before={"status": "active"},
                        after={"status": "dormant"},
                        reason="低重要度且无入边引用",
                    ),
                )
            )

        for overview_id in report.stale_overviews:
            report.intents.append(
                WriteIntent(op="overview_invalidate", scope_id=overview_id, actor="optimizer")
            )
        return report

    # ------------------------------------------------------------------ #

    def _overview_covered(self) -> set[str]:
        """任何 L1 概览覆盖到的记忆。

        "被概览收进去"意味着还能从概览路径抵达——因此**不算不可达**。
        """
        summaries = [o.content for o in self.backend.overview_list()]
        if not summaries:
            return set()

        covered: set[str] = set()
        for record in self.backend.query(status=None, limit=self.scan_limit):
            snippet = (record.abstract or record.content or "")[:40]
            if snippet and any(snippet in text for text in summaries):
                covered.add(record.id)
        return covered

    @staticmethod
    def _has_entity_association(record: MemoryRecord, entity_names: set[str]) -> bool:
        """正文里出现任何已登记实体名 → 该记忆可从实体路径抵达。"""
        if not entity_names:
            return False
        haystack = " ".join(
            filter(None, [record.subject, record.object, record.content, record.abstract])
        ).casefold()
        return any(name in haystack for name in entity_names)

    def _committed_sessions(self) -> set[str]:
        """已结束的会话 id 集合。

        这不是时间判据——它回答的是"这段记忆是否已经定形"。
        """
        committed: set[str] = set()
        for record in self.backend.query(status=None, limit=self.scan_limit):
            session_id = record.source_session
            if not session_id or session_id in committed:
                continue
            row = self.backend.session_get(session_id)
            if row is not None and row.get("status") == "committed":
                committed.add(session_id)
        return committed

    @staticmethod
    def _is_settled(record: MemoryRecord, committed_sessions: set[str]) -> bool:
        """记忆是否已"定形"，够资格参与治理判定。

        - 没有来源会话（全局 / 核心类记忆）→ 已定形
        - 来源会话已 ``committed`` → 已定形
        - 会话仍在进行中 → **不定形**，不参与判定

        这条守门很重要：会话还没结束就断言某条记忆"孤立"，会把**刚写下、
        还没来得及建立关联**的正常记忆误判为垃圾。用一个**生命周期**判据
        而不是时间判据，才符合 D-23 的纪律。
        """
        if not record.source_session:
            return True
        return record.source_session in committed_sessions
