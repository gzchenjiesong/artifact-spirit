"""核心层门面（LLD-AL2 §2.1）。

**AL1 与 AL5 只依赖本模块暴露的这一个类**——有了单一可替换门面，
AL1 才能用 fake 核心层做独立验收（评审 P0-1）。

## 数据流

```
一轮对话输入
   └─▶ sensory.filter()        显著性打分（零 LLM）
         ├─ 低于阈值 → working.ingest()（仅会话内，不落长期记忆）
         └─ 达标    → extractor.extract()（LLM）→ schema 校验
                        └─▶ dedup.decide()（规则 / LLM 仲裁）
                              └─▶ 产出 WriteIntent（不落库，交 AL5 writer）
   └─▶ activator.co_activate() 共激活建边
   └─▶ decayer.touch            记录访问
```

**本类不落库、不起线程、不做 I/O**（R5）。它的输出是"意图"，
把意图变成事实是 AL5 writer 的工作。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common import first_sentence, now_iso
from ..extract.dedup import DedupDecision, Deduplicator
from ..extract.extractor import Extractor
from ..model.base import EmbeddingProvider, LLMProvider
from ..store.base import AuditEvent, MemoryBackend, MemoryRecord
from ..store.ids import new_memory_id
from .activation import Activator
from .base import (
    Clock,
    ConsolidationReport,
    CrossCheckReport,
    DecayReport,
    EvolutionReport,
    HealthReport,
    IdGen,
    OptimizationReport,
    RecallQuery,
    RecallWeights,
    Scored,
    TransferReport,
    TurnContext,
    TurnEvent,
    WriteIntent,
)
from .consolidation import Consolidator, Optimizer
from .crosscheck import CrossChecker
from .decay import Decayer
from .evolution import Evolver
from .layers.core_memory import CoreMemoryLayer
from .layers.episodic import EpisodicLayer
from .layers.procedural import ProceduralLayer
from .layers.semantic import SemanticLayer
from .layers.sensory import SensoryLayer
from .layers.working import WorkingLayer
from .progressive import ProgressiveLoader
from .recall import DecayParams, Recaller
from .review import Reviewer
from .salience import SalienceConfig, SalienceScorer
from .temporal import TemporalService
from .transfer import Transferrer

__all__ = ["MEMORY_WRITE_ACTOR", "ArtifactSpiritCore", "CoreSettings"]

MEMORY_WRITE_ACTOR = "extractor"


@dataclass(slots=True)
class CoreSettings:
    """核心层可调参数（全部集中于此，不散落魔法数字 —— C9）。"""

    recall: RecallWeights = field(default_factory=RecallWeights)
    decay: DecayParams = field(default_factory=DecayParams)
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    candidate_k: int = 24
    diffusion_threshold: float = 0.2
    evolution_enabled: bool = True
    """记忆进化开关（T-AL2-20）。**关掉即退回纯 ADD**——代价大的能力必须能关。"""
    activation_eta: float = 0.3
    promotion_threshold: int = 2
    consolidate_limit: int = 5000


class ArtifactSpiritCore:
    """``CoreFacade`` 的实现。"""

    def __init__(
        self,
        *,
        backend: MemoryBackend,
        clock: Clock = now_iso,
        id_gen: IdGen = new_memory_id,
        embedding: EmbeddingProvider | None = None,
        llm: LLMProvider | None = None,
        settings: CoreSettings | None = None,
    ) -> None:
        self.backend = backend
        self.clock = clock
        self.id_gen = id_gen
        self.embedding = embedding
        self.llm = llm
        self.settings = settings or CoreSettings()

        self.recaller = Recaller(
            backend=backend,
            clock=clock,
            weights=self.settings.recall,
            decay=self.settings.decay,
            candidate_k=self.settings.candidate_k,
            diffusion_threshold=self.settings.diffusion_threshold,
        )
        self.progressive = ProgressiveLoader(backend=backend, clock=clock, llm=llm)
        self.reviewer = Reviewer(backend=backend, clock=clock)
        self.temporal = TemporalService(backend=backend, clock=clock)
        self.crosschecker = CrossChecker(
            backend=backend, temporal=self.temporal, llm=llm, clock=clock
        )
        self.evolver = Evolver(
            backend=backend, llm=llm, clock=clock, enabled=self.settings.evolution_enabled
        )
        self.consolidator = Consolidator(
            backend=backend,
            clock=clock,
            id_gen=id_gen,
            promotion_threshold=self.settings.promotion_threshold,
        )
        self.optimizer = Optimizer(backend=backend, clock=clock, scan_limit=self.settings.consolidate_limit)
        self.decayer = Decayer(
            backend=backend, clock=clock, params=self.settings.decay, limit=self.settings.consolidate_limit
        )
        self.activator = Activator(
            backend=backend, clock=clock, eta=self.settings.activation_eta
        )
        self.extractor = Extractor(llm=llm)
        self.deduplicator = Deduplicator(backend=backend, llm=llm)

        # 层服务
        # 注意：显著性打分器必须拿到 embedding——否则新颖度分量会被置零
        # （那是"embedding 不可用"的降级路径），分数会被系统性压低，
        # 表现为"大多数正常内容都过不了阈值"。
        self.sensory = SensoryLayer(
            backend=backend,
            clock=clock,
            config=self.settings.salience,
            scorer=SalienceScorer(
                backend=backend, embedding=embedding, config=self.settings.salience
            ),
        )
        self.working = WorkingLayer(backend=backend, clock=clock)
        self.episodic = EpisodicLayer(backend=backend, clock=clock)
        self.semantic = SemanticLayer(backend=backend, clock=clock)
        self.procedural = ProceduralLayer(backend=backend, clock=clock)
        self.core_memory = CoreMemoryLayer(backend=backend, clock=clock)

        self._pending: list[WriteIntent] = []

    # ================================================================== #
    # 写入路径
    # ================================================================== #

    def ingest_turn(self, event: TurnEvent) -> list[WriteIntent]:
        """处理一轮对话，产出写意图。**不做任何 I/O，不抛异常到宿主。**"""
        ctx = TurnContext(
            session_id=event.session_id,
            ts=event.ts or self.clock(),
            user=event.user or "",
            assistant=event.assistant or "",
        )
        intents: list[WriteIntent] = []

        # 1. 显著性（零 LLM；唯一的模型依赖是新颖度）
        salience = self.sensory.filter(ctx)

        # 2. 工作记忆（会话内，永远做）
        intents.extend(self.working.ingest(ctx, salience.score))

        # 3. 上一轮召回过的记忆 → 记录访问 + 共激活建边
        if event.recalled:
            for mem_id in event.recalled:
                intents.append(
                    WriteIntent(
                        op="touch",
                        mem_id=mem_id,
                        ts=ctx.ts,
                        strength=_rehearsed_strength(self.backend, mem_id),
                        actor="system",
                    )
                )
            intents.extend(self.activator.co_activate(list(event.recalled)))
            self._invalidate_overviews_for(list(event.recalled), intents)

        # 4. 达标才提取（低于阈值只留在工作记忆里）
        if salience.passes_threshold(self.settings.salience):
            intents.extend(self._extract_and_dedup(ctx, salience.embedding))

        return intents

    def _extract_and_dedup(self, ctx: TurnContext, embedding: list[float] | None) -> list[WriteIntent]:
        intents: list[WriteIntent] = []
        result = self.extractor.extract(
            ctx.text,
            scope={"type": "session", "id": ctx.session_id},
            # 本轮时间随正文一起给模型：没有它，「昨天」无从换算成绝对日期。
            now=ctx.ts,
        )

        if result.fallback_only:
            # F1 / F3：**仅存原文**——记忆不丢，只是未结构化
            record = MemoryRecord(
                id=self.id_gen("episodic"),
                layer="episodic",
                type="event",
                content=result.raw_text,
                abstract=first_sentence(result.raw_text),
                scope={"type": "session", "id": ctx.session_id},
                confidence=0.5,
                source_session=ctx.session_id,
                created_at=ctx.ts,
                updated_at=ctx.ts,
            )
            intents.append(
                WriteIntent(
                    op="put",
                    record=record,
                    embed_text=record.abstract or record.content,
                    actor=MEMORY_WRITE_ACTOR,
                    audit=AuditEvent(
                        op="add",
                        actor=MEMORY_WRITE_ACTOR,
                        target_kind="memory",
                        target_id=record.id,
                        after={"degraded": result.degraded},
                        reason=f"提取降级：{result.degraded}",
                        session_id=ctx.session_id,
                    ),
                )
            )
            return intents

        for candidate, record in zip(
            result.candidates,
            self.extractor.to_records(result.candidates, id_gen=self.id_gen, now=ctx.ts),
            strict=False,
        ):
            decision = self.deduplicator.decide(candidate)
            # `op` 与决策**必须一一对应**（P0-6：账本不许撒谎）。这里用穷举表而不是
            # 字典字面量 + 下标——后者在新增一种决策时是 `KeyError`（红得晚、且只在跑到那条分支时才红）。
            audit = AuditEvent(
                op=_AUDIT_OP_FOR_DECISION[decision.decision],
                actor="dedup",
                target_kind="memory",
                target_id=record.id,
                before=_before_snapshot(decision, self.backend),
                after={"decision": decision.decision, "reason": decision.reason},
                reason=decision.reason,
                session_id=ctx.session_id,
            )

            if decision.decision == "IGNORE":
                # "决定不做"也是决定——留痕，否则事后永远解释不了"这条为什么没进来"。
                intents.append(WriteIntent(op="audit", actor="dedup", audit=audit))
                continue

            record.source_session = ctx.session_id

            if decision.decision == "MERGE" and decision.target_id:
                # **真的合并**：目标记录的 `content` 换成信息更全的那条（旧值已在 `before` 留痕）。
                # "记一笔 audit 了事"不算合并——那样两条表述里必有一条的细节**谁也拿不到**。
                merged_content = (decision.merged or {}).get("content") or record.content
                intents.append(
                    WriteIntent(
                        op="update",
                        mem_id=decision.target_id,
                        patch={"content": merged_content},
                        actor="dedup",
                        embed_text=merged_content,
                        audit=audit,
                    )
                )
                continue

            if decision.decision == "INVALIDATE" and decision.target_id:
                # **事实变了**：新条落库 + 旧条失效（不删）+ `supersedes` 边。
                # 三条意图齐备才算把这次变化记完整——少一条，`asof` 或图查询就少一半答案。
                record.content_hash = None
                intents.append(
                    WriteIntent(
                        op="put",
                        record=record,
                        embed_text=record.abstract or record.content,
                        actor=MEMORY_WRITE_ACTOR,
                        audit=audit,
                    )
                )
                intents.extend(
                    self.temporal.invalidate(
                        decision.target_id,
                        superseded_by=record.id,
                        reason=decision.reason or "事实变化",
                        valid_to=ctx.ts,
                        actor="dedup",
                    )
                )
                continue

            if decision.decision == "UPDATE" and decision.target_id:
                intents.append(
                    WriteIntent(
                        op="update",
                        mem_id=decision.target_id,
                        patch=_patch_from_candidate(candidate, ctx),
                        actor="dedup",
                        embed_text=record.abstract or record.content,
                        audit=audit,
                    )
                )
                continue

            record.content_hash = None  # 交给 AL3 按内容计算
            intents.append(
                WriteIntent(
                    op="put",
                    record=record,
                    embed_text=record.abstract or record.content,
                    actor=MEMORY_WRITE_ACTOR,
                    audit=audit,
                )
            )
            # 实体登记：提取器认出的主体要进 entities 表。
            # 两个下游都依赖它——① 召回的"实体匹配"第六因子；② 优化任务的"可达性"判定。
            # 不登记的话，几乎所有记忆在治理时都会被判为"图结构孤立"。
            entity_name = candidate.get("subject") or candidate.get("object")
            if entity_name:
                intents.append(
                    WriteIntent(
                        op="entity_upsert",
                        entity_name=str(entity_name),
                        entity_type=str(candidate.get("type") or "concept"),
                        actor=MEMORY_WRITE_ACTOR,
                    )
                )
        return intents

    def _invalidate_overviews_for(
        self, mem_ids: list[str], intents: list[WriteIntent]
    ) -> None:
        """访问过的记忆所属概览置 stale——**标记式，O(1)，不触发重算**。"""
        scopes: set[tuple[str, str]] = set()
        for mem_id in mem_ids:
            record = self.backend.get(mem_id)
            if record is None:
                continue
            term = record.subject or record.object
            if term:
                matches = self.backend.entity_find(term)
                scopes.add(("entity", matches[0].id) if matches else ("topic", term))
        for scope_kind, scope_id in sorted(scopes):
            intents.append(
                WriteIntent(
                    op="overview_invalidate",
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    actor="system",
                )
            )

    # ================================================================== #
    # 召回（**严格只读**）
    # ================================================================== #

    def recall(self, q: RecallQuery) -> list[Scored]:
        """多路召回 + 融合 + 预算裁剪。

        **只读**——召回永远不写库。这是 ``prefetch`` 能在 300ms 护栏内返回的前提，
        也是"记忆系统绝不阻断宿主"（P6）的实现基础。

        ## 向量在这里补算，而不是交给调用方

        `q.vec` 为空且配置了嵌入时，本方法**自己算**。因为本方法就是"召回的唯一边界"：
        调用方只该给出**线索文本**，不该知道"要不要走向量、怎么算向量"。

        这个补算不是可有可无。早先只有工具面自己算 `vec`，
        于是**从 `core.recall` 进来的调用（评测、宿主直连）一律静默退化成纯关键词检索**：
        嵌入配好了、库里 122 条向量都在、`embedding_available()` 返回 True，
        而召回一次都没用过它们。它的表现是"语义召回不行"，
        于是人会去怀疑**向量模型**，而不是"向量根本没接上"。

        同一个判断散在三处，就一定会漏掉两处——所以它只该有一处。
        """
        if q.vec is None and q.text and self.embedding is not None:
            try:
                q.vec = self.embedding.embed([q.text])[0]
            except Exception:
                # **降级而不抛错**（F2）：嵌入这条路坏了，关键词检索照常工作。
                q.vec = None
        scored = self.recaller.recall(q)
        return _clip(scored, q.token_budget, q.top_k)

    # ================================================================== #
    # 分级加载（P1）
    # ================================================================== #

    def expand(self, ref: str, level: str = "L0", *, hot_path: bool = True) -> str:
        """展开到指定级别。热路径发现 L1 缺失时**降级而不生成**（N-P0-2）。"""
        result = self.progressive.expand(ref, level, hot_path=hot_path)
        self._pending.extend(result.intents)
        return result.text

    # ================================================================== #
    # 审查与干预（P2）
    # ================================================================== #

    def review(
        self, *, layer: str | None = None, since: str | None = None, limit: int = 50
    ) -> list[dict]:
        return self.reviewer.review(layer=layer, since=since, limit=limit)

    def trace(self, mem_id: str) -> list[dict]:
        return self.reviewer.trace(mem_id)

    def correct(self, mem_id: str, patch: dict, *, reason: str) -> list[WriteIntent]:
        return self.reviewer.correct(mem_id, patch, reason=reason)

    def invalidate_since(
        self, mem_id: str, *, from_ts: str, reason: str
    ) -> list[WriteIntent]:
        """**时态干预**：这条从 `from_ts` 起不再成立（T-AL2-22）。"""
        return self.reviewer.invalidate_since(mem_id, from_ts=from_ts, reason=reason)

    def promote(
        self, mem_id: str, *, reason: str, as_type: str = "identity"
    ) -> list[WriteIntent]:
        """**核心记忆干预：升格**（T-AL2-22）——人工升格**不过置信度门槛**。"""
        return self.reviewer.promote(mem_id, reason=reason, as_type=as_type)

    def demote(
        self, mem_id: str, *, reason: str, target_layer: str = "semantic"
    ) -> list[WriteIntent]:
        """**核心记忆干预：降格**（T-AL2-22）——留审计、不删内容。"""
        return self.reviewer.demote(mem_id, reason=reason, target_layer=target_layer)

    def forget(
        self,
        mem_id: str,
        *,
        reason: str,
        source: str,
        purge_snapshot: bool = False,
    ) -> list[WriteIntent]:
        """物理删除的唯一人工收口（D-17 入口 2）。"""
        return self.reviewer.forget(
            mem_id, reason=reason, source=source, purge_snapshot=purge_snapshot
        )

    def restore(self, audit_id: int) -> list[WriteIntent]:
        return self.reviewer.restore(audit_id)

    # ================================================================== #
    # 时态（T-AL2-16 · M3）
    # ================================================================== #

    def asof(self, ref: str, ts: str) -> MemoryRecord | None:
        """取 ``ts`` 时刻**有效**的那一版。

        ``None`` = **那时不存在有效版本**——不抛错，也不回退到当前值：
        调用方必须能区分"那时没有"与"看错了时间"。
        """
        return self.temporal.asof(ref, ts)

    def invalidate(
        self,
        mem_id: str,
        *,
        superseded_by: str,
        reason: str,
        valid_to: str | None = None,
        actor: str = "user",
    ) -> list[WriteIntent]:
        """把一条记忆标记为**失效**（**不是删除** · D-17 / INV-7）。

        产出两条意图：``update``（``valid_to`` + ``superseded_by``）与
        ``link(rel_type='supersedes')``（新 → 旧）——**必须一起写**，
        只写一半等于"追溯链只修了一个方向"。
        """
        return self.temporal.invalidate(
            mem_id,
            superseded_by=superseded_by,
            reason=reason,
            valid_to=valid_to,
            actor=actor,
        )

    # ================================================================== #
    # 交叉验证（T-AL2-18 · M3 · INV-11：高门槛低频）
    # ================================================================== #

    def crosscheck(self, *, now: str | None = None) -> CrossCheckReport:
        """跑一轮交叉验证：找冲突 → LLM 仲裁 → 产写意图。

        **不在热路径**：由 maintenance 周期调用（默认 30 min），或 CLI 显式触发。
        `llm` 为 `None` 时只报告冲突数、不做裁决——**不假装看过了**
        （`report.skipped` 会说清原因）。
        """
        return self.crosschecker.run(now=now)

    # ================================================================== #
    # 记忆进化（T-AL2-20 · M4）
    # ================================================================== #

    def evolve(self, *, limit: int = 20) -> EvolutionReport:
        """跑一轮记忆进化：用新信息完善旧条目的**抽象**。

        **只改 `abstract`，永不改 `content`**——原文是事实来源，抽象是投影。
        与交叉验证同理，**不在热路径**（一次 LLM 调用是秒级的）。
        """
        return self.evolver.run(limit=limit)

    # ================================================================== #
    # 传承重建（T-AL2-23 · M5）
    # ================================================================== #

    def restore_pack(self, pack: dict) -> TransferReport:
        """把一个**已解析的**档案 pack 翻成写意图（传承重建）。

        - **不做 I/O**（R5）：文件读取与解析在 `store.archive.load_archive`；
        - **不重新提取**：档案里每条都是已提取过的结论，再提一遍是用模型的偶然行为
          覆盖用户的资产（这也让"导入过程零 LLM 调用"成为可断言的性质）；
        - **按 `content_hash` 判重**（INV-14）：重复导入零新增。
        """
        return Transferrer(backend=self.backend, id_gen=self.id_gen).plan(pack)

    # ================================================================== #
    # 元认知
    # ================================================================== #

    def on_session_end(self, session_id: str) -> ConsolidationReport:
        return self.consolidate(session_id=session_id)

    def consolidate(self, *, session_id: str) -> ConsolidationReport:
        return self.consolidator.consolidate(session_id=session_id)

    def decay(self, *, now: str | None = None, dry_run: bool = True) -> DecayReport:
        return self.decayer.run(now=now, dry_run=dry_run)

    def optimize(
        self,
        *,
        dry_run: bool = True,
        autonomous: bool = False,
        confirm=None,
    ) -> OptimizationReport:
        return self.optimizer.run(dry_run=dry_run, autonomous=autonomous, confirm=confirm)

    def health(self) -> HealthReport:
        counts = self.backend.count_by_layer()
        total = sum(sum(v.values()) for v in counts.values())
        active = sum(v.get("active", 0) for v in counts.values())
        core_count = sum(counts.get("core", {}).values())
        stale = len(self.backend.overview_stale_list(limit=1000))

        records = self.backend.query(status="active", limit=self.settings.consolidate_limit)
        avg_strength = (
            round(sum(r.strength for r in records) / len(records), 4) if records else 0.0
        )
        return HealthReport(
            layer_counts=counts,
            total=total,
            core_count=core_count,
            stale_overviews=stale,
            avg_strength=avg_strength,
            promoted_candidates=active,
        )

    def system_prompt_block(self, *, token_budget: int = 400) -> str:
        """核心记忆摘要 + 器灵状态。**必须有预算并截断**（C11）。"""
        from ..common import truncate_to_tokens

        parts: list[str] = []
        # 状态行预留：预算本身不足预留时，预留不能超过预算。
        # 旧写法 `max(1, token_budget - 80)` 会凑出"永远无法满足的预算 1"
        # （单字最低成本 2），把 truncate_to_tokens 推进死循环、挂死宿主（DES-REV-009 P0-1）。
        reserve = min(80, token_budget)
        block = self.core_memory.prompt_block(token_budget=token_budget - reserve)
        if block:
            parts.append(block)

        health = self.health()
        active = sum(v.get("active", 0) for v in health.layer_counts.values())
        dormant = sum(v.get("dormant", 0) for v in health.layer_counts.values())
        parts.append(
            f"【记忆状态】长期记忆 {active} 条（休眠 {dormant} 条）"
            f" · 核心记忆 {health.core_count} 条"
        )
        return truncate_to_tokens("\n".join(parts), token_budget)

    # ================================================================== #
    # 待落库意图（读路径产生的）
    # ================================================================== #

    def drain_pending(self) -> list[WriteIntent]:
        """取走读路径（``expand``）产生的写意图，交 AL5 writer 执行。"""
        pending, self._pending = self._pending, []
        return pending

    def refresh_overviews(self, *, limit: int = 20) -> list[WriteIntent]:
        """重算 stale 的 L1 概览（**只应由 maintenance 线程调用**）。

        这是把 L1 生成挪出热路径的落点（N-P0-2）：热路径只读旧缓存或降级 L0，
        真正的重算在这里发生。
        """
        from ..common import estimate_tokens

        intents: list[WriteIntent] = []
        for overview in self.backend.overview_stale_list(limit=limit):
            regenerated = self.progressive.regenerate(
                overview.scope_kind, overview.scope_id, overview.level
            )
            if regenerated is None:
                continue
            text, model_used = regenerated
            intents.append(
                WriteIntent(
                    op="overview_put",
                    scope_kind=overview.scope_kind,
                    scope_id=overview.scope_id,
                    overview_content=text,
                    overview_level=overview.level,
                    token_count=estimate_tokens(text),
                    model=model_used,
                    actor="system",
                )
            )
        return intents


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


_AUDIT_OP_FOR_DECISION: dict[str, str] = {
    "ADD": "add",
    "UPDATE": "update",
    "IGNORE": "ignore",
    "MERGE": "merge",
    "INVALIDATE": "invalidate",
}
"""决策 → 审计 ``op`` 的**穷举映射**（P0-6：账本必须等于真正执行的那个动作）。

用「表 + 下标」而不是 `if/elif` 链：**少一项就是 `KeyError`**——红在第一次遇到该决策时，
而不是"静默少写一条审计"。它与 `Decision` 的取值域必须一一对应，
由 `test_audit_op_covers_every_decision` 钉住。
"""


def _before_snapshot(decision: DedupDecision, backend: MemoryBackend) -> dict | None:
    """决策要改动的那条记录的**旧值快照**。

    P0-6 的教训是"账本不许撒谎"，而撒谎有两种形态：**写错的 op**，以及**空话 before**。
    后者更隐蔽——`before=None` 看着只是"没记"，实际让人永远说不清"改之前长什么样"。
    """
    if not decision.target_id:
        return None
    current = backend.get(decision.target_id)
    if current is None:
        return None
    return {
        "content": current.content,
        "object": current.object,
        "valid_to": current.valid_to,
        "superseded_by": current.superseded_by,
    }


def _clip(scored: list[Scored], token_budget: int, top_k: int) -> list[Scored]:
    from .recall import clip_to_budget

    return clip_to_budget(scored[:top_k], token_budget)


def _rehearsed_strength(backend: MemoryBackend, mem_id: str) -> float | None:
    """间隔重复效应：再次被用到时强度回升。

    计算放在 AL2（领域知识），AL3 只负责存（它不该知道"回升多少"）。
    """
    record = backend.get(mem_id)
    if record is None:
        return None
    return min(1.0, record.strength + 0.2)


def _patch_from_candidate(candidate: dict, ctx: TurnContext) -> dict:
    return {
        "content": candidate["content"],
        "abstract": candidate.get("abstract") or first_sentence(candidate["content"]),
        "subject": candidate.get("subject"),
        "predicate": candidate.get("predicate"),
        "object": candidate.get("object"),
        "scope": candidate.get("scope"),
        "confidence": candidate.get("confidence", 0.7),
        "salience": candidate.get("salience", 0.0),
        "source_session": ctx.session_id,
    }
