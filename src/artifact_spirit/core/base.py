"""AL2 核心层共享数据结构与协议。

本模块**纯声明、零 I/O**（R5）。

关键的架构点：**AL2 不落库**。它产出 :class:`WriteIntent` 列表，由 AL5 的 writer 线程
翻译为 AL3 调用并执行。这样 AL2 的可测性是"断言意图"，不需要真库也不需要线程。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..store.base import AuditEvent, MemoryRecord

__all__ = [
    "Clock",
    "ConsolidationReport",
    "CoreFacade",
    "DecayReport",
    "HealthReport",
    "IdGen",
    "IntentOp",
    "OptimizationReport",
    "RecallQuery",
    "RecallWeights",
    "Scored",
    "TurnContext",
    "TurnEvent",
    "WriteIntent",
]

IntentOp = Literal[
    "put",
    "update",
    "set_status",
    "link",
    "reinforce",
    "touch",
    "forget",
    "audit",
    "overview_invalidate",
    "overview_put",
    "restore",
    "wm_put",
    "wm_delete",
    "session_create",
    "session_end",
    "entity_upsert",
]

Clock = Callable[[], str]
"""时钟签名：``() -> ISO8601 字符串``。

**必须注入**——否则衰减与时序逻辑无法稳定测试（C2）。
"""

IdGen = Callable[[str], str]
"""ID 生成器签名：``(layer) -> str``。同样注入，理由同上。"""


@dataclass(frozen=True, slots=True)
class WriteIntent:
    """一条写意图。**AL2 的唯一输出形式**（除了查询结果）。

    AL5 的 writer 会按 ``op`` 分派到 AL3：

    | op | 对应 AL3 调用 |
    |---|---|
    | ``put`` | ``backend.put(record, embedding)`` |
    | ``update`` | ``backend.update(mem_id, patch, audit=…)`` |
    | ``set_status`` | ``backend.set_status(mem_id, status, reason=…, actor=…)`` |
    | ``link`` | ``backend.link(...)`` |
    | ``reinforce`` | ``backend.reinforce(...)`` |
    | ``touch`` | ``backend.touch(mem_id, ts, strength=…)`` |
    | ``forget`` | ``backend.hard_delete(mem_id, reason=…, purge_snapshot=…)`` |
    | ``overview_invalidate`` | ``backend.overview_invalidate(...)`` |
    | ``session_create`` / ``session_end`` | ``backend.session_create/session_end(session_id, ts)`` |

    ``embed_text`` 非空时，writer 需要先做一次向量化再写入——
    把"要嵌入什么"的决定留在 AL2（领域知识），把"怎么嵌入"留给 AL5/AL4。
    """

    op: IntentOp
    record: MemoryRecord | None = None
    mem_id: str | None = None
    patch: dict | None = None
    status: str | None = None
    reason: str | None = None
    actor: str = "system"
    # 关联
    a_kind: str = "memory"
    a_id: str | None = None
    b_kind: str = "memory"
    b_id: str | None = None
    rel_type: str | None = None
    weight: float = 0.0
    delta: float = 0.0
    # 活性
    ts: str | None = None
    strength: float | None = None
    # 审计
    audit: AuditEvent | None = None
    # 向量化
    embed_text: str | None = None
    # 删除
    purge_snapshot: bool = False
    audit_id: int | None = None
    # 概览失效 / 写入
    scope_kind: str | None = None
    scope_id: str | None = None
    overview_content: str | None = None
    overview_level: str = "L1"
    token_count: int = 0
    model: str = ""
    # 工作记忆（会话内，非真相源）
    session_id: str | None = None
    chunk_key: str | None = None
    chunk_id: str | None = None
    content: str | None = None
    salience: float = 0.0
    # 实体登记（从提取结果同步，供实体匹配因子与"可达性"判定使用）
    entity_name: str | None = None
    entity_type: str = "concept"
    aliases: tuple[str, ...] = ()

    @property
    def target(self) -> str:
        return self.mem_id or self.a_id or (self.record.id if self.record else "") or "?"


@dataclass(frozen=True, slots=True)
class TurnEvent:
    """一轮对话（写入路径的输入）。"""

    session_id: str
    user: str
    assistant: str
    ts: str
    messages: list[dict] = field(default_factory=list)
    recalled: tuple[str, ...] = ()
    """上一轮召回并交付给宿主的记忆 id。

    用于：① 记录访问（``touch``）；② Hebbian 共激活建边。
    宿主不会告诉器灵"模型到底看了哪几条"，因此以**上一轮召回结果**为代理指标——
    这比"什么都不知道"好得多，且成本为零。
    """


@dataclass(slots=True)
class TurnContext:
    """一轮对话在核心层内的上下文（层服务共享）。"""

    session_id: str
    ts: str
    user: str
    assistant: str

    @property
    def text(self) -> str:
        return f"{self.user}\n{self.assistant}".strip()


@dataclass(slots=True)
class Scored:
    """召回结果的一条。

    ``raw`` 必须保留**六个分量**——可解释性是一等公民（C4）：用户要能问
    "这条为什么被召回"，系统要答得出来。
    """

    record: MemoryRecord
    raw: dict[str, float]
    score: float


@dataclass(slots=True)
class RecallQuery:
    """召回请求。"""

    text: str
    vec: list[float] | None = None
    session_id: str = ""
    layers: list[str] | None = None
    token_budget: int = 2000
    top_k: int = 8


@dataclass(slots=True)
class RecallWeights:
    """六因子权重（合计 1.0）。

    D-20：原 ``vitality``（strength 与 access_count 合成）拆为两个**独立**因子——
    ``recency``（时间邻近）与 ``importance``（重要度）。

    原因：``strength`` 内部已含时间衰减，与 recency 语义重叠；而"低频但关键"的记忆
    （身份证号 / 过敏史 / 紧急联系人）**只能靠 importance 救回来**，频率给不了它任何分
    （见 DES-RES-002 §4.1）。
    """

    semantic: float = 0.40
    importance: float = 0.20
    recency: float = 0.15
    entity: float = 0.10
    diffusion: float = 0.10
    core: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "importance": self.importance,
            "recency": self.recency,
            "entity": self.entity,
            "diffusion": self.diffusion,
            "core": self.core,
        }

    def total(self) -> float:
        return sum(self.as_dict().values())


@dataclass(slots=True)
class ConsolidationReport:
    """巩固结果。``intents`` 需要 AL5 writer 落库。"""

    session_id: str
    episode_id: str | None = None
    promoted: list[str] = field(default_factory=list)
    intents: list[WriteIntent] = field(default_factory=list)
    skipped: str | None = None


@dataclass(slots=True)
class DecayReport:
    """衰减报告。

    **注意 ``downgrades`` 恒为空**：D-16/D-23 明确"降级不得由时间或频率触发"，
    状态迁移只由**记忆优化任务**（:meth:`CoreFacade.optimize`）依据图结构不可达判定。
    本报告只提供**排序信号**与**可见的候选清单**，不改状态。
    """

    total: int = 0
    recomputed: int = 0
    low_strength: list[tuple[str, float]] = field(default_factory=list)
    downgrades: list[str] = field(default_factory=list)
    intents: list[WriteIntent] = field(default_factory=list)
    dry_run: bool = True


@dataclass(slots=True)
class OptimizationReport:
    """记忆优化（治理性删除与注意力降级）报告。

    三类触发（D-23）：**错误 / 冲突 / 不可达**。其中"不可达"= 图结构孤立
    （无入边 + 无出边 + 无实体关联 + 不在任何 L1 概览中），**与时间、频率无关**。

    ``notes`` 记录判定过程中**被跳过的原因**——例如"实体表为空，无法判定实体关联"。
    没有这个字段，一次"没找到任何候选"的运行会让人分不清
    "确实干净"和"判据根本没生效"。
    """

    scanned: int = 0
    unreachable: list[str] = field(default_factory=list)
    downgrades: list[str] = field(default_factory=list)
    stale_overviews: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    intents: list[WriteIntent] = field(default_factory=list)
    dry_run: bool = True


@dataclass(slots=True)
class HealthReport:
    """记忆健康度（`reflect` 的数据源）。"""

    layer_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    total: int = 0
    core_count: int = 0
    stale_overviews: int = 0
    avg_strength: float = 0.0
    promoted_candidates: int = 0


@dataclass(slots=True)
class CrossCheckReport:
    """一轮交叉验证的结果（T-AL2-18 · M3）。

    **候选集与处置结果都要可见**——这是"高门槛低频"的能力最容易被做成黑箱的地方：
    用户只看到"巩固完成"，却不知道它裁决了几对、保留了几对、跳过没有。
    """

    conflicts: int = 0
    """发现的冲突对数。"""
    arbitrated: int = 0
    """真正改动了记忆的裁决数（`INVALIDATE` / `MERGE`）。"""
    kept_both: int = 0
    """裁决为「两条都留」的数量。"""
    skipped: str | None = None
    """未执行的原因（如"未配置 LLM"）——**不假装看过了**。"""
    intents: list[WriteIntent] = field(default_factory=list)


@dataclass(slots=True)
class EvolutionReport:
    """一轮记忆进化的结果（T-AL2-20 · M4）。"""

    scanned: int = 0
    """考察过的记忆数。"""
    evolved: int = 0
    """真正改写了抽象的记忆数。"""
    skipped: str | None = None
    """未执行的原因（如"未启用" / "未配置 LLM"）——**不假装做过了**。"""
    intents: list[WriteIntent] = field(default_factory=list)


@dataclass(slots=True)
class TransferReport:
    """一次传承重建的结果（T-AL2-23 · M5）。

    四个计数**各自独立**：只报"导入了 N 条"是不够的——
    用户最需要知道的是"**跳过了多少、为什么**"，那才是"我的资料到底进来没有"的答案。
    """

    imported: int = 0
    """新落库的记忆数。"""
    skipped: int = 0
    """因 `content_hash` 已存在而跳过的记忆数（INV-14 —— 幂等的来源）。"""
    relations: int = 0
    entities: int = 0
    restored_superseded: int = 0
    """回填了 `superseded_by` 的条数（两阶段的第二阶段）。"""
    errors: list[str] = field(default_factory=list)
    """**读得懂、但放不进去**的条目（如关联指向不存在的记忆）。

    与"跳过"分开：跳过是**正常**的（内容已在库里），错误是**异常**的（数据有问题）。
    合并计数会让真正的数据问题被"幂等"这个好消息掩盖掉。
    """
    intents: list[WriteIntent] = field(default_factory=list)


class CoreFacade(Protocol):
    """AL2 对外的唯一门面。**AL1 与 AL5 只依赖它**（评审 P0-1）。

    有了这个单一可替换门面，AL1 才能"用 fake 核心层测试"——否则 AL1 的独立验收
    根本无从谈起。
    """

    # ---- 写入路径（同步产出决策，落库由 AL5 writer 执行）----
    def ingest_turn(self, event: TurnEvent) -> list[WriteIntent]: ...

    # ---- 召回路径（AL1 的 prefetch 调用）----
    def recall(self, q: RecallQuery) -> list[Scored]: ...

    # ---- 分级加载（P1 首要机制）----
    def expand(self, ref: str, level: str = "L0") -> str: ...

    # ---- 审查与主动干预（P2 首要机制）----
    def review(
        self, *, layer: str | None = None, since: str | None = None, limit: int = 50
    ) -> list[dict]: ...

    def trace(self, mem_id: str) -> list[dict]: ...

    def correct(self, mem_id: str, patch: dict, *, reason: str) -> list[WriteIntent]: ...

    # ---- 干预的两种"更细"形态（M4 · T-AL2-22）----
    def invalidate_since(
        self, mem_id: str, *, from_ts: str, reason: str
    ) -> list[WriteIntent]: ...
    """**时态干预**：事实变了（不是记错了）——截断有效期，历史仍可查（INV-7）。"""

    def promote(
        self, mem_id: str, *, reason: str, as_type: str = "identity"
    ) -> list[WriteIntent]: ...
    """核心记忆升格（人工发起，**不过置信度门槛**——门槛是给机器设的）。"""

    def demote(
        self, mem_id: str, *, reason: str, target_layer: str = "semantic"
    ) -> list[WriteIntent]: ...
    """核心记忆降格（留审计、不删内容）。"""

    def forget(
        self,
        mem_id: str,
        *,
        reason: str,
        source: str,
        purge_snapshot: bool = False,
    ) -> list[WriteIntent]: ...

    def restore(self, audit_id: int) -> list[WriteIntent]: ...

    # ---- 时态（T-AL2-16 · M3）----
    def asof(self, ref: str, ts: str) -> MemoryRecord | None: ...
    """取 `ts` 时刻有效的那一版；**`None` = 那时不存在有效版本**。"""

    def invalidate(
        self,
        mem_id: str,
        *,
        superseded_by: str,
        reason: str,
        valid_to: str | None = None,
        actor: str = "user",
    ) -> list[WriteIntent]: ...
    """把一条记忆标记为**失效**（不是删除）——同时写 `superseded_by` 与 `supersedes` 边。"""

    def crosscheck(self, *, now: str | None = None) -> CrossCheckReport: ...
    """跑一轮交叉验证（**不在热路径**，由 maintenance 周期调用）。"""

    def evolve(self, *, limit: int = 20) -> EvolutionReport: ...
    """跑一轮记忆进化（**只改 `abstract`**；同**不在热路径**）。"""

    def restore_pack(self, pack: dict) -> TransferReport: ...
    """传承重建（T-AL2-23）：把**已解析的**档案 pack 翻成写意图。**不做 I/O**（R5）。

    刻意不叫 `import_pack`：那是 AL3 的**直插**接口。两个名字同形会让人
    在 facade 上以为拿到了"导入"，实际拿到的是**导入计划**——而计划不落库。
    """

    # ---- 会话生命周期 ----
    def on_session_end(self, session_id: str) -> ConsolidationReport: ...

    # ---- 元认知（AL5 maintenance 调用）----
    def consolidate(self, *, session_id: str) -> ConsolidationReport: ...

    def decay(self, *, now: str | None = None, dry_run: bool = True) -> DecayReport: ...

    def optimize(self, *, dry_run: bool = True) -> OptimizationReport: ...

    def health(self) -> HealthReport: ...

    # ---- 核心记忆 ----
    def system_prompt_block(self, *, token_budget: int = 400) -> str: ...
