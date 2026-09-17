"""存储层契约：数据结构 + 异常族 + ``MemoryBackend`` 协议。

本模块**零 I/O、零具体依赖**（LLD-AL3 §3）。
上层（AL2 核心层）只依赖本模块，不依赖任何具体实现（R1）。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

# --------------------------------------------------------------------------- #
# 类型别名
# --------------------------------------------------------------------------- #

Layer = Literal["episodic", "semantic", "procedural", "core"]
"""记忆层。感觉记忆与工作记忆不落 ``memories`` 表，故不在此列。"""

Status = Literal["active", "dormant", "forgotten"]
"""生命周期状态（D-17 / D-23）。

- ``active``    —— 全层级参与召回（L0 初筛 → L1 判断 → L2 展开）
- ``dormant``   —— **只保留 L0 参与召回**，详情仍在库（可逆）
- ``forgotten`` —— 物理删除（快照可恢复，合规删除除外）

注意：**没有 ``archived``**。原设计中的"归档丢详情"已取消——
"看到多少细节"属摘要层（L0/L1/L2）职责，不该用不可逆方式实现。
"""

Kind = Literal["memory", "entity"]
"""关联边的端点类型。"""


# --------------------------------------------------------------------------- #
# 异常族（LLD-AL3 §7）
# --------------------------------------------------------------------------- #


class StoreError(Exception):
    """存储层异常基类。"""


class StorageBusyError(StoreError):
    """SQLite 锁超时（F4）。可重试——由 AL5 writer 负责重试。"""


class DimensionMismatchError(StoreError):
    """向量维度不符（F7）。拒绝写入，不得静默截断。"""


class StorageFatalError(StoreError):
    """不可恢复的存储故障（磁盘满 / 库损坏）。

    由 AL1 转换为工具级错误——**绝不崩溃宿主**（P6 / INV 精神）。
    """


class SchemaVersionError(StoreError):
    """schema 版本与代码期望不一致。"""


# --------------------------------------------------------------------------- #
# 版本契约
# --------------------------------------------------------------------------- #

SCHEMA_VERSION = 2
"""当前代码期望的 ``meta.schema_version``。

- v1：初版全量 DDL（M0）
- v2：``mem_fts`` 改为 CJK 归一化的普通 FTS5 表 + 新增 ``delete_snapshots``

**常量放在协议层而非实现层**：AL1 做"库可用性预检"时需要比对这个版本，
若常量只存在于 ``sqlite_backend``，最外层就得 import 实现模块（连 sqlite-vec
一起拖进来），从而把 AL3 的内部结构暴露给 AL1——正是 DES-REV-003 的 R9 要拦的事。
实现层（``sqlite_backend``）与只读探针（``probe``）都从本模块取，源头唯一。
"""


class NotFoundError(StoreError):
    """目标记录不存在。"""


class WhitelistViolation(StoreError):
    """违反删除纪律（D-17 / D-22）。

    物理删除仅两个入口：① 记忆优化任务 ② 用户显式要求。
    缺 ``reason`` / ``source``、或从其他路径调用，均抛本异常。
    """


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MemoryRecord:
    """统一记忆条目。所有层共用此结构（真相源的载体）。"""

    id: str
    layer: Layer
    type: str
    content: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    abstract: str | None = None  # L0 摘要
    scope: dict | None = None  # {"type": "global|project|session", "id": ...}
    confidence: float = 0.7
    salience: float = 0.0
    # 活性（派生，可从 audit 重放重建 —— INV-12）
    strength: float = 1.0
    access_count: int = 0
    last_access_at: str | None = None
    # 时态（双时态；MVP 只写 valid_from）
    valid_from: str | None = None
    valid_to: str | None = None
    # 来源
    source_session: str | None = None
    source_turn: int | None = None
    # 生命周期
    status: Status = "active"
    superseded_by: str | None = None
    # 审计
    created_at: str = ""
    updated_at: str = ""
    # 一致性
    embedding_model: str | None = None
    # 内容寻址指纹（D-21 / INV-14）
    content_hash: str | None = None


class Hit(NamedTuple):
    """单路检索的原生结果。

    ``score`` 是该路的**原生分**，跨路不可比——融合排序在 AL2 的 ``fuse()``。
    ``content`` 是交付给宿主的**文本**（非向量）。
    """

    mem_id: str
    layer: str
    content: str
    score: float
    meta: dict


AUDIT_OPS = frozenset(
    {
        # 记忆生命周期
        "add",
        "update",
        "set_status",
        "touch",
        "forget",
        "restore",
        # 时态（M3 · T-AL3-24）：**被取代不是删除**——写 `valid_to` + `superseded_by`
        "invalidate",
        # 去重决策的"没写"与"合并"分支（M3 · T-AL2-17）。
        # **"决定不做"同样是决定**：不留痕就永远解释不了"这条为什么没进来"。
        "ignore",
        "merge",
        # 关联
        "link",
        "reinforce",
        # 后台流程
        "consolidate",
        "summarize",
        "evolve",
        # 核心记忆的升格 / 降格（M4 · T-AL2-21 / T-AL2-22）。
        # **人格的变化是后果最重的一类写入**：自动升格与人工升降格都必须各留一条，
        # 否则"它是什么时候变成我的底色的"永远查不出来。
        "promote",
        "demote",
        # 维护与传承
        "reindex",
        "reconcile",
    }
)
"""``AuditEvent.op`` 的**唯一权威列举**（评审 P1-20 的处置）。

在此之前，"权威列举"散在**三处**（LLD / `schema.sql` 注释 / 本文件 docstring）且互不相同：
文档独有 5 个（`merge` / `invalidate` / `ignore` / `dormant` / `unlink`），
实现独有 4 个（`set_status` / `touch` / `reinforce` / `reconcile`）。

其中 **3 个已在 M3 补齐实现**（`merge` / `invalidate` / `ignore` —— 去重三态）。
剩下 2 个（`dormant` / `unlink`）**仍然没有实现**，在此**登记为"未落地"而不是删掉**：
删掉会让下一个人以为"从来没有人打算做这件事"——而登记着，它就是一个待还的债。

现在只有这一份，且写入侧**强制校验**（``SQLiteBackend._insert_audit``）：
一列"写什么都收"的 op 会让账本慢慢堆成没人认识的字符串，
而审计是**可溯性的底座**（INV-12）——底座的取值域不能是"随便"。
"""


@dataclass(slots=True)
class AuditEvent:
    """审计事件（append-only —— INV-8）。

    ``op`` 的权威列举是 :data:`AUDIT_OPS`（本文中出现的任何 op 名字都以它为准）。

    ``actor`` 用于区分**自动流程**与**人的干预**（D-17 / M11）：
    ``extractor`` / ``dedup`` / ``consolidator`` / ``decay`` / ``optimizer`` /
    ``user`` / ``cli`` / ``system``。

    注意：删除时的**内容快照不在这里**，而在 ``delete_snapshots`` 表——
    合规删除需要清除内容，而本表 append-only，不可修改（见 schema.sql 说明）。
    """

    op: str
    actor: str
    target_kind: str | None = None
    target_id: str | None = None
    before: dict | None = None
    after: dict | None = None
    reason: str | None = None
    session_id: str | None = None
    ts: str = ""

    # 审计行主键（``audit.id``）。**恢复必须用它**：`restore_from_audit(audit_id)`
    # 认的是这个值，不是"在某个过滤结果里的第几条"。
    #
    # 放在字段末尾且带默认值，是为了不破坏既有的 24 处关键字构造。
    audit_id: int | None = None


@dataclass(slots=True)
class WorkingChunk:
    """工作记忆组块（会话内活跃内容，**不落 ``memories`` 表**）。

    与 :class:`MemoryRecord` 分开，是因为它没有层归属、没有向量、没有审计——
    它是"注意力里的东西"，不是"记住的东西"。会话结束时会固化进情景记忆。
    """

    id: str
    session_id: str
    chunk_key: str
    content: str
    salience: float
    act_count: int
    created_at: str
    last_touched: str


@dataclass(slots=True)
class EntityRecord:
    """实体节点（语义记忆的图结构）。"""

    id: str
    name: str
    type: str
    aliases: list[str]
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class OverviewRecord:
    """L1 概览缓存条目（M9 · P1）。

    ``stale=True`` 表示底层记忆已变更，缓存待重算——**热路径只读不重算**（N-P0-2）。
    """

    id: str
    scope_kind: str
    scope_id: str
    level: str
    content: str
    token_count: int | None = None
    generated_at: str = ""
    stale: bool = False
    model: str | None = None


# --------------------------------------------------------------------------- #
# 协议
# --------------------------------------------------------------------------- #


class MemoryBackend(Protocol):
    """存储后端协议。

    **接口稳定性契约**：一经发布即视为稳定接口。
    AL2 只依赖它；任何实现变更（含新增后端）不得修改既有签名语义。
    """

    # ---------- 生命周期 ----------
    def open(self) -> None: ...
    def close(self) -> None: ...
    def migrate(self) -> None: ...

    # ---------- 写（真相源）----------
    def put(
        self,
        rec: MemoryRecord,
        embedding: list[float] | None = None,
        *,
        audit: AuditEvent | None = None,
    ) -> str: ...
    def update(self, mem_id: str, patch: dict, *, audit: AuditEvent | None = None) -> None: ...
    def set_status(
        self, mem_id: str, status: Status, *, reason: str, actor: str = "system"
    ) -> None: ...
    def hard_delete(
        self,
        mem_id: str,
        *,
        reason: str,
        purge_snapshot: bool = False,
        actor: str = "user",
        source: str = "unknown",
    ) -> None: ...

    # ---------- 读 ----------
    def get(self, mem_id: str) -> MemoryRecord | None: ...
    def asof(self, mem_id: str, ts: str) -> MemoryRecord | None: ...
    """取 ``ts`` 时刻**有效**的那一版（双时态 · M3）。

    **必须在协议里**：AL2 的 `asof` 与 AL1 的 `spirit_asof` 都通过本协议调用它。
    协议少一个成员，上层就"合法地"不知道它存在——AL3 P1-21 的教训。
    """
    def query(
        self,
        *,
        layer: Layer | None = None,
        status: Status | None = "active",
        types: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]: ...
    def count_by_layer(self) -> dict[str, dict[str, int]]: ...

    # ---------- 检索原语（只返回原生分，融合在 AL2）----------
    def vector_search(
        self, vec: list[float], *, layer: Layer | None = None, top_k: int = 8
    ) -> list[Hit]: ...
    def keyword_search(
        self, query: str, *, layer: Layer | None = None, top_k: int = 8
    ) -> list[Hit]: ...

    # ---------- 关联 ----------
    def link(
        self,
        a_kind: Kind,
        a_id: str,
        b_kind: Kind,
        b_id: str,
        rel_type: str,
        weight: float,
    ) -> None: ...
    def reinforce(
        self, a_id: str, b_id: str, delta: float, *, rel_type: str = "co_activation"
    ) -> None: ...
    def outbound_edges(self, mem_ids: list[str]) -> list[dict]: ...
    """批量取这些记忆的出边。**批量而不是逐条**——扩散激活与交叉验证都要扫一圈邻居，
    逐条查会退化成 N+1。

    以下这几个方法原先只有 `SQLiteBackend` 实现、**协议里没声明**：
    调用方拿到的类型检查是红的，而运行时是好的。与 `wm_list` 同一类问题（见 §15 / §18.4）。

    > 这类缺口只有 `mypy` 看得见——`ruff` 看语法与模式，`mypy` 看**契约**。
    > 而契约正是分层边界的那个东西：**协议落后于实现，就等于边界已经松了还没人知道**。
    """

    def inbound_reference_counts(self, mem_ids: list[str]) -> dict[str, int]: ...
    """批量统计每条记忆的**入边引用数**（`importance` 的"被反复引用"分量）。

    **刻意批量**：逐条查会让召回退化成 N+1，而召回是每轮都跑的热路径。
    """

    def core_memory_terms(self) -> list[str]: ...
    """核心记忆的**主题词**集合——`core` 因子靠它判断"这条与人格是否同题"。

    返回词而不是记忆本身：调用方要的是**比对**，不是读内容。
    """

    def all_relations(self) -> list[dict]: ...
    """全量关联边——**诊断与优化**用（找孤立点、对账）。

    它**刻意没有 `limit`**：用它的人要的是"全貌"，分页反而会让人误以为看到的就是全部。
    所以它只该在维护路径上调，不进热路径。
    """

    def set_vector(self, mem_id: str, vec: list[float]) -> None: ...
    """单独更新某条记忆的向量（内容被修改后重算）。

    **必须做维度校验**：写入与检索用了不同嵌入模型时，这是最后一道闸门。
    """

    def neighbors(
        self,
        kind: Kind,
        node_id: str,
        *,
        rel_type: str | None = None,
        min_weight: float = 0.0,
        limit: int = 20,
    ) -> list[tuple[str, str, float]]: ...
    def relations_of(self, mem_id: str) -> list[dict]: ...

    # ---------- 实体 ----------
    def entity_upsert(
        self, name: str, type_: str, *, aliases: list[str] | None = None
    ) -> str: ...
    def entity_find(self, text: str) -> list[EntityRecord]: ...
    def entity_get(self, entity_id: str) -> EntityRecord | None: ...

    def entity_list(self) -> list[EntityRecord]: ...
    """全部实体——巩固时用来判断"某个名字是不是已经认识的实体"。

    没有 `limit`，因为调用方要的是**集合成员资格**：分页会让
    "认不认识这个实体"取决于恰好翻到第几页。
    """

    # ---------- 审计（append-only —— INV-8）----------
    def audit(self, ev: AuditEvent) -> int: ...
    def audit_replay(self, *, since: str | None = None) -> Iterator[AuditEvent]: ...
    def audit_get(self, audit_id: int) -> AuditEvent | None: ...
    def audit_for(self, mem_id: str, *, limit: int = 200) -> list[AuditEvent]: ...

    # ---------- 活性 ----------
    def touch(self, mem_id: str, ts: str, *, strength: float | None = None) -> None: ...
    """记录一次访问：`access_count += 1`、刷新 `last_access_at`，可选回升 `strength`。

    `strength` **原先只有实现有、协议里没声明**——与 `wm_list`（§15）、
    `outbound_edges`（§18.4）是同一类：**协议落后于实现**。
    运行时一直是对的（拿到的确实是 `SQLiteBackend`），
    但"按协议编程"的调用方会看不到这个参数——**而协议正是分层边界本身**。
    """

    # ---------- 会话 ----------
    def session_create(self, session_id: str, started_at: str) -> None: ...
    def session_bump_turn(self, session_id: str) -> int: ...
    def session_end(self, session_id: str, ended_at: str) -> None: ...
    def session_get(self, session_id: str) -> dict | None: ...

    # ---------- 工作记忆 / 意图 ----------
    def wm_put(
        self, session_id: str, chunk_key: str, content: str, salience: float
    ) -> str: ...
    def wm_list(self, session_id: str, limit: int = 9) -> list[WorkingChunk]: ...
    def wm_clear(self, session_id: str) -> None: ...
    def intent_put(
        self,
        content: str,
        session_id: str | None = None,
        due_at: str | None = None,
    ) -> str: ...
    def intent_list(self, status: str = "open") -> list[dict]: ...

    # ---------- 元 ----------
    def meta_get(self, key: str) -> str | None: ...
    def meta_set(self, key: str, value: str) -> None: ...

    # ---------- 分级加载（M9）----------
    def overview_get(
        self, scope_kind: str, scope_id: str, level: str = "L1"
    ) -> OverviewRecord | None: ...
    def overview_put(
        self,
        scope_kind: str,
        scope_id: str,
        level: str,
        content: str,
        *,
        token_count: int,
        model: str,
    ) -> None: ...
    def overview_invalidate(
        self, *, scope_kind: str | None = None, scope_id: str | None = None
    ) -> int: ...
    def overview_stale_list(self, *, limit: int = 50) -> list[OverviewRecord]: ...
    def overview_list(self, *, limit: int | None = None) -> list[OverviewRecord]: ...

    # ---------- 内容寻址（D-21）----------
    def find_by_hash(self, content_hash: str) -> list[MemoryRecord]: ...

    def find_by_source_session(
        self, session_id: str, *, layer: Layer | None = None
    ) -> list[MemoryRecord]: ...
    """按**来源会话**查记忆（巩固的幂等判据）。

    用 SQL 回答"这个会话是否已固化过"，比把整层拉进内存再过滤要诚实得多：
    `query()` 默认只看 `active`，会把已休眠的固化结果**悄悄滤掉**——
    于是"固化过没有"这个判断在反复巩固之后开始给出"没有"。
    """

    # ---------- 恢复（D-22）----------
    def restore_from_audit(self, audit_id: int) -> str: ...

    # 可恢复的删除快照清单。`include_payload=True` 时每项额外带 `record`
    # （被删记录的字段）与 `relation_count` —— 审计视图靠它显示"这次删掉的到底是什么"。
    #
    # **清单即安全网**：任何一次未显式要求清除（`purge_snapshot=True`）的删除，都必须
    # 在这里留下一条。闭合性由 `test_every_forget_leaves_a_restorable_snapshot` 守住。
    def delete_snapshots(self, *, include_payload: bool = False) -> list[dict]: ...

    # ---------- 运维 ----------
    def reconcile(self, *, dry_run: bool = True) -> dict: ...
    def replay_derived(self) -> dict: ...
    def rebuild_fts(self) -> int: ...

    # ---------- 传承（P3 / INV-13）----------
    def archive_version(self) -> int: ...
    """当前**代码**能写出的档案格式版本（T-AL3-26）。

    它是**代码常量**而不是库属性——档案格式由"写档案的那份代码"决定，与库里的数据无关。
    放在协议上，是为了让 AL5 的 `doctor` 能报出它而**不必 import 本层的具体模块**（R10）：
    "要不要暴露"这件事，判据是**有没有别的层需要读它**，而不是它看起来像谁的东西。
    """

    def export_archive(self, path: str, *, fmt: str = "markdown") -> None: ...
    def export_pack(self) -> bytes: ...
    def import_pack(self, data: bytes, *, merge: bool = False) -> dict: ...
    def import_archive(self, path: str) -> dict: ...
