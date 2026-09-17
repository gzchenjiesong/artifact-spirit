# LLD-AL3 存储层详细设计

> 上级文档：[DES-000 概要设计 §5 / §3.2](../00-方案设计.md)
> 选型依据：[ADR-002 底层存储选型](../02-底层存储选型.md)
> 状态：初版，待评审

---

## 1. 职责与边界

### 1.1 负责

| 职责 | 说明 |
|---|---|
| **真相源持久化** | 记忆条目、**会话**、工作记忆、意图、实体、关联的唯一存储 |
| **检索原语** | 向量 KNN、关键词 BM25 —— 只提供"原语"，不做融合排序 |
| **审计链** | append-only 的操作事件记录，支持回放 |
| **活性字段维护** | `strength` / `access_count` / `last_access_at` 的读写 |
| **记忆包往返** | `export`（档案 / 完整包）与 `import`（重建）：依 `content_hash` 去重、护住 `parent_ref` 与关联边——INV-13 / INV-14 的落点。**往返保真是 AL3 的责任，不是 AL2 的**（内容寻址指纹是存储层机制） |
| **重嵌入与重放** | `reindex`（换 embedding 模型）、`replay`（从 audit 重建派生字段） |
| **架构探针** | schema 版本探测、库可用性预检等**只读**探针（`probe.py`）：给 AL1 的 `is_available()` 与 `doctor` 用，不迁移、不加锁、不写库 |
| **会话生命周期** | `sessions` 表的建 / 计数 / 结束：`session_create` / `session_bump_turn` / `session_end` / `session_get`。会话是**巩固提升的判定单位**（"在多会话中被反复提及"靠它计数），也是工作记忆的级联宿主 |

> **`probe.py` 与 `sqlite_backend.py` 分开是刻意的**：AL1 的 `provider.py` 只需要
> "库能不能用、schema 版本对不对"，如果那两行 import 的是 `sqlite_backend`，
> 就会把 sqlite-vec 一起拖进最外层——实现细节变成事实上的公开 API。
> 探针是**接口级能力**，实现留在层内（R9）。

### 1.2 不负责（明确排除）

| 不负责 | 归属 |
|---|---|
| 决定"什么值得记"、何时写 | AL2 核心层 |
| 多因子融合打分与排序 | AL2 核心层 `recall.py` |
| 向量化调用 | AL4 模型层 |
| 何时执行衰减/巩固 | AL2 元认知 + AL5 运行时 |
| 线程与队列 | AL5 辅助机制 |
| 记忆语义解释（为什么这条重要） | AL2 核心层 |

> **边界判据**：AL3 是"哑存储"——它不理解记忆的含义，只保证存取正确、一致、可审计。任何"判断"都不属于 AL3。

---

## 2. 对外接口

### 2.1 数据结构（`store/base.py`）

```python
from dataclasses import dataclass
from typing import Protocol, Literal, Iterator, NamedTuple

Layer  = Literal["episodic", "semantic", "procedural", "core"]
Status = Literal["active", "dormant", "forgotten"]   # D-17：取消 archive（两级 + 终态）
Kind   = Literal["memory", "entity"]


@dataclass(slots=True)
class MemoryRecord:
    """统一的记忆条目。所有层共用此结构。"""
    id: str
    layer: Layer
    type: str                                  # fact|preference|event|entity|skill|identity|soul|intent
    content: str                               # 自然语言陈述 = 事实来源
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    abstract: str | None = None                # L0 摘要
    scope: dict | None = None                  # {"type": "global|project|session", "id": ...}
    confidence: float = 0.7
    salience: float = 0.0
    strength: float = 1.0                      # 派生
    access_count: int = 0                      # 派生
    last_access_at: str | None = None          # 派生
    valid_from: str | None = None
    valid_to: str | None = None
    source_session: str | None = None
    source_turn: int | None = None
    status: Status = "active"
    superseded_by: str | None = None
    created_at: str = ""
    updated_at: str = ""
    embedding_model: str | None = None
    content_hash: str | None = None            # 内容寻址指纹（D-21）：往返幂等与去重锚点


class Hit(NamedTuple):
    """单路检索的原生结果。score 是该路原生分，不跨路可比。"""
    mem_id: str
    layer: str
    content: str                               # 交付给宿主的文本（非向量）
    score: float
    meta: dict


@dataclass(slots=True)
class AuditEvent:
    op: str            # 见下方「op 权威列举」——本字段是自由字符串，无运行时校验
    actor: str         # extractor|dedup|consolidator|decay|optimizer|user|cli|system
    target_kind: str | None = None
    target_id: str | None = None
    before: dict | None = None
    after: dict | None = None
    reason: str | None = None
    session_id: str | None = None
    ts: str = ""
    audit_id: int | None = None    # audit.id，恢复入口 restore_from_audit 认它


@dataclass(slots=True)
class WorkingChunk:
    """工作记忆组块（会话内活跃内容，**不落 memories 表**，无向量、无审计）。"""
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
    """L1 概览缓存条目（M9）。``stale=True`` = 底层已变更、待重算。

    它**必须**是 `overview_get` 的返回类型：返回纯 `str` 会丢掉 `stale`，
    调用方（AL2）就无法判断"这份概览还能不能用"。
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
```

**异常族**（`store/base.py`，§7 的落点）：

```python
StoreError                       # 基类
├── StorageBusyError             # F4 锁超时（AL5 writer 可重试）
├── DimensionMismatchError       # F7 维度不符（拒绝写入，不得静默截断）
├── StorageFatalError            # 磁盘满 / 库损坏（AL1 转工具级错误，不崩宿主）
├── SchemaVersionError           # 库版本 ≠ 代码期望
├── NotFoundError                # 目标记录不存在
└── WhitelistViolation           # 违反删除纪律（D-17 / D-22）
```

**版本常量**：`SCHEMA_VERSION = 2`（v1 = 初版全量 DDL；v2 = `mem_fts` 改 CJK 归一化普通表）。
**它住在协议层而非实现层**——AL1 的 `is_available()` / `doctor` 要做"库可用性预检"，
若常量只在 `sqlite_backend`，最外层就得 import 实现模块（连 `sqlite-vec` 一起拖进来），
正是 R9 要拦的事。实现层与只读探针（`probe.py`）都从 `base.py` 取，源头唯一。

**`op` 权威列举**（`AuditEvent.op` 是自由字符串，**没有任何校验点**）：

| 类别 | 取值 |
|---|---|
| 实际写入（9 类） | `add` / `update` / `set_status` / `touch` / `link` / `reinforce` / `forget` / `restore` / `reconcile` |
| 由上游按需写入 | `consolidate` / `summarize` / `evolve` / `reindex` |
| **已废弃（不得再写）** | `merge`（MERGE 已降级为 `add`，见 §M11 与评审 P0-6）、`dormant`（由 `set_status` 取代）、`unlink` / `invalidate` / `ignore`（从未实现） |

> **这是一处已知的契约松弛**：`op` 无枚举约束，三个"列举处"（本文件、`schema.sql` 注释、
> `base.py` docstring）历史上互不相同，靠人工维护一致性。**审计是 INV-8 可审计性的载体，
> 拼写错的 `op` 会静默通过**。收紧方案见评审 P1-20（列为 AL3 待裁决项）。

### 2.2 `MemoryBackend` 协议

```python
class MemoryBackend(Protocol):
    # ---------- 生命周期 ----------
    def open(self) -> None: ...
    def close(self) -> None: ...
    def migrate(self) -> None: ...
    """建表 / schema 版本迁移。**不含维度参数**——维度由实现构造参数决定
       （`SQLiteBackend(path, embedding_dim=...)`），因为 vec0 的维度建表后不可改。"""

    # ---------- 写（真相源）----------
    def put(self, rec: MemoryRecord, embedding: list[float] | None = None,
            *, audit: AuditEvent | None = None) -> str: ...
    def update(self, mem_id: str, patch: dict, *, audit: AuditEvent | None = None) -> None: ...
    def set_status(self, mem_id: str, status: Status, *, reason: str,
                   actor: str = "system") -> None: ...
    def hard_delete(self, mem_id: str, *, reason: str, purge_snapshot: bool = False,
                    actor: str = "user", source: str = "unknown") -> None: ...
    """物理删除。默认在 `delete_snapshots` 落内容快照（可恢复，D-22）；
       `purge_snapshot=True` 仅供合规删除——连快照一并清除（不可恢复）。
       `reason` 缺省即抛 `WhitelistViolation`；`source` 记录调用来源（cli / optimizer / system）。"""

    # ---------- 读 ----------
    def get(self, mem_id: str) -> MemoryRecord | None: ...
    def query(self, *, layer: Layer | None = None, status: Status | None = "active",
              types: list[str] | None = None, since: str | None = None,
              until: str | None = None, limit: int | None = None) -> list[MemoryRecord]: ...
    def count_by_layer(self) -> dict[str, dict[str, int]]: ...   # layer → status → 计数

    # ---------- 检索原语 ----------
    def vector_search(self, vec: list[float], *, layer: Layer | None = None,
                      top_k: int = 8) -> list[Hit]: ...
    def keyword_search(self, query: str, *, layer: Layer | None = None,
                       top_k: int = 8) -> list[Hit]: ...

    # ---------- 关联 ----------
    def link(self, a_kind: Kind, a_id: str, b_kind: Kind, b_id: str,
             rel_type: str, weight: float) -> None: ...
    def reinforce(self, a_id: str, b_id: str, delta: float,
                  *, rel_type: str = "co_activation") -> None: ...
    def neighbors(self, kind: Kind, node_id: str, *, rel_type: str | None = None,
                  min_weight: float = 0.0, limit: int = 20) -> list[tuple[str, str, float]]: ...
    def relations_of(self, mem_id: str) -> list[dict]: ...   # 某条记忆的全部关联边

    # ---------- 实体 ----------
    def entity_upsert(self, name: str, type_: str,
                      *, aliases: list[str] | None = None) -> str: ...
    def entity_find(self, text: str) -> list[EntityRecord]: ...   # 按名称与别名（无 LLM）
    def entity_get(self, entity_id: str) -> EntityRecord | None: ...

    # ---------- 审计 ----------
    def audit(self, ev: AuditEvent) -> int: ...   # **返回 audit.id**——恢复入口要它
    def audit_replay(self, *, since: str | None = None) -> Iterator[AuditEvent]: ...
    def audit_get(self, audit_id: int) -> AuditEvent | None: ...
    def audit_for(self, mem_id: str, *, limit: int = 200) -> list[AuditEvent]: ...

    # ---------- 活性 ----------
    def touch(self, mem_id: str, ts: str) -> None: ...

    # ---------- 会话 ----------
    def session_create(self, session_id: str, started_at: str) -> None: ...
    def session_bump_turn(self, session_id: str) -> int: ...
    def session_end(self, session_id: str, ended_at: str) -> None: ...
    def session_get(self, session_id: str) -> dict | None: ...

    # ---------- 工作记忆 / 意图 ----------
    def wm_put(self, session_id: str, chunk_key: str, content: str, salience: float) -> str: ...
    def wm_list(self, session_id: str, limit: int = 9) -> list[MemoryRecord]: ...
    def wm_clear(self, session_id: str) -> None: ...
    def intent_put(self, content: str, session_id: str | None = None,
                   due_at: str | None = None) -> str: ...
    def intent_list(self, status: str = "open") -> list[dict]: ...

    # ---------- 元 ----------
    def meta_get(self, key: str) -> str | None: ...
    def meta_set(self, key: str, value: str) -> None: ...

    # ---------- 分级加载（M9）----------
    def overview_get(self, scope_kind: str, scope_id: str,
                     level: str = "L1") -> OverviewRecord | None: ...
    def overview_put(self, scope_kind: str, scope_id: str, level: str,
                     content: str, *, token_count: int, model: str) -> None: ...
    def overview_invalidate(self, *, scope_kind: str | None = None,
                            scope_id: str | None = None) -> int: ...   # 置 stale，返回影响行数
    def overview_stale_list(self, *, limit: int = 50) -> list[OverviewRecord]: ...
    def overview_list(self, *, limit: int | None = None) -> list[OverviewRecord]: ...

    # ---------- 内容寻址（D-21）----------
    def find_by_hash(self, content_hash: str) -> list[MemoryRecord]: ...

    # ---------- 恢复（D-22 安全网）----------
    def restore_from_audit(self, audit_id: int) -> str: ...   # 返回恢复出的 mem_id
    def delete_snapshots(self, *, include_payload: bool = False) -> list[dict]: ...

    # ---------- 运维 ----------
    def reconcile(self, *, dry_run: bool = True) -> dict: ...      # 启动对账（对齐向量）
    def replay_derived(self) -> dict: ...                          # 从 audit 重建派生字段（INV-12）
    def rebuild_fts(self) -> int: ...                              # 从 memories 重建 FTS（INV-1）

    # ---------- 传承（P3 / INV-13）----------
    def export(self, *, fmt: str = "markdown") -> str: ...   # 状态投影（纯函数，INV-1）
    def export_archive(self, path: str, *, fmt: str = "markdown") -> None: ...  # 人类可读档案
    def export_pack(self) -> bytes: ...                                        # 含向量与关系的完整包
    def import_pack(self, data: bytes, *, merge: bool = False) -> dict: ...
    def import_archive(self, path: str) -> dict: ...                           # 从档案重建
```

> **上面这份是本协议的唯一权威副本，须与 `store/base.py` 逐字对齐**（评审 P1-21）。
> 历史上它比实现少 19 个方法、8 处签名不符（`audit` 的返回类型、`overview_get` 的返回类型、
> `put` / `update` 的 `audit` 可选性……），而 AL2 只按这份文档对契约——**文档少一个方法，
> 上层就"合法地"不知道它存在**。`reconcile` 在实现层还多一个 `embed_fn` 参数
> （AL5 `lifecycle._safe_reconcile` 需要它来补算向量），协议签名暂不收录，见评审 P2-16。

> **接口稳定性契约**：`MemoryBackend` 一经发布即视为稳定接口。AL2 只依赖它，任何实现变更（含新增后端）不得修改既有签名语义。

---

## 3. 内部结构

```
store/
├── __init__.py          # 包级再导出（零逻辑，R6）
├── base.py              # 数据结构 + 异常族 + SCHEMA_VERSION + MemoryBackend 协议（无实现）
├── schema.sql           # 全部 DDL 与触发器（真相源定义）
├── text.py              # 文本归一化 / FTS 查询构造 / content_hash 转发
├── ids.py               # ULID 与 {abbr}_{ulid} ID 生成
├── sqlite_backend.py    # SQLiteBackend（默认实现）
├── archive.py           # export/import 的渲染与解析（markdown 档案 / JSON 包）
├── reindex.py           # 重嵌入工具：换 embedding 模型后全库重算
└── probe.py             # 只读探针：schema 版本 / 库可用性（给 AL1，不迁移、不加锁、不写库）
```

| 文件 | 职责 | 关键约束 |
|---|---|---|
| `base.py` | 协议、数据类、异常族、版本常量 | 零 I/O，零具体依赖 |
| `schema.sql` | DDL 单一来源 | 版本号写入 `meta.schema_version`；PRAGMA **不在**其中（连接级，见文件头） |
| `text.py` | 归一化与内容寻址 | 纯函数；`content_hash_of` 实现在包根 `common.py`，此处仅转发 |
| `ids.py` | ID 生成 | ULID 单调；前缀 `epi\|sem\|pro\|cor` / `wm` / `int` / `ent` / `ov` |
| `sqlite_backend.py` | 实现全部协议方法 | 一条记忆的写入必须单事务 |
| `archive.py` | 档案渲染 / 包往返 | markdown 产物必须"无器灵代码也可读"（INV-13） |
| `reindex.py` | 批量重嵌入 | 必须支持断点续跑（进度写 `meta`） |
| `probe.py` | 只读探测 | **不迁移、不加锁、不写库**；AL1 的 `is_available()` / `doctor` 只走它 |

> **`probe.py` 必须与 `sqlite_backend.py` 分开**（原文已在 §1.1 说明）：AL1 只需"库能不能用、
> 版本对不对"，若那两行 import 的是 `sqlite_backend`，就会把 `sqlite-vec` 一起拖进最外层——
> **实现细节变成事实上的公开 API**（R9）。

---

## 4. 依赖

| 方向 | 内容 |
|---|---|
| **依赖** | Python 标准库 `sqlite3`；`sqlite-vec` 扩展（MIT/Apache-2.0）；`ulid` 生成（可用标准库自实现） |
| **禁止依赖** | `core/*`、`extract/*`、`model/*`（R2）；`httpx`（AL3 不发起网络请求） |
| **被依赖** | AL2 核心层（仅协议）、AL5 辅助机制、AL1 适配层（间接） |

---

## 5. 关键设计

### M1 单一真相源 + 按需投影（INV-1）

- **真相源 = 结构化表**；文本是 `export()` 的渲染结果
- `export()` 是**纯查询 + 渲染**，无副作用、同输入同输出（"档案投影"——`export_archive` 的内核）
- **不存在**"文本 ↔ 结构化"的双向同步路径
- **归属澄清**：`status` / `layers` **不是 AL3 的方法**，而是 AL1 的 CLI 子命令 + AL5 `observability/status.py` 的渲染（INV-1「文本是投影」）。AL3 只出**结构化计数**（`count_by_layer`），不出面向人的排版

> 取舍：放弃"文本直改生效"，换取单一真相源的简洁。这是 ADR-002 §7.6 的硬约束。

### M2 单库设计（D-01）

活性字段（`strength` / `access_count` / `last_access_at`）收敛为 `memories` 的**派生列**，不另建库。

- **理由**：写记忆与更新活性可同事务提交；消除跨库对账
- **代价**：活性字段与应用数据同生共死——但由 INV-12 的 `replay` 兜底

### M3 记忆 ID 方案（D-08）

```
{abbr}_{ulid}       abbr ∈ epi | sem | pro | cor
```

- ULID 保证**时间有序**，利于范围扫描与调试
- 前缀让日志与审计可读（一眼看出属于哪层）
- 工作记忆 `wm_<ulid>`、意图 `int_<ulid>`、实体 `ent_<ulid>`

### M4 完整 DDL

```sql
-- 注：PRAGMA 均为**连接级**设置，不属于 schema，实际由 `sqlite_backend._configure_connection()`
-- 在每条连接上施加：`journal_mode=WAL` / `synchronous=NORMAL` / `foreign_keys=ON` / `busy_timeout=5000`。
-- 此处列出只为交代取值来源，`schema.sql` 内**不含** PRAGMA。

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
-- 约定键：schema_version / spirit_id / spirit_name
--        embedding_model / embedding_dim / vec_capability / created_at

CREATE TABLE IF NOT EXISTS sessions (
  id         TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  ended_at   TEXT,
  status     TEXT NOT NULL DEFAULT 'active',
  turn_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status, started_at DESC);

CREATE TABLE IF NOT EXISTS working_memory (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  chunk_key    TEXT NOT NULL,
  content      TEXT NOT NULL,
  salience     REAL NOT NULL DEFAULT 0.0,
  act_count    INTEGER NOT NULL DEFAULT 1,
  created_at   TEXT NOT NULL,
  last_touched TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory(session_id, last_touched DESC);

CREATE TABLE IF NOT EXISTS intents (
  id          TEXT PRIMARY KEY,
  session_id  TEXT,
  content     TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'open',
  due_at      TEXT,
  created_at  TEXT NOT NULL,
  resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_intents_status ON intents(status, due_at);

CREATE TABLE IF NOT EXISTS memories (
  id              TEXT PRIMARY KEY,
  layer           TEXT NOT NULL,
  type            TEXT NOT NULL,
  subject         TEXT,
  predicate       TEXT,
  object          TEXT,
  content         TEXT NOT NULL,
  abstract        TEXT,
  scope           TEXT,
  confidence      REAL NOT NULL DEFAULT 0.7,
  salience        REAL NOT NULL DEFAULT 0.0,
  strength        REAL NOT NULL DEFAULT 1.0,
  access_count    INTEGER NOT NULL DEFAULT 0,
  last_access_at  TEXT,
  valid_from      TEXT,
  valid_to        TEXT,
  source_session  TEXT,
  source_turn     INTEGER,
  status          TEXT NOT NULL DEFAULT 'active',
  superseded_by   TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  embedding_model TEXT,
  content_hash    TEXT,                     -- D-21：内容寻址指纹
  FOREIGN KEY (superseded_by) REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS idx_mem_layer_status ON memories(layer, status);
CREATE INDEX IF NOT EXISTS idx_mem_subject      ON memories(subject);
CREATE INDEX IF NOT EXISTS idx_mem_updated      ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_mem_access       ON memories(last_access_at);
CREATE INDEX IF NOT EXISTS idx_mem_valid        ON memories(valid_from, valid_to);
-- D-21：非唯一索引——允许相同内容在不同情境下共存，但导入时按指纹幂等判定
CREATE INDEX IF NOT EXISTS idx_mem_chash        ON memories(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(
  content, abstract, subject, object,
  tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
  mem_id    TEXT PRIMARY KEY,
  embedding float[__EMBEDDING_DIM__]
);

CREATE TABLE IF NOT EXISTS entities (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  type       TEXT,
  aliases    TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ent_name_type ON entities(name, type);

CREATE TABLE IF NOT EXISTS relations (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  src_kind   TEXT NOT NULL,
  src_id     TEXT NOT NULL,
  dst_kind   TEXT NOT NULL,
  dst_id     TEXT NOT NULL,
  rel_type   TEXT NOT NULL,
  weight     REAL NOT NULL DEFAULT 0.0,
  co_count   INTEGER NOT NULL DEFAULT 1,
  valid_from TEXT,
  valid_to   TEXT,
  last_co_at TEXT,
  UNIQUE (src_kind, src_id, dst_kind, dst_id, rel_type)
);
CREATE INDEX IF NOT EXISTS idx_rel_src ON relations(src_kind, src_id, weight DESC);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON relations(dst_kind, dst_id, weight DESC);

-- ============ L1 概览缓存（分级加载 · M9）============
-- 注意：这是**缓存**不是真相源（INV-1）——可从底层记忆重算
CREATE TABLE IF NOT EXISTS overviews (
  id           TEXT PRIMARY KEY,
  scope_kind   TEXT NOT NULL,               -- entity | topic | layer
  scope_id     TEXT NOT NULL,
  level        TEXT NOT NULL DEFAULT 'L1',  -- 预留多级
  content      TEXT NOT NULL,
  token_count  INTEGER,
  generated_at TEXT NOT NULL,
  stale        INTEGER NOT NULL DEFAULT 0,  -- 底层变更后置 1，下次读取重算
  model        TEXT,
  UNIQUE (scope_kind, scope_id, level)
);
CREATE INDEX IF NOT EXISTS idx_ov_stale ON overviews(stale, generated_at);

CREATE TABLE IF NOT EXISTS audit (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  ts          TEXT NOT NULL,
  op          TEXT NOT NULL,
  actor       TEXT NOT NULL,
  target_kind TEXT,
  target_id   TEXT,
  before      TEXT,
  after       TEXT,
  reason      TEXT,
  session_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts     ON audit(ts);
CREATE INDEX IF NOT EXISTS idx_audit_target ON audit(target_kind, target_id);

-- ============ 删除快照（D-22 安全网）============
CREATE TABLE IF NOT EXISTS delete_snapshots (
  audit_id   INTEGER PRIMARY KEY,           -- 对应 audit.id
  mem_id     TEXT NOT NULL,
  payload    TEXT NOT NULL,                 -- JSON：MemoryRecord + 关联边
  created_at TEXT NOT NULL
);
```

> **为什么 `delete_snapshots` 必须单独一张表，而不是塞进 `audit.before`**：
> 合规删除要求"数据真的不存在"（须清除内容），而 `audit` 是 append-only（INV-8，触发器拒删）。
> 把**内容**（可清除）与**账本**（永不删）分成两张表，两个约束才不打架：
> `audit` = 谁 / 何时 / 因何 / 对哪条（永不删）；`delete_snapshots` = 恢复用的内容快照（可被合规清除）。
> **`hard_delete` 默认写的是这张表，不是 `audit.before`**——`audit.before` 只记元数据级的差异。

> **`mem_fts` 为什么是普通 FTS5 表而非外部内容表**：外部内容表要求索引内容与 `memories`
> 逐字一致，而器灵需要对中文做逐字归一化（`store/text.py`）。改为普通表后由触发器写入
> **归一化后**的文本，所以删除必须用 `DELETE FROM mem_fts`，**不能**用外部内容表的
> `INSERT INTO mem_fts(mem_fts, ...) VALUES('delete', ...)` 语法。
> 它仍是**派生索引**：`rebuild_fts()` 可从 `memories` 全量重建（INV-1）。

**FTS5 同步触发器**（普通 FTS5 表模式，必须成对实现；`spirit_norm()` 是连接级注册的 Python 函数）：

```sql
CREATE TRIGGER IF NOT EXISTS trg_mem_ai AFTER INSERT ON memories BEGIN
  INSERT INTO mem_fts(rowid, content, abstract, subject, object)
  VALUES (new.rowid, spirit_norm(new.content), spirit_norm(new.abstract),
          spirit_norm(new.subject), spirit_norm(new.object));
END;
CREATE TRIGGER IF NOT EXISTS trg_mem_ad AFTER DELETE ON memories BEGIN
  DELETE FROM mem_fts WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS trg_mem_au AFTER UPDATE ON memories BEGIN
  DELETE FROM mem_fts WHERE rowid = old.rowid;
  INSERT INTO mem_fts(rowid, content, abstract, subject, object)
  VALUES (new.rowid, spirit_norm(new.content), spirit_norm(new.abstract),
          spirit_norm(new.subject), spirit_norm(new.object));
END;
```

**append-only 保护**（INV-8）：

```sql
CREATE TRIGGER IF NOT EXISTS trg_audit_no_update BEFORE UPDATE ON audit BEGIN
  SELECT RAISE(ABORT, 'audit is append-only');
END;
CREATE TRIGGER IF NOT EXISTS trg_audit_no_delete BEFORE DELETE ON audit BEGIN
  SELECT RAISE(ABORT, 'audit is append-only');
END;
```

### M5 事务边界

| 操作 | 事务范围 |
|---|---|
| 写入一条记忆 | `memories` + `vec_memories` + `mem_fts`（触发器） + `audit` **同一事务** |
| 更新状态 | `memories.status` + `audit` 同事务 |
| 加固关联 | `relations` upsert + `audit` 同事务 |
| 检索 | **只读**，不开事务，不加写锁 |

### M6 层与状态正交

- **层间提升** = 新建一条记录 + 一条 `relations(rel_type='derived_from')` 保留来源链
- **分级降级** = 只改 `status`（`active` ↔ `dormant`），**不跨层搬移**，不删 `layer`
- **物理删除**（`forgotten`）只由白名单触发（D-17），且必须与 `audit(op='forget')` **同事务**（INV-7 / INV-8）

> 为什么：层的语义是"这是什么类型的记忆"，状态是"它还活跃吗"。混在一起会让检索谓词变得复杂且易错。

### M7 vec0 的 TEXT 主键（已裁决：**无退化路径，明确失败**）

`sqlite-vec` ≥ 0.1.9 支持 `mem_id TEXT PRIMARY KEY`。`migrate()` 时探测一次，结果写入 `meta.vec_capability`：

```
支持 text pk  → 写入 meta.vec_capability='text_pk'，正常建表
不支持        → 写 'rowid' 后**抛错终止**，要求升级扩展
```

**明确放弃"退回 INTEGER rowid + 映射表"的退化方案**（原 M7 的第二分支，已删除）。理由来自实现（`_require_text_pk` 的 docstring，**这是原设计漏掉的那一条**）：

> 该分支在 sqlite-vec 0.1.9（实测支持 TEXT 主键）下**无法被测试**，
> 而它位于**所有向量读写的关键路径**上——**引入不可测的分支比明确失败更危险**。

补三条推论（说明"为什么不能靠'以后再说'绕过"）：

- 映射表会把"ID 即主键"变成"ID → rowid → 向量"的双层间接，所有涉及 `vec_memories` 的代码都要分叉；
- 它的收益只是兼容一个**器灵本就声明不支持的旧版本**（扩展随包分发，没有"用不了新版"的现实场景）；
- 而"静默退化"会让性能与不变量**都悄悄变差且无从察觉**（P1-3 / P2-14 同型的病）。

> 因此 §7 里"向量索引缺失 → 按 `mem_id` 差集补算"的对账逻辑可以直接按主键对齐，
> 不必考虑 rowid 映射；探测结果仍缓存进 `meta` 以免每次启动重复探测。

### M8 与 Graphiti 三表模型的对应（评审 P1-2 补回）

RES-001 §4.3 提出借鉴 Graphiti 的 `nodes` / `edges` / `facts` 三表。器灵的落地是 `entities` / `relations` + `memories`——**名字不同，需要说清对应关系与归并理由**：

| Graphiti | 器灵落点 | 说明 |
|---|---|---|
| `nodes`（实体节点） | `entities` | 一一对应 |
| `edges`（关系边，带 `valid_from/valid_to`） | `relations` | 一一对应，同样带双时态字段 |
| `facts`（事实三元组） | **`memories`（`layer='semantic'`）** | **归并** |

**为什么把 `facts` 并进 `memories`**：

- `facts` 与 `memories` 本质是同一个东西——"带 `subject` / `predicate` / `object` 的结构化陈述"
- 若拆成两表，会出现**两套事实真相源**，直接违反 **INV-1**（单一真相源）
- 归并后职责清晰：**事实只存 `memories` 一份**；`entities` 表达"有哪些实体"；`relations` 表达"它们怎么关联"

> 一句话：**`memories` 是真相源，`entities` + `relations` 是它的图结构投影。不存在两套事实。**

### M9 分级加载的存储支持（P1 · 首要机制）

三级内容各有归属，关键是**分清哪些是真相源、哪些是缓存**：

| 级 | 存储位置 | 性质 |
|---|---|---|
| **L0** | `memories.abstract` | **真相源的一部分**（随记忆写入，异步生成） |
| **L1** | `overviews` 表 | **纯缓存**——可从底层记忆重算，丢失无害 |
| **L2** | `memories.content` + `relations` + `audit` | **真相源**，实时组装，不额外存储 |

**失效传播**：

```
任何写操作影响某条记忆
  └─▶ overview_invalidate(scope_kind, scope_id)
        └─▶ 相关 overviews.stale = 1
              └─▶ 下次 overview_get() 发现 stale → 触发重算（由 AL2 调度）
```

**硬约束**：

- `overviews` **可以随时清空重建**，不得有任何真相源依赖它（这是"缓存"的定义）
- 当 `abstract` 为空时，`overview_get` 的上游（AL2）应能退化到 `content` 截断
- 失效传播是**标记式**（O(1) 置位），重算是**惰性**的（读取时触发）——避免写路径变重

### M10 可迁移导出格式（P3 · INV-13）

> 直接针对"换个工具记忆归零"这一痛点。**一个只存在于 SQLite 二进制里的记忆，等于没有记忆资产。**

**两种产物，用途不同**：

| 产物 | 格式 | 用途 | 依赖 |
|---|---|---|---|
| **人类可读档案** | Markdown（默认）/ JSONL | 长期存档、换工具、人工审阅、迁移到别的系统 | **无**——文本编辑器即可读 |
| **完整包** | 自有二进制包（含向量 + 关系） | 器灵快速接续 / 快速恢复 | 器灵运行时 |

**档案必须自描述**，至少包含：

```
# Artifact Spirit Memory Archive
schema_version / spirit_name / spirit_id / exported_at / memory_count
---
## <层> · <type>
- content: …（完整原文）
- abstract: …（L0）
- created_at / valid_from / valid_to
- confidence / salience / status
- source_session
- relations: […]
```

**红线**：

| # | 约束 |
|---|---|
| 1 | 档案**不依赖** `sqlite-vec` / 器灵代码，任何文本编辑器可读 |
| 2 | 档案**自描述版本**，未来可被更高版本器灵或第三方工具解析 |
| 3 | 导入时**核心信息无损**（内容 / 层 / 时间 / 置信度 / 关联）；派生字段（`strength` 等）可重算 |
| 4 | 导出是**纯函数投影**：同库状态导出结果一致（INV-1） |
| 5 | 向量不必写入可读档案（体积大且不可读）——需要时由 `reindex` 重算 |

> **与 ADR-002 §7.6 的关系**：`export()` 是"当前状态的投影"（给人看），`export_archive()` 是"可迁移的资产快照"（给未来用）。二者都是纯函数，但用途不同、格式可以不同。

### M11 内容寻址与可恢复删除（D-21 / D-22）

#### 一、`content_hash`：往返幂等的锚点（D-21）

**问题**：主键 `{abbr}_{ulid}` 是**随机唯一 ID**。导出再导入会生成**新 ID** → 往返不幂等、去重失效、关联断裂。这直接削弱 G2（自由迁移）。

**做法**：新增 `content_hash` 作为**内容寻址指纹**。

```python
# 实现在包根 common.py（不是 AL3）——理由见该函数的 docstring：
# 它是 AL2 的提取契约与 AL3 的写入校验**同时**需要的领域常量，
# 住在 store/ 会让 AL2 为了算指纹而依赖存储层（R2 边界）。
def content_hash_of(*, content: str, subject=None, predicate=None,
                    object_=None, scope=None) -> str:
    if subject and predicate and object_:      # 注意：要求**三者齐备**
        payload = {                            # 任一字段单独归一化后再序列化
            "subject": normalize_content(subject),
            "predicate": normalize_content(predicate),
            "object": normalize_content(object_),
            "scope": scope or {},
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    else:                                      # 三元组缺失（episodic 事件）→ 退化
        raw = normalize_content(content)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"sha256:{digest}"
```

| 细节 | 取值 | 为什么 |
|---|---|---|
| 三元组判定 | **三者齐备**才走三元组路径 | 半截三元组（只有 subject）序列化出来是"半条事实"，会把不同事实撞成同一指纹 |
| 归一化位置 | 每个字段**各自**归一化 | 否则"标点/大小写差异"会被 `json.dumps` 固化进指纹 |
| 摘要长度 | `sha256` **截断 32 hex**（128 bit） | 指纹只做等值判定、不进审计证据链，128 bit 的碰撞概率已远低于误判成本；短一半的键让索引更小 |
| 前缀 | `sha256:` | 自描述算法，未来换算法可区分 |

| 决策 | 取值 | 理由 |
|---|---|---|
| 索引类型 | **非唯一索引** | 允许相同文本在不同情境下共存；幂等由**导入逻辑**保证，不由 DB 约束强压 |
| 用途 1 | **导入幂等** | `import` 时 `find_by_hash()` 命中则跳过/合并，而非新建 |
| 用途 2 | **去重锚点** | 作为提取去重的候选召回信号，替代/补充向量相似度 |
| 是否入档案 | ✅ 写入档案 | 使往返可判定；且档案自描述 |

**往返幂等定义**：`export → import → export` 的第二次导出与第一次**等价**（记忆集合、关联、层级、时间一致）；`import` 执行两次**不产生重复**。

#### 二、删除可恢复（D-22 安全网）

**核心观念转变**：删除的安全性**不靠"禁止删除"，而靠"可恢复"**。

| 环节 | 实现 |
|---|---|
| 删除前 | 写 `audit(op='forget')` **只记账本**（谁 / 何时 / 因何 / 对哪条）；**内容快照写进 `delete_snapshots` 表**（`audit_id` 为键，`payload` = `MemoryRecord` + 关联边）。两者与 `hard_delete` **同事务** |
| 恢复 | `restore_from_audit(audit_id)` 按 `audit_id` 读 `delete_snapshots.payload` 重建记录**与关联边**；**恢复的 ID 沿用原 ID**（避免关联断裂）；若原 ID 已被占用则新建并补 `superseded_by` 说明 |
| 恢复留痕 | 写 `audit(op='restore', actor='user', reason=...)` |
| 快照清点 | `delete_snapshots()` 列出全部快照（`include_payload=True` 才带内容），供合规审计与 CLI 查看 |

> **为什么不放 `audit.before`**：`audit` 是 append-only（INV-8 触发器拒删），而合规删除要求
> "数据真的不存在"。把**内容**与**账本**分成两张表，两个约束才不打架（见 §M4 的说明）。
> `audit.before/after` 仍保留，用来记**元数据级**的差异（如 `set_status` 的 `{status: active} → {dormant}`）。

**唯一例外 · 合规删除**：

| 场景 | `purge_snapshot` | 可恢复性 |
|---|---|---|
| 普通删除（优化删除 / 用户要求） | `False`（默认） | ✅ 可恢复 |
| **合规删除**（隐私 / 被遗忘权） | `True` | ❌ **不可恢复**——因为合规要求"数据真的不存在" |

> **合规删除仍需留痕**：记录"何时、因何合规要求执行了不可恢复删除"，但**不留内容快照**（否则删除不彻底）。

---

## 6. 编码注意事项

| # | 注意点 | 说明 |
|---|---|---|
| C1 | **连接与线程的关系必须显式交代** | 实现取的是**第三条路**：`check_same_thread=False` + `threadsafety == 3`（串行化模式）下**共享连接**，并区分只读 / 读写两条连接。**不要**按"每线程一条连接"实现——那会与 AL5 的单写者线程模型冲突（R8：并发原语只在 `runtime/`） |
| C2 | **`row_factory = sqlite3.Row`** | 避免下标取值，全部按列名访问 |
| C3 | **时间统一 ISO8601 带时区** | 所有时间字段用同一格式，禁止混用时间戳与字符串 |
| C4 | **`scope` / `aliases` / `before` / `after` 为 JSON 字符串** | 读写时显式 `json.dumps/loads`，不要依赖隐式转换 |
| C5 | **FTS5 删除走 `DELETE FROM mem_fts`** | 因为 `mem_fts` 是**普通 FTS5 表**（非外部内容表，见 M4）。外部内容表的 `'delete'` 语法在此**不适用**——两者不可混用 |
| C6 | **`vec_memories` 写入前必须校验维度** | 维度不符应抛错（F7），不要静默截断 |
| C7 | **`status` 变更必须同时写 `audit`** | 由 `set_status` / `hard_delete` 内部**自动**记账（调用方无需、也不该手写这两条的审计）。**注意**：`put` / `update` 的 `audit` 参数是**可选**的——强制必填会把"内部自动记账"与"调用方补记"两种来源混为一谈 |
| C8 | **`hard_delete` 是危险操作** | 必须显式传 `reason`；建议在 CLI 层默认 dry-run |
| C9 | **`export()` 必须是纯函数** | 不得写库、不得生成随机/时间相关输出（INV-1 可测试性） |
| C10 | **迁移必须幂等** | `migrate()` 可重复执行不报错 |
| C11 | **禁止在 AL3 内拼接用户输入为 SQL** | 一律参数化查询 |

---

## 7. 错误处理与降级

| 故障 | 检测 | AL3 行为 |
|---|---|---|
| **F4** SQLite 锁超时 | `busy_timeout` 到期 | 上抛 `StorageBusyError`，由 AL5 writer 重试（最多 3 次） |
| **F7** 向量维度不匹配 | 建表 / 写入时校验 | 抛 `DimensionMismatchError`，拒绝启动或拒绝写入 |
| **vec0 不支持 TEXT 主键** | `migrate()` 探测一次，结果缓存进 `meta.vec_capability` | 抛 `SchemaVersionError` 并提示升级 `sqlite-vec`——**明确失败，无退化路径**（M7） |
| 向量索引缺失 | **启动对账**（AL5 `lifecycle._safe_reconcile` 调 `reconcile(dry_run=False)`） | 按 `mem_id` 差集补算（需要 `embed_fn` 来重算向量） |
| 孤儿向量 | 启动对账（同上） | 删除 `vec_memories` 中无对应 `memories` 的行 |
| `embedding_model` 变更 | 启动校验（`assert_embedding_model`） | 抛错并提示执行 `reindex`（INV-2） |
| 派生字段可疑 | 手动（`replay_derived`）/ 启动检测 | 从 `audit` 重建 `strength` / `access_count` 等（INV-12） |
| 磁盘满 / DB 损坏 | SQLite 异常 | 抛 `StorageFatalError`，由 AL1 转为工具级错误，**不崩溃宿主** |

> **对账默认是 `dry_run=True`**（只报告不修改）——启动时由 AL5 显式传 `dry_run=False` 才真正修复。
> 这是刻意的：**"只读探针"与"修复动作"不能共用一个默认值**，否则任何一次查看都会写库。

---

## 8. 独立验收标准

**测试环境**：临时 SQLite 文件 + 已加载的 `sqlite-vec` 扩展。**不需要**任何其他层。

- [ ] `migrate()` 可在空库上建全部表；重复执行幂等
- [ ] `PRAGMA journal_mode` 返回 `wal`
- [ ] `put()` 返回符合 `{abbr}_{ulid}` 格式的 ID，且 ULID 单调递增
- [ ] `put()` 后 `get()` 返回的 `MemoryRecord` 字段与写入一致
- [ ] **单事务断言**：注入 audit 写入失败 → 记忆主体不落库（无半写状态）
- [ ] `vector_search()` 对已知向量返回正确 top-k 顺序（用可控的合成向量）
- [ ] `keyword_search()` 能命中中文与英文关键词
- [ ] 更新 `memories` 后 `mem_fts` 同步生效（三个触发器；且**索引到的是 `spirit_norm()` 归一化后的文本**——用"全角/半角、大小写不同的同一关键词"检索命中来断言）
- [ ] 删除 `memories` 后 `mem_fts` 无残留（普通 FTS5 表：`DELETE FROM mem_fts WHERE rowid=?`）
- [ ] `rebuild_fts()` 从 `memories` 全量重建后，检索结果与触发器维护的结果**一致**（证明 FTS 确是派生索引，INV-1）
- [ ] `audit` 的 UPDATE / DELETE 被触发器拒绝（INV-8）
- [ ] `audit_replay()` 返回事件序列与写入顺序一致
- [ ] `replay` 重建的派生字段与原始值一致（INV-12）
- [ ] 维度不符的写入被拒绝并抛 `DimensionMismatchError`（F7）；**且库中原有数据不被改动**（拒绝写入 ≠ 半写）
- [ ] `export()` 连续调用两次输出完全一致（纯函数，INV-1）；**且输出中不含时间戳等非确定性字段**（"两次一致"不能靠恰好在同一秒内调用通过）
- [ ] `export_archive()` 的产物 = `export()` 的投影 + 自描述头部（二者必须同源，不允许各写一套渲染）
- [ ] `set_status()` 后 `query(status='active')` 不再返回该条，但 `get()` 仍可取到
- [ ] `neighbors()` 按 weight 降序返回
- [ ] `export_pack()` → `import_pack()` 往返后数据等价
- [ ] **分级加载**：`overview_put` / `overview_get` 往返正确，`level` 维度生效
- [ ] `overview_invalidate` 返回受影响行数，且 `overview_get` 能读到 `stale=1`
- [ ] **清空 `overviews` 表后一切功能不受影响**（缓存性质断言，INV-1）
- [ ] `export_archive()` 产物**用纯文本方式可读**，含全部自描述头部（INV-13）
- [ ] `export_archive()` 连续两次输出一致（纯函数，INV-1）
- [ ] `export_archive()` → `import_archive()` 往返后，核心信息无损（INV-13）
- [ ] 档案中**不含向量**，且导入后 `reindex` 能补回——补回的判据是**"该条能被 `vector_search` 命中"**，不是"reindex 跑完没报错"
- [ ] **`Status` 的取值域恰为 `{active, dormant, forgotten}`**（D-17 两级断言）。**白名单式断言**：`set(typing.get_args(Status)) == {...}`——`"archived" not in str(Status)` 这类断言在取值域被改成任意其它名字时**依然通过**，不能作为 D-17 的证据
- [ ] `set_status(mem_id,'dormant')` 后 `query(status='active')` 不再返回该条，但 `get()` 仍可取到
- [ ] `hard_delete()` **不带 `reason` 时拒绝执行**（D-17 白名单纪律）
- [ ] `hard_delete()` 与 `audit(op='forget')` + `delete_snapshots` **同事务**：注入 audit 失败 → 记忆未被删除（⚠ 当前**无用例**，见评审 P2-17）
- [ ] `hard_delete()` 后 `audit` 中关于该条的历史**仍然存在**（审计不级联删除，INV-8）
- [ ] **`content_hash` 稳定**（D-21）：同一记录重复计算得到同一指纹
- [ ] `find_by_hash()` 命中相同指纹的记录（非唯一索引，允许返回多条）
- [ ] **往返幂等**：`export_pack` → `import_pack` → 再 `import_pack` **不产生重复记录**
- [ ] **恢复**（D-22）：`hard_delete` → `restore_from_audit` 后记录与删除前**等价**（含 ID 与关联）
- [ ] 恢复本身产生 `audit(op='restore')`
- [ ] `hard_delete(purge_snapshot=True)` 后**该条内容不可恢复**：`delete_snapshots` 中无该 `audit_id` 的行，且内容**不出现在 `audit` 的任何字段里**（否则合规删除是假的）
- [ ] **迁移**：`meta.schema_version` 从 1 升到当前版本后，库内容不被破坏（在 v1 库上跑 `migrate()` → 数据仍在、`mem_fts` 已按新模式重建）
- [ ] **协议对齐**：`MemoryBackend` Protocol 的方法集合与本文档 §2.2 **逐字一致**（白名单式 `inspect` 断言，两个方向都比）
- [ ] 代码扫描：AL3 无 `httpx` / `core` 导入

---

## 9. 集成验收关注点

| 关注点 | 验证方式 |
|---|---|
| 与 AL2 的协议契合 | AL2 用真实 `SQLiteBackend` 跑五类记忆写入与召回 |
| 与 AL4 的向量契约 | AL4 按**配置维度**产出的向量可直接落 `vec_memories`（默认 2560，见 `KNOWN_EMBEDDING_DIMS`）；维度不符必须被拒绝而非截断 |
| 与 AL5 的线程安全 | 单写者线程下并发写入无 `SQLITE_BUSY` |
| 与 AL1 的路径约定 | DB 文件确实落在 `{hermes_home}/spirit/`（INV-9） |
| 崩溃恢复 | 写入过程中 kill 进程 → 重启后数据一致、无半写 |

---

## 10. 待决策 / 未决项

| # | 事项 | 现状 | 影响 |
|---|---|---|---|
| 1 | `sqlite-vec` TEXT 主键支持 | **✅ 已裁决**：强制要求 TEXT 主键（≥0.1.9），不支持即**明确失败**；无 INTEGER rowid 退化路径（M7 已重写） | 建表语句 |
| 2 | 是否启用 binary quantization | **✅ 已决定**：暂不启用（省实现复杂度） | 存储占用与检索速度 |
| 3 | `export()` 投影的输出格式 | **✅ 已定**：markdown 纯函数，实现于 `store/archive.py`；**`status` / `layers` 的渲染不在此处**（归 AL5 `observability/`，INV-1） | 可观测性 |
| 4 | `schema_version` 迁移框架 | **✅ 已实现**：手写迁移函数表 `_MIGRATIONS = {1: _apply_v1, 2: _apply_v2}`，版本号门控、幂等；v1 = 全量 DDL，v2 = `mem_fts` 改 CJK 归一化普通表 | 长期可维护性 |
| 5 | **档案格式**：Markdown vs JSONL 为默认 | **✅ 已定（部分）**：Markdown 为默认且**唯一已实现**格式；`fmt` 参数已预留，JSONL 未实现 | P3 可迁移性 |
| 6 | **`overviews` 是否随档案导出** | **✅ 已裁决**：**不导出**（是缓存，可重算）；导入也不触碰 `overviews` | 档案体积 |
| 7 | **`AuditEvent.op` 无枚举校验** | **⏳ 待裁决**（评审 P1-20）：`op` 是自由字符串，三处"列举处"（本文档 / `schema.sql` 注释 / `base.py` docstring）历史上互不相同，拼错的 `op` 静默通过。建议提为 `Literal` 或用模块级白名单常量 + 写入前校验 | INV-8 可审计性 |
| 8 | **`reconcile` 的 `embed_fn` 参数** | **⏳ 待裁决**（评审 P2-16）：实现层需要它补算向量，AL5 调用点也传了，但协议签名没有——AL2 若通过 `MemoryBackend` 调用会类型不符。建议收进协议（代价：协议层要描述一个可调用签名） | 协议完整性 |
