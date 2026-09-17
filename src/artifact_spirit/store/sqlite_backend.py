"""SQLite + sqlite-vec 后端（默认实现）。

覆盖 AL3 的全部任务（T-AL3-01 ~ 22）。设计要点：

- PRAGMA 是**连接级**设置，由 :meth:`SQLiteBackend._configure_connection` 在每个连接上施加
- 迁移按版本号门控，**幂等**；已应用过的版本不会重复执行
- ``vec0`` 的能力探测结果缓存到 ``meta.vec_capability``，避免每次启动重复探测
- ``spirit_norm()`` 是连接级注册的 Python 函数，供 FTS 触发器调用（见 ``store/text.py``）
- 写操作全部走 :meth:`_tx`（``BEGIN IMMEDIATE``），保证「记忆 + 向量 + FTS + 审计」原子提交
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..common import now_iso
from . import ids as _ids
from .base import (
    AUDIT_OPS,
    SCHEMA_VERSION,
    AuditEvent,
    DimensionMismatchError,
    EntityRecord,
    Hit,
    Layer,
    MemoryRecord,
    NotFoundError,
    OverviewRecord,
    SchemaVersionError,
    Status,
    StorageBusyError,
    StorageFatalError,
    StoreError,
    WhitelistViolation,
    WorkingChunk,
)
from .text import (
    FTS_COLUMN_WEIGHTS,
    content_hash_of,
    fts_match_query,
    normalize_for_fts,
)

__all__ = ["META_KEYS", "SCHEMA_VERSION", "SQLiteBackend"]


def _strictly_newer(ts: str, floor: str | None) -> str:
    """让 ``last_touched`` 在**同一会话内严格递增**。

    Windows 上 ``datetime.now()`` 的时钟粒度约 1ms（远粗于微秒），同一毫秒内的多次
    触碰会拿到**完全相同**的 ``last_touched``，排序随即退化为 ``rowid`` 顺序——而工作
    记忆的排序**完全依赖"最近触碰"**（见 :meth:`SQLiteBackend.wm_put`）。

    做法：只在"新时间戳没有真正变新"时把 floor 推后 1 微秒；时钟真实前进时不受影响，
    因此不改变时间戳的语义，只消除同刻并列。
    """
    if floor is None or ts > floor:
        return ts
    return (datetime.fromisoformat(floor) + timedelta(microseconds=1)).isoformat()


SCHEMA_FILE = Path(__file__).with_name("schema.sql")

_EMBEDDING_DIM_TOKEN = "__EMBEDDING_DIM__"
_DEFAULT_EMBEDDING_DIM = 2560

# meta 约定键
META_SCHEMA_VERSION = "schema_version"
META_SPIRIT_ID = "spirit_id"
META_SPIRIT_NAME = "spirit_name"
META_EMBEDDING_MODEL = "embedding_model"
META_EMBEDDING_DIM = "embedding_dim"
META_VEC_CAPABILITY = "vec_capability"
META_CREATED_AT = "created_at"
META_LAST_RECONCILE = "last_reconcile_at"

META_KEYS = (
    META_SCHEMA_VERSION,
    META_SPIRIT_ID,
    META_SPIRIT_NAME,
    META_EMBEDDING_MODEL,
    META_EMBEDDING_DIM,
    META_VEC_CAPABILITY,
    META_CREATED_AT,
)

# vec0 能力取值
VEC_CAP_TEXT_PK = "text_pk"
VEC_CAP_ROWID = "rowid"

# 检索时可见的状态：遗忘态不参与任何召回
_SEARCHABLE = ("active", "dormant")

_MEMORY_COLUMNS = (
    "id, layer, type, subject, predicate, object, content, abstract, scope, "
    "confidence, salience, strength, access_count, last_access_at, "
    "valid_from, valid_to, source_session, source_turn, status, superseded_by, "
    "created_at, updated_at, embedding_model, content_hash"
)

# update() 允许的补丁字段白名单（列名 → 是否 JSON 编码）
_PATCHABLE: dict[str, bool] = {
    "layer": False,
    "type": False,
    "subject": False,
    "predicate": False,
    "object": False,
    "content": False,
    "abstract": False,
    "scope": True,
    "confidence": False,
    "salience": False,
    "strength": False,
    "access_count": False,
    "last_access_at": False,
    "valid_from": False,
    "valid_to": False,
    "source_session": False,
    "source_turn": False,
    "status": False,
    "superseded_by": False,
    "embedding_model": False,
    "content_hash": False,
}


class SQLiteBackend:
    """SQLite + sqlite-vec 后端。

    Args:
        path: 数据库文件路径；``":memory:"`` 表示内存库（测试用）。
        embedding_dim: 向量维度。**建表后不可更改**，故必须在首次迁移前确定。
        embedding_model: embedding 模型名（INV-2 的比对基据）。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
        embedding_model: str | None = None,
    ) -> None:
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim 必须是正整数，收到 {embedding_dim!r}")
        self.path = str(path)
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self._conn: sqlite3.Connection | None = None
        self._write_conn: sqlite3.Connection | None = None
        self._write_tid: int | None = None
        self._write_lock = threading.RLock()
        self._in_tx = False
        self.connections_opened = 0
        """已建立的连接数（测试用它断言"读连接 + 一个写连接"，不多也不少）。"""

    # ================================================================== #
    # 生命周期（T-AL3-01 / T-AL3-04）
    # ================================================================== #

    def open(self) -> None:
        """建连 → 施加 PRAGMA → 注册函数 → 加载扩展 → 迁移。

        **打开两条连接**：

        - **读连接**（``self.conn``）：召回与查询用，可被任意线程使用
        - **写连接**（``self._tx`` 内部按需创建）：**只由单写者线程使用**（D-03）

        为什么要分开：SQLite 的连接不是线程安全的，而器灵的架构又决定了
        "主线程只读、writer 线程只写"。共用一个连接会在 writer 线程里直接报
        "SQLite objects created in a thread can only be used in that same thread"。
        分开之后，WAL 让读与写真正并发，且**单写者纪律仍然成立**。
        """
        if self._conn is not None:
            return

        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        conn = self._new_connection()
        self._conn = conn

        try:
            self.migrate()
        except Exception:
            conn.close()
            self._conn = None
            raise

    def _new_connection(self) -> sqlite3.Connection:
        """建一条配置完整的连接。

        **关于 ``check_same_thread=False``**：器灵的架构决定了"主线程只读、
        writer 线程只写"，而两者都要碰 SQLite。Python 的 sqlite3 在本环境下以
        ``SQLITE_THREADSAFE=1``（串行化）编译（``sqlite3.threadsafety == 3``），
        同一连接可被多线程共享，访问由 SQLite 自己的互斥量串行化。

        真正防止写竞争的不是这个标志，而是 **D-03 单写者 + 写连接独立 + 写锁**
        这三件事。这里只是把"跨线程访问"从异常变成受控行为。
        """
        if sqlite3.threadsafety < 2:  # pragma: no cover - 环境不支持时明确失败
            raise StorageFatalError(
                f"当前 Python 的 sqlite3 线程安全级别为 {sqlite3.threadsafety}（需 >= 2）。"
                "器灵依赖串行化模式的 SQLite 才能在读写线程间共享连接。"
            )
        conn = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        self._configure_connection(conn)
        self._register_functions(conn)
        self._load_extensions(conn)
        self.connections_opened += 1
        return conn

    def _writer_connection(self) -> sqlite3.Connection:
        """取**当前线程**的写连接。

        D-03 保证同一时刻只有一个写线程，因此这里最多只会存在一条写连接；
        若写线程变了（例如无队列的 CLI 场景），旧连接会被关掉再建。

        **内存库例外**：``:memory:`` 的每条连接各自是一个独立数据库，
        分连接会让写连接看不到读连接建的表。因此内存库共用一条连接——
        它只用于测试，没有跨线程写入的场景。
        """
        if self.path == ":memory:":
            return self.conn

        tid = threading.get_ident()
        if self._write_conn is not None and self._write_tid == tid:
            return self._write_conn
        if self._write_conn is not None:
            try:
                self._write_conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass
        self._write_conn = self._new_connection()
        self._write_tid = tid
        return self._write_conn

    def close(self) -> None:
        """WAL checkpoint + 关闭全部连接。**幂等**。"""
        if self._conn is None and self._write_conn is None:
            return
        read_conn, self._conn = self._conn, None
        write_conn, self._write_conn = self._write_conn, None
        self._write_tid = None
        for conn in (write_conn, read_conn):
            if conn is None:
                continue
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError:
                pass  # 内存库不支持
            finally:
                conn.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """**读连接**。未 open 时抛 :class:`StorageFatalError`。"""
        if self._conn is None:
            raise StorageFatalError("后端尚未 open()，无法访问连接")
        return self._conn

    # ================================================================== #
    # 连接配置
    # ================================================================== #

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        """施加连接级 PRAGMA（LLD-AL3 §5 M4 文件头）。

        这些设置**不随库持久化**，每个连接都必须重新施加。
        """
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")

    @staticmethod
    def _register_functions(conn: sqlite3.Connection) -> None:
        """注册 ``spirit_norm``，供 FTS 触发器把正文转成 CJK 逐字形式。

        必须在 ``migrate()`` 之前完成——否则 ``CREATE TRIGGER`` 会因
        函数不存在而失败。
        """
        conn.create_function("spirit_norm", 1, normalize_for_fts, deterministic=True)

    @staticmethod
    def _load_extensions(conn: sqlite3.Connection) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover
            raise StorageFatalError(
                "缺少 sqlite-vec 扩展。请执行：pip install sqlite-vec"
            ) from exc

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)

    # ================================================================== #
    # 事务
    # ================================================================== #

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """显式事务：``BEGIN IMMEDIATE`` → 提交 / 回滚。

        使用 IMMEDIATE 是为了**立即取得写锁**，把锁冲突暴露在事务开头而非中途。
        SQLITE_BUSY 统一翻译为 :class:`StorageBusyError`（AL5 writer 会重试）。

        事务跑在**:写连接**上（见 :meth:`_writer_connection`），
        并由 ``_write_lock`` 串行化——D-03 已经保证只有一个写线程，
        这把锁是"就算哪天有人违反了 D-03 也不会静默出错"的保险。
        """
        with self._write_lock:
            if self._in_tx:
                raise StorageFatalError("检测到嵌套事务——_tx() 不可重入")
            conn = self._writer_connection()
            self._in_tx = True
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                self._in_tx = False
                raise self._translate(exc) from exc

            try:
                yield conn
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:  # pragma: no cover
                    pass
                self._in_tx = False
                raise
            else:
                try:
                    conn.execute("COMMIT")
                except sqlite3.OperationalError as exc:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:  # pragma: no cover
                        pass
                    self._in_tx = False
                    raise self._translate(exc) from exc
                self._in_tx = False

    @staticmethod
    def _translate(exc: sqlite3.Error) -> StoreError:
        """把 sqlite3 异常翻译为存储层异常族。"""
        msg = str(exc).lower()
        if "locked" in msg or "busy" in msg:
            return StorageBusyError(f"数据库繁忙：{exc}")
        return StorageFatalError(f"存储故障：{exc}")

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """**读**语句（走读连接）。"""
        try:
            return self.conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            raise self._translate(exc) from exc
        except sqlite3.IntegrityError:
            raise

    def _write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """**写**语句（走写连接，受 ``_write_lock`` 保护）。

        单条自动提交的写也走这里——否则它会跑到读连接上，与写连接上的事务
        抢锁，表现为偶发的 ``database is locked``。
        """
        with self._write_lock:
            conn = self._writer_connection()
            try:
                return conn.execute(sql, params)
            except sqlite3.OperationalError as exc:
                raise self._translate(exc) from exc
            except sqlite3.IntegrityError:
                raise

    # ================================================================== #
    # 迁移框架（T-AL3-01 / T-AL3-06）
    # ================================================================== #

    def migrate(self) -> None:
        """按版本号施加迁移。**幂等**——已应用版本不会重复执行。

        迁移是写操作，因此跑在**写连接**上并持有写锁——
        否则启动时可能与正在跑的写事务抢锁。
        """
        with self._write_lock:
            conn = self._writer_connection()
            self._migrate_on(conn)

    def _migrate_on(self, conn: sqlite3.Connection) -> None:
        current = self._read_schema_version(conn)

        if current > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"库的 schema 版本为 {current}，高于代码期望的 {SCHEMA_VERSION}。"
                "请升级器灵，或使用匹配版本的代码。"
            )

        for version in sorted(v for v in _MIGRATIONS if v > current):
            _MIGRATIONS[version](self, conn)
            self._write_schema_version(conn, version)

        self._ensure_vec_capability(conn)
        self._record_embedding_meta(conn)

    def _read_schema_version(self, conn: sqlite3.Connection) -> int:
        if not self._table_exists(conn, "meta"):
            return 0
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (META_SCHEMA_VERSION,)
        ).fetchone()
        return int(row["value"]) if row else 0

    @staticmethod
    def _write_schema_version(conn: sqlite3.Connection, version: int) -> None:
        conn.execute(
            """INSERT INTO meta(key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (META_SCHEMA_VERSION, str(version)),
        )

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _apply_schema(self, conn: sqlite3.Connection) -> None:
        """执行全量 schema.sql（全部 DDL 均为 ``IF NOT EXISTS``）。"""
        ddl = SCHEMA_FILE.read_text(encoding="utf-8").replace(
            _EMBEDDING_DIM_TOKEN, str(self.embedding_dim)
        )
        conn.executescript(ddl)

    def _apply_v1(self, conn: sqlite3.Connection) -> None:
        """v1：全量 schema。先探测 vec0 能力再建表。"""
        capability = self._probe_vec_capability(conn)
        self._require_text_pk(capability)

        self._apply_schema(conn)

        for key, value in (
            (META_VEC_CAPABILITY, capability),
            (META_CREATED_AT, now_iso()),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value)
            )

    def _apply_v2(self, conn: sqlite3.Connection) -> None:
        """v2：把 ``mem_fts`` 从外部内容表重建为 CJK 归一化的普通 FTS5 表。

        外部内容表要求索引内容与 ``memories`` 逐字一致，无法承载逐字归一化
        （见 ``store/text.py`` 文件头）。重建后由 ``rebuild_fts()`` 补数据。
        """
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS trg_mem_ai;
            DROP TRIGGER IF EXISTS trg_mem_ad;
            DROP TRIGGER IF EXISTS trg_mem_au;
            DROP TABLE IF EXISTS mem_fts;
            """
        )
        # 重新执行全量 schema：mem_fts 刚被删除，会被重建为新的形态
        self._apply_schema(conn)

    # ================================================================== #
    # vec0 能力探测（T-AL3-03）
    # ================================================================== #

    def _ensure_vec_capability(self, conn: sqlite3.Connection) -> str:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (META_VEC_CAPABILITY,)
        ).fetchone()
        if row is not None:
            return row["value"]

        capability = self._probe_vec_capability(conn)
        self._require_text_pk(capability)
        conn.execute(
            """INSERT INTO meta(key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (META_VEC_CAPABILITY, capability),
        )
        return capability

    @staticmethod
    def _require_text_pk(capability: str) -> None:
        """当前设计依赖 ``vec0`` 的 TEXT 主键——不支持则**明确失败**。

        **实现决策（偏离 T-AL3-03 的"退化路径"）**：原规格要求不支持时退回
        ``INTEGER rowid`` + 映射表。该分支在 sqlite-vec 0.1.9（实测支持 TEXT
        主键）下**无法被测试**，而它位于所有向量读写的关键路径上——引入不可测
        的分支比明确失败更危险。故改为探测 → 不支持则抛错并指明所需版本。
        """
        if capability != VEC_CAP_TEXT_PK:
            raise StorageFatalError(
                "当前 sqlite-vec 不支持 vec0 的 TEXT 主键"
                f"（探测结果：{capability}）。器灵要求 sqlite-vec >= 0.1.9。"
                "请升级扩展后重试。"
            )

    @staticmethod
    def _probe_vec_capability(conn: sqlite3.Connection) -> str:
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE temp.__vec_probe USING vec0("
                "  mem_id TEXT PRIMARY KEY,"
                "  embedding float[8]"
                ")"
            )
            conn.execute("DROP TABLE temp.__vec_probe")
            return VEC_CAP_TEXT_PK
        except sqlite3.OperationalError:
            return VEC_CAP_ROWID

    # ================================================================== #
    # meta（T-AL3-02）
    # ================================================================== #

    def meta_get(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def meta_set(self, key: str, value: str) -> None:
        """写入或覆盖。"""
        self._write(
            """INSERT INTO meta(key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (key, str(value)),
        )

    def ensure_spirit_id(self) -> str:
        """返回 ``spirit_id``；不存在则生成并落库。

        **已存在时不覆盖**（C6）——否则器灵等于"换了身份"。
        """
        existing = self.meta_get(META_SPIRIT_ID)
        if existing:
            return existing
        spirit_id = _ids.new_ulid()
        self._write(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            (META_SPIRIT_ID, spirit_id),
        )
        return self.meta_get(META_SPIRIT_ID) or spirit_id

    # ================================================================== #
    # 记忆写入（T-AL3-05 / T-AL3-07 / T-AL3-16）
    # ================================================================== #

    def put(
        self,
        rec: MemoryRecord,
        embedding: list[float] | None = None,
        *,
        audit: AuditEvent | None = None,
    ) -> str:
        """写入一条记忆。

        **原子性（T-AL3-16）**：``memories`` + ``vec_memories`` + ``mem_fts``（触发器）
        + ``audit`` 在同一事务内提交。任一步失败则全部回滚，不留半写状态。
        """
        if not rec.id:
            rec.id = _ids.new_memory_id(rec.layer)

        ts = now_iso()
        if not rec.created_at:
            rec.created_at = ts
        if not rec.updated_at:
            rec.updated_at = ts
        # 双时态（T-AL3-24 / M3 启用）：`valid_from` 缺省 = `created_at`。
        # **不能留空**——as-of 判定读的就是它，留空会让"这条没有时态信息"与
        # "它从很久以前就有效"在查询里长得一模一样，无法区分。
        if not rec.valid_from:
            rec.valid_from = rec.created_at
        if rec.content_hash is None:
            rec.content_hash = content_hash_of(
                content=rec.content,
                subject=rec.subject,
                predicate=rec.predicate,
                object_=rec.object,
                scope=rec.scope,
            )
        if embedding is not None:
            self._check_dim(embedding)
            if rec.embedding_model is None:
                rec.embedding_model = self.embedding_model

        with self._tx() as conn:
            conn.execute(
                f"INSERT INTO memories ({_MEMORY_COLUMNS}) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                self._record_params(rec),
            )
            if embedding is not None:
                conn.execute(
                    "INSERT INTO vec_memories(mem_id, embedding) VALUES (?, ?)",
                    (rec.id, json.dumps(embedding)),
                )
            self._insert_audit(
                conn,
                audit
                or AuditEvent(
                    op="add",
                    actor="system",
                    target_kind="memory",
                    target_id=rec.id,
                    after={"layer": rec.layer, "type": rec.type},
                ),
            )
        return rec.id

    def update(
        self, mem_id: str, patch: dict, *, audit: AuditEvent | None = None
    ) -> None:
        """按白名单字段更新，并**同步刷新 ``updated_at``**。"""
        unknown = set(patch) - set(_PATCHABLE)
        if unknown:
            raise ValueError(f"不可更新的字段：{sorted(unknown)}")

        # 双时态（T-AL3-24）：`superseded_by` **不得自指**。
        # 写环会让"谁取代谁"这条链无解——而 :meth:`asof` 正是沿着它走的：
        # 一旦成环，那条记忆的"某时刻有效版本"就永远查不出来，**只是一个 None，
        # 没有任何报错**。所以宁可在写入侧就炸掉。
        if patch.get("superseded_by") == mem_id:
            raise StoreError(
                f"superseded_by 不能指向自己：{mem_id}——被谁取代必须指向另一条记录，"
                "否则时态链成环、as-of 查询永远查不出结果"
            )

        before = self.get(mem_id)
        if before is None:
            raise NotFoundError(f"记忆不存在：{mem_id}")

        assignments: list[str] = []
        values: list[Any] = []
        for key, value in patch.items():
            assignments.append(f"{key} = ?")
            values.append(json.dumps(value, ensure_ascii=False) if _PATCHABLE[key] else value)

        if "content_hash" not in patch and any(
            k in patch for k in ("content", "subject", "predicate", "object", "scope")
        ):
            merged = asdict(before)
            merged.update(patch)
            assignments.append("content_hash = ?")
            values.append(
                content_hash_of(
                    content=merged["content"],
                    subject=merged.get("subject"),
                    predicate=merged.get("predicate"),
                    object_=merged.get("object"),
                    scope=merged.get("scope"),
                )
            )

        assignments.append("updated_at = ?")
        values.append(now_iso())
        values.append(mem_id)

        with self._tx() as conn:
            conn.execute(
                f"UPDATE memories SET {', '.join(assignments)} WHERE id = ?",
                tuple(values),
            )
            self._insert_audit(
                conn,
                audit
                or AuditEvent(
                    op="update",
                    actor="system",
                    target_kind="memory",
                    target_id=mem_id,
                    before={"patch": sorted(patch)},
                ),
            )

    def set_status(
        self, mem_id: str, status: Status, *, reason: str, actor: str = "system"
    ) -> None:
        """状态迁移（``active`` ↔ ``dormant`` / → ``forgotten``）。

        本方法**只改状态**，不做物理删除——删除的唯一入口是 :meth:`hard_delete`。
        """
        before = self.get(mem_id)
        if before is None:
            raise NotFoundError(f"记忆不存在：{mem_id}")

        with self._tx() as conn:
            conn.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                (status, now_iso(), mem_id),
            )
            self._insert_audit(
                conn,
                AuditEvent(
                    op="set_status",
                    actor=actor,
                    target_kind="memory",
                    target_id=mem_id,
                    before={"status": before.status},
                    after={"status": status},
                    reason=reason,
                ),
            )

    def touch(
        self, mem_id: str, ts: str, *, strength: float | None = None
    ) -> None:
        """记录一次访问：``access_count += 1``、刷新 ``last_access_at``，可选回升 ``strength``。

        写入 ``audit(op='touch')`` 并**记录三个派生字段的绝对值**——这是
        :meth:`replay_derived` 能精确重建派生字段的前提（INV-12）。
        """
        row = self._execute(
            "SELECT strength, access_count FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"记忆不存在：{mem_id}")

        new_strength = float(strength if strength is not None else row["strength"])
        new_count = int(row["access_count"]) + 1

        with self._tx() as conn:
            conn.execute(
                "UPDATE memories SET strength = ?, access_count = ?, "
                "last_access_at = ?, updated_at = ? WHERE id = ?",
                (new_strength, new_count, ts, ts, mem_id),
            )
            self._insert_audit(
                conn,
                AuditEvent(
                    op="touch",
                    actor="system",
                    target_kind="memory",
                    target_id=mem_id,
                    after={
                        "strength": new_strength,
                        "access_count": new_count,
                        "last_access_at": ts,
                    },
                ),
            )

    # ================================================================== #
    # 记忆读取（T-AL3-05）
    # ================================================================== #

    def get(self, mem_id: str) -> MemoryRecord | None:
        row = self._execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def asof(self, mem_id: str, ts: str) -> MemoryRecord | None:
        """取 ``ts`` 时刻**有效**的那一版（双时态 · T-AL3-24）。

        语义（跟 ``superseded_by`` 链走）：

        1. 从 ``mem_id`` 出发，沿 ``superseded_by`` **向前**（走向更新的版本）；
        2. 返回第一条满足 ``valid_from <= ts AND (valid_to IS NULL OR valid_to > ts)`` 的记录；
        3. 走完整条链都没有 → ``None``（= 该时刻不存在有效版本）。

        **判定落在 SQL 里**：不把链捞进内存再比时间——那是**静默的全表扫描**，
        而库里几百条之后没人会注意到它变成了热点。

        ``valid_from IS NULL`` 视为"**一直有效**"：那是 M3 之前写入的存量记录
        （当时只写 ``created_at``），把它们判成"从来没有效"比放宽更糟。

        ``seen`` 用来防链上出现环——写入侧已经拒绝自指（见 :meth:`update`），
        这里是第二道闸：**同一个错误不要只靠一处防**。
        """
        current: str | None = mem_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            row = self._execute(
                f"""SELECT {_MEMORY_COLUMNS} FROM memories
                    WHERE id = ?
                      AND (valid_from IS NULL OR valid_from <= ?)
                      AND (valid_to IS NULL OR valid_to > ?)""",
                (current, ts, ts),
            ).fetchone()
            if row is not None:
                return self._row_to_record(row)
            nxt = self._execute(
                "SELECT superseded_by FROM memories WHERE id = ?", (current,)
            ).fetchone()
            current = nxt["superseded_by"] if nxt else None
        return None

    def query(
        self,
        *,
        layer: Layer | None = None,
        status: Status | None = "active",
        types: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        """组合过滤查询。``status=None`` 表示不限状态。"""
        where: list[str] = []
        params: list[Any] = []

        if layer is not None:
            where.append("layer = ?")
            params.append(layer)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if types:
            where.append(f"type IN ({','.join('?' * len(types))})")
            params.extend(types)
        if since is not None:
            where.append("created_at >= ?")
            params.append(since)
        if until is not None:
            where.append("created_at <= ?")
            params.append(until)

        sql = f"SELECT {_MEMORY_COLUMNS} FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY rowid DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        rows = self._execute(sql, tuple(params)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def find_by_source_session(
        self, session_id: str, *, layer: Layer | None = None
    ) -> list[MemoryRecord]:
        """按来源会话查找（巩固的幂等判据）。

        AL2 的巩固需要回答"这个会话是否已经固化过"——用 SQL 回答比把整层拉进内存再过滤
        要诚实得多（也避免 `query()` 的默认 ``status='active'`` 过滤掉已休眠的固化结果）。
        """
        sql = f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE source_session = ?"
        params: list[Any] = [session_id]
        if layer is not None:
            sql += " AND layer = ?"
            params.append(layer)
        sql += " ORDER BY id ASC"
        rows = self._execute(sql, tuple(params)).fetchall()
        return [self._row_to_record(r) for r in rows]

    def inbound_reference_counts(self, mem_ids: list[str]) -> dict[str, int]:
        """批量统计每条记忆的**入边引用数**。

        用于 `importance` 的"被反复引用"分量。**刻意批量**——逐条查询会让召回
        退化成 N+1，而召回是每轮都跑的热路径。
        """
        if not mem_ids:
            return {}
        placeholders = ",".join("?" * len(mem_ids))
        rows = self._execute(
            f"""SELECT dst_id AS mid, COUNT(*) AS n FROM relations
                WHERE dst_kind = 'memory' AND dst_id IN ({placeholders})
                GROUP BY dst_id""",
            tuple(mem_ids),
        ).fetchall()
        counts = {row["mid"]: int(row["n"]) for row in rows}
        return {mid: counts.get(mid, 0) for mid in mem_ids}

    def outbound_edges(self, mem_ids: list[str]) -> list[dict]:
        """批量取这些记忆的出边（扩散激活用，避免 N+1）。"""
        if not mem_ids:
            return []
        placeholders = ",".join("?" * len(mem_ids))
        rows = self._execute(
            f"""SELECT src_id, dst_kind, dst_id, rel_type, weight FROM relations
                WHERE src_kind = 'memory' AND src_id IN ({placeholders})""",
            tuple(mem_ids),
        ).fetchall()
        return [dict(r) for r in rows]

    def core_memory_terms(self) -> list[str]:
        """核心记忆的索引词（`core` 一致性因子用）。

        取每条的 `subject` 与 `content` 中出现的实体名——**不需要 LLM**。
        """
        rows = self._execute(
            "SELECT subject, object, content FROM memories WHERE layer = 'core'"
        ).fetchall()
        terms: set[str] = set()
        for row in rows:
            for field in (row["subject"], row["object"]):
                if field:
                    terms.add(str(field).casefold())
        return sorted(terms)

    def set_vector(self, mem_id: str, vec: list[float]) -> None:
        """单独更新某条记忆的向量（内容被修改后重算）。

        **维度校验先于写入**——它是防"写入与检索向量空间不一致"的最后一道闸门。
        """
        self._check_dim(vec)
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vec_memories(mem_id, embedding) VALUES (?, ?)",
                (mem_id, json.dumps(vec)),
            )

    def count_by_layer(self) -> dict[str, dict[str, int]]:
        """各层的状态计数（``status`` 面板的数据来源）。"""
        rows = self._execute(
            "SELECT layer, status, COUNT(*) AS n FROM memories GROUP BY layer, status"
        ).fetchall()
        result: dict[str, dict[str, int]] = {}
        for row in rows:
            result.setdefault(row["layer"], {})[row["status"]] = int(row["n"])
        return result

    # ================================================================== #
    # 检索原语（T-AL3-08）
    # ================================================================== #

    def vector_search(
        self, vec: list[float], *, layer: Layer | None = None, top_k: int = 8
    ) -> list[Hit]:
        """KNN 向量检索。**只返回原生分**（距离越小越近），融合排序在 AL2。

        遗忘态（``forgotten``）不参与检索；``dormant`` 仍在结果中，由 AL2 决定其
        是否只取 L0（D-23）。
        """
        self._check_dim(vec)
        if top_k <= 0:
            return []

        # KNN 的"取前 k 条"必须写成 **`AND k = ?`**，不能写 `ORDER BY distance LIMIT ?`。
        #
        # 这不是风格问题：两者在新旧 SQLite 上行为不同。实测
        #   SQLite 3.53 + sqlite-vec 0.1.9：两种写法都可以
        #   SQLite 3.40 + sqlite-vec 0.1.9：只有 `k = ?` 可以，
        #     LIMIT（无论字面量还是绑定参数）直接报
        #     "A LIMIT or 'k = ?' constraint is required on vec0 knn queries"
        # 后者正是 Debian 12 / Python 3.11 系统自带的 SQLite——也就是**部署目标机器**。
        # 只在开发机上跑，这个缺陷永远不会出现。
        sql = f"""
            SELECT {', '.join('m.' + c.strip() for c in _MEMORY_COLUMNS.split(','))},
                   v.distance AS _distance
            FROM (SELECT mem_id, distance FROM vec_memories
                  WHERE embedding MATCH ? AND k = ?) v
            JOIN memories m ON m.id = v.mem_id
            WHERE m.status IN ({','.join('?' * len(_SEARCHABLE))})
        """
        params: list[Any] = [json.dumps(vec), int(top_k), *_SEARCHABLE]
        if layer is not None:
            sql += " AND m.layer = ?"
            params.append(layer)
        sql += " ORDER BY v.distance ASC"

        rows = self._execute(sql, tuple(params)).fetchall()
        # 距离 → 相似度：cosine 距离落在 [0,2]，转成 [0,1] 的"越大越好"
        return [
            self._row_to_hit(r, score=1.0 - float(r["_distance"])) for r in rows
        ]

    def keyword_search(
        self, query: str, *, layer: Layer | None = None, top_k: int = 8
    ) -> list[Hit]:
        """BM25 关键词检索。查询中的 CJK 会被归一化为短语（见 ``store/text.py``）。"""
        match = fts_match_query(query)
        if match is None or top_k <= 0:
            return []

        weights = ", ".join(str(w) for w in FTS_COLUMN_WEIGHTS)
        sql = f"""
            SELECT {', '.join('m.' + c.strip() for c in _MEMORY_COLUMNS.split(','))},
                   bm25(mem_fts, {weights}) AS _rank
            FROM mem_fts
            JOIN memories m ON m.rowid = mem_fts.rowid
            WHERE mem_fts MATCH ?
              AND m.status IN ({','.join('?' * len(_SEARCHABLE))})
        """
        params: list[Any] = [match, *_SEARCHABLE]
        if layer is not None:
            sql += " AND m.layer = ?"
            params.append(layer)
        sql += " ORDER BY _rank ASC LIMIT ?"
        params.append(int(top_k))

        try:
            rows = self._execute(sql, tuple(params)).fetchall()
        except sqlite3.OperationalError:
            # FTS5 语法异常（罕见）→ 关键词路降级为空，绝不阻断召回
            return []

        # bm25 越小越相关；取负使"越大越好"，与 vector_search 同向
        return [self._row_to_hit(r, score=-float(r["_rank"])) for r in rows]

    # ================================================================== #
    # 关联（T-AL3-09）
    # ================================================================== #

    def link(
        self,
        a_kind: str,
        a_id: str,
        b_kind: str,
        b_id: str,
        rel_type: str,
        weight: float,
    ) -> None:
        """建立关联边。**重复调用不产生重复边**（UNIQUE 约束 + INSERT OR IGNORE）。"""
        ts = now_iso()
        with self._tx() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO relations
                   (src_kind, src_id, dst_kind, dst_id, rel_type, weight, co_count,
                    valid_from, last_co_at)
                   VALUES (?,?,?,?,?,?,1,?,?)""",
                (a_kind, a_id, b_kind, b_id, rel_type, float(weight), ts, ts),
            )
            self._insert_audit(
                conn,
                AuditEvent(
                    op="link",
                    actor="system",
                    target_kind="relation",
                    target_id=f"{a_id}->{b_id}",
                    after={"rel_type": rel_type, "weight": float(weight)},
                ),
            )

    def reinforce(
        self, a_id: str, b_id: str, delta: float, *, rel_type: str = "co_activation"
    ) -> None:
        """Hebbian 共激活强化（LLD-AL2 §5 M4）：

        ``w ← w + η·(1 − w)``，并累加 ``co_count``。边不存在时以 ``η`` 为初值创建。
        """
        ts = now_iso()
        eta = max(0.0, min(1.0, float(delta)))
        with self._tx() as conn:
            row = conn.execute(
                """SELECT weight, co_count FROM relations
                   WHERE src_kind='memory' AND src_id=? AND dst_kind='memory'
                     AND dst_id=? AND rel_type=?""",
                (a_id, b_id, rel_type),
            ).fetchone()

            if row is None:
                conn.execute(
                    """INSERT INTO relations
                       (src_kind, src_id, dst_kind, dst_id, rel_type, weight, co_count,
                        valid_from, last_co_at)
                       VALUES ('memory',?,'memory',?,?,?,1,?,?)""",
                    (a_id, b_id, rel_type, eta, ts, ts),
                )
                new_weight, new_count = eta, 1
            else:
                new_weight = float(row["weight"]) + eta * (1.0 - float(row["weight"]))
                new_count = int(row["co_count"]) + 1
                conn.execute(
                    """UPDATE relations SET weight=?, co_count=?, last_co_at=?
                       WHERE src_kind='memory' AND src_id=? AND dst_kind='memory'
                         AND dst_id=? AND rel_type=?""",
                    (new_weight, new_count, ts, a_id, b_id, rel_type),
                )

            self._insert_audit(
                conn,
                AuditEvent(
                    op="reinforce",
                    actor="system",
                    target_kind="relation",
                    target_id=f"{a_id}->{b_id}",
                    after={"weight": new_weight, "co_count": new_count},
                ),
            )

    def neighbors(
        self,
        kind: str,
        node_id: str,
        *,
        rel_type: str | None = None,
        min_weight: float = 0.0,
        limit: int = 20,
    ) -> list[tuple[str, str, float]]:
        """按 ``weight DESC`` 返回邻居 ``(kind, id, weight)``（双向）。"""
        sql = """
            SELECT dst_kind AS k, dst_id AS i, weight AS w FROM relations
            WHERE src_kind = ? AND src_id = ? AND weight >= ?
            UNION ALL
            SELECT src_kind AS k, src_id AS i, weight AS w FROM relations
            WHERE dst_kind = ? AND dst_id = ? AND weight >= ?
        """
        params: list[Any] = [kind, node_id, min_weight, kind, node_id, min_weight]
        if rel_type is not None:
            sql = sql.replace("AND weight >= ?", "AND weight >= ? AND rel_type = ?")
            params = [
                kind, node_id, min_weight, rel_type,
                kind, node_id, min_weight, rel_type,
            ]

        rows = self._execute(sql, tuple(params)).fetchall()
        merged: dict[tuple[str, str], float] = {}
        for row in rows:
            key = (row["k"], row["i"])
            if key == (kind, node_id):
                continue
            merged[key] = max(merged.get(key, 0.0), float(row["w"]))

        ordered = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
        return [(k, i, w) for (k, i), w in ordered[:limit]]

    def relations_of(self, mem_id: str) -> list[dict]:
        """该记忆的全部关联边（导出与删除快照用）。"""
        rows = self._execute(
            """SELECT src_kind, src_id, dst_kind, dst_id, rel_type, weight, co_count
               FROM relations WHERE (src_kind='memory' AND src_id=?)
                                 OR (dst_kind='memory' AND dst_id=?)""",
            (mem_id, mem_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # ================================================================== #
    # 实体（T-AL3-10）
    # ================================================================== #

    def entity_upsert(
        self,
        name: str,
        type_: str,
        *,
        aliases: list[str] | None = None,
        entity_id: str | None = None,
    ) -> str:
        """同名同类型 upsert（不重复建节点）。

        ``entity_id`` 供导入时**沿用原 ID**——这样含实体引用的关联边在往返后不断裂。
        """
        ts = now_iso()
        alias_json = json.dumps(sorted(set(aliases or [])), ensure_ascii=False)
        with self._tx() as conn:
            row = conn.execute(
                "SELECT id FROM entities WHERE name = ? AND type = ?", (name, type_)
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE entities SET aliases = ?, updated_at = ? WHERE id = ?",
                    (alias_json, ts, row["id"]),
                )
                return row["id"]

            if entity_id and conn.execute(
                "SELECT 1 FROM entities WHERE id = ?", (entity_id,)
            ).fetchone() is None:
                new_id = entity_id
            else:
                new_id = _ids.new_entity_id()

            conn.execute(
                """INSERT INTO entities(id, name, type, aliases, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (new_id, name, type_, alias_json, ts, ts),
            )
            return new_id

    def entity_get(self, entity_id: str) -> EntityRecord | None:
        row = self._execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def entity_find(self, text: str) -> list[EntityRecord]:
        """在文本中找到匹配的实体（名称命中或别名命中）。

        **无需 LLM**——实体在提取环节已建好，这里只是字符串匹配（D-10）。
        """
        if not text:
            return []
        rows = self._execute("SELECT * FROM entities").fetchall()
        found: list[EntityRecord] = []
        lowered = text.casefold()
        for row in rows:
            entity = self._row_to_entity(row)
            names = [entity.name, *entity.aliases]
            if any(n and n.casefold() in lowered for n in names):
                found.append(entity)
        return found

    def entity_list(self) -> list[EntityRecord]:
        """全部实体（导出用）。"""
        rows = self._execute("SELECT * FROM entities ORDER BY id").fetchall()
        return [self._row_to_entity(r) for r in rows]

    def all_relations(self) -> list[dict]:
        """全部关联边（导出用）。"""
        rows = self._execute(
            "SELECT src_kind, src_id, dst_kind, dst_id, rel_type, weight, co_count "
            "FROM relations ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ================================================================== #
    # 审计（T-AL3-12）
    # ================================================================== #

    def audit(self, ev: AuditEvent) -> int:
        """追加审计事件。本表 append-only（INV-8）。"""
        with self._tx() as conn:
            return self._insert_audit(conn, ev)

    def _insert_audit(self, conn: sqlite3.Connection, ev: AuditEvent) -> int:
        # 审计 `op` 的**唯一权威列举**是 `base.AUDIT_OPS`（评审 P1-20）：
        # 在此之前"权威"散在三处且互不相同，而实现里在用的 4 个 op 连文档都没有。
        # 校验放在**唯一写入点**——放在别处等于没放。
        if ev.op not in AUDIT_OPS:
            raise StoreError(
                f"未知的审计 op={ev.op!r}：请把它登记进 store.base.AUDIT_OPS"
                f"（当前已知：{sorted(AUDIT_OPS)}）"
            )
        ts = ev.ts or now_iso()
        cursor = conn.execute(
            """INSERT INTO audit(ts, op, actor, target_kind, target_id,
                                 before, after, reason, session_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                ts,
                ev.op,
                ev.actor,
                ev.target_kind,
                ev.target_id,
                json.dumps(ev.before, ensure_ascii=False) if ev.before is not None else None,
                json.dumps(ev.after, ensure_ascii=False) if ev.after is not None else None,
                ev.reason,
                ev.session_id,
            ),
        )
        return int(cursor.lastrowid or 0)

    def audit_replay(self, *, since: str | None = None) -> Iterator[AuditEvent]:
        """按时间序（同一时刻按 id 序）重放审计事件（INV-12 的基础）。"""
        sql = "SELECT * FROM audit"
        params: tuple = ()
        if since is not None:
            sql += " WHERE ts >= ?"
            params = (since,)
        sql += " ORDER BY ts ASC, id ASC"

        cursor = self._execute(sql, params)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield self._row_to_audit(row)

    def audit_get(self, audit_id: int) -> AuditEvent | None:
        row = self._execute("SELECT * FROM audit WHERE id = ?", (audit_id,)).fetchone()
        return self._row_to_audit(row) if row else None

    def audit_for(self, mem_id: str, *, limit: int = 200) -> list[AuditEvent]:
        rows = self._execute(
            """SELECT * FROM audit WHERE target_id = ?
               ORDER BY ts ASC, id ASC LIMIT ?""",
            (mem_id, int(limit)),
        ).fetchall()
        return [self._row_to_audit(r) for r in rows]

    def last_audit_id(self) -> int:
        row = self._execute("SELECT MAX(id) AS m FROM audit").fetchone()
        return int(row["m"] or 0)

    # ================================================================== #
    # 会话 / 工作记忆 / 意图（T-AL3-11 / T-AL3-17）
    # ================================================================== #

    def session_create(self, session_id: str, started_at: str) -> None:
        self._write(
            "INSERT OR IGNORE INTO sessions(id, started_at, status, turn_count) "
            "VALUES (?,?, 'active', 0)",
            (session_id, started_at),
        )

    def session_bump_turn(self, session_id: str) -> int:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(id, started_at, status, turn_count) "
                "VALUES (?,?, 'active', 0)",
                (session_id, now_iso()),
            )
            conn.execute(
                "UPDATE sessions SET turn_count = turn_count + 1 WHERE id = ?",
                (session_id,),
            )
            row = conn.execute(
                "SELECT turn_count FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return int(row["turn_count"]) if row else 0

    def session_end(self, session_id: str, ended_at: str) -> None:
        self._write(
            "UPDATE sessions SET status = 'committed', ended_at = ? WHERE id = ?",
            (ended_at, session_id),
        )

    def session_get(self, session_id: str) -> dict | None:
        row = self._execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def wm_put(
        self, session_id: str, chunk_key: str, content: str, salience: float
    ) -> str:
        """写入或累积一个工作记忆组块（按 ``session_id + chunk_key`` 归并）。

        ``last_touched`` 用**微秒精度**——工作记忆的排序完全依赖"最近触碰"，
        秒级精度会让同一秒内的多次触碰退化为插入顺序（验收要求按 ``last_touched DESC``）。
        为抵消平台时钟粒度（见 :func:`_strictly_newer`），同刻写入会被推后 1 微秒，
        保证会话内严格递增。
        """
        ts = now_iso("microseconds")
        with self._tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(id, started_at, status, turn_count) "
                "VALUES (?,?, 'active', 0)",
                (session_id, ts),
            )
            newest = conn.execute(
                "SELECT MAX(last_touched) AS m FROM working_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            ts = _strictly_newer(ts, newest["m"] if newest else None)
            row = conn.execute(
                "SELECT id, act_count FROM working_memory WHERE session_id = ? AND chunk_key = ?",
                (session_id, chunk_key),
            ).fetchone()
            if row is not None:
                chunk_id = row["id"]
                conn.execute(
                    "UPDATE working_memory SET act_count = act_count + 1, "
                    "salience = ?, last_touched = ?, content = ? WHERE id = ?",
                    (float(salience), ts, content, chunk_id),
                )
                return chunk_id

            chunk_id = _ids.new_working_id()
            conn.execute(
                """INSERT INTO working_memory
                   (id, session_id, chunk_key, content, salience, act_count,
                    created_at, last_touched)
                   VALUES (?,?,?,?,?,1,?,?)""",
                (chunk_id, session_id, chunk_key, content, float(salience), ts, ts),
            )
            return chunk_id

    def wm_list(self, session_id: str, limit: int = 9) -> list[WorkingChunk]:
        rows = self._execute(
            "SELECT * FROM working_memory WHERE session_id = ? "
            "ORDER BY last_touched DESC, rowid DESC LIMIT ?",
            (session_id, int(limit)),
        ).fetchall()
        return [
            WorkingChunk(
                id=r["id"],
                session_id=r["session_id"],
                chunk_key=r["chunk_key"],
                content=r["content"],
                salience=float(r["salience"]),
                act_count=int(r["act_count"]),
                created_at=r["created_at"],
                last_touched=r["last_touched"],
            )
            for r in rows
        ]

    def wm_clear(self, session_id: str) -> None:
        """仅清当前会话的工作记忆。"""
        self._write("DELETE FROM working_memory WHERE session_id = ?", (session_id,))

    def intent_put(
        self, content: str, session_id: str | None = None, due_at: str | None = None
    ) -> str:
        intent_id = _ids.new_intent_id()
        self._write(
            """INSERT INTO intents(id, session_id, content, status, due_at, created_at)
               VALUES (?,?,?, 'open', ?, ?)""",
            (intent_id, session_id, content, due_at, now_iso()),
        )
        return intent_id

    def intent_list(self, status: str = "open") -> list[dict]:
        rows = self._execute(
            "SELECT * FROM intents WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ================================================================== #
    # 分级加载（T-AL3-13）
    # ================================================================== #

    def overview_get(
        self, scope_kind: str, scope_id: str, level: str = "L1"
    ) -> OverviewRecord | None:
        row = self._execute(
            "SELECT * FROM overviews WHERE scope_kind=? AND scope_id=? AND level=?",
            (scope_kind, scope_id, level),
        ).fetchone()
        return self._row_to_overview(row) if row else None

    def overview_put(
        self,
        scope_kind: str,
        scope_id: str,
        level: str,
        content: str,
        *,
        token_count: int,
        model: str,
    ) -> None:
        """写入/刷新概览缓存，并**清除 stale 标记**。"""
        ts = now_iso()
        self._write(
            """INSERT INTO overviews(id, scope_kind, scope_id, level, content,
                                     token_count, generated_at, stale, model)
               VALUES (?,?,?,?,?,?,?,0,?)
               ON CONFLICT(scope_kind, scope_id, level) DO UPDATE SET
                 content = excluded.content,
                 token_count = excluded.token_count,
                 generated_at = excluded.generated_at,
                 stale = 0,
                 model = excluded.model""",
            (
                _ids.new_overview_id(),
                scope_kind,
                scope_id,
                level,
                content,
                int(token_count),
                ts,
                model,
            ),
        )

    def overview_invalidate(
        self, *, scope_kind: str | None = None, scope_id: str | None = None
    ) -> int:
        """**标记式**失效（O(1) 置 ``stale=1``），不触发重算。返回受影响行数。"""
        where: list[str] = []
        params: list[Any] = []
        if scope_kind is not None:
            where.append("scope_kind = ?")
            params.append(scope_kind)
        if scope_id is not None:
            where.append("scope_id = ?")
            params.append(scope_id)

        sql = "UPDATE overviews SET stale = 1"
        if where:
            sql += " WHERE " + " AND ".join(where)
        cursor = self._write(sql, tuple(params))
        return int(cursor.rowcount or 0)

    def overview_stale_list(self, *, limit: int = 50) -> list[OverviewRecord]:
        """待重算的概览（AL5 maintenance 的输入）。"""
        rows = self._execute(
            "SELECT * FROM overviews WHERE stale = 1 "
            "ORDER BY generated_at ASC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [self._row_to_overview(r) for r in rows]

    def overview_list(self, *, limit: int | None = None) -> list[OverviewRecord]:
        """全部概览缓存（可观测性与"可达性"判定用）。"""
        sql = "SELECT * FROM overviews ORDER BY scope_kind, scope_id"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        rows = self._execute(sql, params).fetchall()
        return [self._row_to_overview(r) for r in rows]

    # ================================================================== #
    # 内容寻址（T-AL3-14）
    # ================================================================== #

    def find_by_hash(self, content_hash: str) -> list[MemoryRecord]:
        """按指纹查找。**可能返回多条**——索引非唯一（同事实不同情境允许共存）。"""
        rows = self._execute(
            f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE content_hash = ?",
            (content_hash,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def content_hash_of_record(self, rec: MemoryRecord) -> str:
        return content_hash_of(
            content=rec.content,
            subject=rec.subject,
            predicate=rec.predicate,
            object_=rec.object,
            scope=rec.scope,
        )

    # ================================================================== #
    # 删除与恢复（T-AL3-18 / D-22）
    # ================================================================== #

    def hard_delete(
        self,
        mem_id: str,
        *,
        reason: str,
        purge_snapshot: bool = False,
        actor: str = "user",
        source: str = "unknown",
    ) -> None:
        """物理删除记忆，**保留可恢复快照**（D-22）。

        - ``reason`` 必填——删除请求本身必须可审计（INV-7）
        - 快照写入 ``delete_snapshots``（内容）与 ``audit``（账本），**与删除同事务**
        - ``purge_snapshot=True`` 仅用于**合规删除**：不留内容快照，且清除该条历史快照
          ——这是**唯一不可恢复**的删除（因为要求"数据真的不存在"）
        """
        if not reason or not reason.strip():
            raise WhitelistViolation(
                "物理删除必须给出 reason——删除请求本身要可审计（INV-7 / D-22）"
            )

        rec = self.get(mem_id)
        if rec is None:
            raise NotFoundError(f"记忆不存在：{mem_id}")

        relations = self.relations_of(mem_id)
        light_snapshot = {
            k: v
            for k, v in asdict(rec).items()
            if k not in ("content", "abstract")
        }

        with self._tx() as conn:
            audit_id = self._insert_audit(
                conn,
                AuditEvent(
                    op="forget",
                    actor=actor,
                    target_kind="memory",
                    target_id=mem_id,
                    before=light_snapshot,
                    reason=reason,
                ),
            )
            if purge_snapshot:
                # 合规：不留内容快照，并清除该条已有的历史快照
                conn.execute("DELETE FROM delete_snapshots WHERE mem_id = ?", (mem_id,))
            else:
                payload = json.dumps(
                    {"record": asdict(rec), "relations": relations},
                    ensure_ascii=False,
                )
                conn.execute(
                    "INSERT OR REPLACE INTO delete_snapshots"
                    "(audit_id, mem_id, payload, created_at) VALUES (?,?,?,?)",
                    (audit_id, mem_id, payload, now_iso()),
                )

            conn.execute(
                "DELETE FROM relations WHERE (src_kind='memory' AND src_id=?) "
                "OR (dst_kind='memory' AND dst_id=?)",
                (mem_id, mem_id),
            )
            conn.execute("DELETE FROM vec_memories WHERE mem_id = ?", (mem_id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))

    def restore_from_audit(self, audit_id: int) -> str:
        """从删除快照恢复记忆（D-22 安全网）。

        恢复**沿用原 ID**（避免关联断裂）；原 ID 已被占用时新建。
        恢复本身写 ``audit(op='restore')``，形成完整链条。

        **向量不随快照**——恢复后由 ``reindex`` / ``reconcile`` 补算（向量是可重建的派生数据）。
        """
        row = self._execute(
            "SELECT * FROM delete_snapshots WHERE audit_id = ?", (audit_id,)
        ).fetchone()
        if row is None:
            event = self.audit_get(audit_id)
            if event is not None and event.op == "forget":
                raise NotFoundError(
                    f"审计 {audit_id} 的快照已被清除（合规删除），不可恢复"
                )
            raise NotFoundError(f"没有可用于恢复的删除快照：audit_id={audit_id}")

        payload = json.loads(row["payload"])
        record = MemoryRecord(**self._decode_record_dict(payload["record"]))
        relations: list[dict] = payload.get("relations", [])

        original_id = record.id
        new_id = original_id
        existing = self.get(original_id)
        if existing is not None:
            # **先判幂等，再考虑换 ID。**
            #
            # 原实现见到"原 ID 已被占用"就新建一条，于是**同一条快照恢复两次会产出
            # 两条记忆** —— 使用者在审计里发现不对、多点了一次恢复，记忆库就多了一份
            # 重复内容。而按设计约定，删除是可审计、可恢复的，那么"恢复"本身也必须是
            # 幂等的：重复请求不该改变结果。
            #
            # 判据是"这个 ID 上的记录是否**就是这条快照恢复出来的**"——依据是 restore
            # 事件里记录的 `restored_from_audit`（写审计时就带上了，见本函数末尾）。
            if self._was_restored_from(original_id, audit_id):
                return original_id  # 已经恢复过 → 原样返回，不再新建
            new_id = _ids.new_memory_id(record.layer)
            record.id = new_id

        ts = now_iso()
        record.updated_at = ts

        with self._tx() as conn:
            conn.execute(
                f"INSERT INTO memories ({_MEMORY_COLUMNS}) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                self._record_params(record),
            )
            for rel in relations:
                src_id = new_id if rel["src_id"] == original_id else rel["src_id"]
                dst_id = new_id if rel["dst_id"] == original_id else rel["dst_id"]
                conn.execute(
                    """INSERT OR IGNORE INTO relations
                       (src_kind, src_id, dst_kind, dst_id, rel_type, weight, co_count)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        rel["src_kind"],
                        src_id,
                        rel["dst_kind"],
                        dst_id,
                        rel["rel_type"],
                        rel["weight"],
                        rel["co_count"],
                    ),
                )
            self._insert_audit(
                conn,
                AuditEvent(
                    op="restore",
                    actor="user",
                    target_kind="memory",
                    target_id=new_id,
                    after={"restored_from_audit": audit_id, "relations": len(relations)},
                    reason=f"restore from audit {audit_id}",
                ),
            )
        return new_id

    def _was_restored_from(self, mem_id: str, audit_id: int) -> bool:
        """``mem_id`` 上的记录是否由 ``audit_id`` 这条快照恢复而来。

        供 :meth:`restore_from_audit` 做**幂等判定**：同一快照被恢复多次时，
        第二次应当原样返回、而不是再插一条。

        依据是 restore 事件里已经写下的 ``after.restored_from_audit``。
        取**最近一条** restore 事件：如果最近那次恢复不是来自这条快照，
        说明该 ID 当前的内容来自别的版本，此时新建才是对的。
        """
        row = self._execute(
            "SELECT after FROM audit WHERE op = 'restore' AND target_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (mem_id,),
        ).fetchone()
        if row is None or not row["after"]:
            return False
        try:
            payload = json.loads(row["after"])
        except (TypeError, ValueError):  # pragma: no cover - 坏审计不该让恢复崩
            return False
        return payload.get("restored_from_audit") == audit_id

    def delete_snapshots(self, *, include_payload: bool = False) -> list[dict]:
        """当前可恢复的删除快照清单（供 `aspirit audit` 展示）。

        ``include_payload=True`` 时额外带上被删记录的字段——审计视图要靠它显示
        "这次删掉的到底是什么内容"。

        为什么默认不带：payload 里是整条记录 + 关联边的 JSON，体积比清单本身大得多，
        而大多数调用方（`reconcile` / `doctor` / 状态统计）只关心"有几个、都是谁"。
        默认 False 让常规路径不付这份代价，需要内容的审计路径显式开口要。
        """
        columns = "audit_id, mem_id, created_at"
        if include_payload:
            columns += ", payload"
        rows = self._execute(
            f"SELECT {columns} FROM delete_snapshots ORDER BY created_at DESC"
        ).fetchall()

        out: list[dict] = []
        for row in rows:
            item = {
                "audit_id": row["audit_id"],
                "mem_id": row["mem_id"],
                "created_at": row["created_at"],
            }
            if include_payload:
                try:
                    blob = json.loads(row["payload"])
                except (TypeError, ValueError):  # pragma: no cover - 坏快照不该让审计视图崩
                    blob = {}
                # payload 的形状是 {"record": {...}, "relations": [...]}
                item["record"] = blob.get("record") or {}
                item["relation_count"] = len(blob.get("relations") or [])
            out.append(item)
        return out

    # ================================================================== #
    # 运维：重放 / 对账 / 重建（T-AL3-19 / T-AL3-21）
    # ================================================================== #

    def replay_derived(self) -> dict:
        """从 ``audit`` 重放重建派生字段（INV-12）。

        重建对象：``strength`` / ``access_count`` / ``last_access_at``。
        顺序严格按 ``(ts, id)``——与 :meth:`audit_replay` 一致。
        """
        updated = 0
        missing = 0
        # **先把事件读完再开事务**：事务跑在写连接上，而 audit_replay 走读连接；
        # 边读边写会看到不一致的快照。
        events = list(self.audit_replay())
        with self._tx() as conn:
            conn.execute(
                "UPDATE memories SET strength = 1.0, access_count = 0, "
                "last_access_at = NULL"
            )
            for event in events:
                if event.op == "add" and event.target_id:
                    cursor = conn.execute(
                        "UPDATE memories SET strength = 1.0, access_count = 0, "
                        "last_access_at = NULL WHERE id = ?",
                        (event.target_id,),
                    )
                    updated += cursor.rowcount or 0
                elif event.op == "touch" and event.target_id and event.after:
                    cursor = conn.execute(
                        "UPDATE memories SET strength = ?, access_count = ?, "
                        "last_access_at = ? WHERE id = ?",
                        (
                            event.after.get("strength", 1.0),
                            event.after.get("access_count", 0),
                            event.after.get("last_access_at"),
                            event.target_id,
                        ),
                    )
                    if cursor.rowcount == 0:
                        missing += 1
                    else:
                        updated += cursor.rowcount
        return {"updated": updated, "orphan_events": missing}

    def reconcile(
        self, *, dry_run: bool = True, embed_fn: Callable[[list[str]], list[list[float]]] | None = None
    ) -> dict:
        """启动对账：修复 ``memories`` 与 ``vec_memories`` 的不一致。

        - **有记忆无向量** → 有 ``embed_fn`` 则补算，否则只报告（需 reindex）
        - **孤儿向量** → 清理
        - ``dry_run=True`` 只报告，不产生任何写操作
        """
        missing_rows = self._execute(
            """SELECT id, content, abstract FROM memories
               WHERE status != 'forgotten'
                 AND id NOT IN (SELECT mem_id FROM vec_memories)"""
        ).fetchall()
        orphan_rows = self._execute(
            """SELECT mem_id FROM vec_memories
               WHERE mem_id NOT IN (SELECT id FROM memories)"""
        ).fetchall()

        report = {
            "missing_vectors": [r["id"] for r in missing_rows],
            "orphan_vectors": [r["mem_id"] for r in orphan_rows],
            "repaired": 0,
            "dry_run": dry_run,
        }
        if dry_run:
            return report

        if orphan_rows:
            with self._tx() as conn:
                for row in orphan_rows:
                    conn.execute(
                        "DELETE FROM vec_memories WHERE mem_id = ?", (row["mem_id"],)
                    )
                self._insert_audit(
                    conn,
                    AuditEvent(
                        op="reconcile",
                        actor="system",
                        target_kind="system",
                        target_id="vec_memories",
                        after={"orphans_removed": len(orphan_rows)},
                    ),
                )

        if missing_rows and embed_fn is not None:
            texts = [r["abstract"] or r["content"] for r in missing_rows]
            vectors = embed_fn(texts)
            with self._tx() as conn:
                for row, vec in zip(missing_rows, vectors, strict=False):
                    self._check_dim(vec)
                    conn.execute(
                        "INSERT OR REPLACE INTO vec_memories(mem_id, embedding) "
                        "VALUES (?, ?)",
                        (row["id"], json.dumps(vec)),
                    )
                    report["repaired"] += 1

        self.meta_set(META_LAST_RECONCILE, now_iso())
        return report

    def rebuild_fts(self) -> int:
        """从 ``memories`` 重建全文索引（FTS 是派生索引，INV-1）。

        注意：本表是**普通 FTS5 表**（非 contentless / 非外部内容表），
        因此用 ``DELETE FROM`` 清空；``'delete-all'`` 命令对这类表不适用。
        """
        with self._tx() as conn:
            conn.execute("DELETE FROM mem_fts")
            cursor = conn.execute(
                """INSERT INTO mem_fts(rowid, content, abstract, subject, object)
                   SELECT rowid, spirit_norm(content), spirit_norm(abstract),
                          spirit_norm(subject), spirit_norm(object) FROM memories"""
            )
            return int(cursor.rowcount or 0)

    # ================================================================== #
    # 传承（T-AL3-15 / T-AL3-22）——委托给 archive 模块
    # ================================================================== #

    def archive_version(self) -> int:
        from .archive import ARCHIVE_VERSION

        return ARCHIVE_VERSION

    def export_archive(self, path: str, *, fmt: str = "markdown") -> None:
        from . import archive as _archive

        _archive.export_archive(self, path, fmt=fmt)

    def export(self, *, fmt: str = "markdown") -> str:
        """状态投影（纯函数：同库状态同输出）。"""
        from . import archive as _archive

        return _archive.render_archive(self, fmt=fmt)

    def export_pack(self) -> bytes:
        from . import archive as _archive

        return _archive.export_pack(self)

    def import_pack(self, data: bytes, *, merge: bool = False) -> dict:
        from . import archive as _archive

        return _archive.import_pack(self, data, merge=merge)

    def import_archive(self, path: str) -> dict:
        from . import archive as _archive

        return _archive.import_archive(self, path)

    # ================================================================== #
    # 内部：行 → 对象
    # ================================================================== #

    def _check_dim(self, vec: list[float]) -> None:
        """写入前校验维度，不符则**拒绝**（不得静默截断）。"""
        if len(vec) != self.embedding_dim:
            raise DimensionMismatchError(
                f"向量维度不符：期望 {self.embedding_dim}，收到 {len(vec)}。"
                "拒绝写入——不得静默截断（C6 / INV-2）。"
            )

    def assert_embedding_model(self, model: str | None) -> str | None:
        """校验配置的 embedding 模型与库中记录一致（INV-2 / F7）。

        不一致时**拒绝启动**并提示 ``reindex``——换模型必须全库重嵌入，
        否则写入与检索的向量空间不一致（铁律：写入与检索必须同一模型）。
        """
        stored = self.meta_get(META_EMBEDDING_MODEL)
        if stored and model and stored != model:
            raise SchemaVersionError(
                f"embedding 模型不一致：库中为 {stored!r}，配置为 {model!r}。"
                "换模型会导致写入与检索的向量空间不一致，请先执行 aspirit reindex。"
            )
        return stored

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(**self._decode_record_dict(dict(row)))

    @staticmethod
    def _decode_record_dict(data: dict) -> dict:
        scope = data.get("scope")
        if isinstance(scope, str) and scope:
            try:
                data["scope"] = json.loads(scope)
            except json.JSONDecodeError:  # pragma: no cover
                data["scope"] = None
        return {k: v for k, v in data.items() if k in _RECORD_FIELDS}

    def _row_to_hit(self, row: sqlite3.Row, *, score: float) -> Hit:
        record = self._row_to_record(row)
        meta = {"record": asdict(record)}
        return Hit(
            mem_id=record.id,
            layer=record.layer,
            content=record.content,
            score=float(score),
            meta=meta,
        )

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> EntityRecord:
        aliases = row["aliases"]
        if isinstance(aliases, str) and aliases:
            try:
                aliases = json.loads(aliases)
            except json.JSONDecodeError:  # pragma: no cover
                aliases = []
        return EntityRecord(
            id=row["id"],
            name=row["name"],
            type=row["type"] or "",
            aliases=list(aliases or []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_overview(row: sqlite3.Row) -> OverviewRecord:
        return OverviewRecord(
            id=row["id"],
            scope_kind=row["scope_kind"],
            scope_id=row["scope_id"],
            level=row["level"],
            content=row["content"],
            token_count=row["token_count"],
            generated_at=row["generated_at"],
            stale=bool(row["stale"]),
            model=row["model"],
        )

    @staticmethod
    def _row_to_audit(row: sqlite3.Row) -> AuditEvent:
        def _load(value: str | None) -> dict | None:
            if not value:
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError:  # pragma: no cover
                return {"_raw": value}

        return AuditEvent(
            op=row["op"],
            actor=row["actor"],
            target_kind=row["target_kind"],
            target_id=row["target_id"],
            before=_load(row["before"]),
            after=_load(row["after"]),
            reason=row["reason"],
            session_id=row["session_id"],
            ts=row["ts"],
            # 必须带上真实主键：`restore_from_audit` 认的是它。审计视图若拿不到它，
            # 就只好用"过滤结果里的第几条"凑数——而 `--since` 一过滤就错位，
            # 用户照着一个错位的编号去恢复，会恢复错东西。
            audit_id=row["id"],
        )

    @staticmethod
    def _record_params(rec: MemoryRecord) -> tuple:
        return (
            rec.id,
            rec.layer,
            rec.type,
            rec.subject,
            rec.predicate,
            rec.object,
            rec.content,
            rec.abstract,
            json.dumps(rec.scope, ensure_ascii=False) if rec.scope is not None else None,
            float(rec.confidence),
            float(rec.salience),
            float(rec.strength),
            int(rec.access_count),
            rec.last_access_at,
            rec.valid_from,
            rec.valid_to,
            rec.source_session,
            rec.source_turn,
            rec.status,
            rec.superseded_by,
            rec.created_at,
            rec.updated_at,
            rec.embedding_model,
            rec.content_hash,
        )

    # ================================================================== #
    # 内部：嵌入元信息
    # ================================================================== #

    def _record_embedding_meta(self, conn: sqlite3.Connection) -> None:
        """记录 embedding 模型与维度（INV-2 的比对基据）。"""
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (META_EMBEDDING_DIM,)
        ).fetchone()
        if row is not None:
            existing_dim = int(row["value"])
            if existing_dim != self.embedding_dim:
                raise SchemaVersionError(
                    f"向量维度不一致：库中为 {existing_dim}，配置为 "
                    f"{self.embedding_dim}。请执行 aspirit reindex 重建索引。"
                )

        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            (META_EMBEDDING_DIM, str(self.embedding_dim)),
        )
        if self.embedding_model:
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                (META_EMBEDDING_MODEL, self.embedding_model),
            )


_RECORD_FIELDS = frozenset(MemoryRecord.__dataclass_fields__)


# --------------------------------------------------------------------------- #
# 迁移注册表（版本 → 应用函数）
# --------------------------------------------------------------------------- #

_MIGRATIONS: dict[int, Callable[[SQLiteBackend, sqlite3.Connection], None]] = {
    1: SQLiteBackend._apply_v1,
    2: SQLiteBackend._apply_v2,
}
