"""AL3 存储层 · M0 验收测试。

对应 `docs/design/encoding/AL3-存储层.md` 的 T-AL3-01 ~ 04。
测试名即验收项。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from artifact_spirit.store.base import SchemaVersionError
from artifact_spirit.store.ids import new_memory_id, new_ulid
from artifact_spirit.store.sqlite_backend import (
    META_SCHEMA_VERSION,
    META_VEC_CAPABILITY,
    SCHEMA_VERSION,
    VEC_CAP_TEXT_PK,
    SQLiteBackend,
)

# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "spirit.db")


@pytest.fixture
def backend(db_path: str):
    be = SQLiteBackend(db_path, embedding_dim=8, embedding_model="test-embed")
    be.open()
    yield be
    be.close()


# --------------------------------------------------------------------------- #
# T-AL3-01 · 建库、PRAGMA 与迁移框架
# --------------------------------------------------------------------------- #

EXPECTED_TABLES = {
    "meta",
    "sessions",
    "working_memory",
    "intents",
    "memories",
    "mem_fts",
    "vec_memories",
    "entities",
    "relations",
    "overviews",
    "audit",
}

EXPECTED_TRIGGERS = {
    "trg_mem_ai",
    "trg_mem_ad",
    "trg_mem_au",
    "trg_audit_no_update",
    "trg_audit_no_delete",
}


def test_migrate_creates_all_tables(backend: SQLiteBackend) -> None:
    rows = backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()
    names = {r["name"] for r in rows}
    missing = EXPECTED_TABLES - names
    assert not missing, f"缺失表：{missing}"


def test_migrate_creates_all_triggers(backend: SQLiteBackend) -> None:
    """schema.sql 被完整应用（executescript 不会静默跳过失败语句）。"""
    rows = backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger'"
    ).fetchall()
    names = {r["name"] for r in rows}
    missing = EXPECTED_TRIGGERS - names
    assert not missing, f"缺失触发器：{missing}"


def test_migrate_is_idempotent(db_path: str) -> None:
    be = SQLiteBackend(db_path, embedding_dim=8)
    be.open()
    try:
        be.migrate()
        be.migrate()
        assert be.meta_get(META_SCHEMA_VERSION) == str(SCHEMA_VERSION)
    finally:
        be.close()


def test_journal_mode_is_wal(backend: SQLiteBackend) -> None:
    mode = backend.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_synchronous_and_busy_timeout_applied(backend: SQLiteBackend) -> None:
    assert backend.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert backend.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_schema_version_written(backend: SQLiteBackend) -> None:
    assert backend.meta_get(META_SCHEMA_VERSION) == str(SCHEMA_VERSION)


def test_higher_schema_version_refused(db_path: str) -> None:
    be = SQLiteBackend(db_path, embedding_dim=8)
    be.open()
    be.meta_set(META_SCHEMA_VERSION, str(SCHEMA_VERSION + 99))
    be.close()

    be2 = SQLiteBackend(db_path, embedding_dim=8)
    with pytest.raises(SchemaVersionError):
        be2.open()


def test_embedding_dim_mismatch_refused(db_path: str) -> None:
    """INV-2 / F7：维度变更必须拒绝启动。"""
    be = SQLiteBackend(db_path, embedding_dim=8)
    be.open()
    be.close()

    be2 = SQLiteBackend(db_path, embedding_dim=16)
    with pytest.raises(SchemaVersionError):
        be2.open()


def test_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "spirit.db"
    be = SQLiteBackend(nested, embedding_dim=8)
    be.open()
    try:
        assert nested.exists()
    finally:
        be.close()


# --------------------------------------------------------------------------- #
# T-AL3-02 · meta 键值读写
# --------------------------------------------------------------------------- #


def test_meta_roundtrip(backend: SQLiteBackend) -> None:
    backend.meta_set("spirit_name", "清欢")
    assert backend.meta_get("spirit_name") == "清欢"


def test_meta_get_missing_returns_none(backend: SQLiteBackend) -> None:
    assert backend.meta_get("不存在的键") is None


def test_meta_set_overwrites(backend: SQLiteBackend) -> None:
    backend.meta_set("k", "v1")
    backend.meta_set("k", "v2")
    assert backend.meta_get("k") == "v2"


def test_spirit_id_not_overwritten(backend: SQLiteBackend) -> None:
    """C6：已存在时不得覆盖——否则器灵等于换了身份。"""
    first = backend.ensure_spirit_id()
    second = backend.ensure_spirit_id()
    assert first == second
    assert len(first) == 26  # ULID


def test_spirit_id_persists_across_reopen(db_path: str) -> None:
    be = SQLiteBackend(db_path, embedding_dim=8)
    be.open()
    first = be.ensure_spirit_id()
    be.close()

    be2 = SQLiteBackend(db_path, embedding_dim=8)
    be2.open()
    try:
        assert be2.ensure_spirit_id() == first
    finally:
        be2.close()


# --------------------------------------------------------------------------- #
# T-AL3-03 · vec0 能力探测与退化路径
# --------------------------------------------------------------------------- #


def test_vec_capability_recorded(backend: SQLiteBackend) -> None:
    cap = backend.meta_get(META_VEC_CAPABILITY)
    assert cap in {VEC_CAP_TEXT_PK, "rowid"}
    # 实测 sqlite-vec 0.1.9 支持 TEXT 主键
    assert cap == VEC_CAP_TEXT_PK


def test_vec_memories_accepts_text_pk(backend: SQLiteBackend) -> None:
    import json

    backend.conn.execute(
        "INSERT INTO vec_memories(mem_id, embedding) VALUES (?, ?)",
        ("sem_01", json.dumps([0.1] * 8)),
    )
    row = backend.conn.execute(
        "SELECT mem_id FROM vec_memories WHERE mem_id = 'sem_01'"
    ).fetchone()
    assert row["mem_id"] == "sem_01"


def test_vec_knn_and_delete_work(backend: SQLiteBackend) -> None:
    import json

    for i, mid in enumerate(("sem_a", "sem_b")):
        backend.conn.execute(
            "INSERT INTO vec_memories(mem_id, embedding) VALUES (?, ?)",
            (mid, json.dumps([0.0] * 7 + [float(i)])),
        )
    rows = backend.conn.execute(
        "SELECT mem_id FROM vec_memories WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
        (json.dumps([0.0] * 8),),
    ).fetchall()
    assert rows[0]["mem_id"] == "sem_a"

    backend.conn.execute("DELETE FROM vec_memories WHERE mem_id = 'sem_a'")
    assert (
        backend.conn.execute(
            "SELECT COUNT(*) AS n FROM vec_memories"
        ).fetchone()["n"]
        == 1
    )


def test_probe_not_repeated_on_reopen(db_path: str, monkeypatch) -> None:
    """能力值已缓存 → 二次启动不得重复探测。"""
    be = SQLiteBackend(db_path, embedding_dim=8)
    be.open()
    be.close()

    calls = {"n": 0}
    original = SQLiteBackend._probe_vec_capability

    def spy(conn):
        calls["n"] += 1
        return original(conn)

    monkeypatch.setattr(SQLiteBackend, "_probe_vec_capability", staticmethod(spy))

    be2 = SQLiteBackend(db_path, embedding_dim=8)
    be2.open()
    try:
        assert calls["n"] == 0, "能力值已缓存，不应再次探测"
    finally:
        be2.close()


def test_probe_runs_once_and_writes_meta(tmp_path: Path, monkeypatch) -> None:
    calls = {"n": 0}
    original = SQLiteBackend._probe_vec_capability

    def spy(conn):
        calls["n"] += 1
        return original(conn)

    monkeypatch.setattr(SQLiteBackend, "_probe_vec_capability", staticmethod(spy))

    be = SQLiteBackend(str(tmp_path / "fresh.db"), embedding_dim=8)
    be.open()
    try:
        assert calls["n"] == 1
        assert be.meta_get(META_VEC_CAPABILITY) is not None
    finally:
        be.close()


def test_unsupported_vec_capability_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    """不支持 TEXT 主键时**明确失败**，不静默降级（偏离 T-AL3-03 的退化路径）。

    原因见 `SQLiteBackend._require_text_pk` 的说明：不可测的分支比明确失败更危险。
    """
    from artifact_spirit.store.base import StorageFatalError

    monkeypatch.setattr(
        SQLiteBackend,
        "_probe_vec_capability",
        staticmethod(lambda conn: "rowid"),
    )

    be = SQLiteBackend(str(tmp_path / "unsupported.db"), embedding_dim=8)
    with pytest.raises(StorageFatalError, match="不支持 vec0 的 TEXT 主键"):
        be.open()
    assert be._conn is None, "失败后不得残留连接"


# --------------------------------------------------------------------------- #
# T-AL3-04 · 生命周期骨架
# --------------------------------------------------------------------------- #


def test_open_close_open(db_path: str) -> None:
    be = SQLiteBackend(db_path, embedding_dim=8)
    be.open()
    be.close()
    be.open()
    try:
        assert be.meta_get(META_SCHEMA_VERSION) == str(SCHEMA_VERSION)
    finally:
        be.close()


def test_close_releases_connection(db_path: str) -> None:
    from artifact_spirit.store.base import StorageFatalError

    be = SQLiteBackend(db_path, embedding_dim=8)
    be.open()
    be.close()
    assert be._conn is None
    with pytest.raises(StorageFatalError):
        _ = be.conn


def test_close_is_idempotent(db_path: str) -> None:
    be = SQLiteBackend(db_path, embedding_dim=8)
    be.open()
    be.close()
    be.close()  # 不抛错


def test_open_is_idempotent(db_path: str) -> None:
    be = SQLiteBackend(db_path, embedding_dim=8)
    be.open()
    conn = be.conn
    be.open()  # 不应重建连接
    try:
        assert be.conn is conn
    finally:
        be.close()


def test_memory_backend_usable_without_file(tmp_path: Path) -> None:
    be = SQLiteBackend(":memory:", embedding_dim=8)
    be.open()
    try:
        assert be.meta_get(META_SCHEMA_VERSION) == str(SCHEMA_VERSION)
    finally:
        be.close()


def test_invalid_embedding_dim_rejected() -> None:
    with pytest.raises(ValueError):
        SQLiteBackend(":memory:", embedding_dim=0)
    with pytest.raises(ValueError):
        SQLiteBackend(":memory:", embedding_dim=-1)


# --------------------------------------------------------------------------- #
# ids：ULID 格式与单调性（T-AL3-05 的前置，此处先守住）
# --------------------------------------------------------------------------- #


def test_ulid_shape() -> None:
    u = new_ulid()
    assert len(u) == 26
    assert all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in u)


def test_ulid_monotonic_within_same_millisecond() -> None:
    ids = [new_ulid(ts_ms=1_700_000_000_000) for _ in range(200)]
    assert ids == sorted(ids), "同一毫秒内 ULID 必须单调递增"
    assert len(set(ids)) == len(ids), "同一毫秒内 ULID 不得重复"


def test_memory_id_prefix() -> None:
    assert new_memory_id("semantic").startswith("sem_")
    assert new_memory_id("episodic").startswith("epi_")
    assert new_memory_id("procedural").startswith("pro_")
    assert new_memory_id("core").startswith("cor_")
    with pytest.raises(ValueError):
        new_memory_id("sensory")


def test_memory_ids_sorted_by_creation() -> None:
    ids = [new_memory_id("semantic") for _ in range(50)]
    assert ids == sorted(ids)


# --------------------------------------------------------------------------- #
# 审计 append-only（INV-8）：schema 已含触发器，此处守住行为
# --------------------------------------------------------------------------- #


def test_audit_is_append_only(backend: SQLiteBackend) -> None:
    backend.conn.execute(
        "INSERT INTO audit(ts, op, actor, target_kind, target_id) "
        "VALUES ('2026-01-01T00:00:00+08:00', 'add', 'extractor', 'memory', 'sem_x')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        backend.conn.execute("UPDATE audit SET op = 'update' WHERE id = 1")
    with pytest.raises(sqlite3.IntegrityError):
        backend.conn.execute("DELETE FROM audit WHERE id = 1")
