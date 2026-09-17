"""AL3 存储层 · M1 验收（T-AL3-05 ~ 16）。

验收项即测试名（ENC-000 §3.4）。
"""

from __future__ import annotations

import sqlite3

import pytest
from conftest import make_record, vec

from artifact_spirit.store import (
    AuditEvent,
    DimensionMismatchError,
    NotFoundError,
    SQLiteBackend,
)
from artifact_spirit.store.text import content_hash_of

# --------------------------------------------------------------------------- #
# T-AL3-05 memories CRUD
# --------------------------------------------------------------------------- #


def test_memory_id_format_is_abbr_ulid(backend):
    rec = make_record(layer="episodic", type="event", content="一次会话")
    backend.put(rec)
    assert rec.id.startswith("epi_")
    assert len(rec.id) == len("epi_") + 26
    assert rec.id[4:].isalnum()


def test_ulid_monotonic_within_same_millisecond(backend):
    ids = [make_record(content=f"c{i}").id for i in range(5)]
    ids = []
    for i in range(5):
        rec = make_record(content=f"c{i}")
        backend.put(rec)
        ids.append(rec.id)
    assert ids == sorted(ids), "同毫秒内 ULID 必须单调递增"


def test_put_then_get_roundtrip_all_fields(backend):
    rec = make_record(
        id="",
        layer="semantic",
        type="fact",
        content="项目 RAGFlow 使用 Python 3.12",
        subject="RAGFlow",
        predicate="uses_python_version",
        object="3.12",
        abstract="RAGFlow 用 Python 3.12",
        scope={"type": "project", "id": "ragflow"},
        confidence=0.98,
        salience=0.7,
        source_session="s1",
        source_turn=3,
        valid_from="2026-01-01T00:00:00+08:00",
    )
    backend.put(rec, vec(1))
    got = backend.get(rec.id)
    assert got is not None
    assert got.content == rec.content
    assert got.subject == "RAGFlow"
    assert got.predicate == "uses_python_version"
    assert got.object == "3.12"
    assert got.abstract == "RAGFlow 用 Python 3.12"
    assert got.scope == {"type": "project", "id": "ragflow"}
    assert got.confidence == pytest.approx(0.98)
    assert got.salience == pytest.approx(0.7)
    assert got.source_session == "s1"
    assert got.source_turn == 3
    assert got.valid_from == "2026-01-01T00:00:00+08:00"
    assert got.status == "active"


def test_query_filters_by_status(backend):
    a = make_record(content="活跃的")
    backend.put(a)
    b = make_record(content="休眠的")
    backend.put(b)
    backend.set_status(b.id, "dormant", reason="测试")
    active = backend.query(status="active")
    assert [r.id for r in active] == [a.id]
    assert {r.id for r in backend.query(status=None)} == {a.id, b.id}
    assert [r.id for r in backend.query(status="dormant")] == [b.id]


def test_query_filters_by_types_and_layer(backend):
    backend.put(make_record(layer="semantic", type="fact", content="事实"))
    backend.put(make_record(layer="semantic", type="preference", content="偏好"))
    backend.put(make_record(layer="episodic", type="event", content="事件"))
    assert len(backend.query(types=["fact"])) == 1
    assert len(backend.query(types=["fact", "preference"])) == 2
    assert len(backend.query(layer="episodic")) == 1


def test_query_since_until_filters(backend):
    backend.put(make_record(content="早", created_at="2026-01-01T00:00:00+08:00"))
    backend.put(make_record(content="晚", created_at="2026-06-01T00:00:00+08:00"))
    assert len(backend.query(since="2026-03-01T00:00:00+08:00")) == 1
    assert len(backend.query(until="2026-03-01T00:00:00+08:00")) == 1


def test_query_limit(backend):
    for i in range(5):
        backend.put(make_record(content=f"c{i}"))
    assert len(backend.query(limit=2)) == 2


def test_count_by_layer(backend):
    backend.put(make_record(layer="semantic", type="fact", content="a"))
    backend.put(make_record(layer="episodic", type="event", content="b"))
    counts = backend.count_by_layer()
    assert counts["semantic"]["active"] == 1
    assert counts["episodic"]["active"] == 1


def test_update_patch_and_updated_at(backend):
    rec = make_record(content="原文")
    backend.put(rec)
    before = backend.get(rec.id)
    backend.update(rec.id, {"content": "新文", "confidence": 0.5})
    after = backend.get(rec.id)
    assert after.content == "新文"
    assert after.confidence == pytest.approx(0.5)
    assert after.updated_at >= before.updated_at


def test_update_rejects_unknown_field(backend):
    rec = make_record()
    backend.put(rec)
    with pytest.raises(ValueError):
        backend.update(rec.id, {"evil_column": 1})


def test_update_missing_record_raises(backend):
    with pytest.raises(NotFoundError):
        backend.update("sem_不存在", {"content": "x"})


# --------------------------------------------------------------------------- #
# T-AL3-06 FTS5 触发器
# --------------------------------------------------------------------------- #


def test_fts_indexes_on_insert(backend):
    backend.put(make_record(content="项目使用 PostgreSQL 数据库"))
    assert len(backend.keyword_search("PostgreSQL")) == 1


def test_fts_syncs_on_update(backend):
    rec = make_record(content="项目使用 Python 3.12")
    backend.put(rec)
    backend.update(rec.id, {"content": "项目已迁移到 Python 3.13"})
    assert backend.keyword_search("3.12") == []
    assert len(backend.keyword_search("3.13")) == 1


def test_fts_clears_on_delete(backend):
    rec = make_record(content="唯一无二的关键词Xyzzy")
    backend.put(rec)
    assert len(backend.keyword_search("Xyzzy")) == 1
    backend.hard_delete(rec.id, reason="测试删除")
    assert backend.keyword_search("Xyzzy") == []


def test_fts_matches_both_chinese_and_english(backend):
    backend.put(make_record(content="项目 RAGFlow 使用 Python 3.12"))
    backend.put(make_record(content="用户偏好深色主题"))
    assert len(backend.keyword_search("RAGFlow")) == 1
    assert len(backend.keyword_search("Python")) == 1
    # 2 字中文词是关键场景——FTS5 的 trigram 分词器在这里会失败
    assert len(backend.keyword_search("偏好")) == 1
    assert len(backend.keyword_search("深色主题")) == 1


def test_fts_search_ignores_forgotten(backend):
    rec = make_record(content="将被删除的内容")
    backend.put(rec)
    backend.set_status(rec.id, "forgotten", reason="测试")
    assert backend.keyword_search("删除的内容") == []


def test_fts_search_includes_dormant(backend):
    """dormant 仍可被**显式搜索**命中（D-23）。"""
    rec = make_record(content="休眠但仍可搜到的内容")
    backend.put(rec)
    backend.set_status(rec.id, "dormant", reason="测试")
    assert len(backend.keyword_search("休眠")) == 1


def test_rebuild_fts_restores_index(backend):
    backend.put(make_record(content="用户偏好深色主题"))
    backend.conn.execute("DELETE FROM mem_fts")
    assert backend.keyword_search("深色主题") == []
    backend.rebuild_fts()
    assert len(backend.keyword_search("深色主题")) == 1


# --------------------------------------------------------------------------- #
# T-AL3-07 向量读写与维度校验
# --------------------------------------------------------------------------- #


def test_vector_write_and_knn_order(backend):
    for i in range(3):
        backend.put(make_record(content=f"记录{i}"), vec(i + 1))
    hits = backend.vector_search(vec(1), top_k=3)
    assert hits[0].content == "记录0"
    assert hits[0].score >= hits[-1].score


def test_vector_dimension_mismatch_raises_and_no_half_write(backend):
    rec = make_record(content="维度错误的记忆")
    with pytest.raises(DimensionMismatchError):
        backend.put(rec, [0.1, 0.2, 0.3])
    # 未产生半写
    assert backend.get(rec.id) is None
    assert backend.query() == []


def test_vector_search_dimension_mismatch_raises(backend):
    with pytest.raises(DimensionMismatchError):
        backend.vector_search([0.1, 0.2])


def test_embedding_model_recorded_on_row(backend):
    rec = make_record(content="带模型的记忆")
    backend.put(rec, vec(1))
    assert backend.get(rec.id).embedding_model == "test-embed"


# --------------------------------------------------------------------------- #
# T-AL3-08 检索原语
# --------------------------------------------------------------------------- #


def test_search_returns_native_scores_not_fused(backend):
    backend.put(make_record(content="甲的记录"), vec(1))
    v = backend.vector_search(vec(1), top_k=1)[0]
    k = backend.keyword_search("甲的记录", top_k=1)[0]
    # 两路原生分不同量纲——AL3 不做融合
    assert v.score != k.score


def test_hit_fields_complete(backend):
    backend.put(make_record(layer="episodic", type="event", content="命中的内容"), vec(1))
    hit = backend.vector_search(vec(1), top_k=1)[0]
    assert hit.mem_id
    assert hit.layer == "episodic"
    assert hit.content == "命中的内容"
    assert isinstance(hit.score, float)
    assert "record" in hit.meta


def test_search_layer_filter(backend):
    backend.put(make_record(layer="semantic", type="fact", content="语义层的内容"), vec(1))
    backend.put(make_record(layer="episodic", type="event", content="情景层的内容"), vec(2))
    assert len(backend.vector_search(vec(1), layer="semantic", top_k=5)) == 1
    assert len(backend.keyword_search("内容", layer="episodic", top_k=5)) == 1


def test_keyword_search_empty_query_returns_empty(backend):
    backend.put(make_record(content="一些内容"))
    assert backend.keyword_search("") == []
    assert backend.keyword_search("   ") == []


# --------------------------------------------------------------------------- #
# T-AL3-09 relations
# --------------------------------------------------------------------------- #


def test_link_is_idempotent(backend):
    a = make_record(content="A")
    b = make_record(content="B")
    backend.put(a)
    backend.put(b)
    backend.link("memory", a.id, "memory", b.id, "co_activation", 0.3)
    backend.link("memory", a.id, "memory", b.id, "co_activation", 0.9)
    assert len(backend.relations_of(a.id)) == 1


def test_reinforce_grows_weight_and_count(backend):
    a = make_record(content="A")
    b = make_record(content="B")
    backend.put(a)
    backend.put(b)
    backend.link("memory", a.id, "memory", b.id, "co_activation", 0.2)
    backend.reinforce(a.id, b.id, 0.5)
    weight = backend.neighbors("memory", a.id)[0][2]
    # Hebbian：w ← w + η(1−w) = 0.2 + 0.5·0.8 = 0.6
    assert weight == pytest.approx(0.6)
    rel = backend.relations_of(a.id)[0]
    assert rel["co_count"] == 2


def test_reinforce_twice_approaches_one_but_bounded(backend):
    a = make_record(content="A")
    b = make_record(content="B")
    backend.put(a)
    backend.put(b)
    for _ in range(50):
        backend.reinforce(a.id, b.id, 0.5)
    weight = backend.neighbors("memory", a.id)[0][2]
    assert 0.0 < weight < 1.0
    assert weight > 0.99


def test_neighbors_sorted_by_weight_desc(backend):
    a = make_record(content="A")
    backend.put(a)
    for i, w in enumerate([0.1, 0.9, 0.5]):
        other = make_record(content=f"O{i}")
        backend.put(other)
        backend.link("memory", a.id, "memory", other.id, "co_activation", w)
    weights = [w for _, _, w in backend.neighbors("memory", a.id)]
    assert weights == sorted(weights, reverse=True)


def test_neighbors_min_weight_filter(backend):
    a = make_record(content="A")
    b = make_record(content="B")
    backend.put(a)
    backend.put(b)
    backend.link("memory", a.id, "memory", b.id, "co_activation", 0.2)
    assert backend.neighbors("memory", a.id, min_weight=0.5) == []


# --------------------------------------------------------------------------- #
# T-AL3-10 entities
# --------------------------------------------------------------------------- #


def test_entity_upsert_dedupes_same_name_type(backend):
    id1 = backend.entity_upsert("RAGFlow", "project", aliases=["ragflow"])
    id2 = backend.entity_upsert("RAGFlow", "project", aliases=["ragflow", "RAG"])
    assert id1 == id2
    assert len(backend.entity_list()) == 1


def test_entity_find_by_name_and_alias(backend):
    backend.entity_upsert("RAGFlow", "project", aliases=["ragflow"])
    assert [e.name for e in backend.entity_find("我在用 RAGFlow 做检索")] == ["RAGFlow"]
    assert [e.name for e in backend.entity_find("ragflow 不错")] == ["RAGFlow"]
    assert backend.entity_find("完全无关的句子") == []


def test_entity_find_requires_no_llm(backend):
    """实体匹配是纯字符串操作——不得触发任何模型调用（D-10 成本为零）。"""
    backend.entity_upsert("张三", "person")
    assert backend.entity_find("张三说了什么")[0].name == "张三"


# --------------------------------------------------------------------------- #
# T-AL3-11 工作记忆与意图槽
# --------------------------------------------------------------------------- #


def test_wm_put_merges_same_chunk_key(backend):
    backend.wm_put("s1", "topic-a", "第一段", 0.5)
    backend.wm_put("s1", "topic-a", "第二段", 0.6)
    chunks = backend.wm_list("s1")
    assert len(chunks) == 1
    assert chunks[0].act_count == 2
    assert chunks[0].content == "第二段"


def test_wm_list_orders_by_last_touched_desc(backend):
    backend.wm_put("s1", "a", "A", 0.1)
    backend.wm_put("s1", "b", "B", 0.1)
    backend.wm_put("s1", "c", "C", 0.1)
    # 再触碰 a
    backend.wm_put("s1", "a", "A2", 0.2)
    assert backend.wm_list("s1")[0].chunk_key == "a"


def test_wm_clear_only_current_session(backend):
    backend.wm_put("s1", "a", "A", 0.1)
    backend.wm_put("s2", "b", "B", 0.1)
    backend.wm_clear("s1")
    assert backend.wm_list("s1") == []
    assert len(backend.wm_list("s2")) == 1


def test_delete_session_cascades_working_memory(backend):
    backend.wm_put("s1", "a", "A", 0.1)
    backend.conn.execute("DELETE FROM sessions WHERE id = 's1'")
    assert backend.wm_list("s1") == []


def test_intent_put_and_list(backend):
    backend.intent_put("下周提醒我发版", session_id="s1", due_at="2026-09-20T09:00:00+08:00")
    intents = backend.intent_list()
    assert len(intents) == 1
    assert intents[0]["content"] == "下周提醒我发版"
    assert intents[0]["status"] == "open"


# --------------------------------------------------------------------------- #
# T-AL3-12 audit
# --------------------------------------------------------------------------- #


def test_audit_write_and_replay_order(backend):
    backend.put(make_record(content="A"))
    backend.put(make_record(content="B"))
    ops = [e.op for e in backend.audit_replay()]
    assert ops == ["add", "add"]


def test_audit_replay_since_filter(backend):
    backend.audit(AuditEvent(op="add", actor="system", ts="2026-01-01T00:00:00+08:00"))
    backend.audit(AuditEvent(op="add", actor="system", ts="2026-06-01T00:00:00+08:00"))
    assert len(list(backend.audit_replay(since="2026-03-01T00:00:00+08:00"))) == 1


def test_audit_is_append_only_update_rejected(backend):
    backend.audit(AuditEvent(op="add", actor="system"))
    with pytest.raises(sqlite3.DatabaseError):
        backend.conn.execute("UPDATE audit SET op = 'forget'")


def test_audit_is_append_only_delete_rejected(backend):
    backend.audit(AuditEvent(op="add", actor="system"))
    with pytest.raises(sqlite3.DatabaseError):
        backend.conn.execute("DELETE FROM audit")


def test_audit_for_returns_history_of_one_memory(backend):
    rec = make_record(content="有历史的记忆")
    backend.put(rec)
    backend.update(rec.id, {"content": "改过一次"})
    events = backend.audit_for(rec.id)
    assert [e.op for e in events] == ["add", "update"]
    assert events[0].actor == "system"


# --------------------------------------------------------------------------- #
# T-AL3-13 分级加载存储
# --------------------------------------------------------------------------- #


def test_overview_put_get_roundtrip_by_level(backend):
    backend.overview_put("entity", "ent_1", "L1", "关于 X 的概览", token_count=10, model="m")
    ov = backend.overview_get("entity", "ent_1", "L1")
    assert ov is not None
    assert ov.content == "关于 X 的概览"
    assert ov.stale is False
    assert backend.overview_get("entity", "ent_1", "L0") is None


def test_overview_invalidate_returns_affected_rows(backend):
    backend.overview_put("entity", "ent_1", "L1", "a", token_count=1, model="m")
    backend.overview_put("entity", "ent_2", "L1", "b", token_count=1, model="m")
    assert backend.overview_invalidate(scope_kind="entity") == 2
    assert backend.overview_invalidate(scope_kind="entity", scope_id="ent_1") == 1
    assert len(backend.overview_stale_list()) == 2


def test_overview_put_clears_stale_flag(backend):
    backend.overview_put("entity", "e", "L1", "旧", token_count=1, model="m")
    backend.overview_invalidate(scope_kind="entity", scope_id="e")
    assert backend.overview_get("entity", "e").stale is True
    backend.overview_put("entity", "e", "L1", "新", token_count=1, model="m")
    assert backend.overview_get("entity", "e").stale is False


def test_overview_is_pure_cache_clearing_is_harmless(backend):
    """INV-1：overviews 是缓存——清空后一切功能不受影响。"""
    rec = make_record(content="用户偏好深色主题", abstract="偏好深色")
    backend.put(rec, vec(1))
    backend.overview_put("entity", "e", "L1", "概览", token_count=1, model="m")
    backend.conn.execute("DELETE FROM overviews")
    assert backend.get(rec.id) is not None
    assert len(backend.keyword_search("深色主题")) == 1
    assert len(backend.vector_search(vec(1), top_k=1)) == 1
    assert len(backend.query()) == 1


# --------------------------------------------------------------------------- #
# T-AL3-14 内容寻址指纹
# --------------------------------------------------------------------------- #


def test_content_hash_is_deterministic(backend):
    h1 = backend.content_hash_of_record(make_record(content="同一条内容"))
    h2 = backend.content_hash_of_record(make_record(content="同一条内容"))
    assert h1 == h2


def test_content_hash_differs_by_scope(backend):
    base = dict(  # noqa: C408 - 与紧随其后的那处构造保持同形，便于逐字对照
        content="项目使用 Python 3.12",
        subject="X",
        predicate="uses",
        object_="3.12",
    )
    h1 = content_hash_of(**base, scope={"type": "project", "id": "a"})
    h2 = content_hash_of(**base, scope={"type": "project", "id": "b"})
    assert h1 != h2, "三元组相同但 scope 不同 → 指纹必须不同"


def test_content_hash_same_triple_different_wording(backend):
    """同一事实、不同措辞 → 同一指纹（这是导入幂等的基础）。"""
    h1 = content_hash_of(
        content="项目 RAGFlow 使用 Python 3.12",
        subject="RAGFlow",
        predicate="uses_python_version",
        object_="3.12",
        scope={"type": "project", "id": "ragflow"},
    )
    h2 = content_hash_of(
        content="RAGFlow 的 Python 版本是 3.12",
        subject="RAGFlow",
        predicate="uses_python_version",
        object_="3.12",
        scope={"type": "project", "id": "ragflow"},
    )
    assert h1 == h2


def test_find_by_hash_may_return_multiple(backend):
    a = make_record(content="内容一")
    b = make_record(content="内容二")
    backend.put(a)
    backend.put(b)
    backend.update(a.id, {"content_hash": "sha256:fixed"})
    backend.update(b.id, {"content_hash": "sha256:fixed"})
    assert len(backend.find_by_hash("sha256:fixed")) == 2


# --------------------------------------------------------------------------- #
# T-AL3-15 可读档案导出
# --------------------------------------------------------------------------- #


def test_archive_header_self_describing(backend):
    backend.put(make_record(content="一条记忆"))
    text = backend.export()
    for field in (
        "format",
        "schema_version",
        "spirit_name",
        "spirit_id",
        "data_as_of",
        "memory_count",
    ):
        assert f"{field}:" in text, f"档案头部缺少 {field}"


def test_archive_contains_core_fields(backend):
    rec = make_record(
        content="项目 RAGFlow 使用 Python 3.12",
        abstract="RAGFlow 用 3.12",
        subject="RAGFlow",
        confidence=0.9,
        salience=0.5,
        source_session="s1",
    )
    backend.put(rec)
    text = backend.export()
    for fragment in (rec.id, rec.content, rec.abstract, "confidence", "content_hash"):
        assert fragment in text


def test_archive_excludes_vectors(backend):
    backend.put(make_record(content="带有向量的记忆"), vec(1))
    text = backend.export()
    # 向量字段不应出现在档案中——既大又不可读
    assert "embedding:" not in text
    assert "0.01" not in text


def test_archive_is_deterministic(backend):
    backend.put(make_record(content="A"))
    backend.put(make_record(content="B"))
    assert backend.export() == backend.export()


def test_export_archive_writes_readable_file(backend, tmp_path):
    backend.put(make_record(content="内容甲"))
    path = tmp_path / "nested" / "archive.md"
    backend.export_archive(str(path))
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "内容甲" in text
    assert "artifact-spirit-archive" in text


# --------------------------------------------------------------------------- #
# T-AL3-16 写入原子性
# --------------------------------------------------------------------------- #


def test_put_atomic_when_audit_fails(backend, monkeypatch):
    """注入 audit 写入失败 → 记忆主体不得落库。"""
    original = SQLiteBackend._insert_audit

    def boom(self, conn, ev):
        if ev.op == "add":
            raise sqlite3.OperationalError("注入的审计写入失败")
        return original(self, conn, ev)

    monkeypatch.setattr(SQLiteBackend, "_insert_audit", boom)
    rec = make_record(content="不应落库")
    # 这里**刻意**用宽类型：要验的正是「**任何**异常都不能留下半写」，
    # 指定成某个具体异常，反而会把「抛了个意外异常但也留了半写」放过去。
    with pytest.raises(Exception):  # noqa: B017
        backend.put(rec, vec(1))
    assert backend.get(rec.id) is None


def test_put_atomic_when_vector_write_fails(backend, monkeypatch):
    rec = make_record(content="向量写失败")
    original = SQLiteBackend._record_params

    def boom(r):
        if r.content == "向量写失败":
            raise sqlite3.OperationalError("注入的向量写入失败")
        return original(r)

    # 直接让向量写入路径抛错：维度校验通过但 json 序列化失败
    bad_vec = [object()] * 8
    with pytest.raises(Exception):  # noqa: B017 - 同上：异常类型不重要，「不留半写」才重要
        backend.put(rec, bad_vec)
    assert backend.get(rec.id) is None


def test_put_success_writes_all_four_places(backend):
    rec = make_record(content="四处一致的记忆")
    backend.put(rec, vec(1))
    assert backend.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id = ?", (rec.id,)
    ).fetchone()[0] == 1
    assert backend.conn.execute(
        "SELECT COUNT(*) FROM vec_memories WHERE mem_id = ?", (rec.id,)
    ).fetchone()[0] == 1
    assert backend.conn.execute(
        "SELECT COUNT(*) FROM mem_fts WHERE rowid = "
        "(SELECT rowid FROM memories WHERE id = ?)",
        (rec.id,),
    ).fetchone()[0] == 1
    assert backend.conn.execute(
        "SELECT COUNT(*) FROM audit WHERE target_id = ? AND op = 'add'", (rec.id,)
    ).fetchone()[0] == 1
