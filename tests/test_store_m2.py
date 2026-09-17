"""AL3 存储层 · M2 验收（T-AL3-17 ~ 23）：治理、恢复与传承。"""

from __future__ import annotations

import pytest
from conftest import make_record, vec

from artifact_spirit.store import NotFoundError, WhitelistViolation
from artifact_spirit.store import archive as archive_mod
from artifact_spirit.store import reindex as reindex_mod

# --------------------------------------------------------------------------- #
# T-AL3-17 sessions 生命周期
# --------------------------------------------------------------------------- #


def test_session_end_sets_committed_and_ended_at(backend):
    backend.session_create("s1", "2026-09-14T10:00:00+08:00")
    backend.session_end("s1", "2026-09-14T11:00:00+08:00")
    row = backend.session_get("s1")
    assert row["status"] == "committed"
    assert row["ended_at"] == "2026-09-14T11:00:00+08:00"


def test_session_turn_count_accumulates(backend):
    assert backend.session_bump_turn("s1") == 1
    assert backend.session_bump_turn("s1") == 2
    assert backend.session_bump_turn("s1") == 3
    assert backend.session_get("s1")["turn_count"] == 3


# --------------------------------------------------------------------------- #
# T-AL3-18 删除与恢复
# --------------------------------------------------------------------------- #


def test_delete_keeps_audit_history(backend):
    """删除记忆 ≠ 删除关于它的审计（INV-8）。"""
    rec = make_record(content="会被删掉的记忆")
    backend.put(rec)
    backend.hard_delete(rec.id, reason="测试", actor="user", source="cli")
    history = [e.op for e in backend.audit_replay() if e.target_id == rec.id]
    assert history == ["add", "forget"]


def test_restore_recovers_record_with_same_id_and_relations(backend):
    a = make_record(content="A 主体")
    b = make_record(content="B 邻居")
    backend.put(a)
    backend.put(b)
    backend.link("memory", a.id, "memory", b.id, "co_activation", 0.42)
    backend.hard_delete(a.id, reason="误删", actor="user", source="cli")

    snapshot = backend.delete_snapshots()[0]
    restored_id = backend.restore_from_audit(snapshot["audit_id"])

    assert restored_id == a.id, "恢复必须沿用原 ID，否则关联断裂"
    restored = backend.get(restored_id)
    assert restored.content == "A 主体"
    assert restored.confidence == a.confidence
    rels = backend.relations_of(restored_id)
    assert len(rels) == 1
    assert rels[0]["weight"] == pytest.approx(0.42)


def test_restore_writes_restore_audit(backend):
    rec = make_record(content="可恢复")
    backend.put(rec)
    backend.hard_delete(rec.id, reason="测试")
    snapshot = backend.delete_snapshots()[0]
    backend.restore_from_audit(snapshot["audit_id"])
    assert [e.op for e in backend.audit_replay()][-1] == "restore"


def test_purge_snapshot_makes_restore_impossible(backend):
    """合规删除是**唯一不可恢复**的路径（D-22 例外）。"""
    rec = make_record(content="合规删除的内容")
    backend.put(rec)
    backend.hard_delete(
        rec.id, reason="合规：被遗忘权", purge_snapshot=True, actor="user", source="cli"
    )
    assert [s for s in backend.delete_snapshots() if s["mem_id"] == rec.id] == []

    forget_ids = [
        e for e in backend.audit_for(rec.id, limit=50) if e.op == "forget"
    ]
    assert forget_ids, "审计仍须记录'发生过一次删除'"


def test_purge_snapshot_removes_content_from_audit(backend):
    rec = make_record(content="敏感内容不应留在审计里")
    backend.put(rec)
    backend.hard_delete(
        rec.id, reason="合规", purge_snapshot=True, actor="user", source="cli"
    )
    forget = [e for e in backend.audit_for(rec.id) if e.op == "forget"][-1]
    assert "content" not in (forget.before or {})
    assert "敏感内容" not in str(forget.before)


def test_delete_without_reason_is_rejected(backend):
    rec = make_record(content="x")
    backend.put(rec)
    with pytest.raises(WhitelistViolation):
        backend.hard_delete(rec.id, reason="")
    with pytest.raises(WhitelistViolation):
        backend.hard_delete(rec.id, reason="   ")
    assert backend.get(rec.id) is not None, "拒绝时必须什么都没发生"


def test_delete_missing_record_raises(backend):
    with pytest.raises(NotFoundError):
        backend.hard_delete("sem_不存在", reason="测试")


def test_restore_unknown_snapshot_raises(backend):
    with pytest.raises(NotFoundError):
        backend.restore_from_audit(99999)


def test_restore_uses_new_id_when_original_taken(backend):
    rec = make_record(content="同 ID 冲突")
    backend.put(rec)
    original_id = rec.id
    backend.hard_delete(original_id, reason="测试")
    # 占住原 ID
    backend.put(make_record(id=original_id, content="占位"))
    snapshot = backend.delete_snapshots()[0]
    new_id = backend.restore_from_audit(snapshot["audit_id"])
    assert new_id != original_id
    assert backend.get(new_id).content == "同 ID 冲突"


# --------------------------------------------------------------------------- #
# T-AL3-19 派生字段重放重建
# --------------------------------------------------------------------------- #


def test_replay_restores_derived_fields(backend):
    rec = make_record(content="有访问历史的记忆")
    backend.put(rec)
    backend.touch(rec.id, "2026-09-14T21:00:00+08:00", strength=0.5)
    backend.touch(rec.id, "2026-09-14T21:05:00+08:00", strength=0.7)
    expected = backend.get(rec.id)

    # 破坏派生字段
    backend.update(rec.id, {"strength": 0.0, "access_count": 999, "last_access_at": None})
    backend.replay_derived()

    actual = backend.get(rec.id)
    assert actual.strength == pytest.approx(expected.strength)
    assert actual.access_count == expected.access_count
    assert actual.last_access_at == expected.last_access_at


def test_replay_reports_orphan_events(backend):
    backend.audit(
        __import__("artifact_spirit.store", fromlist=["AuditEvent"]).AuditEvent(
            op="touch",
            actor="system",
            target_id="sem_早已不存在",
            after={"strength": 1.0, "access_count": 1},
        )
    )
    report = backend.replay_derived()
    assert report["orphan_events"] == 1


# --------------------------------------------------------------------------- #
# T-AL3-20 重嵌入
# --------------------------------------------------------------------------- #


def test_reindex_rebuilds_all_vectors(backend):
    for i in range(3):
        backend.put(make_record(content=f"记录{i}"), vec(i))
    res = reindex_mod.reindex(backend, lambda ts: [[0.5] * 8 for _ in ts], model="m2", dim=8)
    assert res["done"] == 3
    assert res["rebuilt_table"] is True
    assert len(backend.vector_search([0.5] * 8, top_k=10)) == 3


def test_reindex_resumes_without_reprocessing(backend):
    calls: list[list[str]] = []

    def embed(texts):
        calls.append(texts)
        return [[0.5] * 8 for _ in texts]

    for i in range(4):
        backend.put(make_record(content=f"记录{i}"), vec(i))

    reindex_mod.reindex(backend, embed, model="m2", dim=8, batch_size=2)
    first_calls = len(calls)
    assert first_calls == 2

    # 再跑一次：游标已在末尾，不应重复处理
    res = reindex_mod.reindex(backend, embed, model="m2", dim=8, batch_size=2)
    assert res["done"] == 0
    assert len(calls) == first_calls, "续跑不应重复处理已完成项"


def test_reindex_updates_meta_model_and_dim(backend):
    backend.put(make_record(content="x"), vec(1))
    reindex_mod.reindex(backend, lambda ts: [[0.1] * 16 for _ in ts], model="new-model", dim=16)
    assert backend.meta_get("embedding_model") == "new-model"
    assert backend.meta_get("embedding_dim") == "16"
    assert backend.embedding_dim == 16


def test_reindex_writes_audit(backend):
    backend.put(make_record(content="x"), vec(1))
    reindex_mod.reindex(backend, lambda ts: [[0.1] * 8 for _ in ts], model="m2", dim=8)
    assert any(e.op == "reindex" for e in backend.audit_replay())


# --------------------------------------------------------------------------- #
# T-AL3-21 启动对账
# --------------------------------------------------------------------------- #


def test_reconcile_fixes_missing_vectors(backend):
    rec = make_record(content="缺向量的记忆")
    backend.put(rec, vec(1))
    backend.conn.execute("DELETE FROM vec_memories WHERE mem_id = ?", (rec.id,))
    assert backend.reconcile(dry_run=True)["missing_vectors"] == [rec.id]

    def embed(texts):
        return [[0.2] * 8 for _ in texts]

    report = backend.reconcile(dry_run=False, embed_fn=embed)
    assert report["repaired"] == 1
    assert backend.reconcile(dry_run=True)["missing_vectors"] == []


def test_reconcile_removes_orphan_vectors(backend):
    backend.conn.execute(
        "INSERT INTO vec_memories(mem_id, embedding) VALUES (?, ?)",
        ("sem_孤儿", "[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]"),
    )
    assert backend.reconcile(dry_run=True)["orphan_vectors"] == ["sem_孤儿"]
    backend.reconcile(dry_run=False)
    assert backend.reconcile(dry_run=True)["orphan_vectors"] == []


def test_reconcile_dry_run_writes_nothing(backend):
    backend.conn.execute(
        "INSERT INTO vec_memories(mem_id, embedding) VALUES (?, ?)",
        ("sem_孤儿", "[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]"),
    )
    backend.reconcile(dry_run=True)
    assert len(backend.reconcile(dry_run=True)["orphan_vectors"]) == 1


def test_fresh_db_reconcile_is_clean(backend):
    backend.put(make_record(content="正常记录"), vec(1))
    report = backend.reconcile(dry_run=True)
    assert report["missing_vectors"] == []
    assert report["orphan_vectors"] == []


# --------------------------------------------------------------------------- #
# T-AL3-22 记忆包导出/导入与幂等
# --------------------------------------------------------------------------- #


def test_import_pack_twice_produces_no_duplicates(backend):
    for i in range(3):
        backend.put(make_record(content=f"记忆{i}"))
    pack = archive_mod.export_pack(backend)

    first = backend.import_pack(pack)
    second = backend.import_pack(pack)
    assert first["imported"] == 0
    assert second["imported"] == 0
    assert first["skipped"] == 3
    assert len(backend.query(status=None)) == 3


def test_import_pack_into_empty_db_is_lossless(backend, tmp_path):
    a = make_record(content="迁移的内容", abstract="摘要", confidence=0.88)
    backend.put(a)
    backend.link(
        "memory", a.id, "memory", backend.put(make_record(content="邻居")), "co_activation", 0.5
    )
    pack = archive_mod.export_pack(backend)

    from artifact_spirit.store import SQLiteBackend

    fresh = SQLiteBackend(str(tmp_path / "fresh.db"), embedding_dim=8)
    fresh.open()
    stats = fresh.import_pack(pack)
    assert stats["imported"] == 2
    got = fresh.get(a.id)
    assert got is not None
    assert got.content == "迁移的内容"
    assert got.abstract == "摘要"
    assert got.confidence == pytest.approx(0.88)
    assert len(fresh.relations_of(a.id)) == 1
    fresh.close()


def test_export_import_export_is_equivalent(backend, tmp_path):
    for i in range(3):
        backend.put(make_record(content=f"内容{i}", abstract=f"摘要{i}"))
    backend.entity_upsert("RAGFlow", "project", aliases=["ragflow"])
    markdown = backend.export()

    path = tmp_path / "archive.md"
    backend.export_archive(str(path))

    from artifact_spirit.store import SQLiteBackend

    fresh = SQLiteBackend(str(tmp_path / "fresh.db"), embedding_dim=8)
    fresh.open()
    fresh.import_archive(str(path))
    assert fresh.export() == markdown, "导出→导入→再导出必须等价（INV-14）"
    fresh.close()


def test_import_rejects_foreign_payload(backend):
    with pytest.raises(ValueError):
        backend.import_pack(b'{"format": "something-else"}')


def test_archive_is_readable_without_any_code(backend, tmp_path):
    """V3：档案必须能用纯文本编辑器读懂——不依赖器灵运行时。"""
    backend.put(
        make_record(
            content="用户偏好深色主题，不要用浅色界面",
            abstract="偏好深色主题",
            subject="用户",
            predicate="prefers",
            object="深色主题",
        )
    )
    path = tmp_path / "archive.md"
    backend.export_archive(str(path))
    text = path.read_text(encoding="utf-8")
    assert "用户偏好深色主题" in text
    assert "记忆档案" in text
    # 不含任何二进制 / base64 块
    assert "base64" not in text


# --------------------------------------------------------------------------- #
# T-AL3-23 架构与合规（细节见 test_arch.py）
# --------------------------------------------------------------------------- #


def test_store_layer_has_no_illegal_imports():
    from artifact_spirit.compliance import check_architecture
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT

    violations = [v for v in check_architecture(PACKAGE_ROOT) if "store" in v["file"]]
    assert violations == []


def test_fts_is_derived_and_rebuildable(backend):
    """FTS 是派生索引，不是真相源（INV-1）——可随时重建。"""
    for i in range(3):
        backend.put(make_record(content=f"记录{i} 关键词Kwd{i}"))
    assert backend.keyword_search("Kwd1") != []
    backend.conn.execute("DELETE FROM mem_fts")
    assert backend.keyword_search("Kwd1") == []
    backend.rebuild_fts()
    assert backend.keyword_search("Kwd1") != []
