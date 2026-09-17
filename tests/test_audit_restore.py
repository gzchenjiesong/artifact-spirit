"""审计 → 恢复这条链的守门测试。

**为什么单独一个文件**：设计上**不做删除前确认**（删除默认预演、需显式 `confirm`），
安全性完全交给"事后可审计 + 可恢复"。这个取舍成立的前提是那条链**真的可靠** ——
而它一旦断掉，用户的记忆就是真的没了，且不会有任何报错提醒。

所以这里测的不是"某个函数能跑"，而是三件必须成立的承诺：

1. **闭合性**：每一次非清除删除，都在快照清单里留下了一条（没有"静默无快照"的删除）
2. **保真性**：恢复出来的记录与删掉之前**逐字段一致**（含原 ID 与关联边）
3. **可操作性**：审计视图说清了"删了什么、为什么、还能不能救、怎么救"，
   而且它给出的恢复编号**真的能用**

第 3 条尤其重要：它是使用者唯一的介入点。审计视图如果只说"forget user <id>"，
那么"发现不对可以要求恢复"就落不了地 —— 使用者既不知道删的是什么，
也不知道能不能救。
"""

from __future__ import annotations

from artifact_spirit.observability import (
    AuditView,
    format_audit_text,
    format_restorable_text,
)
from artifact_spirit.store.base import MemoryRecord

# 派生字段：由 audit 重放重建（INV-12）。快照里保留的是**删除当时**的值，
# 恢复也原样带回；但若之后跑过 replay_derived，它们会被重算。
# 这里不做 replay，所以理论上也应一致 —— 单列出来是为了让断言失败时更好定位。
_DERIVED_FIELDS = {"strength", "access_count", "last_access_at"}


# --------------------------------------------------------------------------- #
# 1. 闭合性：没有一条删除是静默无快照的
# --------------------------------------------------------------------------- #


def test_every_forget_leaves_a_restorable_snapshot(backend, record_factory):
    """**闭合性断言**：每条 `forget` 要么有快照、要么是显式合规清除。

    这是"事前不确认"能被接受的前提。如果某条删除**既没有快照、又不是合规清除**，
    那它就是真的丢了 —— 而且是事后才知道。
    """
    ids: list[str] = []
    for index in range(5):
        record = record_factory(content=f"第 {index} 条待删内容")
        backend.put(record, None)
        ids.append(record.id)

    # 两条普通删除
    for mem_id in ids[:2]:
        backend.hard_delete(mem_id, reason="用户要求删除", actor="user", source="cli")
    # 两条优化器删除（另一条路径）
    for mem_id in ids[2:4]:
        backend.hard_delete(mem_id, reason="不可达", actor="optimizer", source="system")
    # 一条合规清除：**唯一**允许没有快照的情形，且必须显式声明
    backend.hard_delete(
        ids[4], reason="合规：被遗忘权", purge_snapshot=True, actor="user", source="cli"
    )

    forgets = [event for event in backend.audit_replay() if event.op == "forget"]
    snapshot_ids = {snapshot["mem_id"] for snapshot in backend.delete_snapshots()}

    assert len(forgets) == 5, "5 条删除应留下 5 条审计"

    for event in forgets:
        has_snapshot = event.target_id in snapshot_ids
        explicitly_purged = bool(event.reason and "合规" in event.reason)
        assert has_snapshot or explicitly_purged, (
            f"删除 {event.target_id} 既没有可恢复快照、也不是显式合规清除 —— "
            "这条记忆真的丢了，而使用者只能事后发现"
        )

    # 反向：快照不该凭空多出来
    assert len(snapshot_ids) == 4, f"应有 4 条快照（5 删 1 清除），实得 {len(snapshot_ids)}"


def test_snapshot_payload_can_rebuild_the_original_text(backend, record_factory):
    """快照里的 payload 必须**足以重建原文**（不是只留一个 id）。"""
    record = record_factory(content="这句原文必须能从快照里读回来")
    backend.put(record, None)
    backend.hard_delete(record.id, reason="测试", actor="user", source="cli")

    snapshots = backend.delete_snapshots(include_payload=True)
    assert len(snapshots) == 1
    payload_record = snapshots[0]["record"]
    assert payload_record.get("content") == "这句原文必须能从快照里读回来"
    assert payload_record.get("id") == record.id


# --------------------------------------------------------------------------- #
# 2. 保真性：恢复出来的与删掉之前逐字段一致
# --------------------------------------------------------------------------- #


def test_restore_is_field_faithful(backend, record_factory):
    """恢复必须**逐字段**一致，不只是 content 对得上。"""
    other = record_factory(content="关联的另一条")
    backend.put(other, None)

    record = record_factory(
        content="逐字比对的内容",
        abstract="摘要也在",
        subject="主体",
        predicate="谓词",
        object="客体",
        scope={"type": "project", "id": "demo"},
        confidence=0.87,
        salience=0.42,
        source_session="s-faithful",
    )
    backend.put(record, None)
    backend.link("memory", record.id, "memory", other.id, "co_activation", 0.5)

    original = backend.get(record.id)
    assert original is not None

    backend.hard_delete(record.id, reason="测试", actor="user", source="cli")
    snapshot = backend.delete_snapshots()[0]
    restored = backend.get(backend.restore_from_audit(snapshot["audit_id"]))
    assert restored is not None

    differences = []
    for field in MemoryRecord.__dataclass_fields__:
        before = getattr(original, field)
        after = getattr(restored, field)
        if before != after:
            differences.append(f"{field}: {before!r} → {after!r}")
    assert not differences, "恢复后字段不一致：\n" + "\n".join(differences)

    # 关联边也必须回来（否则恢复出来的记忆在图上孤立）
    neighbors = backend.neighbors("memory", restored.id)
    assert any(node_id == other.id for _, node_id, _ in neighbors), (
        f"恢复后关联边丢了：{neighbors}"
    )


def test_restored_memory_is_searchable_again(backend, record_factory):
    """恢复之后必须重新可检索（FTS 索引要跟着回来）——否则"恢复了但搜不到"。"""
    record = record_factory(content="恢复后应当能被检索到的独特词汇 ZedUnique")
    backend.put(record, None)
    assert backend.keyword_search("ZedUnique", top_k=5), "前置：写入后应能检索到"

    backend.hard_delete(record.id, reason="测试", actor="user", source="cli")
    assert not backend.keyword_search("ZedUnique", top_k=5), "前置：删除后应检索不到"

    snapshot = backend.delete_snapshots()[0]
    restored_id = backend.restore_from_audit(snapshot["audit_id"])
    hits = backend.keyword_search("ZedUnique", top_k=5)
    assert any(hit.mem_id == restored_id for hit in hits), "恢复后全文索引没有跟上"


def test_restoring_twice_does_not_duplicate(backend, record_factory):
    """同一条快照恢复两次，不得产生两条记忆。"""
    record = record_factory(content="只应存在一份")
    backend.put(record, None)
    backend.hard_delete(record.id, reason="测试", actor="user", source="cli")

    audit_id = backend.delete_snapshots()[0]["audit_id"]
    first = backend.restore_from_audit(audit_id)
    second = backend.restore_from_audit(audit_id)

    active = [r for r in backend.query(status=None, limit=100) if r.content == "只应存在一份"]
    assert len(active) == 1, f"恢复两次产生了 {len(active)} 份"
    assert first == record.id and second, "两次恢复都应返回有效 id"
    assert not backend.keyword_search("只应存在一份", top_k=5) or len(
        backend.keyword_search("只应存在一份", top_k=5)
    ) == 1, "全文索引出现了重复条目"


# --------------------------------------------------------------------------- #
# 3. 可操作性：审计视图说清"删了什么、为什么、能不能救、怎么救"
# --------------------------------------------------------------------------- #


def test_audit_view_shows_what_was_deleted_and_why(backend, record_factory):
    """删除条目必须带**内容摘要**与**原因** —— 使用者靠这两样判断操作对不对。"""
    record = record_factory(content="用户的 RAGFlow 项目使用 Docker 部署")
    backend.put(record, None)
    backend.hard_delete(
        record.id, reason="被新事实取代：部署方式改为 Podman", actor="user", source="tool"
    )

    row = AuditView(backend).forgetting()[0]
    assert row["op"] == "forget"
    assert row["reason"] == "被新事实取代：部署方式改为 Podman"
    assert "Docker 部署" in row["deleted_preview"]
    assert row["deleted_layer"] == "semantic"


def test_audit_view_restore_command_actually_works(backend, record_factory):
    """审计视图给出的恢复编号**必须真的能用** —— 这是介入点能落地的判据。"""
    record = record_factory(content="我要把它救回来")
    backend.put(record, None)
    backend.hard_delete(record.id, reason="误删", actor="user", source="tool")

    view = AuditView(backend)
    row = view.forgetting()[0]
    assert row["restorable"] is True
    assert row["restore_command"] == f"aspirit restore {row['audit_id']}"

    # 真按这个编号恢复
    restored_id = backend.restore_from_audit(row["audit_id"])
    restored = backend.get(restored_id)
    assert restored is not None and restored.content == "我要把它救回来"


def test_audit_id_is_the_real_primary_key_not_a_position(backend, record_factory):
    """审计视图给的编号必须是**真实主键**，不能是"过滤结果里的第几条"。

    曾经用 ``enumerate`` 的序号当 ``audit_id``：不带过滤时它恰好与真实 id 一致，
    一带 ``--since`` 就整体错位 —— 使用者照着一个错位的编号去恢复，
    拿到的是**另一条**记忆。这类错误不报错，只是让人拿到错的东西。
    """
    for index in range(6):
        record = record_factory(content=f"第 {index} 条")
        backend.put(record, None)
    ids = [record.id for record in backend.query(status=None, limit=20)]
    backend.hard_delete(ids[0], reason="先删一条", actor="user", source="cli")

    events = list(backend.audit_replay())
    assert len(events) >= 7

    # 用中点时刻做 since，滤掉前半段
    mid_ts = events[len(events) // 2].ts
    filtered = AuditView(backend).entries(since=mid_ts)
    assert filtered, "since 之后应当仍能取到事件"

    real_ids = {event.audit_id for event in events}
    for row in filtered:
        assert row["audit_id"] in real_ids, f"{row['audit_id']} 不是真实主键"

    # 最有力的一条：过滤后最大的编号必须等于最后一条事件的真实 id。
    # 若实现用的是位置序号，这里只会是个很小的数（例如 4），立刻暴露。
    assert max(row["audit_id"] for row in filtered) == events[-1].audit_id


# --------------------------------------------------------------------------- #
# 4. 合规清除的例外：必须显式写明"救不回来"
# --------------------------------------------------------------------------- #


def test_purged_delete_is_marked_unrestorable_and_says_why(backend, record_factory):
    """合规清除是**唯一**不可恢复的删除，视图必须写明这一点而不是留白。"""
    record = record_factory(content="依法必须彻底删除的内容")
    backend.put(record, None)
    backend.hard_delete(
        record.id, reason="合规：被遗忘权", purge_snapshot=True, actor="user", source="cli"
    )

    view = AuditView(backend)
    row = view.forgetting()[0]
    assert row["restorable"] is False
    assert row["restore_command"] is None
    assert not row.get("deleted_preview"), "已清除的内容不该还能从快照里读出来"

    text = format_audit_text([row])
    assert "可恢复：否" in text
    assert "依法必须彻底删除的内容" not in text, "合规清除后内容不该出现在审计文本里"


def test_no_restorable_snapshots_reads_clearly(backend):
    """没有可恢复项时，救援面板必须给出明确结论而不是空白。"""
    text = format_restorable_text(AuditView(backend).restorable())
    assert "无" in text


def test_restorable_panel_lists_content_and_command(backend, record_factory):
    """救援面板：列出被删内容 + 给出恢复命令。"""
    record = record_factory(content="面板上要看得见这条内容")
    backend.put(record, None)
    backend.hard_delete(record.id, reason="测试", actor="user", source="cli")

    text = format_restorable_text(AuditView(backend).restorable())
    assert "面板上要看得见这条内容" in text
    assert "aspirit restore" in text
    assert str(backend.delete_snapshots()[0]["audit_id"]) in text
