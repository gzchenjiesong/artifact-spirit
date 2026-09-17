"""AL2 双时态（T-AL2-16 · M3）：失效标记与 as-of 查询。

## 这些用例各自堵什么

| 用例 | 堵的失效模式 |
|---|---|
| `test_invalidate_marks_without_deleting` | 把"被取代"实现成"被删除"——用户就再也答不出"我上个月填的地址是什么" |
| `test_invalidate_also_writes_supersedes_edge` | 只写 `superseded_by` 不写关联边：**追溯链只修了一个方向** |
| `test_invalidate_audit_op_matches_the_action` | 账本写一个 op、实际做另一个（P0-6 的老毛病） |
| `test_invalidate_rejects_self_reference` | 时态链成环 → as-of 永远查不出结果，而**那只是一个 `None`** |
| `test_invalidate_rejects_backwards_interval` | `valid_to <= valid_from` → "有效期"是空区间，任何时刻都查不到 |
| `test_invalidate_missing_record_raises` | 静默产出一条指向空气的意图（写进去也永远无效） |
| `test_asof_through_the_facade` | 门面没接通时态能力（存储有、上层用不到） |
"""

from __future__ import annotations

import pytest
from conftest import make_record, write_intents

from artifact_spirit.core import ArtifactSpiritCore
from artifact_spirit.store import NotFoundError

TS_OLD = "2026-01-01T00:00:00+08:00"
TS_MID = "2026-06-01T00:00:00+08:00"
TS_IN_BETWEEN = "2026-03-01T00:00:00+08:00"
TS_AFTER = "2026-07-01T00:00:00+08:00"


def _core(backend) -> ArtifactSpiritCore:
    return ArtifactSpiritCore(backend=backend)


def _pair(backend):
    """造一对"旧事实 → 新事实"。"""
    old = make_record(content="旧事实：住在上海", created_at=TS_OLD, valid_from=TS_OLD)
    backend.put(old)
    new = make_record(content="新事实：搬到北京", created_at=TS_MID, valid_from=TS_MID)
    backend.put(new)
    return old, new


# --------------------------------------------------------------------------- #
# 失效（INVALIDATE）
# --------------------------------------------------------------------------- #


def test_invalidate_marks_without_deleting(backend):
    """`INVALIDATE` 写 `valid_to` + `superseded_by`——**不是删除**（D-17 / INV-7）。"""
    old, new = _pair(backend)
    core = _core(backend)

    write_intents(
        backend,
        core.invalidate(old.id, superseded_by=new.id, reason="事实更新", valid_to=TS_MID),
    )

    stored = backend.get(old.id)
    assert stored is not None, "失效不是删除——记录必须还在"
    assert stored.valid_to == TS_MID
    assert stored.superseded_by == new.id
    assert stored.status == "active", "失效是时态概念，不改生命周期 status"


def test_invalidate_also_writes_supersedes_edge(backend):
    """`superseded_by` 与 `supersedes` 边**必须同时存在**、方向是「新 → 旧」。

    只写一个：`superseded_by` 答得出"被谁取代"，答不出"它取代了谁"（反之亦然）——
    而"这条记忆的来龙去脉"正是 V2 的卖点，**追溯链修一半等于没修**。
    """
    old, new = _pair(backend)
    core = _core(backend)

    write_intents(
        backend,
        core.invalidate(old.id, superseded_by=new.id, reason="事实更新", valid_to=TS_MID),
    )

    edges = [e for e in backend.all_relations() if e["rel_type"] == "supersedes"]
    assert len(edges) == 1, f"应当恰好一条 supersedes 边，实际 {len(edges)}"
    assert edges[0]["src_id"] == new.id, "边方向必须是「新 → 旧」（新取代旧）"
    assert edges[0]["dst_id"] == old.id


def test_invalidate_audit_op_matches_the_action(backend):
    """审计 op 必须是 `invalidate`——**账本要与真正执行的动作一致**（P0-6 的口径）。"""
    old, new = _pair(backend)
    core = _core(backend)

    write_intents(
        backend,
        core.invalidate(old.id, superseded_by=new.id, reason="事实更新", valid_to=TS_MID),
    )

    ops = [e.op for e in backend.audit_replay() if e.target_id == old.id]
    assert "invalidate" in ops, f"没有 invalidate 审计事件，只有 {ops}"
    event = next(e for e in backend.audit_replay() if e.op == "invalidate")
    assert event.before is not None and event.after is not None, "变更前后都要留痕"


def test_invalidate_rejects_self_reference(backend):
    """自指 → 时态链成环 → as-of 永远查不出结果，而**那只是一个 `None`、不报错**。"""
    rec = make_record(content="不该被自己取代")
    backend.put(rec)

    with pytest.raises(ValueError):
        _core(backend).invalidate(rec.id, superseded_by=rec.id, reason="自指")


def test_invalidate_rejects_backwards_interval(backend):
    """`valid_to` 必须晚于 `valid_from`——否则"有效期"是空区间，任何时刻都查不到它。"""
    rec = make_record(content="有效期起点在未来", created_at=TS_MID, valid_from=TS_MID)
    backend.put(rec)
    other = make_record(content="另一条")
    backend.put(other)

    with pytest.raises(ValueError):
        _core(backend).invalidate(
            rec.id, superseded_by=other.id, reason="时间倒流", valid_to=TS_OLD
        )


def test_invalidate_missing_record_raises(backend):
    """目标不存在 → **报错**，不静默产出一条指向空气的意图。"""
    other = make_record(content="存在的记录")
    backend.put(other)

    with pytest.raises(NotFoundError):
        _core(backend).invalidate("sem_不存在", superseded_by=other.id, reason="幽灵")


# --------------------------------------------------------------------------- #
# as-of 查询（穿过门面）
# --------------------------------------------------------------------------- #


def test_asof_through_the_facade(backend):
    """门面接通时态能力：**同一 `ref` 在两个时刻返回不同内容**。

    只断言"能返回一条"是不够的——返回当前值同样能过，而那意味着时态根本没起作用。
    """
    old, new = _pair(backend)
    core = _core(backend)
    write_intents(
        backend,
        core.invalidate(old.id, superseded_by=new.id, reason="事实更新", valid_to=TS_MID),
    )

    at_march = core.asof(old.id, TS_IN_BETWEEN)
    assert at_march is not None and at_march.content == "旧事实：住在上海"

    at_july = core.asof(old.id, TS_AFTER)
    assert at_july is not None and at_july.id == new.id, "7 月应沿链找到取代它的新事实"


def test_asof_before_first_write_is_none(backend):
    """`None` = **那时不存在有效版本**（不抛错、也不回退当前值）。"""
    rec = make_record(content="2026 年才写下的", created_at=TS_OLD, valid_from=TS_OLD)
    backend.put(rec)

    assert _core(backend).asof(rec.id, "2020-01-01T00:00:00+08:00") is None


def test_invalidate_intents_are_two_and_both_needed(backend):
    """失效产出**两条**意图（`update` + `link`）——少一条就少一半追溯。"""
    old, new = _pair(backend)
    intents = _core(backend).invalidate(
        old.id, superseded_by=new.id, reason="事实更新", valid_to=TS_MID
    )

    ops = sorted(i.op for i in intents)
    assert ops == ["link", "update"], f"期望 update + link，实际 {ops}"
