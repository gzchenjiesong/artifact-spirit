"""AL3 存储层 · M3 验收（T-AL3-24 ~ 25）：双时态启用与往返。

## 关于"启用"而不是"迁移"（2026-09-17 核实）

任务书原条款写的是"v1 → v2 schema 迁移（双时态字段）"——**那是错的**：

- `SCHEMA_VERSION` 当前是 **2**（v1 = 全量 DDL，v2 = `mem_fts` 重建为 CJK 归一化的普通 FTS5 表）；
- `valid_from` / `valid_to` / `superseded_by` **早已在 v1 的全量 DDL 里**，
  `MemoryRecord` 与 `put` / `update` 的字段白名单也都覆盖了它们；
- `schema.sql` 的注释本来就写明"MVP 只写 `valid_from`，`valid_to` 待 M3 的 INVALIDATE"。

所以本里程碑的存储侧工作是**启用**这三个字段：**被写、被查、被往返**。

## 这些用例各自堵什么

| 用例 | 堵的失效模式 |
|---|---|
| `test_put_defaults_valid_from_to_created_at` | 字段"占着位但没人写"——as-of 判定随即退化成"没有时态信息" |
| `test_asof_returns_the_version_valid_at_that_time` | 只断言"能返回"的话，**返回当前值也能过**；必须断言它**随时间变化** |
| `test_asof_before_first_write_returns_none` | 把"无有效版本"实现成"返回当前值"或抛错 |
| `test_asof_sql_calls_do_not_scale_with_table_size` | 把表捞进内存再比时间——**静默的全表扫描**，几百条之后才显形 |
| `test_superseded_by_self_reference_is_rejected` | 链成环 → as-of 永远查不出，且**只是一个 `None`、不报错** |
| `test_import_pack_roundtrips_superseded_by` | 导出带着它、导入漏了 → "被谁取代"在往返中**静默消失** |
"""

from __future__ import annotations

import pytest
from conftest import make_record

from artifact_spirit.store import SQLiteBackend, StoreError

TS_OLD = "2026-01-01T00:00:00+08:00"
TS_MID = "2026-06-01T00:00:00+08:00"
TS_IN_BETWEEN = "2026-03-01T00:00:00+08:00"
TS_AFTER = "2026-07-01T00:00:00+08:00"


# --------------------------------------------------------------------------- #
# T-AL3-24 双时态写入与 as-of 查询
# --------------------------------------------------------------------------- #


def test_put_defaults_valid_from_to_created_at(backend):
    """`valid_from` 缺省 == `created_at`，且 `valid_to` 为 `NULL`（至今有效）。"""
    rec = make_record(content="偏好深色主题")
    backend.put(rec)

    stored = backend.get(rec.id)
    assert stored is not None
    assert stored.valid_from, "valid_from 不能留空——as-of 判定读的就是它"
    assert stored.valid_from == stored.created_at
    assert stored.valid_to is None, "新写入的记录应当「至今有效」"


def test_asof_returns_the_version_valid_at_that_time(backend):
    """**同一 `ref` 在两个时刻返回不同内容**——这才是"时态真的生效"的证据。

    只断言"能返回一条"是不够的：返回当前值同样能过，而那意味着时态根本没起作用。
    """
    old = make_record(content="旧事实：住在上海", created_at=TS_OLD, valid_from=TS_OLD)
    backend.put(old)

    new = make_record(content="新事实：搬到北京", created_at=TS_MID, valid_from=TS_MID)
    backend.put(new)

    # 旧事实在 6 月失效，被新事实取代
    backend.update(old.id, {"valid_to": TS_MID, "superseded_by": new.id})

    at_march = backend.asof(old.id, TS_IN_BETWEEN)
    assert at_march is not None, "旧事实在 3 月应当仍然有效"
    assert at_march.content == "旧事实：住在上海"

    at_july = backend.asof(old.id, TS_AFTER)
    assert at_july is not None, "7 月应当沿着 superseded_by 链找到新事实"
    assert at_july.id == new.id
    assert at_july.content == "新事实：搬到北京"


def test_asof_before_first_write_returns_none(backend):
    """`ts` 早于首次写入 → **无有效版本**（不是返回当前值，也不抛错）。"""
    rec = make_record(content="2026 年才写下的记忆", created_at=TS_OLD, valid_from=TS_OLD)
    backend.put(rec)

    assert backend.asof(rec.id, "2020-01-01T00:00:00+08:00") is None


def test_asof_returns_none_after_invalidation_without_successor(backend):
    """失效但**没有后继**（如合规清除留下的空档）→ 该时刻之后无有效版本。"""
    rec = make_record(content="只失效、无取代者", created_at=TS_OLD, valid_from=TS_OLD)
    backend.put(rec)
    backend.update(rec.id, {"valid_to": TS_MID})

    assert backend.asof(rec.id, TS_IN_BETWEEN) is not None, "失效前应当有效"
    assert backend.asof(rec.id, TS_AFTER) is None, "失效后且无后继 → 无有效版本"


def test_asof_sql_calls_do_not_scale_with_table_size(backend, monkeypatch):
    """**as-of 判定必须落在 SQL 里**：查询次数与库大小无关。

    反例是把表捞进内存再逐条比时间——它"结果正确"，但代价随库增长，
    而且**没有任何症状**，直到某个用户的库大到召回变慢。
    """
    for i in range(50):
        backend.put(make_record(content=f"填充记忆 {i}"))

    target = make_record(content="目标记忆", created_at=TS_MID, valid_from=TS_MID)
    backend.put(target)

    calls = {"n": 0}
    original = backend._execute

    def counting(sql, params=()):
        calls["n"] += 1
        return original(sql, params)

    monkeypatch.setattr(backend, "_execute", counting)
    found = backend.asof(target.id, TS_AFTER)
    monkeypatch.undo()

    assert found is not None
    assert calls["n"] <= 3, (
        f"as-of 发了 {calls['n']} 次查询——像是把表捞进内存再过滤（库里有 51 条）"
    )


def test_superseded_by_self_reference_is_rejected(backend):
    """自指会让时态链成环 → as-of 永远查不出结果，而**那只是一个 `None`、不报错**。"""
    rec = make_record(content="不该被自己取代")
    backend.put(rec)

    with pytest.raises(StoreError):
        backend.update(rec.id, {"superseded_by": rec.id})


# --------------------------------------------------------------------------- #
# T-AL3-25 完整包往返（时态字段必须一起走）
# --------------------------------------------------------------------------- #


def test_import_pack_roundtrips_superseded_by(backend, tmp_path):
    """往返必须带上 `superseded_by`。

    导出侧走 `asdict()`、**一直带着它**；导入侧曾漏掉——于是"被谁取代"
    在导入后静默消失，而 V3 承诺的正是"不丢核心信息"。
    """
    old = make_record(content="旧事实", created_at=TS_OLD, valid_from=TS_OLD)
    backend.put(old)
    new = make_record(content="新事实", created_at=TS_MID, valid_from=TS_MID)
    backend.put(new)
    backend.update(old.id, {"valid_to": TS_MID, "superseded_by": new.id})

    pack = backend.export_pack()

    fresh = SQLiteBackend(str(tmp_path / "fresh.db"), embedding_dim=8)
    fresh.open()
    try:
        fresh.import_pack(pack)
        restored = fresh.get(old.id)
    finally:
        fresh.close()

    assert restored is not None, "往返后记录不见了"
    assert restored.superseded_by == new.id, "`superseded_by` 在往返中丢了"
    assert restored.valid_to == TS_MID
    assert restored.valid_from == TS_OLD


def test_import_pack_roundtrip_is_idempotent_for_temporal_fields(backend, tmp_path):
    """同一包导入两次：**零新增**，且时态字段不变（INV-14 + 幂等）。"""
    rec = make_record(content="会被导入两次的记忆", created_at=TS_OLD, valid_from=TS_OLD)
    backend.put(rec)
    pack = backend.export_pack()

    fresh = SQLiteBackend(str(tmp_path / "fresh2.db"), embedding_dim=8)
    fresh.open()
    try:
        fresh.import_pack(pack)
        before = len(fresh.query(status=None))
        fresh.import_pack(pack)
        after = len(fresh.query(status=None))
        restored = fresh.get(rec.id)
    finally:
        fresh.close()

    assert after == before, "重复导入不应产生新记录"
    assert restored.valid_from == TS_OLD
