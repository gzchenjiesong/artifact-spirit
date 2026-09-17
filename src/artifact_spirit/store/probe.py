"""AL3 只读探针：**不打开完整后端**地问一句"这个库还能用吗"。

## 为什么单独一个模块

AL1 的 `is_available()` 需要在**宿主装载器阶段**回答"库是否可用"，
而这个时机不允许迁移、不允许建连池、不允许任何副作用（契约要求返回 bool、不抛）。
完整后端（`sqlite_backend.SQLiteBackend`）做不到——`open()` 会施加 PRAGMA、
加载 sqlite-vec、跑迁移，全是副作用。

于是这里给出**最小只读面**：只用标准库 `sqlite3` 开一条 `mode=ro` 连接读
`meta.schema_version`。它与 `sqlite_backend` 共享同一个 `SCHEMA_VERSION`
（来自 `base.py`），因此不存在"两份版本真相"。

## 纪律

- **只读**：连接串固定 `mode=ro`，不写库、不加写锁、不迁移。
- **不抛**：任何异常都收敛成 `None` / `False`——调用方是"探活"，不是"开库"。
- 本模块**不得** import `sqlite_backend`（那会把 sqlite-vec 拖进 AL1 的预检路径）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .base import SCHEMA_VERSION

__all__ = ["probe_schema_version", "schema_version_matches"]

# ``meta.schema_version`` 的键名是 AL3 的存储约定；探针内部自带一份字面量，
# 避免 import 完整的 ``sqlite_backend``（见模块开头"不得 import"的纪律）。
# 该字面量与 ``sqlite_backend.META_SCHEMA_VERSION`` 的一致性由 test_store_probe 守住。
_META_SCHEMA_VERSION = "schema_version"


def probe_schema_version(db_path: str | Path) -> int | None:
    """用**只读**连接读取库内 ``meta.schema_version``；不可读/无表/无值则返回 ``None``。

    调用方不必知道 ``meta`` 表长什么样、键名是什么——那属于 AL3 的存储知识，
    复刻到别处就会出现两份会各自漂移的真相。
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (_META_SCHEMA_VERSION,)
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def schema_version_matches(db_path: str | Path) -> bool:
    """``probe_schema_version(db_path) == SCHEMA_VERSION`` 的语义化封装。

    这是 AL1 `is_available()` 里"DB 中 schema_version 与代码期望一致"（LLD-AL1 验收 3）
    的唯一判定入口。
    """
    return probe_schema_version(db_path) == SCHEMA_VERSION
