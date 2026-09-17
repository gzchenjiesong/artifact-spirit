"""AL3 存储层：契约（``base``）+ 默认实现（``sqlite_backend``）+ 只读探针（``probe``）+ 运维工具。

分层规则 **R2**：本包**不得**依赖 ``core`` / ``extract`` / ``model`` / ``httpx``。
"""

from __future__ import annotations

from .base import (
    SCHEMA_VERSION,
    AuditEvent,
    DimensionMismatchError,
    EntityRecord,
    Hit,
    Layer,
    MemoryBackend,
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
from .ids import new_memory_id, new_ulid
from .probe import probe_schema_version, schema_version_matches
from .sqlite_backend import SQLiteBackend
from .text import content_hash_of, fts_match_query, normalize_for_fts

__all__ = [
    "SCHEMA_VERSION",
    "AuditEvent",
    "DimensionMismatchError",
    "EntityRecord",
    "Hit",
    "Layer",
    "MemoryBackend",
    "MemoryRecord",
    "NotFoundError",
    "OverviewRecord",
    "SQLiteBackend",
    "SchemaVersionError",
    "Status",
    "StorageBusyError",
    "StorageFatalError",
    "StoreError",
    "WhitelistViolation",
    "WorkingChunk",
    "content_hash_of",
    "fts_match_query",
    "new_memory_id",
    "new_ulid",
    "normalize_for_fts",
    "probe_schema_version",
    "schema_version_matches",
]
