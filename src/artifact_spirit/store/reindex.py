"""全库重嵌入（T-AL3-20 · INV-2）。

**为什么必须有这个工具**：向量空间与 embedding 模型强绑定——换模型后旧向量全部失效。
没有重嵌入能力，使用者一旦换模型就只能"记忆归零"，那正好违背核心价值 V3。
所以它不是运维便利，而是**资产可迁移性的组成部分**。

支持**断点续跑**：进度以 ``meta.reindex_state`` 记录（按 ``rowid`` 游标），
中断后重启从断点继续，不重复处理已完成项。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

from ..common import now_iso
from .sqlite_backend import META_EMBEDDING_DIM, META_EMBEDDING_MODEL, SQLiteBackend

__all__ = ["META_REINDEX_STATE", "reindex", "reindex_state"]

META_REINDEX_STATE = "reindex_state"


def reindex_state(backend: SQLiteBackend) -> dict:
    """读取当前重嵌入进度。"""
    raw = backend.meta_get(META_REINDEX_STATE)
    return json.loads(raw) if raw else {}


def _write_state(backend: SQLiteBackend, state: dict) -> None:
    backend.meta_set(META_REINDEX_STATE, json.dumps(state, ensure_ascii=False))


def _rebuild_vector_table(backend: SQLiteBackend, dim: int) -> None:
    """按新维度重建 ``vec_memories``（vec0 的维度建表后不可改）。"""
    with backend._tx() as conn:
        conn.execute("DROP TABLE IF EXISTS vec_memories")
        conn.execute(
            "CREATE VIRTUAL TABLE vec_memories USING vec0("
            "  mem_id TEXT PRIMARY KEY,"
            f"  embedding float[{int(dim)}]"
            ")"
        )


def reindex(
    backend: SQLiteBackend,
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    model: str | None = None,
    dim: int | None = None,
    batch_size: int = 32,
    resume: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """全库重嵌入。

    Args:
        embed_fn: ``(texts) -> vectors``，由 AL4 的 embedding provider 提供。
        model: 目标模型名；与 ``dim`` 一起决定是否需要重建向量表。
        dim: 目标维度。
        resume: 断点续跑。``False`` 表示从头开始（忽略已有游标）。

    Returns:
        ``{"total":…, "done":…, "skipped":…, "rebuilt_table": bool, "model":…, "dim":…}``
    """
    target_model = model or backend.embedding_model or ""
    target_dim = int(dim or backend.embedding_dim)

    state = reindex_state(backend)
    same_task = (
        state.get("model") == target_model
        and int(state.get("dim") or 0) == target_dim
    )

    rebuilt = False
    if not same_task:
        # 新任务：重建向量表（维度可能已变），游标归零
        _rebuild_vector_table(backend, target_dim)
        backend.embedding_dim = target_dim
        backend.embedding_model = target_model or backend.embedding_model
        state = {"model": target_model, "dim": target_dim, "last_rowid": 0}
        rebuilt = True
    cursor = int(state.get("last_rowid") or 0) if resume else 0

    total_row = backend.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
    total = int(total_row["n"])

    done = 0
    batch: list[tuple[int, str]] = []

    def _flush(rows: list[tuple[int, str]]) -> None:
        nonlocal done, cursor
        if not rows:
            return
        texts = [text for _, text in rows]
        vectors = embed_fn(texts)
        if len(vectors) != len(rows):
            raise ValueError(
                f"embedding 返回数量不符：期望 {len(rows)}，收到 {len(vectors)}"
            )
        with backend._tx() as conn:
            for (rowid, _), vec in zip(rows, vectors, strict=False):
                if len(vec) != target_dim:
                    raise ValueError(
                        f"向量维度不符：期望 {target_dim}，收到 {len(vec)}"
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO vec_memories(mem_id, embedding) "
                    "SELECT id, ? FROM memories WHERE rowid = ?",
                    (json.dumps(vec), rowid),
                )
        done += len(rows)
        cursor = rows[-1][0]
        _write_state(backend, {"model": target_model, "dim": target_dim, "last_rowid": cursor})
        if on_progress is not None:
            on_progress(done, total)

    rows = backend.conn.execute(
        "SELECT rowid, id, content, abstract FROM memories "
        "WHERE rowid > ? AND status != 'forgotten' ORDER BY rowid",
        (cursor,),
    )
    for row in rows:
        batch.append((int(row["rowid"]), row["abstract"] or row["content"]))
        if len(batch) >= max(1, batch_size):
            _flush(batch)
            batch = []
    _flush(batch)

    # 更新元信息：模型与维度
    if target_model:
        backend.meta_set(META_EMBEDDING_MODEL, target_model)
    backend.meta_set(META_EMBEDDING_DIM, str(target_dim))

    with backend._tx() as conn:
        conn.execute(
            """INSERT INTO audit(ts, op, actor, target_kind, target_id, after, reason)
               VALUES (?,?,?,?,?,?,?)""",
            (
                now_iso(),
                "reindex",
                "cli",
                "system",
                "vec_memories",
                json.dumps(
                    {"model": target_model, "dim": target_dim, "done": done},
                    ensure_ascii=False,
                ),
                "全库重嵌入",
            ),
        )

    return {
        "total": total,
        "done": done,
        "skipped": max(0, total - done),
        "rebuilt_table": rebuilt,
        "model": target_model,
        "dim": target_dim,
        "resumed_from": cursor if resume and not rebuilt else 0,
    }


def iter_records(backend: SQLiteBackend) -> Iterable[dict]:  # pragma: no cover - 便捷入口
    """遍历全部记忆（供外部工具复用）。"""
    for row in backend.conn.execute("SELECT * FROM memories ORDER BY rowid"):
        yield dict(row)
