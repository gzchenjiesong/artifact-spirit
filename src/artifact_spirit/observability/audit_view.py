"""审计视图（LLD-AL5 §3 · 只读投影）。

**可读 ≠ 可溯**：`status` 给"现在是什么状态"，审计给"它是怎么变成这样的"。

这是 V2「记忆可见、可审、可干预」的复盘入口——用户问"器灵到底背着我干了什么"，
答案在这里。

## 本模块真正要紧的是"删除账本"

设计上**不做删除前确认**：`spirit_forget` 默认预演、需显式 `confirm`，而安全性
交给**事后可审计 + 可恢复**兜底。这个取舍要成立，审计视图就必须同时做到两件事：

1. **说清删了什么、为什么** —— 否则用户没法判断这个操作对不对；
2. **说清还能不能恢复、怎么恢复** —— 否则"发现不对可以要求恢复"落不了地。

只要其中一件缺失，那条取舍就变成了赌博。所以 ``entries()`` 对删除类事件会补上
``restorable`` / ``deleted_preview`` / ``restore_command`` 三个字段，文本渲染也把它们
显式打出来 —— 使用者扫一眼就知道"哪条要质疑、能怎么救"。

（这也是被真机逼出来的：模型曾自主删掉 9 条被它判定为"被新事实取代"的记忆。
那次删除有留痕、有快照、reason 具体，所以毫无争议地可复盘 —— 但那次的审计视图
只显示 ``forget user <id> · 原因``，用户看不出"能不能救回来"。补的就是这一点。）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..store.base import MemoryBackend

__all__ = ["AuditView", "format_audit_text", "format_restorable_text"]

# 删除类操作。目前只有 forget；一旦 M3 引入"失效标记"（INVALIDATE），
# 它**不该**被算进这里——失效是可逆的语义变更，不是删除。
_DELETE_OPS = frozenset({"forget"})

# 删除内容在账本里的展示长度。够看清"删的是什么"，又不至于把一行撑爆。
_PREVIEW_CHARS = 60


def _preview(text: str | None) -> str:
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= _PREVIEW_CHARS:
        return collapsed
    return collapsed[: _PREVIEW_CHARS - 1] + "…"


@dataclass(slots=True)
class AuditView:
    """审计日志的只读视图。"""

    backend: MemoryBackend

    def entries(
        self, *, since: str | None = None, limit: int = 50, actor: str | None = None
    ) -> list[dict]:
        """按时间**倒序**返回审计条目（最近的在最前面）。

        删除类条目会额外带上：

        - ``restorable``：是否还有可恢复快照（``False`` 只可能是显式合规清除）
        - ``deleted_preview``：被删内容的摘要（来自快照）
        - ``restore_command``：可直接复制执行的恢复命令
        - ``relation_count``：被删时一并快照的关联边数量

        快照**一次取完**再做索引，避免每个删除事件查一次库。
        """
        # 一次取齐（含 payload），建 audit_id → 快照 的索引
        snapshots = {
            snapshot["audit_id"]: snapshot
            for snapshot in self.backend.delete_snapshots(include_payload=True)
        }

        rows: list[dict] = []
        for event in self.backend.audit_replay(since=since):
            # 用**真实的** audit.id，不是"过滤结果里的第几条"。
            # 用 enumerate 凑数会在 `--since` 下错位，而用户照着一个错位的编号
            # 去 restore，恢复的就是另一条记忆。
            audit_id = event.audit_id
            row: dict = {
                "audit_id": audit_id,
                "ts": event.ts,
                "op": event.op,
                "actor": event.actor,
                "target_kind": event.target_kind,
                "target_id": event.target_id,
                "reason": event.reason,
                "session_id": event.session_id,
            }
            if event.op in _DELETE_OPS:
                snapshot = snapshots.get(audit_id) if audit_id is not None else None
                record = (snapshot or {}).get("record") or {}
                row["restorable"] = snapshot is not None
                row["relation_count"] = int((snapshot or {}).get("relation_count") or 0)
                row["deleted_preview"] = _preview(record.get("content"))
                row["deleted_layer"] = record.get("layer")
                row["restore_command"] = (
                    f"aspirit restore {audit_id}" if snapshot is not None else None
                )
            rows.append(row)

        if actor:
            rows = [row for row in rows if row["actor"] == actor]
        return list(reversed(rows))[:limit]

    def forgetting(self, *, limit: int = 50) -> list[dict]:
        """只看删除事件——**这是最容易出事、也最需要被看见的一类操作**。"""
        return [row for row in self.entries(limit=1000) if row["op"] in _DELETE_OPS][:limit]

    def restorable(self) -> list[dict]:
        """当前可恢复的删除快照（有快照 = 还能救回来），含被删内容。"""
        return self.backend.delete_snapshots(include_payload=True)


def format_audit_text(entries: list[dict], *, now: str | None = None) -> str:
    """渲染成人可读文本。删除类条目**额外渲染内容、原因与可恢复性**。"""
    del now  # 保留参数位：调用方可能传入"当前时间"用于相对时间标注
    if not entries:
        return "（审计日志为空）"

    lines = [f"审计日志 · 最近 {len(entries)} 条（新的在前）", ""]
    for row in entries:
        target = row.get("target_id") or row.get("target_kind") or ""
        reason = f" · {row['reason']}" if row.get("reason") else ""
        lines.append(
            f"[{row['ts']}] {row['op']:<12} {row['actor']:<12} {target}{reason}"
        )

        if row["op"] in _DELETE_OPS:
            if row.get("deleted_preview"):
                layer = row.get("deleted_layer") or "?"
                lines.append(f"    ↳ 删除内容（{layer}）：{row['deleted_preview']}")
            if row.get("relation_count"):
                lines.append(f"    ↳ 一并快照的关联边：{row['relation_count']} 条")
            if row.get("restorable"):
                lines.append(f"    ↳ 可恢复：是 · 执行 `{row['restore_command']}`")
            else:
                lines.append(
                    "    ↳ 可恢复：否 —— 该删除已显式要求清除内容（合规删除，不可重建）"
                )
    return "\n".join(lines)


def format_restorable_text(snapshots: list[dict], *, limit: int = 50) -> str:
    """渲染"可恢复清单"——出事时的救援面板。"""
    if not snapshots:
        return "可恢复的删除：无（没有待恢复的快照）"

    shown = snapshots[:limit]
    lines = [
        f"可恢复的删除 · 共 {len(snapshots)} 条（新的在前）",
        "",
    ]
    for snapshot in shown:
        record = snapshot.get("record") or {}
        content = _preview(record.get("content"))
        layer = record.get("layer") or "?"
        rels = snapshot.get("relation_count") or 0
        rel_note = f" · 关联边 {rels}" if rels else ""
        lines.append(
            f"  #{snapshot['audit_id']:<5} [{layer}]{rel_note}  {content}"
        )
    if len(snapshots) > len(shown):
        lines.append(f"  …另有 {len(snapshots) - len(shown)} 条")
    lines += [
        "",
        "恢复某一条：aspirit restore <编号>",
        "（恢复会沿用原 ID 与原关联边，并在审计里记一条 restore）",
    ]
    return "\n".join(lines)
