"""审查、溯源与主动干预（LLD-AL2 §5 M11 · **P2 首要机制**）。

这是器灵的另一个原始动因：**运行一段时间后能逐条看到 Agent 记住了什么**，
从而具备复盘与主动优化的能力。人类可读不是"附带好处"，是**一等公民**。

| 能力 | 接口 | 一句话 |
|---|---|---|
| **审查** | :meth:`Reviewer.review` | 逐条列出：层 / 内容 / 摘要 / 时间 / 置信度 / 来源 / 状态 / **importance 及其构成** |
| **溯源** | :meth:`Reviewer.trace` | 该条记忆的完整变更史（从 ``audit`` 重放） |
| **干预** | :meth:`Reviewer.correct` / :meth:`Reviewer.forget` | 修正内容 / 调层 / 改状态 / 调置信度 / 删除 |

## 两条红线

1. **干预必须走正常写入路径**（产出 ``WriteIntent`` → AL5 writer → AL3），
   **不得绕过 ``audit``**——否则"可审查"就成了空话。
2. **``forget`` 是物理删除的唯一人工收口**（D-17 入口 2），``reason`` 必填、
   ``source`` 必填——**删除请求本身也是可审计要素**（用户明确指出）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..store.base import AuditEvent, MemoryBackend, MemoryRecord
from .base import Clock, WriteIntent
from .layers.core_memory import CORE_TYPES
from .recall import IMPORTANCE_WEIGHTS, REFERENCE_SATURATION, score_importance

__all__ = ["COERCIBLE_PATCH_KEYS", "Reviewer"]

COERCIBLE_PATCH_KEYS = frozenset(
    {
        "content",
        "abstract",
        "subject",
        "predicate",
        "object",
        "scope",
        "confidence",
        "salience",
        "layer",
        "status",
        "valid_from",
        "valid_to",
    }
)
"""人工干预允许修改的字段。**不含 `id` / `created_at` / `content_hash`**——
前者是身份，中者是历史，后者是派生值（改了会让幂等锚点失真）。"""

_DEMOTE_TARGETS = ("episodic", "semantic", "procedural")
"""降格的合法目标层。**不含 `core`**（那是升格）——也不含 `sensory` / `working`：
那两层不落表，降到哪里等于"这条记忆消失了"，而用户要的是"降级"不是"删除"。"""


@dataclass(slots=True)
class Reviewer:
    """审查 / 溯源 / 干预。"""

    backend: MemoryBackend
    clock: Clock
    confirm: Callable[[str], bool] | None = None
    """可选的确认回调——破坏性干预（删除）在无人值守时需要它。"""

    # ------------------------------------------------------------------ #
    # 审查
    # ------------------------------------------------------------------ #

    def review(
        self, *, layer: str | None = None, since: str | None = None, limit: int = 50
    ) -> list[dict]:
        """逐条审查视图。

        ``importance`` 与它的**构成**一并给出（D-20 明文要求）——用户要能看懂
        "为什么这条排在前面"，否则"可审查"只完成了一半。
        """
        records = self.backend.query(layer=layer, status=None, since=since, limit=limit)
        return [self._review_row(record) for record in records]

    def _review_row(self, record: MemoryRecord) -> dict:
        refs = self.backend.inbound_reference_counts([record.id]).get(record.id, 0)
        breakdown = {
            "user_label": 0.0,
            "confidence": IMPORTANCE_WEIGHTS["confidence"] * record.confidence,
            "salience": IMPORTANCE_WEIGHTS["salience"] * record.salience,
            "referenced": IMPORTANCE_WEIGHTS["referenced"]
            * min(1.0, refs / REFERENCE_SATURATION),
        }
        return {
            "id": record.id,
            "layer": record.layer,
            "type": record.type,
            "content": record.content,
            "abstract": record.abstract,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "confidence": record.confidence,
            "salience": record.salience,
            "status": record.status,
            "source_session": record.source_session,
            "content_hash": record.content_hash,
            "importance": round(score_importance(record, inbound_refs=refs), 4),
            "importance_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "inbound_refs": refs,
        }

    # ------------------------------------------------------------------ #
    # 溯源
    # ------------------------------------------------------------------ #

    def trace(self, mem_id: str) -> list[dict]:
        """某条记忆的完整变更史（按时间序）。"""
        events = self.backend.audit_for(mem_id)
        return [
            {
                "audit_id": index,
                "ts": event.ts,
                "op": event.op,
                "actor": event.actor,
                "reason": event.reason,
                "before": event.before,
                "after": event.after,
            }
            for index, event in enumerate(events, start=1)
        ]

    # ------------------------------------------------------------------ #
    # 干预
    # ------------------------------------------------------------------ #

    def correct(
        self, mem_id: str, patch: dict, *, reason: str
    ) -> list[WriteIntent]:
        """修正某条记忆。**产出 ``WriteIntent``，不直接写库**。"""
        record = self.backend.get(mem_id)
        if record is None:
            raise KeyError(f"记忆不存在：{mem_id}")
        if not reason or not reason.strip():
            raise ValueError("干预必须给出 reason——可审查的前提是每次都留痕（P5）")

        rejected = sorted(set(patch) - COERCIBLE_PATCH_KEYS)
        if rejected:
            raise ValueError(f"不允许人工修改这些字段：{rejected}")

        intents = [
            WriteIntent(
                op="update",
                mem_id=mem_id,
                patch=dict(patch),
                actor="user",
                audit=AuditEvent(
                    op="update",
                    actor="user",
                    target_kind="memory",
                    target_id=mem_id,
                    before={"changed": sorted(patch)},
                    after={"changed": sorted(patch)},
                    reason=reason,
                ),
            )
        ]
        # 内容变了 → 相关 L1 概览失效（否则"记忆可审"与"概览"会各说各话）
        if {"content", "abstract", "subject", "object"} & set(patch):
            scope_kind, scope_id = self._scope_of(record)
            intents.append(
                WriteIntent(
                    op="overview_invalidate",
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    actor="user",
                )
            )
        return intents

    def invalidate_since(self, mem_id: str, *, from_ts: str, reason: str) -> list[WriteIntent]:
        """**时态干预**：这条从 `from_ts` 起不再成立（T-AL2-22）。

        与 :meth:`correct` 的区别在**语义**：

        - `correct` 说"**我们记错了**" → 改这条记录本身；
        - 时态干预说"**事实变了**" → 只截断有效期。

        后者**不覆盖历史**：`asof` 在 `from_ts` 之前依然返回它（INV-7）。
        把"搬家"当成"改错别字"处理，正是 M3 之前那个把两件事混为一谈的老毛病。
        """
        if not reason or not reason.strip():
            raise ValueError("时态干预必须给出 reason——干预一律留痕（P5）")
        record = self.backend.get(mem_id)
        if record is None:
            raise KeyError(f"记忆不存在：{mem_id}")
        if record.valid_from and from_ts <= record.valid_from:
            raise ValueError(
                f"截断时刻（{from_ts}）必须晚于该条的 valid_from（{record.valid_from}）"
                "——否则它的有效期会成为空区间（或负数），任何时刻都查不到"
            )
        return [
            WriteIntent(
                op="update",
                mem_id=mem_id,
                patch={"valid_to": from_ts},
                actor="user",
                reason=reason,
                audit=AuditEvent(
                    op="invalidate",
                    actor="user",
                    target_kind="memory",
                    target_id=mem_id,
                    before={"valid_to": record.valid_to},
                    after={"valid_to": from_ts},
                    reason=reason,
                ),
            )
        ]

    def promote(
        self, mem_id: str, *, reason: str, as_type: str = "identity"
    ) -> list[WriteIntent]:
        """**核心记忆干预：升格**（T-AL2-22）——「这条是我的底色」。

        与自动升格（`CoreMemoryLayer.propose`）的关键差别：**这里不看置信度门槛**。
        门槛是给机器设的——用户说它是人格的一部分，它就是。

        `as_type` **不能省**：`prompt_block` 是按 `type ∈ {identity, soul}` 取内容的，
        只改 `layer` 不改 `type` 的结果是——**升格"成功"了，系统提示里却看不见它**。
        那正是"看起来做了、实际没做"的典型形态。
        """
        if not reason or not reason.strip():
            raise ValueError("升格必须给出 reason——干预一律留痕（P5）")
        record = self.backend.get(mem_id)
        if record is None:
            raise KeyError(f"记忆不存在：{mem_id}")
        if record.layer == "core":
            raise ValueError(f"{mem_id} 已经是核心记忆——重复升格只会多一条没有信息量的审计")
        if as_type not in CORE_TYPES:
            raise ValueError(f"核心记忆的 type 只能是 {CORE_TYPES}，收到 {as_type!r}")
        return [
            WriteIntent(
                op="update",
                mem_id=mem_id,
                patch={"layer": "core", "type": as_type},
                actor="user",
                reason=reason,
                audit=AuditEvent(
                    op="promote",
                    actor="user",
                    target_kind="memory",
                    target_id=mem_id,
                    before={"layer": record.layer, "type": record.type},
                    after={"layer": "core", "type": as_type},
                    reason=reason,
                ),
            )
        ]

    def demote(
        self, mem_id: str, *, reason: str, target_layer: str = "semantic"
    ) -> list[WriteIntent]:
        """**核心记忆干预：降格**——「它不再是我的底色了」。

        降格**也要留审计**："它曾经是"本身是人格史的一部分（INV-12）。
        它也**不删内容**——只是放回普通层，照样参与召回。
        """
        if not reason or not reason.strip():
            raise ValueError("降格必须给出 reason——干预一律留痕（P5）")
        record = self.backend.get(mem_id)
        if record is None:
            raise KeyError(f"记忆不存在：{mem_id}")
        if record.layer != "core":
            raise ValueError(f"只有核心记忆能降格——{mem_id} 现在是 {record.layer}")
        if target_layer not in _DEMOTE_TARGETS:
            raise ValueError(f"降格目标层只能是 {_DEMOTE_TARGETS}，收到 {target_layer!r}")
        return [
            WriteIntent(
                op="update",
                mem_id=mem_id,
                patch={"layer": target_layer},
                actor="user",
                reason=reason,
                audit=AuditEvent(
                    op="demote",
                    actor="user",
                    target_kind="memory",
                    target_id=mem_id,
                    before={"layer": "core"},
                    after={"layer": target_layer},
                    reason=reason,
                ),
            )
        ]

    def forget(
        self,
        mem_id: str,
        *,
        reason: str,
        source: str,
        purge_snapshot: bool = False,
        require_confirm: bool = False,
    ) -> list[WriteIntent]:
        """**物理删除的唯一人工收口**（D-17 入口 2）。

        - ``reason`` 与 ``source`` **必填**——删除请求本身要可审计（用户明确指出）
        - 默认保留快照 → 可经 :meth:`restore` 恢复（D-22 安全网）
        - ``purge_snapshot=True`` 仅用于**合规删除**（不可恢复，D-22 唯一例外）
        """
        if not reason or not reason.strip():
            raise ValueError("删除必须给出 reason（D-17）")
        if not source or not source.strip():
            raise ValueError("删除必须给出 source——'从哪个入口发起的'也要留痕（D-22）")
        if self.backend.get(mem_id) is None:
            raise KeyError(f"记忆不存在：{mem_id}")
        if require_confirm and self.confirm is not None and not self.confirm(mem_id):
            return []

        return [
            WriteIntent(
                op="forget",
                mem_id=mem_id,
                reason=reason,
                actor="user",
                purge_snapshot=purge_snapshot,
                audit=AuditEvent(
                    op="forget",
                    actor="user",
                    target_kind="memory",
                    target_id=mem_id,
                    reason=reason,
                    after={"source": source, "purge_snapshot": purge_snapshot},
                ),
            )
        ]

    def restore(self, audit_id: int) -> list[WriteIntent]:
        """从删除快照恢复（D-22）。恢复本身也写 ``audit(op='restore')``。"""
        return [
            WriteIntent(
                op="restore",
                audit_id=audit_id,
                actor="user",
                audit=AuditEvent(
                    op="restore",
                    actor="user",
                    target_kind="memory",
                    after={"restored_from_audit": audit_id},
                    reason=f"restore from audit {audit_id}",
                ),
            )
        ]

    # ------------------------------------------------------------------ #

    def _scope_of(self, record: MemoryRecord) -> tuple[str, str]:
        for term in (record.subject, record.object):
            if term:
                matches = self.backend.entity_find(term)
                if matches:
                    return "entity", matches[0].id
                return "topic", term
        return "memory", record.id


# 人可读渲染（原 `format_review_text` / `format_trace_text`）**已移至 AL5**
# `observability/render.py`：按 INV-1「文本是投影」，人类可读属于投影关注点，
# 不是领域逻辑。AL2 只负责产出数据结构（`review()` / `trace()` 的 `list[dict]`），
# 渲染由 AL5 承担，从而消除 AL5 → AL2 具体模块的反向依赖（DES-000 §4.1）。
