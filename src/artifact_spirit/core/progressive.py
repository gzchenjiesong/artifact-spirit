"""分级加载 L0 / L1 / L2（LLD-AL2 §5 M9 · **P1 首要机制**）。

这是器灵**存在的原始动因之一**：贴合人类"模糊印象 → 大致判断 → 回想细节"的回忆过程，
同时优化 LLM 注意力——**不把几十年的记忆一次性灌进上下文**。

**依据**：稀缺的是**注意力**而非存储；同时在线的信息量有上限，所以第一手段是
**抽象与压缩**而不是删除——抽象提高的是有效容量（见 ``docs/design/10-神经科学依据与机制映射.md``
§2，DES-RES-003）。判据是**注意力效率**（被引用条数 / 注入条数）。

| 级 | 是什么 | 生成时机 | 存储 |
|---|---|---|---|
| **L0** | 单条记忆的一句话摘要 | 写入后异步（与提取同批） | ``memories.abstract`` |
| **L1** | 主题 / 实体级概览——"关于 X 我大致记得什么" | 冷路径生成 + 缓存 | ``overviews`` 表 |
| **L2** | 原文 + 来源 + 关联链 | 实时组装（只读） | 不新增存储 |

## 三条硬约束

1. **L1 不得在热路径生成**（评审 N-P0-2）：``prefetch`` 有 300ms 护栏，
   同步调 LLM 生成 L1 会直接击穿它。热路径发现 ``stale`` 时**只读旧缓存或降级到 L0**，
   重算投递给 AL5 maintenance。
2. **L0 缺失时退化**为 ``content`` 首句截断，不得让召回失败。
3. **L2 只读**，不触发任何写入。

三级在**信息上层层包含**（L2 ⊃ L1 ⊃ L0），因此体量天然单调——
这让"预算不足时先舍细级"有明确的操作含义。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common import estimate_tokens, first_sentence, truncate_to_tokens
from ..model.base import LLMError, LLMProvider, ProviderUnavailableError
from ..store.base import MemoryBackend, MemoryRecord
from .base import Clock, WriteIntent

__all__ = ["LEVELS", "OVERVIEW_LEVEL_L1", "ExpandResult", "ProgressiveLoader"]

LEVELS = ("L0", "L1", "L2")
OVERVIEW_LEVEL_L1 = "L1"

OVERVIEW_PIECE_BUDGET = 24
"""L1 概览里每条素材的 token 上限。

没有这个上限，"关于 X 我大致记得…"就会退化成"把 X 的原文抄一遍"，
分级加载的注意力收益随之归零（V1）。
"""


@dataclass(slots=True)
class ExpandResult:
    """展开结果。

    ``actual_level`` 可能低于 ``level``——这正是降级路径的体现（发现了要如实说出来）。
    """

    ref: str
    level: str
    text: str
    actual_level: str = "L0"
    degraded: str | None = None
    intents: list[WriteIntent] = field(default_factory=list)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)


@dataclass(slots=True)
class ProgressiveLoader:
    """三级展开。

    Args:
        backend: 存储协议。
        clock: 注入时钟。
        llm: 可选。**只有冷路径**会用它生成 L1；热路径绝不调用。
        model: L1 概览生成所用的模型名（写进 ``overviews.model``，便于追溯）。
    """

    backend: MemoryBackend
    clock: Clock
    llm: LLMProvider | None = None
    model: str = ""

    # ------------------------------------------------------------------ #
    # 对外
    # ------------------------------------------------------------------ #

    def expand(
        self, ref: str, level: str = "L0", *, hot_path: bool = True
    ) -> ExpandResult:
        """展开某条记忆到指定级别。

        Args:
            hot_path: ``True``（默认，对应 ``prefetch`` / 工具热路径）时**绝不生成 L1**；
                ``False``（``aspirit review`` 之类的冷路径）允许现场生成并缓存。
        """
        if level not in LEVELS:
            raise ValueError(f"未知级别：{level!r}，应为 {LEVELS}")

        record = self.backend.get(ref)
        if record is None:
            return ExpandResult(
                ref=ref, level=level, text="", actual_level="L0", degraded="not_found"
            )

        if level == "L0":
            return self._level0(record)

        if level == "L1":
            return self._level1(record, hot_path=hot_path)

        return self._level2(record, hot_path=hot_path)

    # ------------------------------------------------------------------ #
    # 冷路径重算（供 AL5 maintenance 调用）
    # ------------------------------------------------------------------ #

    def regenerate(
        self, scope_kind: str, scope_id: str, level: str = OVERVIEW_LEVEL_L1
    ) -> tuple[str, str] | None:
        """为一个 scope 重新生成概览文本。

        **只应由 maintenance 线程调用**——它可能做一次 LLM 调用，
        放在热路径上会直接击穿 ``prefetch`` 的 300ms 护栏（N-P0-2）。
        """
        records = self._scope_records(scope_kind, scope_id, _placeholder(scope_id))
        if not records:
            return None
        text, model_used, _ = self._compose_overview(records[0], records)
        return text, model_used

    # ------------------------------------------------------------------ #
    # 三级实现
    # ------------------------------------------------------------------ #

    def _level0(self, record: MemoryRecord) -> ExpandResult:
        """一句话摘要。缺失时**退化**为首句截断——不报错（硬约束 2）。"""
        text = record.abstract
        degraded = None
        if not text:
            text = first_sentence(record.content)
            degraded = "abstract_missing"
        return ExpandResult(
            ref=record.id, level="L0", text=text, actual_level="L0", degraded=degraded
        )

    def _level1(self, record: MemoryRecord, *, hot_path: bool) -> ExpandResult:
        scope_kind, scope_id = self._scope_of(record)
        cached = self.backend.overview_get(scope_kind, scope_id, OVERVIEW_LEVEL_L1)

        if cached is not None and not cached.stale:
            return ExpandResult(
                ref=record.id,
                level="L1",
                text=cached.content,
                actual_level="L1",
            )

        if hot_path:
            # N-P0-2：热路径只读旧缓存或降级到 L0，**不产生 LLM 调用**
            if cached is not None:
                return ExpandResult(
                    ref=record.id,
                    level="L1",
                    text=cached.content,
                    actual_level="L1",
                    degraded="stale_cache",
                    intents=[self._recompute_intent(scope_kind, scope_id)],
                )
            level0 = self._level0(record)
            return ExpandResult(
                ref=record.id,
                level="L1",
                text=level0.text,
                actual_level="L0",
                degraded="overview_missing_hot_path",
                intents=[self._recompute_intent(scope_kind, scope_id)],
            )

        return self._generate_level1(record, scope_kind, scope_id)

    def _generate_level1(
        self, record: MemoryRecord, scope_kind: str, scope_id: str
    ) -> ExpandResult:
        """冷路径：现场生成 L1 并请求缓存写入。"""
        related = self._scope_records(scope_kind, scope_id, record)
        text, model_used, degraded = self._compose_overview(record, related)
        intent = WriteIntent(
            op="overview_put",
            scope_kind=scope_kind,
            scope_id=scope_id,
            overview_content=text,
            overview_level=OVERVIEW_LEVEL_L1,
            token_count=estimate_tokens(text),
            model=model_used,
            actor="system",
        )
        return ExpandResult(
            ref=record.id,
            level="L1",
            text=text,
            actual_level="L1",
            degraded=degraded,
            intents=[intent],
        )

    def _compose_overview(
        self, record: MemoryRecord, related: list[MemoryRecord]
    ) -> tuple[str, str, str | None]:
        """组合出"关于 X 我大致记得…"。

        有 LLM 就交给它（质量更好）；不可用时**退化为摘要拼接**——
        一处朴素的拼接远好过一个失败的分级加载（P 档优先于 A 档）。
        """
        scope_label = record.subject or record.object or record.type
        # 每条素材都截断——L1 是「大致记得」，不是「逐条复述全文」
        pieces = [
            truncate_to_tokens(r.abstract or first_sentence(r.content), OVERVIEW_PIECE_BUDGET)
            for r in related
        ]
        pieces = [p for p in pieces if p]

        if self.llm is not None and pieces:
            try:
                text = self.llm.complete(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你是记忆系统的概览生成器。把这些记忆要点压缩成一段"
                                "不超过 200 字的中文概览，保留关键事实与时间线索，"
                                "不要编造、不要评论。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"主题：{scope_label}\n要点：\n"
                            + "\n".join(f"- {p}" for p in pieces),
                        },
                    ]
                )
                return f"关于「{scope_label}」：{text.strip()}", self.model or "llm", None
            except (LLMError, ProviderUnavailableError):
                pass  # 落到拼接路径

        body = "；".join(pieces) if pieces else first_sentence(record.content)
        return (
            f"关于「{scope_label}」：{body}",
            "heuristic",
            "llm_unavailable",
        )

    def _level2(self, record: MemoryRecord, *, hot_path: bool) -> ExpandResult:
        """原文 + 来源 + 关联链。**只读**（硬约束 3）。

        L2 **包含** L1——"努力回想"拿到的是要点加上细节，而不是取代要点。
        """
        l1 = self._level1(record, hot_path=hot_path)
        lines = [l1.text, "", "【详细】", record.content]

        if record.valid_from or record.source_session:
            source = []
            if record.source_session:
                source.append(f"会话 {record.source_session}")
            if record.valid_from:
                source.append(f"始于 {record.valid_from}")
            source.append(f"置信度 {record.confidence:.2f}")
            lines.extend(["", "【来源】" + " · ".join(source)])

        relations = self.backend.neighbors("memory", record.id, limit=10)
        if relations:
            lines.extend(["", "【关联】"])
            for kind, node_id, weight in relations:
                if kind != "memory":
                    continue
                lines.append(f"- {node_id}（权重 {weight:.2f}）")

        return ExpandResult(
            ref=record.id,
            level="L2",
            text="\n".join(lines),
            actual_level="L2",
            degraded=l1.degraded,
            intents=list(l1.intents),
        )

    # ------------------------------------------------------------------ #
    # scope 解析
    # ------------------------------------------------------------------ #

    def _scope_of(self, record: MemoryRecord) -> tuple[str, str]:
        """L1 概览的归属：优先实体，其次主题名，最后退化到记忆自身。

        以 ``entities`` 为主、``scope`` 为辅（LLD-AL2 §10 未决项 12 的建议）。
        """
        for term in (record.subject, record.object):
            if not term:
                continue
            matches = self.backend.entity_find(term)
            if matches:
                return "entity", matches[0].id
            return "topic", term
        scope = record.scope or {}
        if scope.get("id"):
            return "scope", str(scope["id"])
        return "memory", record.id

    def _scope_records(
        self, scope_kind: str, scope_id: str, record: MemoryRecord
    ) -> list[MemoryRecord]:
        """该 scope 下的相关记忆（生成 L1 的素材）。"""
        if scope_kind == "entity":
            entity = self.backend.entity_get(scope_id)
            if entity is not None:
                found = [
                    r
                    for r in self.backend.query(status=None, limit=200)
                    if entity.name.casefold() in f"{r.subject or ''} {r.object or ''} {r.content}".casefold()
                ]
                if found:
                    return found[:20]
        if scope_kind == "topic":
            found = self.backend.query(status=None, limit=200)
            matched = [
                r
                for r in found
                if scope_id.casefold() in f"{r.subject or ''} {r.object or ''} {r.content}".casefold()
            ]
            if matched:
                return matched[:20]
        return [record]

    def _recompute_intent(self, scope_kind: str, scope_id: str) -> WriteIntent:
        """请求 maintenance 重算该 scope 的 L1（热路径不自己动手）。"""
        return WriteIntent(
            op="overview_invalidate",
            scope_kind=scope_kind,
            scope_id=scope_id,
            actor="system",
        )


def _placeholder(scope_id: str) -> MemoryRecord:
    """``_scope_records`` 需要一个占位记录以处理"没有对应记忆"的情形。"""
    return MemoryRecord(
        id=scope_id, layer="semantic", type="entity", content=scope_id
    )
