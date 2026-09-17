"""记忆进化（T-AL2-20 · M4）。

借鉴 A-MEM：**新记忆到来时允许改写旧记忆的抽象与关联**，而不是只做 ADD。

## 为什么值得做

只做 ADD 的记忆系统会**只增不减地变臃肿**：同一件事的十次表述各占一条，
每条都只讲了其中一部分。进化让"后来的信息"能反过来完善"先前的条目"——
这正是人类记忆的运作方式（新经验会重写旧经验的概括）。

## 三条硬边界

1. **只改 `abstract`，永不改 `content`** —— 原文是事实来源，抽象是投影。
   这条线一旦松掉，"用户的原话"就没有权威副本了（与 INV-1 同一条精神）。
2. **每次改写留审计（含 `before` / `after`）** —— 抽象被改了却查不出来，
   比不改更糟：用户会看到一句自己没说过、也追溯不到来源的"自己的总结"（INV-12）。
3. **可以关掉** —— `enabled=False` 时行为完全退回纯 ADD。代价大的能力必须能关，
   否则一次误判会变成"用户关不掉的持续扣费"。

## 位置：不在热路径

与交叉验证同理（INV-11）：一次 LLM 调用是秒级的，塞进 `sync_turn` 直接违反 INV-4。
由 maintenance 周期调用。

## 降级

LLM 不可用、输出非法、或模型回答"不需要改" → **一律不改**。
理由与交叉验证一致：**代价不对称**——不改只是少了一点点信息增益，
改错却会污染一条本来正确的记忆。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common import now_iso, truncate_to_tokens
from ..model.base import LLMError, LLMProvider, ProviderUnavailableError
from ..store.base import MemoryBackend, MemoryRecord
from .base import AuditEvent, Clock, EvolutionReport, WriteIntent

__all__ = ["Evolver"]

_SCHEMA = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "abstract": {"type": "string"},
    },
    "required": ["abstract"],
}


@dataclass(slots=True)
class Evolver:
    """记忆进化。**只产出写意图，不落库**（R5）。"""

    backend: MemoryBackend
    llm: LLMProvider | None = None
    clock: Clock | None = None
    enabled: bool = True
    limit: int = 5
    """单次最多改写几条邻居——进化是"锦上添花"，不该把一轮维护撑爆。"""
    max_abstract_tokens: int = 60
    """新抽象的预算。抽象必须比正文短，否则 L0 就退化成 L2（V1 的意义随之消失）。"""

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = now_iso

    # ------------------------------------------------------------------ #

    def evolve_text(self, *, old_abstract: str, new_content: str) -> str | None:
        """问模型"新信息是否让这条旧抽象需要更新"。返回新抽象，或 `None`（不改）。"""
        if self.llm is None:
            return None
        try:
            payload = self.llm.complete_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你在一句话摘要一条记忆的旧概括。现在有一条相关的新记忆，"
                            "判断它是否让旧概括需要更新（补全、纠正、收窄都算）。"
                            '只输出 JSON：{"changed": true|false, "abstract": "…"}。'
                            "**不要编造旧概括里没有、新记忆里也没有的内容**；"
                            "若新记忆与旧概括无关，就把 changed 设为 false。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"旧概括：{old_abstract}\n新记忆：{new_content}",
                    },
                ],
                schema=_SCHEMA,
            )
        except (LLMError, ProviderUnavailableError):
            return None

        new_abstract = str(payload.get("abstract") or "").strip()
        if not new_abstract or new_abstract == old_abstract.strip():
            return None
        if payload.get("changed") is False:
            return None
        # 抽象超预算就截断；截到空则视为"不值得改"
        return truncate_to_tokens(new_abstract, self.max_abstract_tokens) or None

    def evolve(
        self, new_record: MemoryRecord, *, neighbors: list[MemoryRecord]
    ) -> list[WriteIntent]:
        """用新记忆去完善它的邻居。**只动 `abstract`**。"""
        if not self.enabled:
            return []

        intents: list[WriteIntent] = []
        for old in neighbors[: self.limit]:
            if old.id == new_record.id or not old.abstract:
                continue
            updated = self.evolve_text(
                old_abstract=old.abstract, new_content=new_record.content
            )
            if updated is None:
                continue
            reason = f"记忆进化：{new_record.id} 带来新信息"
            intents.append(
                WriteIntent(
                    op="update",
                    mem_id=old.id,
                    patch={"abstract": updated},
                    actor="consolidator",
                    reason=reason,
                    embed_text=updated,
                    audit=AuditEvent(
                        op="update",
                        actor="consolidator",
                        target_kind="memory",
                        target_id=old.id,
                        before={"abstract": old.abstract},
                        after={"abstract": updated},
                        reason=reason,
                    ),
                )
            )
        return intents

    # ------------------------------------------------------------------ #

    def run(self, *, limit: int = 20) -> EvolutionReport:
        """批量：对最近活跃的记忆，尝试进化它们的邻居。"""
        report = EvolutionReport()
        if not self.enabled:
            report.skipped = "进化未启用"
            return report
        if self.llm is None:
            report.skipped = "未配置 LLM"
            return report

        recent = self.backend.query(status="active", limit=limit)
        report.scanned = len(recent)
        for record in recent:
            neighbors = [
                n for n in self._neighbor_records(record) if n.id != record.id and n.abstract
            ]
            if not neighbors:
                continue
            intents = self.evolve(record, neighbors=neighbors)
            if intents:
                report.evolved += len(intents)
                report.intents.extend(intents)
        return report

    def _neighbor_records(self, record: MemoryRecord) -> list[MemoryRecord]:
        """这条记忆的一跳邻居（只取记忆节点，实体节点跳过）。"""
        out: list[MemoryRecord] = []
        for kind, node_id, _weight in self.backend.neighbors("memory", record.id, limit=10):
            if kind != "memory":
                continue
            found = self.backend.get(node_id)
            if found is not None:
                out.append(found)
        return out
