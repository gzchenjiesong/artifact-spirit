"""传承重建（T-AL2-23 · M5）。

把一个**已解析的档案 pack** 翻译成**写意图序列**——与在线写入走同一条路径
（写意图 → 单写者 → AL3），而不是另开一条"批量直插"。

## 为什么必须走同一条路径

直插看起来只是"快一点"，实际会**绕过三样东西**：写队列的优先级、
单写者的线程约束、以及——最要紧的——**审计**。而传承导入恰恰是最需要留痕的操作：
它是"一批外来的记忆一次性进入本器灵"，事后要能回答"这批是什么时候、从哪进来的"。

## 三条纪律

1. **不重新提取**：`spirit_import` 是**重建**，不是 ingest。档案里每条记忆都是
   已经提取过的结论；再提一遍等于用模型的偶然行为覆盖用户的资产。
   这条也是"导入过程零 LLM 调用"能成为验收项的底气。
2. **以 `content_hash` 为锚**（INV-14）：判重不靠 id、不靠时间，靠内容指纹。
   于是"把同一个档案导进一个已有三条的库"不会变成六条——**重复导入零新增**。
3. **两阶段**：`superseded_by` 指向 `memories` 自己，而包内顺序任意。
   先全部落库（不带该字段），再统一回填。走写队列时这条尤其重要：
   队列是**按优先级而非提交顺序**消费的，所以顺序只能由**意图序列本身**保证——
   把"先 put 后 update"写进序列里，而不是指望"提交顺序会被保留"。

## 不做的 I/O

本模块**不读文件**（R5）。文件读取与解析在 `store/archive.load_archive`，
调用方（CLI / 工具面）把解析好的 dict 递进来。

分开的收益是可测的：**两条入口共用同一个解析器**，
不会出现"工具面读得懂、CLI 读不懂"这类只在跨版本时才暴露的分歧。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..common import content_hash_of
from ..store.base import MemoryBackend, MemoryRecord
from .base import Clock, TransferReport, WriteIntent

__all__ = ["Transferrer"]

_RECORD_FIELDS = frozenset(MemoryRecord.__dataclass_fields__)

_ACTOR = "cli"
"""传承导入的 actor。用 `cli` 而不是 `user`：它确实是**命令行工具**发起的批量操作，
而不是"某个人一条条改的"。事后区分"这批是导进来的"与"这些是用户手工改的"，靠的就是它。"""


@dataclass(slots=True)
class Transferrer:
    """传承重建。**只产出写意图**（R5）。"""

    backend: MemoryBackend
    id_gen: Callable[[str], str]
    """`(layer) -> id`。**必须在这里分配**而不能留给落库那一刻：
    `relations` 与 `superseded_by` 都引用记录 id，而它们要在同一批意图里就被解析完——
    计划阶段不知道 id，就只能让调用方去猜。"""
    clock: Clock | None = None
    force_new_ids: bool = False
    """为真时**连原 id 也不沿用**，一律重新分配。

    默认 `False`（INV-14 要求"沿用原 id"，这样跨库的关联关系才对得上）。
    但"把档案导回**同一个**库"这种场景下，原 id 已被占用，
    沿用就变成覆盖——所以那条路径由调用方显式打开这个开关。"""

    # ------------------------------------------------------------------ #

    def plan(self, pack: dict) -> TransferReport:
        """把一个 pack 翻译成写意图。**读库判重，但不写库**。"""
        report = TransferReport()
        id_map: dict[str, str] = {}
        deferred_superseded: list[tuple[str, str]] = []
        known: set[str] = set()

        for raw in pack.get("memories", []):
            if not isinstance(raw, dict):
                report.errors.append(f"记忆条目不是对象：{type(raw).__name__}")
                continue
            record = MemoryRecord(**{k: v for k, v in raw.items() if k in _RECORD_FIELDS})
            record.content_hash = record.content_hash or content_hash_of(
                content=record.content,
                subject=record.subject,
                predicate=record.predicate,
                object_=record.object,
                scope=record.scope,
            )
            digest = record.content_hash or ""
            pending_target = record.superseded_by
            record.superseded_by = None  # 第二阶段回填，见模块头说明

            hits = self.backend.find_by_hash(digest)
            if hits:
                # **已有就不动**：不改写、不新写（INV-14 的幂等）。
                # 这里刻意**不**做"内容不同就更新"——档案是外来的，
                # 用外来的内容覆写库内已有的同指纹记录，会把"用户后来手工改过的版本"冲掉。
                report.skipped += 1
                id_map[record.id] = hits[0].id
                known.add(hits[0].id)
                if pending_target:
                    deferred_superseded.append((hits[0].id, pending_target))
                continue

            original_id = record.id
            if self.force_new_ids or (original_id and self.backend.get(original_id) is not None):
                record.id = ""
            if not record.id:
                record.id = self.id_gen(record.layer or "semantic")
            id_map[original_id] = record.id
            known.add(record.id)
            if pending_target:
                deferred_superseded.append((record.id, pending_target))

            report.intents.append(
                WriteIntent(
                    op="put",
                    record=record,
                    embed_text=record.abstract or record.content,
                    actor=_ACTOR,
                    reason="传承重建",
                )
            )
            report.imported += 1

        # ---- 第二阶段：回填"被谁取代" ----
        # 单独一轮，且**排在所有 put 之后**——见模块头的第 3 条纪律。
        for mem_id, target_id in deferred_superseded:
            resolved = id_map.get(target_id, target_id)
            if resolved not in known:
                report.errors.append(f"{mem_id} 的 superseded_by 指向不存在的记忆：{target_id}")
                continue
            report.intents.append(
                WriteIntent(
                    op="update",
                    mem_id=mem_id,
                    patch={"superseded_by": resolved},
                    actor=_ACTOR,
                    reason="传承重建：回填被取代关系",
                )
            )
            report.restored_superseded += 1

        self._plan_entities(pack, report)
        self._plan_relations(pack, id_map, known, report)
        return report

    # ------------------------------------------------------------------ #

    def _plan_entities(self, pack: dict, report: TransferReport) -> None:
        for ent in pack.get("entities", []):
            name = (ent or {}).get("name")
            if not name:
                report.errors.append(f"实体缺少 name：{ent!r}")
                continue
            aliases = tuple(str(a) for a in (ent.get("aliases") or ()))
            report.intents.append(
                WriteIntent(
                    op="entity_upsert",
                    entity_name=str(name),
                    entity_type=str(ent.get("type") or ""),
                    aliases=aliases,
                    actor=_ACTOR,
                )
            )
            report.entities += 1

    def _plan_relations(
        self,
        pack: dict,
        id_map: dict[str, str],
        known: set[str],
        report: TransferReport,
    ) -> None:
        """关联边。**两端都必须真实存在**——指向空气的边比丢边更坏：
        它会让后续的图查询与可达性判定把不存在的节点算进去。
        """
        for rel in pack.get("relations", []):
            src_kind = str(rel.get("src_kind") or "memory")
            dst_kind = str(rel.get("dst_kind") or "memory")
            if src_kind != "memory" or dst_kind != "memory":
                # **如实报告能力边界**，而不是静默丢边。
                # 实体节点走 `entity_upsert`，而那条意图**不携带 id**——
                # 也就是说实体落地时会拿到新 id，指向它的边随即指向空气。
                # 静默跳过会让"导入成功"这个结论**包含了"少了若干净关联"**，
                # 而用户看不出来。
                report.errors.append(
                    f"涉及实体节点的关联暂不支持重放（本版只重放记忆↔记忆）："
                    f"{rel.get('rel_type')} {src_kind}:{rel.get('src_id')} "
                    f"-> {dst_kind}:{rel.get('dst_id')}"
                )
                continue
            src_id = id_map.get(rel.get("src_id", ""), rel.get("src_id", ""))
            dst_id = id_map.get(rel.get("dst_id", ""), rel.get("dst_id", ""))
            if src_id not in known or dst_id not in known:
                report.errors.append(
                    f"关联指向不存在的记忆：{rel.get('rel_type')} {src_id} -> {dst_id}"
                )
                continue
            report.intents.append(
                WriteIntent(
                    op="link",
                    a_kind="memory",
                    a_id=src_id,
                    b_kind="memory",
                    b_id=dst_id,
                    rel_type=str(rel.get("rel_type") or "co_activation"),
                    weight=float(rel.get("weight") or 1.0),
                    actor=_ACTOR,
                )
            )
            report.relations += 1
