"""多因子融合召回（LLD-AL2 §5 M5 · D-10 / D-16 / D-20）。

```
score = α·semantic + β·importance + γ·recency + δ·entity + ε·diffusion + ζ·core
```

三条纪律：

1. **各路原生分必须先归一化到 [0,1]**（C3）——不同量纲的分数相加没有意义
2. **保留 `raw` 六分量**（C4）——"这条为什么被召回"必须答得出来
3. **向量路不可用时跳过该路并重归一化权重**，绝不抛错（F2 降级）

本模块是**纯函数 + 一个薄协调器**（:class:`Recaller`），协调器只经 `MemoryBackend`
协议访问数据，不写库、不起线程。

**依据**：回忆依赖**线索**——线索与记忆在编码期的匹配程度决定能否想起，因此需要
多路召回而不是单路相似度；扩散激活是**瞬时**的促进，持久联结另有字段承载
（见 ``docs/design/10-神经科学依据与机制映射.md`` §4.4 / §4.5，DES-RES-003）。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ..common import estimate_tokens
from ..store.base import Hit, MemoryBackend, MemoryRecord
from .base import Clock, RecallQuery, RecallWeights, Scored

__all__ = [
    "FACTORS",
    "DecayParams",
    "Recaller",
    "choose_level",
    "clip_to_budget",
    "fuse",
    "normalize_minmax",
    "score_core_alignment",
    "score_diffusion",
    "score_entity",
    "score_importance",
    "score_recency",
    "score_semantic",
    "strength_at",
]

FACTORS = ("semantic", "importance", "recency", "entity", "diffusion", "core")
"""六因子名。``Scored.raw`` 的键集合——**顺序即展示顺序**。"""

IMPORTANCE_WEIGHTS: dict[str, float] = {
    "user_label": 0.40,
    "confidence": 0.25,
    "salience": 0.20,
    "referenced": 0.15,
}
"""``importance`` 的构成（D-20 已裁决）。

**关键**：四项里没有任何一项是 `access_count`——重要度与访问频率**完全解耦**。
否则"身份证号 / 过敏史"这类低频关键记忆会被系统性淘汰。
"""

REFERENCE_SATURATION = 5
"""被引用次数达到该值时 `referenced` 分量饱和为 1.0。"""


# --------------------------------------------------------------------------- #
# 衰减（只作排序信号 —— D-16）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DecayParams:
    """Wixted 混合衰减参数（近期快衰 + 远期长尾）。

    **用途限定**：只喂给 :func:`score_recency`，**不参与任何删除或降级判定**（D-16）。
    """

    w: float = 0.6
    tau_fast: float = 7.0
    beta: float = 0.5
    base_retention: float = 1.0


def strength_at(
    record: MemoryRecord, *, now: str, params: DecayParams, last_access: str | None = None
) -> float:
    """计算时刻 ``now`` 的记忆强度。

    ``strength(t) = base × ( w·exp(−Δt/τ_fast) + (1−w)·(1+Δt)^(−β) )``
    """
    reference = last_access or record.last_access_at or record.created_at
    delta_days = _days_between(reference, now)
    if delta_days <= 0:
        return float(params.base_retention)
    exponent = params.w * math.exp(-delta_days / max(params.tau_fast, 1e-6))
    tail = (1.0 - params.w) * (1.0 + delta_days) ** (-params.beta)
    return float(params.base_retention) * (exponent + tail)


def _days_between(earlier: str | None, later: str | None) -> float:
    if not earlier or not later:
        return 0.0
    try:
        a = datetime.fromisoformat(earlier)
        b = datetime.fromisoformat(later)
    except ValueError:
        return 0.0
    return max(0.0, (b - a).total_seconds() / 86400.0)


# --------------------------------------------------------------------------- #
# 六个打分函数（全部返回 [0,1]）
# --------------------------------------------------------------------------- #


def score_semantic(similarity: float | None) -> float:
    """语义相似度。``vector_search`` 已把余弦距离转成 [0,1] 的相似度。"""
    if similarity is None:
        return 0.0
    return _clip01(similarity)


def score_importance(
    record: MemoryRecord,
    *,
    inbound_refs: int = 0,
    user_labeled: bool = False,
    weights: Mapping[str, float] | None = None,
) -> float:
    """重要度。**与 ``access_count`` 完全解耦**（D-20）。

    构成：``用户标注 > 置信度 > 显著性 > 被引用次数``（归一化加权）。

    与频率解耦的必要性见 DES-RES-002 §4.1：按频率排序会**系统性优先淘汰
    "低频但关键"的记忆**——身份证号、血型、过敏史、紧急联系人，调用频率全都极低，
    但一旦需要就必须有。
    """
    w = weights or IMPORTANCE_WEIGHTS
    referenced = min(1.0, max(0, inbound_refs) / REFERENCE_SATURATION)
    value = (
        w["user_label"] * (1.0 if user_labeled else 0.0)
        + w["confidence"] * _clip01(record.confidence)
        + w["salience"] * _clip01(record.salience)
        + w["referenced"] * referenced
    )
    return _clip01(value)


def score_recency(
    record: MemoryRecord,
    *,
    now: str,
    params: DecayParams | None = None,
    last_access: str | None = None,
) -> float:
    """时间邻近度。内含 Wixted 衰减形状，**只喂给排序**（D-16）。

    ``strength`` 到这里的唯一去处就是本函数——它不再触发任何删除。
    """
    return _clip01(strength_at(record, now=now, params=params or DecayParams(), last_access=last_access))


def score_entity(
    record: MemoryRecord, query: RecallQuery, entity_index: Mapping[str, float]
) -> float:
    """实体命中度（D-10 第六因子）。

    成本为零：实体在提取环节已建好，这里只是把预先算好的命中强度取出来——
    **不做 LLM 调用**。
    """
    return _clip01(entity_index.get(record.id, 0.0))


def score_diffusion(
    record: MemoryRecord, seeds: set[str], neighbors: Mapping[str, float]
) -> float:
    """扩散激活值（一拍扩散）。

    作为种子的记忆本身给满分；其余取其到种子集合的最大边权。
    """
    if record.id in seeds:
        return 1.0
    return _clip01(neighbors.get(record.id, 0.0))


def score_core_alignment(record: MemoryRecord, core_index: Mapping[str, float]) -> float:
    """与核心记忆的一致性。

    ``core_index`` 由调用方预计算（核心记忆自身 → 1.0；主题/实体与核心记忆重合的 → 部分分）。
    """
    if record.layer == "core":
        return 1.0
    return _clip01(core_index.get(record.id, 0.0))


# --------------------------------------------------------------------------- #
# 融合与裁剪
# --------------------------------------------------------------------------- #


def fuse(items: Sequence[Scored], weights: RecallWeights) -> list[Scored]:
    """加权融合。

    **缺失分量会被跳过，权重在剩余分量上重归一化**——这样向量路不可用时
    召回依然成立（只是退化），而不是整体失效或抛错。
    """
    weight_map = weights.as_dict()
    for item in items:
        present = {k: v for k, v in item.raw.items() if k in weight_map}
        total_weight = sum(weight_map[k] for k in present)
        if total_weight <= 0:
            item.score = 0.0
            continue
        item.score = sum(weight_map[k] * _clip01(v) for k, v in present.items()) / total_weight
    return sorted(items, key=lambda s: (-s.score, s.record.id))


def clip_to_budget(
    items: Sequence[Scored],
    token_budget: int,
    *,
    cost_of: Callable[[Scored], int] | None = None,
) -> list[Scored]:
    """按 token 预算裁剪。

    **预算按 L0（摘要）计量**——这是设计选择而非权宜：粗细筛用的就是 L0，
    让 L2 全文去挤占粗筛预算等于"因为想细看而看不见更多"。需要全文时由
    :meth:`CoreFacade.expand` 显式展开（P1 分级加载）。

    调用方若想按别的级别计量，传 ``cost_of``。
    """
    scorer = cost_of or (lambda s: estimate_tokens(s.record.abstract or s.record.content))
    kept: list[Scored] = []
    used = 0
    for item in items:
        cost = scorer(item)
        if used + cost > token_budget and kept:
            break
        kept.append(item)
        used += cost
    return kept


def choose_level(
    items: Sequence[Scored], token_budget: int
) -> str:
    """在预算内选择能负担的**最高**展示级别：``L2`` > ``L1`` > ``L0``。

    超预算时**先舍细级**（L2 → L1 → L0），而不是先丢记忆——这是分级加载的
    注意力收益所在（V1）。
    """
    if not items:
        return "L0"
    l2_cost = sum(estimate_tokens(s.record.content) for s in items)
    if l2_cost <= token_budget:
        return "L2"
    l1_cost = sum(estimate_tokens(s.record.abstract or s.record.content) for s in items)
    if l1_cost <= token_budget:
        return "L1"
    return "L0"


def normalize_minmax(values: Sequence[float]) -> list[float]:
    """归一化到 [0,1]。全部相等时返回全 1（而非全 0）——避免"都重要"变成"都不重要"。"""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def _clip01(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - 防御性
        return 0.0
    if math.isnan(number):
        return 0.0
    return max(0.0, min(1.0, number))


# --------------------------------------------------------------------------- #
# 协调器
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Recaller:
    """把六路信号拼起来的薄协调器。

    它**只读**——召回永远不写库（只读是 `prefetch` 能在 300ms 护栏内返回的前提）。
    """

    backend: MemoryBackend
    clock: Clock
    weights: RecallWeights = field(default_factory=RecallWeights)
    decay: DecayParams = field(default_factory=DecayParams)
    candidate_k: int = 24
    diffusion_threshold: float = 0.2

    def recall(self, q: RecallQuery) -> list[Scored]:
        hits: dict[str, Hit] = {}
        vector_scores: dict[str, float] = {}

        if q.vec:
            for hit in self.backend.vector_search(
                q.vec, layer=None, top_k=self.candidate_k
            ):
                hits[hit.mem_id] = hit
                vector_scores[hit.mem_id] = hit.score

        lexical_hits = self.backend.keyword_search(q.text, layer=None, top_k=self.candidate_k)
        lexical_scores = {h.mem_id: h.score for h in lexical_hits}
        for hit in lexical_hits:
            hits.setdefault(hit.mem_id, hit)

        # 会话内工作记忆也参与（"刚才说过什么"）
        for chunk in self.backend.wm_list(q.session_id, limit=5) if q.session_id else []:
            _ = chunk  # 工作记忆以文本形式并入下文，不单独打分

        if not hits:
            return []

        records = {mid: _record_of(hit) for mid, hit in hits.items()}
        if q.layers:
            allow = set(q.layers)
            records = {k: v for k, v in records.items() if v.layer in allow}
        if not records:
            return []

        # 休眠态只以 L0 参与（D-23）——排序上给一个温和的降权
        dormant = {mid for mid, rec in records.items() if rec.status == "dormant"}

        ids = list(records)
        now = self.clock()

        semantic = _merge_semantic(
            {mid: vector_scores.get(mid) for mid in ids},
            {mid: lexical_scores.get(mid) for mid in ids},
        )
        refs = self.backend.inbound_reference_counts(ids)
        importance = {
            mid: score_importance(records[mid], inbound_refs=refs.get(mid, 0))
            for mid in ids
        }
        recency = {mid: score_recency(records[mid], now=now, params=self.decay) for mid in ids}
        entity_index = self._entity_index(q)
        core_index = self._core_index(records, q)
        neighbors, seeds = self._diffusion_inputs(ids, semantic)

        items: list[Scored] = []
        for mid in ids:
            record = records[mid]
            raw = {
                "semantic": semantic.get(mid, 0.0),
                "importance": importance[mid],
                "recency": recency[mid],
                "entity": score_entity(record, q, entity_index),
                "diffusion": score_diffusion(record, seeds, neighbors),
                "core": score_core_alignment(record, core_index),
            }
            if mid in dormant:
                # 休眠只保留 L0 参与：影响力削减但不消失
                raw = {k: (v * 0.6 if k != "importance" else v) for k, v in raw.items()}
            items.append(Scored(record=record, raw=raw, score=0.0))

        fused = fuse(items, self.weights)
        below = [s for s in fused if s.score > 0.0]
        return below or fused

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _entity_index(self, q: RecallQuery) -> dict[str, float]:
        """query 命中的实体所"提及"的记忆 → 命中强度（D-10）。

        两条来源：① 提取期建的 ``mentions`` 关联边；② 正文包含实体名的兜底匹配。
        """
        index: dict[str, float] = {}
        entities = self.backend.entity_find(q.text)
        lowered = q.text.casefold()
        for entity in entities:
            for kind, node_id, weight in self.backend.neighbors(
                "entity", entity.id, limit=self.candidate_k
            ):
                if kind == "memory":
                    index[node_id] = max(index.get(node_id, 0.0), min(1.0, weight or 0.6))
            for record in self.backend.query(status=None, limit=500):
                for name in [entity.name, *entity.aliases]:
                    if name and name.casefold() in (record.content or "").casefold():
                        exact = name.casefold() in lowered
                        index[record.id] = max(index.get(record.id, 0.0), 0.8 if exact else 0.5)
        return index

    def _core_index(
        self, records: Mapping[str, MemoryRecord], q: RecallQuery
    ) -> dict[str, float]:
        """核心记忆一致性：核心自身 → 1.0；与核心主题重合 → 0.8。"""
        index: dict[str, float] = {}
        terms = {t for t in self.backend.core_memory_terms() if t}
        for mid, record in records.items():
            if record.layer == "core":
                index[mid] = 1.0
                continue
            candidates = {
                (record.subject or "").casefold(),
                (record.object or "").casefold(),
            }
            if terms & {c for c in candidates if c}:
                index[mid] = 0.8
        return index

    def _diffusion_inputs(
        self, ids: list[str], semantic: Mapping[str, float]
    ) -> tuple[dict[str, float], set[str]]:
        """一拍扩散：种子 = 语义分最高的若干条；邻居取其到种子的最大边权。"""
        ranked = sorted(ids, key=lambda m: semantic.get(m, 0.0), reverse=True)
        seeds = {m for m in ranked[:3] if semantic.get(m, 0.0) > 0}
        if not seeds:
            return {}, set()

        neighbors: dict[str, float] = {}
        for edge in self.backend.outbound_edges(sorted(seeds)):
            if edge["dst_kind"] != "memory":
                continue
            weight = float(edge["weight"])
            if weight < self.diffusion_threshold:
                continue
            neighbors[edge["dst_id"]] = max(neighbors.get(edge["dst_id"], 0.0), weight)
        return neighbors, seeds


def _merge_semantic(
    vector: Mapping[str, float | None], lexical: Mapping[str, float | None]
) -> dict[str, float]:
    """语义 + BM25 混合。

    BM25 是无界分，先做 min-max 归一化；随后两路**取大**——
    任一路强命中就应该被算作强命中（"或"语义而非"与"）。
    """
    lex_values = [v for v in lexical.values() if v is not None]
    lex_norm = dict(
        zip(
            [k for k, v in lexical.items() if v is not None],
            normalize_minmax(lex_values),
            strict=False,
        )
    )
    merged: dict[str, float] = {}
    for mid in set(vector) | set(lexical):
        vec = vector.get(mid)
        lex = lex_norm.get(mid, 0.0)
        merged[mid] = max(_clip01(vec) if vec is not None else 0.0, lex)
    return merged


def _record_of(hit: Hit) -> MemoryRecord:
    """从 ``Hit.meta`` 还原记录。

    AL3 的 ``Hit.meta['record']`` 携带完整列值，因此这里**不需要再查一次库**——
    召回是热路径，N+1 查询是不可接受的。
    """
    payload = hit.meta.get("record")
    if isinstance(payload, MemoryRecord):  # pragma: no cover - 兼容
        return payload
    return MemoryRecord(**payload)
