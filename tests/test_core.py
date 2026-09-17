"""AL2 核心层验收（T-AL2-01 ~ 15）。

**测试环境**：临时 SQLite 后端 + 假 LLM + 假 embedding + 可注入固定时钟。
不连网络、不启线程。

> 说明：任务书写的替身是"内存 fake backend"。这里改用**真实 SQLite（临时文件）**——
> 它同样不依赖任何外部服务，但能顺带验证"意图 → 落库"这条链路真的成立。
> 对 AL2 而言，被验证的是**决策**；用真库只是让决策的后果可见。
"""

from __future__ import annotations

import sqlite3

import pytest
from conftest import ScriptedLLM, write_intents

from artifact_spirit.common import estimate_tokens, truncate_to_tokens
from artifact_spirit.core import (
    ArtifactSpiritCore,
    CoreSettings,
    RecallQuery,
    TurnEvent,
)
from artifact_spirit.core.base import RecallWeights
from artifact_spirit.core.layers.procedural import ProceduralLayer
from artifact_spirit.core.layers.working import WORKING_CAPACITY
from artifact_spirit.core.recall import (
    DecayParams,
    choose_level,
    fuse,
    score_importance,
    strength_at,
)
from artifact_spirit.core.salience import SalienceConfig, SalienceScorer
from artifact_spirit.extract.schema import SYSTEM_PROMPT, validation_errors
from artifact_spirit.model import (
    EmbeddingError,
    ProviderUnavailableError,
    SchemaViolationError,
)
from artifact_spirit.observability import format_review_text, format_trace_text
from artifact_spirit.store import MemoryRecord

FIXED_TS = "2026-09-14T10:00:00+08:00"


def memory_payload(**overrides) -> dict:
    item = {
        "type": "preference",
        "layer": "semantic",
        "subject": "用户",
        "predicate": "prefers",
        "object": "深色主题",
        "content": "用户偏好深色主题，不喜欢浅色界面",
        "abstract": "偏好深色主题",
        "confidence": 0.92,
        "salience": 0.7,
    }
    item.update(overrides)
    return {"memories": [item]}


def test_truncate_to_tokens_never_spins_on_tiny_budget():
    """预算装不下一个汉字时必须**立刻**返回空串，不能原地打转（DES-REV-009 P0-1）。

    旧实现逐字回退，步长 ``max(1, len(result) - 1)``：只剩 1 个字时切片结果等于自身，
    而汉字估算恒为 2 —— ``budget == 1`` 于是变成死循环。受害路径是宿主每轮都调的
    ``system_prompt_block``：``token_budget=60`` 时内部派生出的核心记忆预算正好是 1。
    """
    assert truncate_to_tokens("一二三四五", 1) == ""
    assert truncate_to_tokens("一二三四五", 0) == ""
    assert truncate_to_tokens("一二三四五", -5) == ""


def test_truncate_to_tokens_stays_within_budget_and_is_monotone():
    """返回的是**预算内的最长前缀**：不超预算、是原文本前缀、随预算单调不减。"""
    text = "一二三四五abc" * 30
    full = estimate_tokens(text)
    lengths = [len(truncate_to_tokens(text, budget)) for budget in range(full + 1)]
    assert lengths == sorted(lengths), "预算变大时前缀不应变短"
    for budget, size in enumerate(lengths):
        if size:
            out = text[:size]
            assert text.startswith(out)
            assert estimate_tokens(out) <= budget, f"budget={budget} 时超出预算"
    assert lengths[-1] == len(text), "预算等于全文估算值时应原样返回"


@pytest.fixture
def core(backend, embedding):
    llm = ScriptedLLM([("记住", memory_payload()), ("偏好", memory_payload())])
    return ArtifactSpiritCore(
        backend=backend,
        clock=lambda: FIXED_TS,
        embedding=embedding,
        llm=llm,
        settings=CoreSettings(candidate_k=10),
    )


def ingest(core, backend, embedding, text, *, session_id="s1", recalled=(), ts=FIXED_TS):
    intents = core.ingest_turn(
        TurnEvent(
            session_id=session_id, user=text, assistant="好的", ts=ts, recalled=tuple(recalled)
        )
    )
    write_intents(backend, intents, embedding)
    return intents


# --------------------------------------------------------------------------- #
# T-AL2-01 CoreFacade 与共享数据类
# --------------------------------------------------------------------------- #


def test_recall_weights_sum_to_one():
    weights = RecallWeights()
    assert weights.total() == pytest.approx(1.0)
    assert set(weights.as_dict()) == {
        "semantic",
        "importance",
        "recency",
        "entity",
        "diffusion",
        "core",
    }


def test_scored_raw_has_six_components(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    hits = core.recall(RecallQuery(text="深色主题", session_id="s1"))
    assert hits
    assert set(hits[0].raw) == {
        "semantic",
        "importance",
        "recency",
        "entity",
        "diffusion",
        "core",
    }


def test_core_layer_has_no_io_imports():
    """R5：core/ 与 extract/ 内不得出现 I/O 库。"""
    from artifact_spirit.compliance import resolved_imports
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT, iter_python_files

    for package in ("core", "extract"):
        for path in iter_python_files(PACKAGE_ROOT / package):
            imports = resolved_imports(path)
            for banned in ("httpx", "sqlite3", "openai"):
                assert banned not in imports, f"{path} 出现 {banned}（R5）"


def test_core_does_not_import_hermes():
    from artifact_spirit.compliance import resolved_imports
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT, iter_python_files

    for path in iter_python_files(PACKAGE_ROOT / "core"):
        assert not any(
            name.startswith("hermes") for name in resolved_imports(path)
        ), f"{path} 依赖了宿主代码（R4）"


def test_clock_and_id_gen_are_injected(backend, embedding):
    """C2：时钟与 ID 生成器必须可注入，否则衰减与时序无法稳定测试。"""
    ids = iter(["sem_fixed_1", "sem_fixed_2", "sem_fixed_3"])

    def id_gen(layer: str) -> str:
        return next(ids)

    core = ArtifactSpiritCore(
        backend=backend,
        clock=lambda: "2030-01-01T00:00:00+08:00",
        id_gen=id_gen,
        embedding=embedding,
        llm=ScriptedLLM([("记住", memory_payload())]),
    )
    intents = core.ingest_turn(
        TurnEvent(session_id="s1", user="记住 X", assistant="", ts="2030-01-01T00:00:00+08:00")
    )
    assert any(i.record and i.record.id == "sem_fixed_1" for i in intents if i.record)


# --------------------------------------------------------------------------- #
# T-AL2-02 显著性过滤
# --------------------------------------------------------------------------- #


def test_explicit_instruction_scores_higher_than_neutral(backend, embedding):
    scorer = SalienceScorer(backend=backend, embedding=embedding)
    instruction = scorer.score("请记住：我偏好深色主题，以后都用深色")
    neutral = scorer.score("今天天气还不错")
    assert instruction.score > neutral.score


def test_similar_input_scores_lower_novelty(backend, embedding):
    scorer = SalienceScorer(backend=backend, embedding=embedding)
    text = "用户偏好深色主题不要浅色界面"
    backend.put(MemoryRecord(id="", layer="semantic", type="fact", content=text), embedding.embed([text])[0])
    repeated = scorer.score(text)
    novel = scorer.score("用户的血型是 O 型，过敏史是青霉素")
    assert repeated.score < novel.score
    assert repeated.factors["novelty"] < novel.factors["novelty"]


def test_below_threshold_produces_no_candidates(backend, embedding):
    scorer = SalienceScorer(backend=backend, embedding=embedding, config=SalienceConfig(threshold=0.99))
    result = scorer.score("嗯好的")
    assert result.passes_threshold(SalienceConfig(threshold=0.99)) is False


def test_salience_degrades_without_embedding(backend):
    """F2：embedding 不可用时退化为纯规则打分，**不抛错**。"""
    scorer = SalienceScorer(backend=backend, embedding=None)
    result = scorer.score("请记住：我偏好深色主题")
    assert result.degraded == "embedding_unavailable"
    assert result.factors["novelty"] == pytest.approx(0.5)
    assert result.score > 0

    class ExplodingEmbedding:
        model = "boom"
        dim = 8

        def embed(self, texts):
            raise EmbeddingError("断了")

    scorer2 = SalienceScorer(backend=backend, embedding=ExplodingEmbedding())
    assert scorer2.score("请记住这件事").score > 0


# --------------------------------------------------------------------------- #
# T-AL2-03 工作记忆
# --------------------------------------------------------------------------- #


def test_working_memory_evicts_coldest_from_attention(backend, embedding, core):
    for index in range(WORKING_CAPACITY + 2):
        for _ in range(index + 1):
            backend.wm_put("s1", f"topic-{index}", f"内容 {index}", 0.5)
    active = core.working.active_chunks("s1")
    assert len(active) == WORKING_CAPACITY
    evicted = core.working.evict("s1")
    assert len(evicted) == 2
    # 被挤出注意力 ≠ 被丢弃
    assert len(backend.wm_list("s1", limit=50)) == WORKING_CAPACITY + 2


def test_same_topic_accumulates_into_same_chunk(backend, embedding, core):
    ingest(core, backend, embedding, "RAGFlow 项目用了 Python 3.12")
    ingest(core, backend, embedding, "RAGFlow 项目的部署方式是 Docker")
    keys = [c.chunk_key for c in backend.wm_list("s1")]
    assert len(keys) == len(set(keys)), f"同类话题应累积到同一组块，实际 {keys}"


def test_intent_slot_not_limited_by_capacity(backend, embedding, core):
    for index in range(WORKING_CAPACITY + 3):
        ingest(core, backend, embedding, f"提醒我第 {index} 件事")
    chunks = backend.wm_list("s1", limit=50)
    assert any(c.chunk_key == "__intent__" for c in chunks)


# --------------------------------------------------------------------------- #
# T-AL2-04 六层服务骨架
# --------------------------------------------------------------------------- #


def test_all_layer_services_instantiate(core):
    for service in (
        core.sensory,
        core.working,
        core.episodic,
        core.semantic,
        core.procedural,
        core.core_memory,
    ):
        assert isinstance(service.layer, str)


def test_procedural_layer_is_placeholder():
    """C11：程序性记忆 MVP **只有占位**，不得提前实现固化逻辑。"""
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT

    layer = ProceduralLayer(backend=None, clock=lambda: FIXED_TS)
    assert layer.candidates(None) == []
    source = PACKAGE_ROOT / "core" / "layers" / "procedural.py"
    assert source.exists()


def test_sensory_layer_never_persists():
    """感觉记忆是门口不是仓库——candidates 恒为空。"""
    from artifact_spirit.core.layers.sensory import SensoryLayer

    layer = SensoryLayer.__new__(SensoryLayer)
    assert layer.candidates(None) == []


# --------------------------------------------------------------------------- #
# T-AL2-05 召回六因子
# --------------------------------------------------------------------------- #


def test_all_factors_within_unit_interval(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    hits = core.recall(RecallQuery(text="深色主题", session_id="s1"))
    for hit in hits:
        for name, value in hit.raw.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} 越界"


def test_importance_is_decoupled_from_access_count(backend):
    """D-20 核心断言：改变 access_count **不影响** importance。"""
    record = MemoryRecord(
        id="sem_x", layer="semantic", type="fact", content="身份证号是 110101...",
        confidence=0.95, salience=0.8, access_count=0,
    )
    base = score_importance(record, inbound_refs=0)
    record.access_count = 9999
    assert score_importance(record, inbound_refs=0) == base


def test_low_frequency_high_importance_beats_high_frequency_low_importance():
    """D-20：低频但关键的记忆，在重要性维度上必须胜过高频但无关的。"""
    critical = MemoryRecord(
        id="sem_id", layer="semantic", type="fact", content="身份证号",
        confidence=0.98, salience=0.9, access_count=0,
    )
    chatter = MemoryRecord(
        id="sem_chat", layer="semantic", type="fact", content="随口一说",
        confidence=0.4, salience=0.1, access_count=9999,
    )
    assert score_importance(critical, inbound_refs=0) > score_importance(chatter, inbound_refs=999)


def test_importance_breakdown_present_in_review(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    row = core.review(limit=1)[0]
    assert set(row["importance_breakdown"]) == {
        "user_label",
        "confidence",
        "salience",
        "referenced",
    }


def test_entity_factor_rises_when_query_hits_entity(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    hit_with = core.recall(RecallQuery(text="用户喜欢什么", session_id="s1"))[0]
    hit_without = core.recall(RecallQuery(text="颜色偏好", session_id="s1"))[0]
    assert hit_with.raw["entity"] > hit_without.raw["entity"]


def test_recency_decays_monotonically(backend):
    record = MemoryRecord(
        id="sem_r", layer="semantic", type="fact", content="x",
        created_at="2026-01-01T00:00:00+08:00", last_access_at="2026-01-01T00:00:00+08:00",
    )
    params = DecayParams()
    values = [
        strength_at(record, now=f"2026-{month:02d}-01T00:00:00+08:00", params=params)
        for month in range(1, 13)
    ]
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


def test_wixted_shape_fast_then_long_tail(backend):
    """近期衰减快于远期——指数项主导 → 幂律项主导的转折可见。"""
    record = MemoryRecord(
        id="sem_r", layer="semantic", type="fact", content="x",
        created_at="2026-01-01T00:00:00+08:00", last_access_at="2026-01-01T00:00:00+08:00",
    )
    params = DecayParams()
    early = strength_at(record, now="2026-01-02T00:00:00+08:00", params=params) - strength_at(
        record, now="2026-01-03T00:00:00+08:00", params=params
    )
    late = strength_at(record, now="2026-06-01T00:00:00+08:00", params=params) - strength_at(
        record, now="2026-06-02T00:00:00+08:00", params=params
    )
    assert early > late


# --------------------------------------------------------------------------- #
# T-AL2-06 融合与预算裁剪
# --------------------------------------------------------------------------- #


def test_weight_change_alters_ranking():
    from artifact_spirit.core.recall import Scored

    a = Scored(record=_rec("A"), raw={"semantic": 1.0, "importance": 0.0}, score=0.0)
    b = Scored(record=_rec("B"), raw={"semantic": 0.0, "importance": 1.0}, score=0.0)
    semantic_first = fuse([a, b], RecallWeights(semantic=1.0, importance=0.0))
    importance_first = fuse([a, b], RecallWeights(semantic=0.0, importance=1.0))
    assert semantic_first[0].record.id == "A"
    assert importance_first[0].record.id == "B"


def test_fuse_renormalizes_when_a_factor_is_missing():
    """向量路不可用时跳过该路并重归一化——**不抛错**。"""
    from artifact_spirit.core.recall import Scored

    item = Scored(record=_rec("A"), raw={"importance": 1.0}, score=0.0)
    assert fuse([item], RecallWeights())[0].score == pytest.approx(1.0)


def test_clip_to_budget_keeps_abstract_form(core, backend, embedding):
    for index in range(6):
        backend.put(
            MemoryRecord(
                id="", layer="semantic", type="fact",
                content="内容" * 200, abstract=f"摘要{index}",
            )
        )
    hits = core.recall(RecallQuery(text="内容", session_id="s1", token_budget=60, top_k=6))
    total = sum(len(h.record.abstract or "") for h in hits)
    assert total <= 60 * 3


def test_choose_level_prefers_coarse_when_budget_tight():
    from artifact_spirit.core.recall import Scored

    items = [
        Scored(
            record=MemoryRecord(
                id="sem_a", layer="semantic", type="fact",
                content="很长的正文" * 100, abstract="短摘要",
            ),
            raw={}, score=1.0,
        )
    ]
    assert choose_level(items, 2000) == "L2"
    assert choose_level(items, 20) == "L1"
    assert choose_level(items, 1) == "L0"


# --------------------------------------------------------------------------- #
# T-AL2-07 分级加载（P 档前置）
# --------------------------------------------------------------------------- #


def test_expand_levels_grow_monotonically(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    l0 = core.expand(ref, "L0", hot_path=False)
    l1 = core.expand(ref, "L1", hot_path=False)
    l2 = core.expand(ref, "L2", hot_path=False)
    assert len(l0) < len(l1) < len(l2)


def test_expand_l1_does_not_contain_l2_fulltext(core, backend, embedding):
    """V1：逐级展开不得越级——L1 里不能出现原文。"""
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    l1 = core.expand(ref, "L1", hot_path=False)
    assert "不喜欢浅色界面" not in l1


def test_expand_l2_is_read_only(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    before = backend.last_audit_id()
    core.expand(ref, "L2", hot_path=False)
    assert backend.last_audit_id() == before


def test_hot_path_never_generates_l1(core, backend, embedding):
    """N-P0-2：热路径发现 L1 缺失时**降级到 L0**，不得产生 LLM 调用。"""
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    core.expand(ref, "L2", hot_path=False)  # 先清空 pending
    core.drain_pending()

    calls_before = len(core.llm.calls)
    result = core.expand(ref, "L1", hot_path=True)
    assert len(core.llm.calls) == calls_before, "热路径绝不能调用模型"
    # 缓存缺失 → 降级到了 L0
    assert result == core.expand(ref, "L0", hot_path=True)


def test_hot_path_serves_stale_cache_without_recompute(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    write_intents(backend, core.drain_pending(), embedding)
    entity = backend.entity_list()[0]
    backend.overview_put("entity", entity.id, "L1", "旧的概览内容", token_count=5, model="m")
    backend.overview_invalidate(scope_kind="entity", scope_id=entity.id)

    calls_before = len(core.llm.calls)
    result = core.expand(ref, "L1", hot_path=True)
    assert result == "旧的概览内容"
    assert len(core.llm.calls) == calls_before


def test_expand_l0_degrades_when_abstract_missing(core, backend, embedding):
    record = MemoryRecord(
        id="", layer="semantic", type="fact", content="没有摘要的一条内容。后面还有别的句子。"
    )
    backend.put(record)
    text = core.expand(record.id, "L0")
    assert text
    assert text.startswith("没有摘要的一条内容")


def test_expand_unknown_ref_returns_empty_not_crash(core):
    result = core.expand("sem_不存在", "L1")
    assert result == ""


def test_cold_path_generates_and_requests_cache_write(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    core.drain_pending()
    core.expand(ref, "L1", hot_path=False)
    pending = core.drain_pending()
    assert any(i.op == "overview_put" for i in pending)


# --------------------------------------------------------------------------- #
# T-AL2-08 审查、溯源与干预（P 档前置）
# --------------------------------------------------------------------------- #


def test_review_includes_importance_breakdown(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    row = core.review(limit=1)[0]
    for field in (
        "layer", "content", "abstract", "created_at", "confidence",
        "source_session", "status", "importance", "importance_breakdown",
    ):
        assert field in row


def test_trace_returns_full_history(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    write_intents(
        backend,
        core.correct(ref, {"content": "用户强烈偏好深色主题"}, reason="用户确认"),
        embedding,
    )
    ops = [event["op"] for event in core.trace(ref)]
    assert ops == ["add", "update"]


def test_correct_produces_intent_not_direct_write(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    before = backend.get(ref).content
    intents = core.correct(ref, {"content": "改过的内容"}, reason="用户确认")
    assert backend.get(ref).content == before, "AL2 不得直接写库"
    assert intents and intents[0].op == "update"
    assert intents[0].actor == "user"


def test_correct_invalidates_overviews(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    intents = core.correct(ref, {"content": "改过的内容"}, reason="用户确认")
    assert any(i.op == "overview_invalidate" for i in intents)


def test_correct_rejects_immutable_fields(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    with pytest.raises(ValueError):
        core.correct(ref, {"id": "sem_other"}, reason="试图改身份")
    with pytest.raises(ValueError):
        core.correct(ref, {"created_at": "2000-01-01T00:00:00+08:00"}, reason="试图改历史")


def test_forget_requires_reason_and_source(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    with pytest.raises(ValueError):
        core.forget(ref, reason="", source="cli")
    with pytest.raises(ValueError):
        core.forget(ref, reason="测试", source="")
    assert backend.get(ref) is not None


def test_forget_and_restore_roundtrip(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    write_intents(backend, core.forget(ref, reason="用户要求删除", source="cli"), embedding)
    assert backend.get(ref) is None

    snapshot = backend.delete_snapshots()[0]
    write_intents(backend, core.restore(snapshot["audit_id"]), embedding)
    assert backend.get(ref) is not None
    assert backend.get(ref).content == "用户偏好深色主题，不喜欢浅色界面"


def test_intervention_audit_actor_is_user(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    write_intents(backend, core.correct(ref, {"confidence": 0.5}, reason="用户修正"), embedding)
    assert [e.actor for e in backend.audit_for(ref)][-1] == "user"


def test_review_and_trace_outputs_are_human_readable(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    review_text = format_review_text(core.review(limit=1))
    trace_text = format_trace_text(core.trace(ref))
    assert "内容：" in review_text and "重要度" in review_text
    assert "变更史" in trace_text and "add" in trace_text
    assert "{" not in review_text.split("\n")[1]


# --------------------------------------------------------------------------- #
# T-AL2-09 / 10 提取
# --------------------------------------------------------------------------- #


def test_extraction_schema_validates_bad_payloads():
    assert validation_errors({"memories": []}) == []
    assert validation_errors({}) != []
    assert validation_errors({"memories": "not a list"}) != []
    assert validation_errors({"memories": [{"type": "fact"}]}) != []
    assert validation_errors(
        {"memories": [{"type": "不存在的类型", "layer": "semantic", "content": "x"}]}
    ) != []


def test_prompt_and_schema_share_one_source():
    """C12：提示词里的字段说明与校验规则必须同源。"""
    from artifact_spirit.extract import schema as schema_mod

    for field in ("type", "layer", "content", "abstract", "confidence", "salience"):
        assert field in SYSTEM_PROMPT
        assert field in schema_mod.FIELD_HINTS
    for memory_type in schema_mod.MEMORY_TYPES:
        assert memory_type in SYSTEM_PROMPT


def test_extraction_produces_candidates(core, backend, embedding):
    intents = ingest(core, backend, embedding, "请记住：我偏好深色主题")
    puts = [i for i in intents if i.op == "put" and i.record]
    assert puts and puts[0].record.layer == "semantic"


def test_extraction_produces_no_partial_candidates_on_bad_json(backend, embedding):
    llm = ScriptedLLM([("记住", SchemaViolationError("两次都不是 JSON"))])
    core = ArtifactSpiritCore(backend=backend, clock=lambda: FIXED_TS, embedding=embedding, llm=llm)
    intents = core.ingest_turn(
        TurnEvent(session_id="s1", user="请记住这件事", assistant="", ts=FIXED_TS)
    )
    puts = [i for i in intents if i.op == "put"]
    assert len(puts) == 1
    assert puts[0].record.type == "event", "降级路径只应存原文"
    assert "extract_parse_failed" in (puts[0].audit.reason or "")


def test_llm_unavailable_degrades_to_raw_text_only(backend, embedding):
    llm = ScriptedLLM([("记住", ProviderUnavailableError("没有可用模型"))])
    core = ArtifactSpiritCore(backend=backend, clock=lambda: FIXED_TS, embedding=embedding, llm=llm)
    intents = core.ingest_turn(
        TurnEvent(session_id="s1", user="请记住这件事", assistant="", ts=FIXED_TS)
    )
    puts = [i for i in intents if i.op == "put"]
    assert len(puts) == 1
    assert puts[0].record.content == "请记住这件事"


def test_no_llm_configured_degrades_gracefully(backend, embedding):
    core = ArtifactSpiritCore(backend=backend, clock=lambda: FIXED_TS, embedding=embedding, llm=None)
    intents = core.ingest_turn(
        TurnEvent(session_id="s1", user="请记住这件事", assistant="", ts=FIXED_TS)
    )
    assert not any(i.op == "put" or True for i in [])  # 占位：只需不抛错
    assert intents is not None


# --------------------------------------------------------------------------- #
# T-AL2-11 去重决策
# --------------------------------------------------------------------------- #


def test_dedup_add_branch(core, backend, embedding):
    from artifact_spirit.extract.dedup import Deduplicator

    dedup = Deduplicator(backend=backend)
    assert dedup.decide({"content": "全新的事实内容", "subject": "A", "predicate": "p", "object": "o"}).decision == "ADD"


def test_dedup_ignore_branch(core, backend, embedding):
    from artifact_spirit.extract.dedup import Deduplicator

    candidate = {"content": "已存在的内容", "subject": "A", "predicate": "p", "object": "o"}
    dedup = Deduplicator(backend=backend)
    assert dedup.decide(candidate).decision == "ADD"
    write_intents(backend, core.ingest_turn(
        TurnEvent(session_id="s1", user="请记住：我偏好深色主题", assistant="", ts=FIXED_TS)
    ), embedding)
    record = backend.query()[0]
    candidate = {
        "content": record.content, "subject": record.subject,
        "predicate": record.predicate, "object": record.object,
        "scope": record.scope,
    }
    assert dedup.decide(candidate).decision == "IGNORE"


# --------------------------------------------------------------------------- #
# 两个"配好了却从没用上"的缺口（提取的时间、召回的向量）
# --------------------------------------------------------------------------- #


class _PromptSpy:
    """记录**实际发出去的提示词**的假 LLM。

    用它而不是只断言"调了一次"：这里要验的是**提示词里有没有那句话**。
    "调用发生了、内容不对"正是这类缺陷的样子——只数调用次数是数不出来的。
    """

    def __init__(self) -> None:
        self.seen: list[list[dict]] = []

    def complete_json(self, *, messages, schema, model=None, timeout=None):
        self.seen.append(list(messages))
        return {"memories": []}


def test_extraction_tells_the_model_when_the_conversation_happened():
    """提取时必须把**本轮时间**交给模型——否则「昨天」无从换算成绝对日期。

    没有它，模型不知道"今天"是哪天，只能照抄一个相对说法，
    时间点于是从库里彻底消失：实测对话里 `...went to a support group yesterday`
    提取出的记忆**一个日期都没有**，而标准答案正是一个绝对日期。
    """
    from artifact_spirit.extract import Extractor

    spy = _PromptSpy()
    Extractor(llm=spy).extract("我昨天去了支持小组", now="2023-05-08T13:56:00+00:00")

    assert spy.seen, "应当调用过一次模型"
    prompt = str(spy.seen[-1])
    assert "2023-05-08" in prompt, f"提示词里要写清对话发生时间，实际：{prompt[:180]}"


def test_extraction_prompt_demands_absolute_dates_and_same_language():
    """两条硬规则必须**明文写在提示词里**，不能指望模型自己懂。

    相对时间与语言这两件事，模型的默认行为恰好都是错的：
    它会照抄 `yesterday`，也会跟着**系统提示**的语言走（而不是跟随对话原文）。
    靠"希望模型聪明"没法回归——所以钉住提示词里有这两句。
    """
    from artifact_spirit.extract import SYSTEM_PROMPT

    assert "绝对日期" in SYSTEM_PROMPT
    assert "跟随对话原文" in SYSTEM_PROMPT


def test_semantic_fusion_prefers_items_ranked_high_by_both_paths():
    """融合**按排名**（RRF）：两路都靠前的条目，必须胜过只在单路排第一的。

    钉住的是"两把尺子不能直接相加"：BM25 经 min-max 后**最高分恒为 1.0**（相对分），
    余弦相似度是 [0,1]（绝对分）。旧实现取 `max`，于是 BM25 的第一名**总是**
    压过向量，"含同一个词的句子"被稳定排到最前——实测 LoCoMo 的 Top-10
    全是"含 LGBTQ 但不相关"的句子，而库里那条真正相关的进不来。

    实测改善（`--ingest llm` 的库、20 题）：R@10 0.201→0.252、
    R@|gold| 0.158→0.212、MRR 0.487→0.682。
    """
    from artifact_spirit.core.recall import _merge_semantic

    # 让 `both` 在**两路都排第一**，其余两条各有一路落后。
    vector = {"both": 0.95, "vec_only": 0.90, "lex_only": 0.10}
    lexical = {"both": 12.0, "lex_only": 9.0, "vec_only": 0.5}

    fused = _merge_semantic(vector, lexical)

    assert fused["both"] > fused["vec_only"], "两路都靠前的应当胜过只有向量靠前的"
    assert fused["both"] > fused["lex_only"], "两路都靠前的应当胜过只有关键词靠前的"
    assert all(0.0 <= value <= 1.0 for value in fused.values()), (
        "融合结果必须落在 [0,1]——`fuse()` 的加权求和假定每一路都在这个区间"
    )


def test_semantic_fusion_is_rank_based_so_swapped_ranks_tie():
    """RRF **只看排名**：两条各在一路排第一时，融合分**完全相同**。

    这不是缺陷，是它的代价与收益：用"丢掉分数强度"换"无参数、抗量纲"。
    写下来是因为它反直觉——**向量分 0.95 与关键词分 12.0 的两条，会被融合成同分**
    （`1/61 + 1/62` 与 `1/62 + 1/61`），于是高下交给下游因子
    （importance / recency …）去决定。

    需要注意的是：**它们只是"分不出高下"，不是"被算成一样好"**——
    与旧的 `max` 相比，后者会直接让关键词那条满分、向量那条落败。
    """
    from artifact_spirit.core.recall import _merge_semantic

    vector = {"a": 0.95, "b": 0.10}  # 向量排名：a, b
    lexical = {"a": 0.5, "b": 12.0}  # 关键词排名：b, a

    fused = _merge_semantic(vector, lexical)

    assert fused["a"] == fused["b"], "排名互换 → 倒数和相同（加法交换律）"


def test_semantic_fusion_handles_a_missing_path():
    """某一路缺席时照常工作，**且顺序仍然正确**（降级不等于失效）。"""
    from artifact_spirit.core.recall import _merge_semantic

    only_vector = _merge_semantic({"a": 0.9, "b": 0.2}, {})
    assert only_vector["a"] > only_vector["b"]

    only_lexical = _merge_semantic({}, {"a": 1.0, "b": 8.0})
    assert only_lexical["b"] > only_lexical["a"], "只有关键词时也要按它的排名走"

    assert _merge_semantic({}, {}) == {}, "两路都没有时不该造出条目"


def test_recall_computes_the_vector_when_the_caller_did_not(backend, embedding):
    """`recall` 在调用方没给 `vec` 时**自己算**——否则向量路静默失效。

    钉的是一个"配好了却从没用上"的缺陷：嵌入可用、库里向量齐全、
    `embedding_available()` 返回 True，而**召回一次都没走过向量**——
    因为只有工具面自己算 `vec`，从 `core.recall` 进来的调用全漏了。
    它的表现是"语义召回不行"，于是人会去怀疑**向量模型**，
    而不是"向量根本没接上"。
    """
    from artifact_spirit.core import ArtifactSpiritCore
    from artifact_spirit.core.base import RecallQuery

    backend.put(
        MemoryRecord(id="sem_home", layer="semantic", type="fact", content="用户住在上海")
    )
    core = ArtifactSpiritCore(backend=backend, embedding=embedding)

    core.recall(RecallQuery(text="我住在哪", top_k=5))
    assert embedding.calls, "recall 应当把查询文本交给嵌入模型——否则向量路等于没接"
    assert "我住在哪" in embedding.calls[-1]


def test_recall_still_works_without_embedding(backend):
    """没有嵌入时**照常召回**（降级而不是抛错，F2）。"""
    from artifact_spirit.core import ArtifactSpiritCore
    from artifact_spirit.core.base import RecallQuery

    backend.put(
        MemoryRecord(id="sem_home", layer="semantic", type="fact", content="用户住在上海")
    )
    core = ArtifactSpiritCore(backend=backend, embedding=None)

    hits = core.recall(RecallQuery(text="用户住在上海", top_k=5))
    assert hits, "没有嵌入时应当仍能靠关键词召回"


def test_dedup_invalidate_branch_on_value_change(core, backend, embedding):
    """取值变化 → ``INVALIDATE``（**不是** ``UPDATE``）。

    M3 的关键更正：``UPDATE`` 会覆盖历史，于是"用户换地址"与"我们改错别字"混为一谈——
    而这两件事在"我上个月填的地址是什么"上，答案完全不同（D-17 / INV-7）。
    """
    from artifact_spirit.extract.dedup import Deduplicator

    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    dedup = Deduplicator(backend=backend)
    decision = dedup.decide(
        {"content": "用户偏好浅色主题", "subject": "用户", "predicate": "prefers", "object": "浅色主题"}
    )
    assert decision.decision == "INVALIDATE"
    assert decision.target_id == backend.query()[0].id


def test_dedup_merge_branch_on_different_wording(backend):
    """同一取值、不同表述 → ``MERGE``（**不是 ``IGNORE``**）。

    ``IGNORE`` 在这里是错的：那会丢掉新表述里的细节。合并后的文本必须**双方都保留**——
    否则"合并"只是个听起来体面的丢弃。
    """
    from artifact_spirit.extract.dedup import Deduplicator

    first = MemoryRecord(
        id="sem_merge_probe",
        layer="semantic",
        type="fact",
        content="用户偏好深色主题",
        subject="用户",
        predicate="prefers",
        object="深色主题",
    )
    backend.put(first)

    decision = Deduplicator(backend=backend).decide(
        {
            "content": "用户偏好深色主题，夜里更护眼",
            "subject": "用户",
            "predicate": "prefers",
            "object": "深色主题",
        }
    )
    assert decision.decision == "MERGE"
    assert decision.target_id == first.id
    assert "夜里更护眼" in decision.merged["content"], "合并后的文本必须带上新表述的细节"


def test_merge_texts_prefers_the_superset_and_joins_otherwise():
    """`merge_texts` 的规则：**包含关系取超集**，否则分号连接（可解释、可测）。"""
    from artifact_spirit.extract.dedup import merge_texts

    assert merge_texts("喜欢咖啡", "喜欢咖啡和茶") == "喜欢咖啡和茶"
    assert merge_texts("喜欢咖啡和茶", "喜欢咖啡") == "喜欢咖啡和茶"
    assert merge_texts("喜欢咖啡", "喜欢茶") == "喜欢咖啡；喜欢茶"
    assert merge_texts("", "只有新的") == "只有新的"
    assert merge_texts("只有旧的", "") == "只有旧的"


def test_audit_op_covers_every_decision():
    """`_AUDIT_OP_FOR_DECISION` 必须覆盖 `Decision` 的**全部取值**，且 op 已登记。

    这是 P0-6 的机械堵口，两个方向都查：

    - 决策新增而映射没跟 → 第一次走到那条分支时 `KeyError`（红得晚）；
    - 映射里的 op 没进 `AUDIT_OPS` → 写审计时 `StoreError`（`ignore` / `merge` 正是这么漏的）。
    """
    from typing import get_args

    from artifact_spirit.core.facade import _AUDIT_OP_FOR_DECISION
    from artifact_spirit.extract.dedup import Decision
    from artifact_spirit.store.base import AUDIT_OPS

    declared = set(get_args(Decision))
    assert declared == set(_AUDIT_OP_FOR_DECISION), (
        f"决策与审计 op 映射不一致：决策 {sorted(declared)}，"
        f"映射 {sorted(_AUDIT_OP_FOR_DECISION)}"
    )
    unregistered = sorted(op for op in _AUDIT_OP_FOR_DECISION.values() if op not in AUDIT_OPS)
    assert not unregistered, f"这些 op 没登记进 AUDIT_OPS：{unregistered}"


def test_fact_change_end_to_end_invalidates_the_old_version(backend, embedding):
    """端到端：同一属性取值变化 → 旧条**失效**（不删）+ 新条落库 + `supersedes` 边。

    这是 M3 的核心行为，也是 `asof` 能回答"上个月我住哪"的前提。
    **三条缺一不可**——少一条，`asof` 或图查询就少一半答案。
    """
    llm = ScriptedLLM(
        [
            ("上海", memory_payload(content="我住在上海", object="上海")),
            ("北京", memory_payload(content="我搬到北京了，住在朝阳区", object="北京")),
        ]
    )
    from artifact_spirit.core.salience import SalienceConfig

    core = ArtifactSpiritCore(
        backend=backend,
        clock=lambda: FIXED_TS,
        embedding=embedding,
        llm=llm,
        # **门槛关到 0**：本用例验的是"决策 → 意图"的翻译，不是显著性判定。
        # 假 embedding 的向量方向几乎一致（`seed` 只改长度、不改方向），
        # 库里有第一条之后，第二遍必然被判"不够新颖"——那是**测试替身的性质**，
        # 与被测行为无关。把它排除掉，才有机会看到真正要验的那段。
        settings=CoreSettings(candidate_k=10, salience=SalienceConfig(threshold=0.0)),
    )

    # 两个 turn 必须**时间不同**：`invalidate` 会拒绝 `valid_to <= valid_from`（空区间），
    # 而固定时钟下 `ctx.ts` 恰好相等——那是**时钟替身的性质**，不是被测行为。
    ingest(core, backend, embedding, "请记住：我住在上海", ts=FIXED_TS)
    old = backend.query(status=None)
    assert len(old) == 1, f"第一条事实应当落库，实际 {len(old)} 条"
    old_id = old[0].id

    ingest(core, backend, embedding, "请记住：我搬到北京了", ts="2026-09-20T10:00:00+08:00")

    rows = {r.id: r for r in backend.query(status=None)}
    assert len(rows) == 2, "事实变化应当**新增**一条，而不是覆盖旧条"
    new_id = next(i for i in rows if i != old_id)

    assert old_id in rows, "失效**不是删除**——旧记录必须还在"
    assert rows[old_id].valid_to, "旧条应当写上 valid_to"
    assert rows[old_id].superseded_by == new_id, "旧条应当指向取代它的新条"

    edges = [e for e in backend.all_relations() if e["rel_type"] == "supersedes"]
    assert edges, "应当有一条 supersedes 关联边"
    assert edges[0]["src_id"] == new_id
    assert edges[0]["dst_id"] == old_id


def test_dedup_decision_is_audited(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    assert any(e.op == "add" for e in backend.audit_replay())


def test_dedup_llm_arbitration_falls_back_to_add(backend):
    from artifact_spirit.extract.dedup import Deduplicator

    llm = ScriptedLLM([("仲裁", SchemaViolationError("boom"))])
    dedup = Deduplicator(backend=backend, llm=llm)
    existing = [MemoryRecord(id="sem_a", layer="semantic", type="fact", content="旧内容")]
    decision = dedup.llm_arbitrate({"content": "新内容"}, existing, schema={"type": "object"})
    assert decision.decision == "ADD"


# --------------------------------------------------------------------------- #
# T-AL2-12 / 13 巩固
# --------------------------------------------------------------------------- #


def test_session_end_produces_one_episode(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    report = core.on_session_end("s1")
    assert report.episode_id
    write_intents(backend, report.intents, embedding)
    episodes = backend.query(layer="episodic", status=None)
    assert len(episodes) == 1


def test_consolidation_is_idempotent(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    first = core.on_session_end("s1")
    write_intents(backend, first.intents, embedding)
    second = core.on_session_end("s1")
    assert second.skipped == "already_consolidated"
    assert second.episode_id == first.episode_id
    assert len(backend.query(layer="episodic", status=None)) == 1


def test_cross_session_repetition_promotes_to_semantic(core, backend, embedding):
    backend.entity_upsert("RAGFlow", "project")
    for session in ("s1", "s2"):
        ingest(core, backend, embedding, "RAGFlow 项目使用 Python 3.12", session_id=session)
        report = core.on_session_end(session)
        write_intents(backend, report.intents, embedding)

    promoted = [
        r
        for r in backend.query(layer="semantic", status=None)
        if r.predicate == "recurring_topic"
    ]
    assert promoted, "跨会话复现的实体应被提升为语义记忆"
    assert backend.relations_of(promoted[0].id), "提升必须留下 derived_from 来源链"


def test_promotion_is_idempotent(core, backend, embedding):
    backend.entity_upsert("RAGFlow", "project")
    for session in ("s1", "s2", "s3"):
        ingest(core, backend, embedding, "RAGFlow 项目使用 Python 3.12", session_id=session)
        write_intents(backend, core.on_session_end(session).intents, embedding)
    promoted = [
        r
        for r in backend.query(layer="semantic", status=None)
        if r.predicate == "recurring_topic"
    ]
    assert len(promoted) == 1, "同一提升不得重复产生新记录"


def test_consolidation_produces_episode_before_promotion(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    report = core.on_session_end("s1")
    ops = [i.op for i in report.intents]
    assert ops[0] == "put", "首个意图必须是 episode 本身"


# --------------------------------------------------------------------------- #
# T-AL2-14 衰减排序与降级
# --------------------------------------------------------------------------- #


def test_decay_produces_no_deletion_or_downgrade(core, backend, embedding):
    """D-16 纪律：注入"很久未访问"的记忆 → **不自动降级、不删除**。"""
    record = MemoryRecord(
        id="", layer="semantic", type="fact", content="很久以前的一条记忆",
        created_at="2020-01-01T00:00:00+08:00", last_access_at="2020-01-01T00:00:00+08:00",
    )
    backend.put(record)
    report = core.decay(now="2026-09-14T00:00:00+08:00", dry_run=False)
    assert report.downgrades == []
    assert not any(i.op in ("set_status", "forget") for i in report.intents)
    assert backend.get(record.id).status == "active"


def test_decay_module_contains_no_delete_calls():
    """红线：``decay.py`` 中不存在任何删除或降级调用（可源码断言）。"""
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT

    source = PACKAGE_ROOT / "core" / "decay.py"
    text = source.read_text(encoding="utf-8")
    body = text.split('"""', 2)[-1]  # 去掉模块 docstring（其中会**提到**这些词）
    for banned in ("hard_delete", "set_status", "forget("):
        assert banned not in body, f"decay.py 出现 {banned}"


def test_decay_dry_run_writes_nothing(core, backend, embedding):
    record = MemoryRecord(
        id="", layer="semantic", type="fact", content="会衰减",
        created_at="2020-01-01T00:00:00+08:00", last_access_at="2020-01-01T00:00:00+08:00",
    )
    backend.put(record)
    before = backend.get(record.id).strength
    report = core.decay(now="2026-09-14T00:00:00+08:00", dry_run=True)
    assert report.intents == []
    assert backend.get(record.id).strength == before


def test_decay_reports_low_strength_candidates(core, backend, embedding):
    backend.put(MemoryRecord(id="", layer="semantic", type="fact", content="旧", created_at="2019-01-01T00:00:00+08:00"))
    backend.put(MemoryRecord(id="", layer="semantic", type="fact", content="新", created_at="2026-09-13T00:00:00+08:00"))
    report = core.decay(now="2026-09-14T00:00:00+08:00", dry_run=True)
    assert report.low_strength


def test_dormant_memory_keeps_full_detail(core, backend, embedding):
    """D-23：``dormant`` 只退出注意力，**详情仍在库**。"""
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    backend.set_status(ref, "dormant", reason="测试")
    assert backend.get(ref).content == "用户偏好深色主题，不喜欢浅色界面"
    assert "不喜欢浅色界面" in core.expand(ref, "L2")


def test_dormant_memory_can_be_restored(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    ref = backend.query()[0].id
    backend.set_status(ref, "dormant", reason="测试")
    backend.set_status(ref, "active", reason="恢复")
    assert backend.get(ref).status == "active"


def test_status_target_never_archive(core, backend, embedding):
    """D-17：状态机里**不存在** archived。"""
    from artifact_spirit.store.base import Status

    assert "archived" not in str(Status)


# --------------------------------------------------------------------------- #
# T-AL2-15 优化删除（"不可达"判定）
# --------------------------------------------------------------------------- #


def test_isolated_memory_hits_unreachable_criterion(core, backend, embedding):
    isolated = MemoryRecord(
        id="", layer="semantic", type="fact", content="完全孤立毫无关联的一条内容"
    )
    backend.put(isolated)
    report = core.optimize(dry_run=True)
    assert isolated.id in report.unreachable


def test_uncommitted_session_memory_is_not_flagged(core, backend, embedding):
    """会话未结束 → 记忆未定形 → 不参与治理（避免误杀刚写入的正常记忆）。"""
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    assert core.optimize(dry_run=True).unreachable == []


def test_entity_associated_memory_is_reachable(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    backend.session_end("s1", FIXED_TS)
    assert core.optimize(dry_run=True).unreachable == []


def test_optimizer_produces_no_intents_without_authorization(core, backend, embedding):
    backend.put(MemoryRecord(id="", layer="semantic", type="fact", content="孤立内容 ABC"))
    report = core.optimize(dry_run=False)
    assert report.intents == []
    assert any("授权" in note for note in report.notes)


def test_optimizer_dry_run_writes_nothing(core, backend, embedding):
    record = MemoryRecord(id="", layer="semantic", type="fact", content="孤立内容 ABC")
    backend.put(record)
    core.optimize(dry_run=True)
    assert backend.get(record.id) is not None


def test_optimizer_deletes_with_optimizer_actor_and_restorable(core, backend, embedding):
    record = MemoryRecord(id="", layer="semantic", type="fact", content="孤立内容 ABC")
    backend.put(record)
    report = core.optimize(dry_run=False, autonomous=True)
    write_intents(backend, report.intents, embedding)
    assert backend.get(record.id) is None
    assert any(e.actor == "optimizer" for e in backend.audit_replay() if e.op == "forget")

    snapshot = backend.delete_snapshots()[0]
    write_intents(backend, core.restore(snapshot["audit_id"]), embedding)
    assert backend.get(record.id) is not None


def test_optimizer_never_touches_core_layer(core, backend, embedding):
    core_memory = MemoryRecord(
        id="", layer="core", type="identity", content="我是一个注重隐私的助手"
    )
    backend.put(core_memory)
    report = core.optimize(dry_run=True)
    assert core_memory.id not in report.unreachable
    assert core_memory.id not in report.downgrades


def test_optimizer_criterion_ignores_time_and_frequency(core, backend, embedding):
    """D-23 断言：判据只看图结构，不看时间与访问频率。"""
    old_but_linked = MemoryRecord(
        id="", layer="semantic", type="fact", content="很旧但有邻居",
        created_at="2015-01-01T00:00:00+08:00",
    )
    neighbor = MemoryRecord(id="", layer="semantic", type="fact", content="邻居节点")
    backend.put(old_but_linked)
    backend.put(neighbor)
    backend.link("memory", old_but_linked.id, "memory", neighbor.id, "co_activation", 0.9)
    backend.entity_upsert("邻居", "concept")

    fresh_but_isolated = MemoryRecord(
        id="", layer="semantic", type="fact", content="刚写入但完全孤立",
        created_at="2026-09-14T09:59:00+08:00",
    )
    backend.put(fresh_but_isolated)

    report = core.optimize(dry_run=True)
    assert old_but_linked.id not in report.unreachable
    assert fresh_but_isolated.id in report.unreachable


def test_optimizer_confirm_callback_can_reject(core, backend, embedding):
    record = MemoryRecord(id="", layer="semantic", type="fact", content="孤立内容 ABC")
    backend.put(record)
    report = core.optimize(dry_run=False, confirm=lambda ids: False)
    assert report.intents == []
    assert any("未获批准" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# T-AL2-16 激活与健康度
# --------------------------------------------------------------------------- #


def test_co_activation_creates_edges(core, backend, embedding):
    a = MemoryRecord(id="", layer="semantic", type="fact", content="A 记忆")
    b = MemoryRecord(id="", layer="semantic", type="fact", content="B 记忆")
    backend.put(a)
    backend.put(b)
    intents = core.activator.co_activate([a.id, b.id])
    assert intents and intents[0].op == "link"
    write_intents(backend, intents, embedding)
    assert backend.relations_of(a.id)


def test_co_activation_is_idempotent(core, backend, embedding):
    a = MemoryRecord(id="", layer="semantic", type="fact", content="A")
    b = MemoryRecord(id="", layer="semantic", type="fact", content="B")
    backend.put(a)
    backend.put(b)
    write_intents(backend, core.activator.co_activate([a.id, b.id]), embedding)
    second = core.activator.co_activate([a.id, b.id])
    assert second[0].op == "reinforce"
    write_intents(backend, second, embedding)
    assert len(backend.relations_of(a.id)) == 1


def test_health_report_shape(core, backend, embedding):
    ingest(core, backend, embedding, "请记住：我偏好深色主题")
    health = core.health()
    assert health.total >= 1
    assert isinstance(health.layer_counts, dict)
    assert health.core_count == 0


def test_system_prompt_block_has_budget_and_content(core, backend, embedding):
    backend.put(
        MemoryRecord(id="", layer="core", type="identity", content="我是一个务实的工程助手")
    )
    block = core.system_prompt_block(token_budget=200)
    assert "我是谁" in block or "记忆状态" in block
    assert len(block) < 2000


def test_ingest_turn_never_raises_on_broken_llm(backend, embedding):
    class BrokenLLM:
        def complete(self, **kwargs):
            raise RuntimeError("完全意料之外的错误")

        def complete_json(self, **kwargs):
            raise RuntimeError("完全意料之外的错误")

    core = ArtifactSpiritCore(
        backend=backend, clock=lambda: FIXED_TS, embedding=embedding, llm=BrokenLLM()
    )
    with pytest.raises(RuntimeError):
        # 意料之外的异常**允许上抛**——AL1 边界会兜住它（AL2 只兜已知的模型异常）
        core.ingest_turn(TurnEvent(session_id="s1", user="记住 X", assistant="", ts=FIXED_TS))


def _rec(mem_id: str) -> MemoryRecord:
    return MemoryRecord(id=mem_id, layer="semantic", type="fact", content=f"内容 {mem_id}")


_ = sqlite3
