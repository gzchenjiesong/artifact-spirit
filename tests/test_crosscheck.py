"""交叉验证（T-AL2-18 · M3）：LLM 仲裁池的**限频与降级**。

这组用例的重点不是"裁决得对不对"（那是模型的事），而是四条**系统性质**：

| 用例 | 堵的失效模式 |
|---|---|
| `test_no_conflict_means_no_llm_call` | "高门槛低频"退化成"每轮都调"——花的是钱和延迟，且**没有任何症状** |
| `test_conflict_triggers_arbitration_and_invalidates_older` | 冲突永远不被裁决（功能形同虚设） |
| `test_llm_unavailable_keeps_both_and_says_so` | 仲裁失败时**误删记忆**——代价不对称：多留一条无害，误删一条静默消失 |
| `test_already_settled_pairs_are_not_asked_again` | 同一对冲突被反复拿去问模型 |
"""

from __future__ import annotations

from conftest import ScriptedLLM, make_record, write_intents

from artifact_spirit.core import ArtifactSpiritCore

TS_OLD = "2026-01-01T00:00:00+08:00"
TS_MID = "2026-06-01T00:00:00+08:00"


def _conflicting_pair(backend):
    """造一对**真冲突**：同属性、取值不同、都 active。"""
    older = make_record(
        content="用户住在上海",
        subject="用户",
        predicate="住在",
        object="上海",
        created_at=TS_OLD,
        valid_from=TS_OLD,
    )
    backend.put(older)
    newer = make_record(
        content="用户住在北京",
        subject="用户",
        predicate="住在",
        object="北京",
        created_at=TS_MID,
        valid_from=TS_MID,
    )
    backend.put(newer)
    return older, newer


def test_no_conflict_means_no_llm_call(backend):
    """无冲突 → **一次 LLM 调用都不发生**。

    只断言"有冲突时调了"是不够的——真要防的是"每轮都调"。
    所以这里用**调用计数**断言，而不是看结果对不对：
    结果在两种实现下都一样，账单不一样。
    """
    llm = ScriptedLLM([])
    core = ArtifactSpiritCore(backend=backend, llm=llm)
    backend.put(make_record(content="孤零零的一条", subject="用户", predicate="喜欢", object="咖啡"))

    report = core.crosscheck()

    assert report.conflicts == 0
    assert llm.calls == [], f"没有冲突却调了模型：{llm.calls}"


def test_conflict_triggers_arbitration_and_invalidates_older(backend):
    """同属性取值不同 → 触发仲裁；裁为 `INVALIDATE` 时**旧条失效（不删）**。"""
    older, newer = _conflicting_pair(backend)
    llm = ScriptedLLM([("上海", {"decision": "INVALIDATE", "reason": "用户换城市了"})])
    core = ArtifactSpiritCore(backend=backend, llm=llm)

    report = core.crosscheck()

    assert report.conflicts == 1, "这一对应当被认成冲突"
    assert report.arbitrated == 1
    assert llm.calls, "有冲突就必须真的问过模型"
    write_intents(backend, report.intents)

    stored = backend.get(older.id)
    assert stored is not None, "INVALIDATE **不是删除**"
    assert stored.superseded_by == newer.id
    assert stored.valid_to, "旧条应当写上 valid_to"

    edges = [e for e in backend.all_relations() if e["rel_type"] == "supersedes"]
    assert edges and edges[0]["src_id"] == newer.id and edges[0]["dst_id"] == older.id


def test_llm_unavailable_keeps_both_and_says_so(backend):
    """LLM 不可用 → **两条都留**，且报告说清"未配置"。

    这是代价不对称的落点：多留一条只是多占一点空间，
    而误判 `INVALIDATE` 会让一条真实记忆**静默地从召回里消失**。
    """
    older, _ = _conflicting_pair(backend)
    core = ArtifactSpiritCore(backend=backend, llm=None)

    report = core.crosscheck()

    assert report.conflicts == 1, "冲突本身要能被发现——**发现**不依赖模型"
    assert report.intents == [], "没有模型时不得产生任何裁决意图"
    assert report.skipped == "未配置 LLM"
    assert backend.get(older.id).superseded_by is None, "旧条不得被误判失效"


def test_bad_llm_output_falls_back_to_keep_both(backend):
    """模型返回了未知裁决 → 回到 `KEEP_BOTH`，并**在报告里留痕**。"""
    _conflicting_pair(backend)
    llm = ScriptedLLM([("上海", {"decision": "删除旧的那条"})])
    core = ArtifactSpiritCore(backend=backend, llm=llm)

    report = core.crosscheck()

    assert report.arbitrated == 0
    assert report.kept_both == 1
    assert report.intents == [], "未知裁决绝不能变成一次写入"


def test_already_settled_pairs_are_not_asked_again(backend):
    """已有 `superseded_by` / `supersedes` 的对**不再问一遍**。

    冲突是常态（同一属性的历史版本全在库里）——不筛掉已裁决的，
    每轮都会把同一批拿去问模型。
    """
    older, newer = _conflicting_pair(backend)
    backend.update(older.id, {"superseded_by": newer.id, "valid_to": TS_MID})

    llm = ScriptedLLM([("上海", {"decision": "INVALIDATE"})])
    core = ArtifactSpiritCore(backend=backend, llm=llm)

    report = core.crosscheck()

    assert report.conflicts == 0, "已裁决的对不该再进候选集"
    assert llm.calls == [], "已裁决的对不该再问模型"


# --------------------------------------------------------------------------- #
# T-AL4-13 仲裁档位（AL4 侧）
# --------------------------------------------------------------------------- #


def test_crosscheck_task_is_a_separate_slot():
    """交叉验证是一个**独立档位**，且默认走轻量模型（T-AL4-13）。

    独立档位的意义：它能被单独指向另一个模型或另一个 endpoint，
    而不必和提取 / 巩固共用配置——"高门槛低频"的能力不该拖着热路径的配置走。
    """
    from artifact_spirit.model.base import TASKS
    from artifact_spirit.model.resolver import DEFAULT_LLM_TASK_MODELS

    assert "crosscheck" in TASKS, "任务档位表里没有 crosscheck"
    assert "crosscheck" in DEFAULT_LLM_TASK_MODELS
    assert DEFAULT_LLM_TASK_MODELS["crosscheck"] == "glm-5.3-flash", "仲裁走轻量档"


def test_crosscheck_reuses_the_single_http_path():
    """仲裁复用 `complete_json`，**不新增第二条 HTTP 路径**（T-AL4-13）。

    自己写一条 HTTP 调用看起来只是"多几行"，实际绕开了重试、超时、
    schema 校验那一整套护栏——而护栏是在**别人的失败里**总结出来的。
    """
    from pathlib import Path

    source = Path("src/artifact_spirit/core/crosscheck.py").read_text(encoding="utf-8")
    assert "httpx" not in source, "AL2 不得直接碰 HTTP（R5）"
    assert "complete_json" in source, "仲裁应当复用 AL4 的结构化输出通道"
