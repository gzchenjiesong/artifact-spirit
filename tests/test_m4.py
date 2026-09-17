"""M4 验收：扩散激活（T-AL2-19）、记忆进化（T-AL2-20）、核心记忆（T-AL2-21）、审查干预（T-AL2-22）。

## 为什么这层容易"看起来已经做完"

M4 的四项能力里，**扩散激活与核心记忆在 MVP 就已经有实现**（六因子里各占一路），
所以"代码在那儿"不等于"验收过了"。这组用例补的正是那些**从来没有断言过**的性质：
边界（`hops > 1`）、零值语义（孤立记忆）、降级方式（静默 vs 抛错）。
"""

from __future__ import annotations

import pytest
from conftest import make_record

from artifact_spirit.core.activation import Activator
from artifact_spirit.core.recall import score_diffusion

FIXED_TS = "2026-09-14T10:00:00+08:00"


# --------------------------------------------------------------------------- #
# T-AL2-19 扩散激活
# --------------------------------------------------------------------------- #


def test_diffusion_gives_zero_without_edges():
    """没有关联边的记忆，`diffusion` 因子得 **0**。

    不是"平均分"、也不是"兜底最小值"——**图不通就没有扩散**。
    给它一个非零值会让"孤立"这个信号消失，而"孤立"正是优化任务判定不可达的依据之一。
    """
    lonely = make_record(id="sem_lonely", content="孤立记忆")
    assert score_diffusion(lonely, set(), {}) == 0.0


def test_seed_itself_scores_full():
    """种子自身给满分——它是"激活源"，不是"被激活的邻居"。"""
    seed = make_record(id="sem_seed", content="种子记忆")
    assert score_diffusion(seed, {"sem_seed"}, {}) == 1.0


def test_neighbor_takes_the_max_edge_weight_and_is_clipped():
    """邻居取到种子集合的**最大边权**，且裁剪到 [0,1]（越界值不许直接进融合）。"""
    neighbor = make_record(id="sem_nb", content="邻居")
    assert score_diffusion(neighbor, set(), {"sem_nb": 0.42}) == pytest.approx(0.42)
    assert score_diffusion(neighbor, set(), {"sem_nb": 3.0}) == 1.0


def test_hops_beyond_one_is_refused_not_silently_downgraded(backend):
    """`hops > 1` **抛 `NotImplementedError`**，不静默按一跳处理。

    静默降级在这里格外危险：调用方以为拿到了多跳关联，实际只拿到一跳——
    而"少了的那部分"**不产生任何信号**（不是空、不是错，只是少了）。
    """
    activator = Activator(backend=backend, clock=lambda: FIXED_TS)
    with pytest.raises(NotImplementedError):
        activator.activate(["sem_any"], hops=2)


def test_activate_expands_along_edges_and_respects_threshold(backend):
    """沿关联边扩散：**弱于阈值的边不激活**——否则扩散会退化成「全连通」。"""
    seed = make_record(id="sem_seed", content="种子")
    close = make_record(id="sem_close", content="强关联")
    far = make_record(id="sem_far", content="弱关联")
    for rec in (seed, close, far):
        backend.put(rec)
    backend.link("memory", seed.id, "memory", close.id, "co_activation", 0.8)
    backend.link("memory", seed.id, "memory", far.id, "co_activation", 0.05)

    activated = Activator(backend=backend, clock=lambda: FIXED_TS).activate([seed.id])

    assert activated[seed.id] == 1.0
    assert close.id in activated, "强关联应当被激活"
    assert far.id not in activated, "弱于阈值的边不激活"


# --------------------------------------------------------------------------- #
# T-AL2-20 记忆进化
# --------------------------------------------------------------------------- #


def test_evolution_touches_abstract_but_never_content(backend):
    """进化后 `abstract` 变、**`content` 一字不改**。

    原文是事实来源、抽象是投影——这条线一旦松掉，
    "用户的原话"就没有权威副本了（与 INV-1 同一条精神）。
    """
    from conftest import ScriptedLLM, write_intents

    from artifact_spirit.core import ArtifactSpiritCore

    old = make_record(id="sem_old", content="用户喜欢喝咖啡", abstract="喜欢咖啡")
    backend.put(old)
    new = make_record(id="sem_new", content="用户喜欢喝浅烘的咖啡，不加糖")
    backend.put(new)

    llm = ScriptedLLM([("旧概括", {"changed": True, "abstract": "喜欢浅烘咖啡、不加糖"})])
    core = ArtifactSpiritCore(backend=backend, llm=llm)

    intents = core.evolver.evolve(new, neighbors=[old])
    assert intents, "配置了模型且有邻居时应产出改写意图"
    write_intents(backend, intents)

    after = backend.get(old.id)
    assert after.abstract != old.abstract, "抽象应当被更新"
    assert after.content == old.content, "**原文一字不改**——它是事实来源"


def test_evolution_leaves_audit_with_before_and_after(backend):
    """每次改写都能查出"改前 / 改后"（INV-12）。

    抽象被静默改掉比不改更糟：用户会看到一句自己没说过、
    也追溯不到来源的"自己的总结"。
    """
    from conftest import ScriptedLLM, write_intents

    from artifact_spirit.core import ArtifactSpiritCore

    old = make_record(id="sem_old", content="用户喜欢喝咖啡", abstract="喜欢咖啡")
    backend.put(old)
    new = make_record(id="sem_new", content="用户改喝浅烘了")
    backend.put(new)

    llm = ScriptedLLM([("旧概括", {"changed": True, "abstract": "现在喝浅烘"})])
    core = ArtifactSpiritCore(backend=backend, llm=llm)
    write_intents(backend, core.evolver.evolve(new, neighbors=[old]))

    events = [e for e in backend.audit_replay() if e.target_id == old.id]
    evolved = [e for e in events if e.before and e.after]
    assert evolved, f"改写必须留 before/after，实际审计：{[e.op for e in events]}"
    assert evolved[-1].before["abstract"] == "喜欢咖啡"
    assert evolved[-1].after["abstract"] == "现在喝浅烘"


def test_evolution_disabled_degrades_to_add_only(backend):
    """关掉开关 → **一次 LLM 调用都不发生**（配置真的生效，不是装饰）。

    代价大的能力必须能关；"开关在那儿但不起作用"比没有开关更坏——
    用户以为自己关掉了。
    """
    from conftest import ScriptedLLM

    from artifact_spirit.core import ArtifactSpiritCore, CoreSettings

    old = make_record(id="sem_old", content="用户喜欢喝咖啡", abstract="喜欢咖啡")
    backend.put(old)
    new = make_record(id="sem_new", content="用户改喝浅烘了")
    backend.put(new)

    llm = ScriptedLLM([("旧概括", {"changed": True, "abstract": "现在喝浅烘"})])
    core = ArtifactSpiritCore(
        backend=backend, llm=llm, settings=CoreSettings(evolution_enabled=False)
    )

    assert core.evolver.evolve(new, neighbors=[old]) == []
    assert llm.calls == [], "关掉进化后不该调用模型"


def test_evolution_without_llm_says_why_it_did_nothing(backend):
    """未配置 LLM → 不改写，且报告**说清原因**（不假装做过了）。"""
    from artifact_spirit.core import ArtifactSpiritCore

    backend.put(make_record(content="一条记忆", abstract="摘要"))
    report = ArtifactSpiritCore(backend=backend, llm=None).evolve()

    assert report.evolved == 0
    assert report.intents == []
    assert report.skipped == "未配置 LLM"


# --------------------------------------------------------------------------- #
# T-AL2-21 核心记忆
# --------------------------------------------------------------------------- #


def test_core_promotion_is_refused_below_threshold(backend):
    """未达门槛**不升格**——差一点点的候选也不上。

    核心记忆的错误污染的是**人格**，所以这里的门比别处高得多（INV-11）。
    """
    from artifact_spirit.core import ArtifactSpiritCore
    from artifact_spirit.core.layers.core_memory import CORE_UPDATE_THRESHOLD

    core = ArtifactSpiritCore(backend=backend)
    almost = make_record(
        id="core_almost",
        layer="core",
        type="identity",
        content="我大概是个助手",
        confidence=CORE_UPDATE_THRESHOLD - 0.01,
    )
    assert core.core_memory.propose(almost, reason="自动候选") == [], "差一点也不能上"


def test_promotion_is_audited_and_visible_in_the_prompt_block(backend):
    """升格后 **`system_prompt_block` 真的看得到它**，且留了 `promote` 审计。

    "能注入人格"与"留了痕"是同一件事的两面：
    看不到 = 升格没生效；没痕 = 事后说不清它什么时候变成底色的。
    """
    from conftest import write_intents

    from artifact_spirit.core import ArtifactSpiritCore

    core = ArtifactSpiritCore(backend=backend)
    record = make_record(
        id="core_me", layer="core", type="identity", content="我是星空", confidence=0.99
    )
    intents = core.core_memory.propose(record, reason="用户明确要求")
    assert intents, "过了门槛就该产出意图"
    write_intents(backend, intents)

    assert "我是星空" in core.system_prompt_block(), "升格后系统提示里应当看得到它"
    ops = [e.op for e in backend.audit_replay() if e.target_id == "core_me"]
    assert "promote" in ops, f"升格必须留审计，实际：{ops}"


# --------------------------------------------------------------------------- #
# T-AL2-22 审查干预（增强）
# --------------------------------------------------------------------------- #


def test_temporal_intervention_changes_asof_but_keeps_history(backend):
    """时态干预后 `asof` 随之改变，而**历史仍可查**。

    这正是「事实变了」与「我们记错了」的分界：前者只截断有效期，
    后者才改记录本身。把两者混为一谈，`asof` 就答不出"上个月住哪"。
    """
    from conftest import write_intents

    from artifact_spirit.core import ArtifactSpiritCore

    core = ArtifactSpiritCore(backend=backend)
    backend.put(
        make_record(
            id="sem_addr",
            content="用户住在上海",
            created_at="2026-01-01T00:00:00+08:00",
            valid_from="2026-01-01T00:00:00+08:00",
        )
    )
    assert core.asof("sem_addr", "2026-06-01T00:00:00+08:00") is not None, "干预前 6 月有效"

    write_intents(
        backend,
        core.invalidate_since(
            "sem_addr", from_ts="2026-03-01T00:00:00+08:00", reason="用户在 3 月搬走了"
        ),
    )

    assert core.asof("sem_addr", "2026-06-01T00:00:00+08:00") is None, "3 月之后不再有效"
    assert core.asof("sem_addr", "2026-02-01T00:00:00+08:00") is not None, "**历史仍可查**"


def test_interventions_require_a_reason(backend):
    """缺 `reason` 一律拒绝——干预一律留痕（P5 / V2）。

    四种干预逐条验：**只要有一条漏了校验，"可追溯"就有了一个说不清的入口**。
    """
    from artifact_spirit.core import ArtifactSpiritCore

    core = ArtifactSpiritCore(backend=backend)
    backend.put(make_record(id="sem_x", content="随意"))
    for call in (
        lambda: core.correct("sem_x", {"content": "改了"}, reason=""),
        lambda: core.invalidate_since("sem_x", from_ts="2030-01-01T00:00:00+08:00", reason="  "),
        lambda: core.promote("sem_x", reason=""),
        lambda: core.demote("sem_x", reason=""),
    ):
        with pytest.raises(ValueError):
            call()


def test_core_interventions_are_audited_and_change_the_layer(backend):
    """升格 / 降格各有审计，`actor` 都是 `user`，且**层真的变了**。

    `actor='user'` 不是装饰：它区分"人的决定"与"流程的推断"——
    两者在事后复盘时的分量完全不同（D-17 / M11）。
    """
    from conftest import write_intents

    from artifact_spirit.core import ArtifactSpiritCore

    core = ArtifactSpiritCore(backend=backend)
    backend.put(make_record(id="sem_belief", content="用户相信长期主义"))

    write_intents(backend, core.promote("sem_belief", reason="用户说这是他的底色"))
    assert backend.get("sem_belief").layer == "core"
    assert "长期主义" in core.system_prompt_block(), (
        "人工升格也必须**看得见**——只改 layer 不改 type 就会「升格成功但看不见」"
    )

    write_intents(backend, core.demote("sem_belief", reason="用户改主意了"))
    assert backend.get("sem_belief").layer == "semantic"
    assert backend.get("sem_belief").content == "用户相信长期主义", "降格**不删内容**"

    events = [e for e in backend.audit_replay() if e.target_id == "sem_belief"]
    interventions = [e for e in events if e.op in ("promote", "demote")]
    assert {e.op for e in interventions} == {"promote", "demote"}, (
        f"升降格都要留审计，实际：{[e.op for e in events]}"
    )
    assert all(e.actor == "user" for e in interventions), (
        "干预的 actor 必须是 user——它区分「人的决定」与「流程的推断」，"
        "两者事后复盘时的分量完全不同"
    )


def test_demote_refuses_a_non_core_memory(backend):
    """降格只对核心记忆生效——对普通记忆"降格"是没有意义的动作，
    静默成功会让人以为"操作生效了"。"""
    from artifact_spirit.core import ArtifactSpiritCore

    core = ArtifactSpiritCore(backend=backend)
    backend.put(make_record(id="sem_plain", content="普通记忆"))
    with pytest.raises(ValueError):
        core.demote("sem_plain", reason="试试")
