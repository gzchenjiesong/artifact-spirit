"""AL5 运行时层验收（T-AL5-01 ~ 12）。

本文件由 `test_runtime_adapter.py` 拆出（设计评审 P1-6）：原来的单文件同时承载
AL1 宿主契约、AL5 辅助机制、CLI 与价值判决，出问题时无法从文件名判断"红的是哪一层"。
现在 AL5 归本文件，AL1 归 `test_al1_provider.py`，CLI 归 `test_al1_cli.py`，
价值判决与关键不变量归 `test_value.py`。
"""

from __future__ import annotations

import threading
import time

import pytest
from conftest import services_for

from artifact_spirit.compliance import (
    PACKAGE_ROOT,
    check_architecture,
    scan_dependencies,
)
from artifact_spirit.config import (
    ConfigError,
    config_path,
    load,
    save,
    toml_example,
    validate,
)
from artifact_spirit.core.base import TurnEvent
from artifact_spirit.observability import (
    AuditView,
    audit_view,
    doctor,
    doctor_text,
    layers,
    reflect,
    status,
    status_text,
)
from artifact_spirit.runtime import PRIORITY, start
from artifact_spirit.runtime.writer import WriteQueue, intent_priority

# =========================================================================== #
# AL5 · M0 配置
# =========================================================================== #


def test_toml_example_is_valid_and_loadable(tmp_path):
    path = config_path(tmp_path)
    path.write_text(toml_example(), encoding="utf-8")
    cfg = load(tmp_path, env={})
    assert cfg.embedding is not None
    assert cfg.embedding.model == "kinfra-text-embedding-4b"
    assert cfg.embedding_dim == 2560


def test_load_valid_toml(home):
    cfg = load(home, env={})
    assert cfg.name == "测试器灵"
    assert cfg.embedding.model == "test-embed"
    assert cfg.source == "file"


def test_env_overrides_toml(home):
    cfg = load(home, env={"ARTIFACT_SPIRIT_RECALL_TOP_K": "42"})
    assert cfg.recall["top_k"] == 42


def test_missing_config_file_is_not_an_error(tmp_path):
    cfg = load(tmp_path, env={})
    assert cfg.source == "defaults"
    assert cfg.embedding is None


def test_broken_toml_gives_actionable_error(tmp_path):
    config_path(tmp_path).write_text("[spirit\nname = ", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load(tmp_path, env={})
    assert "语法错误" in str(excinfo.value)


def test_config_rejects_literal_secret(tmp_path):
    config_path(tmp_path).write_text(
        '[models.llm]\napi_key = "sk-real-looking-key-value"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError) as excinfo:
        load(tmp_path, env={})
    assert "环境变量" in str(excinfo.value)


def test_config_allows_api_key_env(tmp_path):
    """密钥纪律要的是**变量名**，不是禁止提及密钥。"""
    config_path(tmp_path).write_text(
        '[models.llm]\napi_key_env = "MY_KEY"\n', encoding="utf-8"
    )
    cfg = load(tmp_path, env={})
    assert cfg.llm.api_key_env == "MY_KEY"


def test_validate_blocks_bad_queue_max(home):
    cfg = load(home, env={"ARTIFACT_SPIRIT_WORKER_WRITE_QUEUE_MAX": "0"})
    problems = validate(cfg)
    assert any("write_queue_max" in p for p in problems)


def test_validate_warns_but_does_not_block_on_weight_sum(home):
    from dataclasses import replace

    cfg = load(home, env={})
    broken = replace(cfg, recall={**cfg.recall, "weights": {"semantic": 0.9}})
    problems = validate(broken)
    assert problems and all(p.startswith("警告") for p in problems)


def test_save_whitelist_blocks_everything_else(home):
    result = save(home, {"models.llm.api_key": "sk-leak-me", "spirit.name": "改动"})
    text = result.read_text(encoding="utf-8")
    assert "sk-leak-me" not in text
    assert "api_key = " not in text
    assert "改动" in text


def test_save_does_not_duplicate_sections(home):
    save(home, {"spirit.name": "甲"})
    save(home, {"spirit.name": "乙"})
    text = config_path(home).read_text(encoding="utf-8")
    assert text.count("[spirit]") == 1
    assert load(home, env={}).name == "乙"


def test_spirit_id_not_overwritten(tmp_path):
    from artifact_spirit.store import SQLiteBackend

    be = SQLiteBackend(str(tmp_path / "a.db"), embedding_dim=4)
    be.open()
    first = be.ensure_spirit_id()
    assert be.ensure_spirit_id() == first
    be.close()

    be2 = SQLiteBackend(str(tmp_path / "a.db"), embedding_dim=4)
    be2.open()
    assert be2.ensure_spirit_id() == first
    be2.close()


# =========================================================================== #
# AL5 · M0 装配
# =========================================================================== #


def test_start_and_stop_are_repeatable(home):
    services = services_for(home, start_threads=True)
    assert services.backend.meta_get("schema_version")
    sequence = services.stop()
    assert sequence == [
        "stop_accepting",
        "flush_write_queue",
        "stop_maintenance",
        "stop_writer",
        "close_backend",
    ]
    # 再停一次不报错（幂等），且序列不变
    assert services.stop() == sequence
    again = services_for(home, start_threads=True)
    again.stop()


def test_stop_without_threads_still_reports_sequence(home):
    """没起线程也如实报告"关掉了什么"——不要因为没线程就变成空。"""
    services = services_for(home)
    assert services.stop() == ["stop_accepting", "flush_write_queue", "close_backend"]


def test_stop_order_has_writer_before_maintenance(home):
    services = services_for(home, start_threads=True)
    sequence = services.stop()
    assert sequence.index("stop_maintenance") < sequence.index("stop_writer")


def test_start_refuses_on_embedding_model_mismatch(home):
    services = services_for(home)
    services.backend.meta_set("embedding_model", "old-model")
    services.stop()

    with pytest.raises(Exception) as excinfo:
        services_for(home)
    assert "reindex" in str(excinfo.value)


def test_start_degrades_gracefully_without_models(tmp_path):
    """P6：模型没配好**不该阻止启动**——记忆照存，只是未结构化。"""
    empty = tmp_path / "bare"
    empty.mkdir()
    services = start(str(empty), env={}, start_threads=False)
    assert services.notes, "应当给出降级说明"
    assert services.backend.meta_get("schema_version")
    services.stop()


def test_reconcile_runs_on_start(home, monkeypatch):
    calls = {"n": 0}
    from artifact_spirit.store.sqlite_backend import SQLiteBackend

    original = SQLiteBackend.reconcile

    def spy(self, **kwargs):
        calls["n"] += 1
        return original(self, **kwargs)

    monkeypatch.setattr(SQLiteBackend, "reconcile", spy)
    services_for(home)
    assert calls["n"] >= 1


# =========================================================================== #
# AL5 · M1 writer 与队列
# =========================================================================== #


def test_submit_is_non_blocking_when_full():
    queue = WriteQueue(maxsize=2)
    for index in range(5):
        queue.submit({"n": index}, kind="intent.memory")
    started = time.perf_counter()
    queue.submit({"n": 99}, kind="intent.memory")
    assert (time.perf_counter() - started) < 0.05
    assert queue.depth() <= 2


def test_overflow_drops_lowest_priority_first():
    """溢出时**摘要先于记忆被丢**（LLD-AL5 §5 M3）。"""
    queue = WriteQueue(maxsize=2)
    queue.submit({"kind": "memory"}, kind="intent.memory", priority=PRIORITY["intent.memory"])
    queue.submit({"kind": "summary"}, kind="commit", priority=PRIORITY["commit"])
    queue.submit({"kind": "summary2"}, kind="commit", priority=PRIORITY["commit"])

    kinds = {task.kind for task in queue.drain()}
    assert "intent.memory" in kinds, "记忆写入不得因摘要而被挤掉"
    assert queue.stats.dropped >= 1
    assert "commit" in queue.stats.dropped_by_kind


def test_intent_priority_mapping():
    assert intent_priority("put") < intent_priority("link") < intent_priority("overview_put")


def test_intent_priority_covers_every_intent_op():
    """**AL2 每新增一个写意图，这里就有义务登记优先级**（DES-REV-003 P1-6）。

    修订前用的是 ``PRIORITY.get(op, PRIORITY_SUMMARY)`` 兜底：新加的写意图会
    **静默落到最容易被丢的一档**——队列照常收活、照常干完，只是关键时刻排在
    队尾，没有任何迹象。这类"功能正确、策略失效"的缺陷靠测试很难自然覆盖，
    所以把约束做成穷举比对：缺登记就红。
    """
    from typing import get_args

    from artifact_spirit.core.base import IntentOp
    from artifact_spirit.runtime.writer import INTENT_PRIORITY

    assert set(get_args(IntentOp)) == set(INTENT_PRIORITY), (
        "IntentOp 与 INTENT_PRIORITY 不一致：新增的写意图必须显式登记优先级"
    )


def test_every_task_kind_has_a_priority():
    """C8 的另一半：**任务类型**也不许"默认档兜底"（DES-REV-008 P1-43）。

    ``intent_priority`` 已是穷举表，但 ``WriteQueue.submit`` 里还留着
    ``PRIORITY.get(kind, PRIORITY_MEMORY)``：未登记的类型不报错，静默排到"记忆档"。
    同一个缺陷在 DES-REV-003 里只修了一半，另一半由这条用例钉住。
    """
    from artifact_spirit.runtime.writer import kind_priority

    for kind in PRIORITY:
        assert kind_priority(kind) == PRIORITY[kind]
    assert kind_priority("intent.put") == intent_priority("put"), "intent.<op> 委托，不另长一张表"
    with pytest.raises(ValueError, match="未登记优先级"):
        kind_priority("no.such.kind")
    with pytest.raises(ValueError, match="未登记优先级"):
        WriteQueue(maxsize=4).submit({"n": 1}, kind="no.such.kind")


def test_writer_stop_counts_inflight_tasks():
    """LLD-AL5 §7 F5：收尾的"未完成数"必须含**在途**任务（DES-REV-008 P0-12）。

    修复前 ``Writer.stop`` 只数堆深：任务已被取走、只是还没落库时它报 0——
    恰好在最需要报案的时刻把丢写说成"干净收尾"。``WriteQueue`` 里那条
    "flush 必须连在途那一条一起等"的注释早就写明了，只有 ``stop`` 没跟上。
    """
    from artifact_spirit.runtime.writer import Writer

    queue = WriteQueue(maxsize=8)
    queue.submit({"n": 1}, kind="intent.memory")
    queue.submit({"n": 2}, kind="intent.memory")
    assert queue.get(timeout=0) is not None, "出队一条，制造在途"
    assert queue.depth() == 1, "堆里只剩一条"
    assert queue.pending() == 2, "在途那一条必须计入残留"

    writer = Writer(backend=None, apply=lambda task: None, queue=queue)
    assert writer.stop(drain=False) == 2, "未 drain 时必须报出全部残留任务"


def test_stop_reports_undrained_writes(home):
    """C3 / LLD-AL5 §7 F5："没 drain 干净"必须**报出来**，不能只是返回 0。

    ``Services.stop`` 此前把 ``writer.stop`` 的返回值直接丢掉，"未完成的任务数"
    既没进 sequence 也没进 notes——于是 C3 的验收（"残留任务数为 0"）恒真。
    """
    services = services_for(home)
    for index in range(3):
        services.submit(
            TurnEvent(
                session_id="s1",
                user=f"第 {index} 件",
                assistant="",
                ts="2026-09-14T10:00:00+08:00",
            ),
            kind="turn",
        )
    services.started = True  # 让 stop() 走到 writer.stop（不真起线程，避免竞态）
    sequence = services.stop(drain=False)

    assert "undrained:3" in sequence, sequence
    assert any("未完成" in note for note in services.notes), services.notes


def test_status_surfaces_config_warnings(home):
    """T-AL5-04：配置校验的告警"不阻断，但**要可见**"。

    ``Services.warnings`` 此前没有任何出口——"权重之和不为 1"这类告警写进对象就沉底。
    """
    services = services_for(home)
    services.warnings.append("recall.weights 之和为 1.4，不是 1")
    alerts = status(services)["degradations"]
    assert any("配置告警" in alert for alert in alerts), alerts
    services.stop()


def test_all_writes_happen_in_one_thread(home):
    """D-03 / C10：**所有写操作由同一线程执行**——包括维护线程的产出。

    修订前这条用例有两个洞，恰好都在同一处（DES-REV-008 P0-12 / P1-43）：

    ① 末行写成 ``... or True``，恒真——等于没断言；
    ② 只投过 ``kind="turn"``，而**唯一会跨线程写的那条路径是维护产出**
       （`_maintenance_task` / `_maintenance_cycle` 曾经直接 ``write_now``，
       于是写发生在 maintenance 线程上）。所以用例一路绿到被架空的 D-03。
    """
    services = services_for(home, start_threads=True)
    writer_tid = services.writer.thread_id
    thread_ids: list[int] = []

    original = services.applier.apply

    def spy(intent):
        thread_ids.append(threading.get_ident())
        return original(intent)

    services.applier.apply = spy
    services.writer.apply = lambda task: _apply_with_spy(task, services, spy, original)

    for index in range(5):
        services.submit(
            TurnEvent(session_id="s1", user=f"请记住第 {index} 件事", assistant="", ts="2026-09-14T10:00:00+08:00"),
            kind="turn",
        )
    services.flush(5.0)
    boundary = len(thread_ids)

    # 离线路径：巩固任务由 maintenance 线程消费，但**写必须回到 writer 线程**。
    services.backend.wm_put("s1", "goal", "把记忆系统接进宿主", 0.9)
    assert services.maintenance.schedule(
        {"action": "consolidate", "session_id": "s1"}, dedupe_key="s1"
    ) is True
    deadline = time.time() + 10
    while time.time() < deadline and len(thread_ids) == boundary:
        services.maintenance.trigger_now()
        time.sleep(0.05)
    services.flush(5.0)
    services.stop()

    assert thread_ids, "应当产生写操作"
    assert len(thread_ids) > boundary, "维护任务应当产出写——否则本用例没覆盖到离线路径"
    assert set(thread_ids) == {writer_tid}, f"写操作出现在多个线程：{set(thread_ids)}"
    assert writer_tid is not None


def _apply_with_spy(task, services, spy, original):
    """让 writer 的 turn 任务也走被监视的 applier。"""
    from artifact_spirit.runtime.lifecycle import _apply_task

    if task.kind == "turn":
        for intent in services.core.ingest_turn(task.payload):
            spy(intent)
        return
    _apply_task(task, core=services.core, applier=services.applier, services_holder=[services])


def test_write_now_goes_through_the_writer_thread(home):
    """`write_now` 有 writer 线程时**由 writer 落库**（D-03）。

    DES-REV-008 P0-15：宿主工具路径（`tools/handlers.py` 的六个工具）都调 `write_now`，
    而它原先**无条件直接应用**——写发生在宿主主线程上。`SQLiteBackend._write_connection()`
    按线程 id 分配写连接，换线程写会先关掉对方那条再新建，于是"单写者"在**最热的那条
    路径**上作废，每次工具调用还额外付一次建连接的代价。

    断言口径是**写发生在哪个线程**，而不是"结果对不对"——结果一直是对的，
    这正是它能活这么久的原因。
    """
    services = services_for(home, start_threads=True)
    seen: list[int] = []
    original = services.applier.apply

    def spy(intent):
        seen.append(threading.get_ident())
        return original(intent)

    services.applier.apply = spy
    services.write_now(
        services.core.ingest_turn(
            TurnEvent(
                session_id="s1",
                user="请记住：我偏好深色主题",
                assistant="",
                ts="2026-09-14T10:00:00+08:00",
            )
        )
    )

    assert seen, "写入没有被执行"
    assert set(seen) == {services.writer.thread_id}, (
        f"write_now 的写发生在 {sorted(set(seen))}，"
        f"而不是 writer 线程 {services.writer.thread_id}"
    )
    services.stop()


def test_write_now_applies_directly_without_a_writer_thread(home):
    """没有 writer 线程时（CLI / 确定性测试）**直接应用**——这是保留该入口的理由。"""
    services = services_for(home)  # start_threads=False
    services.write_now(
        services.core.ingest_turn(
            TurnEvent(
                session_id="s1",
                user="请记住：我偏好深色主题",
                assistant="",
                ts="2026-09-14T10:00:00+08:00",
            )
        )
    )
    assert services.writer.queue.depth() == 0, "没有 writer 时不该往队列里放东西"


def test_concurrent_submits_do_not_raise_busy(home):
    """真实 SQLite 下并发提交 → **无 SQLITE_BUSY**（串行化的收益）。"""
    services = services_for(home, start_threads=True)

    def burst(n: int) -> None:
        for index in range(n):
            services.submit(
                TurnEvent(session_id="s1", user=f"记住并发 {n}-{index}", assistant="", ts="2026-09-14T10:00:00+08:00"),
                kind="turn",
            )

    threads = [threading.Thread(target=burst, args=(5,)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    services.flush(10.0)
    stats = services.writer.queue.stats
    services.stop()

    assert stats.failed == 0, f"写入失败：{services.writer.errors}"
    assert not any("busy" in str(e).lower() for _, e in (services.writer.errors or []))


def test_writer_survives_task_exception(home):
    """C4：单任务抛异常 → 线程存活且后续任务继续。"""
    services = services_for(home, start_threads=True)
    boom = {"armed": True}

    original = services.applier.apply

    def flaky(intent):
        if boom["armed"]:
            boom["armed"] = False
            raise RuntimeError("注入的任务级失败")
        return original(intent)

    services.applier.apply = flaky
    services.writer.apply = lambda task: _apply_with_spy(task, services, flaky, original)

    services.submit(
        TurnEvent(session_id="s1", user="请记住第一件事", assistant="", ts="2026-09-14T10:00:00+08:00"),
        kind="turn",
    )
    services.flush(3.0)
    services.submit(
        TurnEvent(session_id="s1", user="请记住第二件事", assistant="", ts="2026-09-14T10:01:00+08:00"),
        kind="turn",
    )
    services.flush(3.0)

    assert services.writer.running, "线程必须存活"
    assert services.writer.queue.stats.failed >= 1
    services.stop()


def test_maintenance_merges_duplicate_tasks(home):
    services = services_for(home, start_threads=True)
    assert services.maintenance.schedule({"action": "consolidate"}, dedupe_key="s1") is True
    assert services.maintenance.schedule({"action": "consolidate"}, dedupe_key="s1") is False
    services.stop()


def test_maintenance_runs_offline_cycle(home):
    """概览重算确实在 maintenance 线程执行（thread id 断言）。"""
    services = services_for(home, start_threads=True)
    seen: list[int] = []
    original = services.core.refresh_overviews

    def spy(**kwargs):
        seen.append(threading.get_ident())
        return original(**kwargs)

    services.core.refresh_overviews = spy
    services.maintenance.interval_seconds = 0.05
    maintenance_tid = services.maintenance.thread_id
    services.maintenance.trigger_now()
    time.sleep(0.2)
    services.stop()

    assert seen, "周期任务应当被执行"
    # 修订前这里是 ``all(tid == services.maintenance.thread_id ...) or len(set(seen)) == 1``：
    # `thread_id` 用的是 `Thread.ident`，`stop()` 之后变成 None，于是所有比较都是 False，
    # 用例靠 `or` 兜住——真正的"跑在维护线程上"其实没被断言（DES-REV-008 P1-44）。
    assert set(seen) == {maintenance_tid}, f"周期任务跑在了别的线程：{set(seen)}"


def test_scheduled_consolidation_is_actually_executed(home):
    """调度出去的**巩固任务必须真的被执行**（T-AL1-10#1 的 AL5 侧）。

    这是 2026-09-16 补 P1-5 时抓到的真实缺陷：`maintenance.schedule(kind="commit")`
    把任务放进 maintenance 队列，而队列的处理函数只认 `refresh_overviews` /
    `decay` / `optimize` —— **巩固任务被静默丢弃**。
    后果是 `on_session_end` / `on_pre_compress` 一律"成功返回、什么都没发生"，
    情景记忆永远是空的；而这三处调用点都写在"投递即完成"的路径上，
    不阻塞、不报错，所以没有任何迹象。
    """
    services = services_for(home, start_threads=True)
    services.backend.wm_put("s-1", "goal", "把记忆系统接进宿主", 0.9)
    assert services.maintenance.schedule(
        {"action": "consolidate", "session_id": "s-1"}, dedupe_key="s-1"
    ) is True

    counts: dict = {}
    deadline = time.time() + 10
    while time.time() < deadline:
        services.maintenance.trigger_now()
        counts = services.backend.count_by_layer()
        if counts.get("episodic", {}).get("active", 0) >= 1:
            break
        time.sleep(0.05)
    services.stop()

    assert counts.get("episodic", {}).get("active", 0) >= 1, (
        "巩固任务被调度后必须产生情景记忆；"
        f"实际层计数：{counts}"
    )


# =========================================================================== #
# AL5 · M1 可观测性
# =========================================================================== #


def test_status_has_five_sections(home):
    services = services_for(home)
    report = status(services)
    assert set(report) >= {"identity", "layers", "health", "models", "degradations"}
    services.stop()


def test_status_shows_model_chain_and_source(home):
    services = services_for(home)
    chain = status(services)["models"]["llm_chain"]
    assert any(segment.get("active") for segment in chain)
    assert chain[0]["source"] == "spirit"
    services.stop()


def test_status_reports_bm25_degradation(home):
    """造 embedding 不可用的场景 → 必须显示降级告警。"""
    services = services_for(home)
    # 让 resolver 认定 embedding 不可用（模拟「配置缺失 / 密钥缺失」）
    services.resolver._embedding_route = None
    services.resolver._embedding_chain = [
        {"source": "spirit", "usable": False, "reason": "测试注入：密钥缺失"}
    ]
    services.core.embedding = None
    alerts = status(services)["degradations"]
    assert any("BM25" in alert for alert in alerts), alerts
    services.stop()


def test_status_is_idempotent(home):
    services = services_for(home)
    assert status(services) == status(services)
    services.stop()


def test_layers_counts_match_backend(home):
    services = services_for(home)
    services.write_now(
        services.core.ingest_turn(
            TurnEvent(session_id="s1", user="请记住：我偏好深色主题", assistant="", ts="2026-09-14T10:00:00+08:00")
        )
    )
    report = layers(services)
    actual = services.backend.count_by_layer()
    assert report["layers"]["semantic"]["active"] == actual.get("semantic", {}).get("active", 0)
    services.stop()


def test_reflect_explains_forgetting_and_recall(home):
    services = services_for(home)
    report = reflect(services)
    assert "衰减只影响排序" in report["why_forget"]["explain"]
    assert set(report["why_recall"]["weights"]) >= {"semantic", "importance"}
    services.stop()


def test_audit_view_filters(home):
    """T-AL5-07：审计视图按 ``since`` / ``limit`` 过滤、且新的在前。

    修订前这里只断言了 ``isinstance(entries, list)``——**恒真**，而且服务是空的，
    "过滤生效"这条验收从未被验证过（DES-REV-008 P1-46：弱断言 = 已覆盖的错觉）。
    """
    services = services_for(home)
    services.write_now(
        services.core.ingest_turn(
            TurnEvent(session_id="s1", user="请记住：我偏好深色主题", assistant="", ts="2026-09-14T10:00:00+08:00")
        )
    )
    entries = audit_view(services, limit=100)
    assert entries, "写入必须留下审计事件——否则下面的过滤断言全是空转"
    assert [entry["ts"] for entry in entries] == sorted(
        (entry["ts"] for entry in entries), reverse=True
    ), "审计视图必须新的在前"

    assert len(audit_view(services, limit=1)) == 1, "limit 必须真的截断"
    assert audit_view(services, limit=1)[0]["audit_id"] == entries[0]["audit_id"]

    # 编号必须是**真实 audit_id**（INV-12 的可溯性底座）：`aspirit restore <编号>`
    # 只认真实 id，而位置编号在 `--since` 下会错位——用户照着错位的编号去恢复，
    # 恢复的就是另一条记忆（DES-REV-008 P1-47）。
    real_ids = {event.audit_id for event in services.backend.audit_replay()}
    assert real_ids, "审计表应当有事件"
    assert {entry["audit_id"] for entry in entries} <= real_ids

    newest = entries[0]["ts"]
    windowed = audit_view(services, since=newest)
    assert windowed, "since 是闭区间下界"
    assert all(entry["ts"] >= newest for entry in windowed)
    assert len(windowed) <= len(entries)
    # **一份账本只许有一份投影**：`audit_view()` 必须与 `AuditView.entries()` 同源，
    # 否则两套编号语义迟早有一处被接上 CLI 而没带注释。
    assert windowed == AuditView(services.backend).entries(since=newest, limit=50)

    view = AuditView(services.backend)
    assert view.restorable() == []
    services.stop()


def test_doctor_reports_each_check_independently(home):
    services = services_for(home)
    report = doctor(services)
    assert report["checks"], "体检项不应为空"
    assert all({"check", "ok", "detail"} <= set(item) for item in report["checks"])
    assert "自检" in doctor_text(report)
    services.stop()


def test_status_text_is_readable(home):
    services = services_for(home)
    text = status_text(status(services))
    for fragment in ("层计数", "健康度", "生效模型链", "降级告警"):
        assert fragment in text
    services.stop()


# =========================================================================== #
# AL5 · M2 合规门禁
# =========================================================================== #


def test_architecture_rules_all_green():
    assert check_architecture(PACKAGE_ROOT) == []


def test_dependency_scan_is_clean():
    assert scan_dependencies(PACKAGE_ROOT) == []


def test_only_runtime_creates_threads():
    """R8 的落地断言——由 check_architecture 的符号扫描覆盖。"""
    violations = [v for v in check_architecture(PACKAGE_ROOT) if v["rule"] == "R8"]
    assert violations == []


def test_write_queue_does_no_business_reasoning():
    """C8：队列不知道"这条记忆重不重要"——优先级只由任务类型决定。"""
    text = (PACKAGE_ROOT / "runtime" / "writer.py").read_text(encoding="utf-8")
    assert "score_importance" not in text
    assert "SalienceScorer" not in text
