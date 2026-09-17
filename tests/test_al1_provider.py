"""AL1 适配层验收（宿主契约 / 工具 / 热路径 / 钩子行为）。

本文件由 `test_runtime_adapter.py` 拆出（设计评审 P1-6）。分层归属：
`test_host_contract.py` = 宿主 ABC 面、本文件 = 器灵侧的适配实现、
`test_al5_runtime.py` = 辅助机制、`test_al1_cli.py` = CLI、`test_value.py` = 价值判决。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import call_tool, services_for

from artifact_spirit.compliance import PACKAGE_ROOT, check_architecture
from artifact_spirit.config import ConfigError, config_path
from artifact_spirit.model.base import ProviderUnavailableError, SchemaViolationError
from artifact_spirit.provider import ArtifactSpiritProvider
from artifact_spirit.store.base import StorageFatalError
from artifact_spirit.tools import TOOL_NAMES, ToolHandlers, schema_names

# =========================================================================== #
# AL1 · M0 契约
# =========================================================================== #


def test_provider_name():
    assert ArtifactSpiritProvider.name == "artifact-spirit"


def test_package_init_has_zero_business_logic():
    """R6：``__init__.py`` 只做 register。"""
    violations = [v for v in check_architecture(PACKAGE_ROOT) if v["rule"] == "R6"]
    assert violations == []
    text = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "def register(" in text


def test_package_layout(tmp_path):
    for name in ("provider.py", "cli.py", "config_schema.py", "plugin.yaml"):
        assert (PACKAGE_ROOT / name).exists(), f"缺少 {name}"


def test_tool_schemas_and_handlers_are_one_to_one(home):
    """**一一对应**：schema 有、handler 没有（悬空工具）是这类插件最常见的故障。"""
    services = services_for(home)
    provider = ArtifactSpiritProvider()
    provider._services = services
    provider._handlers = ToolHandlers(services)

    schema_names_set = {schema["name"] for schema in provider.get_tool_schemas()}
    handler_names = set(provider._handlers.names())
    assert schema_names_set == handler_names == set(TOOL_NAMES)
    services.stop()


def test_tool_names_are_consistent_across_three_places():
    """工具名在 schema / handler 两处必须一致（C8）。

    新增一个工具要同时动 schema、handler 方法名、文档三处；漏掉 handler 方法名的
    表现是 `spirit_xxx` 被调用时回"未知工具"，而不是启动时报错——**静默悬空**。
    """
    assert set(schema_names()) == set(TOOL_NAMES)
    handler_names = {
        name[len("_tool_"):] for name in dir(ToolHandlers) if name.startswith("_tool_")
    }
    assert handler_names == set(TOOL_NAMES)


def test_config_schema_is_a_flat_list_for_the_host():
    """**必须是 list**——宿主 `_normalize_memory_provider_schema` 写着
    `if isinstance(raw, list)`；返回 dict 的后果是配置面板一个字段都不显示。"""
    from artifact_spirit.config_schema import CONFIG_FIELDS

    assert isinstance(CONFIG_FIELDS, list) and CONFIG_FIELDS
    for field in CONFIG_FIELDS:
        assert isinstance(field, dict)
        assert field.get("key") and field.get("description")
        assert field.get("type") in {"text", "integer", "number", "boolean"}


def test_config_schema_keys_are_save_whitelisted():
    """schema 的 key 必须**全部**能被 `save` 写进去。

    否则宿主面板改得动的字段，落盘时会被白名单静默丢弃——"界面说保存成功、
    实际没生效"是最难查的一类问题。
    """
    from artifact_spirit.config import SAVE_WHITELIST
    from artifact_spirit.config_schema import CONFIG_FIELDS

    not_writable = [
        f["key"] for f in CONFIG_FIELDS if f["key"] not in SAVE_WHITELIST and f["key"] != "spirit.id"
    ]
    assert not not_writable, f"面板字段无法落盘：{not_writable}"


def test_config_schema_has_no_secret_field():
    """器灵**没有**密钥字段——密钥只走环境变量，面板里只填变量名。"""
    from artifact_spirit.config_schema import CONFIG_FIELDS

    assert not any(f.get("secret") for f in CONFIG_FIELDS)
    text = json.dumps(CONFIG_FIELDS, ensure_ascii=False)
    assert "api_key_env" in text


# =========================================================================== #
# AL1 · M0 is_available 与 initialize
# =========================================================================== #


def test_is_available_makes_no_network_call(home, monkeypatch):
    """INV-5：`is_available` 全程**无网络调用**（socket 打桩断言）。"""
    import socket

    def explode(*args, **kwargs):
        raise AssertionError("is_available 不得发起任何网络调用（INV-5）")

    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "create_connection", explode)
    monkeypatch.setattr(socket, "getaddrinfo", explode)

    provider = ArtifactSpiritProvider()
    assert provider.is_available(hermes_home=home) is True


def test_is_available_returns_false_for_missing_home():
    assert ArtifactSpiritProvider().is_available(hermes_home="") is False


def test_is_available_returns_false_for_broken_config(tmp_path):
    config_path(tmp_path).write_text("[broken", encoding="utf-8")
    assert ArtifactSpiritProvider().is_available(hermes_home=str(tmp_path)) is False


def test_is_available_is_fast(home):
    provider = ArtifactSpiritProvider()
    provider.is_available(hermes_home=home)
    assert provider.last_check_ms < 500, f"耗时 {provider.last_check_ms:.1f}ms 过慢"


def test_initialize_after_initialize_is_usable(home):
    services = services_for(home)
    assert services.core is not None
    assert services.backend.meta_get("schema_version")
    services.stop()


def test_initialize_raises_config_error_on_bad_config(tmp_path):
    config_path(tmp_path).write_text(
        '[models.llm]\napi_key = "sk-should-be-env"\n', encoding="utf-8"
    )
    provider = ArtifactSpiritProvider()
    with pytest.raises(ConfigError):
        provider.initialize(hermes_home=str(tmp_path), env={})


def test_initialize_does_not_contain_assembly_details():
    """AL1 只是**调用**装配，装配逻辑在 AL5（C10）。"""
    text = (PACKAGE_ROOT / "provider.py").read_text(encoding="utf-8")
    assert "SQLiteBackend(" not in text
    assert "ArtifactSpiritCore(" not in text
    assert "start(" in text


# =========================================================================== #
# AL1 · M1 热路径护栏
# =========================================================================== #


def test_on_session_end_returns_immediately(home, client):
    provider = client(home)
    started = time.perf_counter()
    provider.on_session_end([], session_id="s1")
    elapsed = (time.perf_counter() - started) * 1000
    assert elapsed < 50, f"on_session_end 耗时 {elapsed:.1f}ms，应当立即返回"
    provider.shutdown()


def test_prefetch_returns_non_empty_when_core_raises(home, client):
    """P6 的核心承诺：核心层抛异常时**仍然返回非空内容**。"""
    provider = client(home)

    def boom(*args, **kwargs):
        raise RuntimeError("核心层炸了")

    provider.services.core.recall = boom
    text = provider.prefetch("任何查询", session_id="s1")
    assert text and text.strip(), "prefetch 必须返回非空内容"
    provider.shutdown()


def test_prefetch_returns_fallback_on_timeout(home, client):
    provider = client(home)
    provider.services.config.worker["prefetch_timeout_ms"] = 1

    def slow(*args, **kwargs):
        time.sleep(0.3)
        return []

    provider.services.core.recall = slow
    text = provider.prefetch("会超时的查询", session_id="s1")
    assert text and text.strip()
    assert provider.services.timeout_runner.timeouts >= 1
    provider.shutdown()


def test_prefetch_works_without_embedding(home, client):
    """向量路不可用时仍能返回（仅 BM25）——F2 降级。"""
    provider = client(home)
    provider.services.core.embedding = None
    assert provider.prefetch("任何查询", session_id="s1")
    provider.shutdown()


def test_prefetch_never_raises(home, client):
    provider = client(home)
    provider.services.core.recall = lambda q: (_ for _ in ()).throw(ValueError("x"))
    assert isinstance(provider.prefetch("q"), str)
    provider.shutdown()


def test_prefetch_timeout_discards_late_thread_side_effects(home, client):
    """P1-4：护栏**放弃等待**不等于 action 没执行——迟到线程不得留下状态。

    护栏只是"不等"，那条被遗弃的线程随后仍会把闭包跑完。所以"记下这一轮召回了
    什么"必须由主线程在未超时分支里做；否则一条**本轮根本没注入**的记忆会被算作
    "上一轮注入过"，进而被记访问、建共激活边——静默的活性污染。
    """
    provider = client(home)
    hit = SimpleNamespace(
        record=SimpleNamespace(id="late-1", layer="semantic", abstract="迟到的记忆", content="x")
    )
    provider.services.core.recall = lambda q: [hit]

    def abandoned(action, timeout, fallback, **kwargs):
        """假护栏：**先让 action 真的跑完**（模拟被遗弃的线程），再报超时。"""
        action()
        return fallback, True

    provider.services.run_with_timeout = abandoned  # type: ignore[method-assign]
    text = provider.prefetch("查询", session_id="s1")

    assert text and text.strip(), "超时仍须返回保底内容"
    assert provider._last_recall == (), "迟到线程把召回 id 留给了下一轮"
    assert provider.recall_status() is None, (
        "宿主契约要求 recall_status 只反映最近一次 prefetch——而这一次什么都没注入"
    )
    provider.shutdown()


def test_prefetch_timeout_does_not_attribute_recall_to_next_turn(home, client):
    """P1-4 的另一半：超时那一轮的召回，**不得**被归因给下一轮。"""
    provider = client(home)
    hit = SimpleNamespace(
        record=SimpleNamespace(id="late-2", layer="semantic", abstract="迟到的记忆", content="x")
    )
    provider.services.core.recall = lambda q: [hit]

    def abandoned(action, timeout, fallback, **kwargs):
        action()
        return fallback, True

    provider.services.run_with_timeout = abandoned  # type: ignore[method-assign]
    provider.prefetch("查询", session_id="s1")

    events = []
    provider.services.submit = lambda event, *args, **kwargs: events.append(event)  # type: ignore[method-assign]
    provider.sync_turn("下一轮用户输入", "下一轮助手回复", session_id="s1")

    assert events, "sync_turn 必须投递 TurnEvent"
    assert events[0].recalled == (), (
        f"超时那一轮的召回被错误归因给下一轮：{events[0].recalled}"
    )
    provider.shutdown()


# =========================================================================== #
# AL1 · M1 工具
# =========================================================================== #


def test_invalid_args_return_structured_error(home, client):
    provider = client(home)
    result = call_tool(provider, "spirit_recall", {})
    # 失败形状与宿主 `tool_error` 同构：{"error": ...}
    assert "error" in result and "ok" not in result
    assert result.get("field") == "query"
    provider.shutdown()


def test_unknown_tool_returns_clear_error(home, client):
    provider = client(home)
    result = call_tool(provider, "spirit_nonexistent", {})
    assert "error" in result
    assert "未知工具" in result["error"]
    provider.shutdown()


# --------------------------------------------------------------------------- #
# AL1 · M8 错误翻译（T-AL1-09 验收 1）
#
# 这三个异常族的 docstring 都白纸黑字写着"由 AL1 转换为工具级错误"——契约被写了
# 三遍，实现却一次都没兑现：用户看到的是 `内部错误：StorageFatalError: database is
# locked`，而不是"存储不可用，请稍后重试"。**分层翻译的意图全丢了**（P1-2）。
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("exc", "must_contain"),
    [
        (StorageFatalError("database is locked"), "存储不可用"),
        (SchemaViolationError("缺少 layer 字段"), "未通过校验"),
        (ProviderUnavailableError("连接超时"), "模型不可用"),
        (ConfigError("embedding.dim 必须为正整数"), "配置有误"),
    ],
)
def test_internal_faults_are_translated_to_actionable_tools(home, client, exc, must_contain):
    """T-AL1-09 验收 1：**每一个已登记的异常族都有对应翻译**。

    这件事必须**从工具出口**验证：翻译写在 `translate_fault` 里，但异常要先被
    `handlers.handle()` 的兜底接住——直接在单测里调 `translate_fault` 的话，
    "兜底分支到底有没有用它"没人管得着（评审时这里就绕过一次）。
    """
    provider = client(home)

    def boom(*args, **kwargs):
        raise exc

    provider.services.core.recall = boom
    result = call_tool(provider, "spirit_recall", {"query": "任意查询"})

    assert "error" in result, result
    assert must_contain in result["error"], f"未翻译：{result['error']}"
    assert "内部错误" not in result["error"], "已登记的异常族不得落到兜底文案"
    provider.shutdown()


def test_unregistered_fault_falls_back_without_crashing_host(home, client):
    """T-AL1-09 验收 2/3：**未登记**的异常仍不抛、不崩，只是退回兜底文案。"""
    provider = client(home)
    provider.services.core.recall = lambda q: (_ for _ in ()).throw(RuntimeError("意料之外"))

    result = call_tool(provider, "spirit_recall", {"query": "任意查询"})
    assert "error" in result
    assert "RuntimeError" in result["error"], result["error"]
    provider.shutdown()


def test_storage_failure_returns_actionable_error(home, client):
    """T-AL1-09 验收 3：**存储不可用**（磁盘满 / 库被锁）时的现场响应。

    翻译不只是"说人话"：它要给出下一步动作（重试 / 跑体检）。原始异常文本会
    附在 detail 里——保留它是对的，没有它就查不出"到底是哪种不可用"。
    """
    provider = client(home)

    def boom(*args, **kwargs):
        raise StorageFatalError("database is locked")

    provider.services.core.recall = boom
    result = call_tool(provider, "spirit_recall", {"query": "任何查询"})

    assert "存储不可用" in result["error"]
    assert "aspirit doctor" in result["error"], "翻译必须指向下一步动作"
    assert "database is locked" in result["error"], "原始细节要保留，否则无从排查"
    provider.shutdown()


# =========================================================================== #
# AL1 · M1 工具（其余契约）
# =========================================================================== #


def test_forget_without_confirm_writes_nothing(home, client):
    """`spirit_forget` 不带 `confirm` 时**不得真的删除**（F6：删除要显式）。"""
    provider = client(home)
    mem = call_tool(provider, "spirit_remember", {"content": "待删内容"})["data"]["id"]

    preview = call_tool(provider, "spirit_forget", {"mem_id": mem, "reason": "测试"})
    assert preview["ok"] is True
    assert provider.services.backend.get(mem) is not None, "预览不得落库"

    done = call_tool(
        provider, "spirit_forget", {"mem_id": mem, "reason": "测试", "confirm": True}
    )
    assert done["ok"] is True
    assert provider.services.backend.get(mem) is None
    provider.shutdown()


def test_forget_requires_reason(home, client):
    """删除必须给理由——理由会进审计账本，是"可审"的最小前提。"""
    provider = client(home)
    mem = call_tool(provider, "spirit_remember", {"content": "待删内容"})["data"]["id"]
    result = call_tool(provider, "spirit_forget", {"mem_id": mem})
    assert result["field"] == "reason"
    assert provider.services.backend.get(mem) is not None
    provider.shutdown()


def test_expand_l1_does_not_leak_l2_text(home, client):
    """V1 的机制面：**L1 不得顺手把 L2 全文带出来**（逐级展开是省钱的手段）。"""
    provider = client(home)
    mem = call_tool(
        provider,
        "spirit_remember",
        {"content": "用户偏好深色主题，因为夜里写代码刺眼", "abstract": "偏好深色主题"},
    )["data"]["id"]

    l1 = call_tool(provider, "spirit_expand", {"ref": mem, "level": "L1"})
    assert l1["data"]["level"] == "L1"
    assert "刺眼" not in l1["text"], f"L1 泄漏了 L2 正文：{l1['text']}"

    l2 = call_tool(provider, "spirit_expand", {"ref": mem, "level": "L2"})
    assert "刺眼" in l2["text"]
    provider.shutdown()


def test_asof_tool_distinguishes_no_version_from_current_value(home, client):
    """`spirit_asof`（T-AL1-14）：**"那时没有有效版本"必须是结构化结果**。

    这与 V1 的"不越级"是同一类要求：工具返回的不只是内容，还有**为什么是这个内容**。
    ``found: False`` 与"取到了但内容为空"必须分得开——否则用户判断不了
    是"那时确实没记"，还是"记了但没取到"。
    """
    provider = client(home)
    mem = call_tool(
        provider,
        "spirit_remember",
        {"content": "用户住在上海，魔都潮湿", "abstract": "住在上海"},
    )["data"]["id"]

    # ① 早于首次写入 → 无有效版本（**结构化**，不是错误、也不是空串）
    early = call_tool(
        provider, "spirit_asof", {"ref": mem, "ts": "2020-01-01T00:00:00+08:00"}
    )
    assert early["ok"] is True, "「那时没有」是个答案，不是错误"
    assert early["data"]["found"] is False
    assert "没有有效版本" in early["text"]

    # ② 写入之后 → 取得到，且带上时态字段（让"为什么是这条"可查）
    late = call_tool(
        provider, "spirit_asof", {"ref": mem, "ts": "2030-01-01T00:00:00+08:00"}
    )
    assert late["data"]["found"] is True
    assert late["data"]["mem_id"] == mem
    assert late["data"]["valid_from"], "应当回带 valid_from，否则无法核对该版本的有效区间"
    assert "上海" in late["text"]

    provider.shutdown()


def test_review_trace_correct_loop_is_visible_in_trace(home, client):
    """V2 闭环：`review → trace → correct` 后，**`trace` 里看得到 actor 与 reason**。

    "能改"不是重点——重点是**改完之后查得到是谁、因为什么改的**（INV-12）。
    没有这一步，"可审查"就只是一个好看的列表。
    """
    provider = client(home)
    mem = call_tool(
        provider, "spirit_remember", {"content": "用户住在上海", "abstract": "住在上海"}
    )["data"]["id"]

    reviewed = call_tool(provider, "spirit_review", {})
    assert mem in reviewed["text"], f"审查视图里应当看得到它：{reviewed['text']}"

    corrected = call_tool(
        provider,
        "spirit_correct",
        {"mem_id": mem, "patch": {"content": "用户住在北京"}, "reason": "用户更正"},
    )
    assert corrected["ok"] is True

    traced = call_tool(provider, "spirit_trace", {"mem_id": mem})
    assert "user" in traced["text"], f"trace 里应当看得到 actor：{traced['text']}"
    assert "用户更正" in traced["text"], f"trace 里应当看得到 reason：{traced['text']}"
    provider.shutdown()


def test_core_memory_tool_promotes_and_demotes(home, client):
    """工具面闭环核心记忆干预（T-AL1-15）：升格 → 降格，两步都报成功。"""
    provider = client(home)
    mem = call_tool(
        provider, "spirit_remember", {"content": "我是星空", "abstract": "我是星空"}
    )["data"]["id"]

    promoted = call_tool(
        provider,
        "spirit_core",
        {"mem_id": mem, "action": "promote", "reason": "用户明确要求", "as_type": "identity"},
    )
    assert promoted["ok"] is True, promoted
    assert promoted["data"]["changed"] == 1

    demoted = call_tool(
        provider,
        "spirit_core",
        {"mem_id": mem, "action": "demote", "reason": "用户改主意了"},
    )
    assert demoted["ok"] is True, demoted

    # 两步都留痕，且 actor 是 user（工具面发起 = 人的决定）
    traced = call_tool(provider, "spirit_trace", {"mem_id": mem})
    assert "promote" in traced["text"] and "demote" in traced["text"], traced["text"]
    provider.shutdown()


def test_core_memory_tool_without_reason_is_a_structured_error(home, client):
    """缺 `reason` → **结构化错误**，不是异常穿透到宿主（F10）。

    错误形态与成功形态不同（宿主契约是 `tool_error on failure`）——
    所以这里断言的是"有一条能读的 error 指出是 reason 的问题"，
    而不是照搬成功路径的字段名。
    """
    provider = client(home)
    result = call_tool(provider, "spirit_core", {"mem_id": "whatever", "action": "promote"})
    assert "error" in result, f"应当返回结构化错误：{result}"
    assert "reason" in str(result).casefold(), f"应当指出是哪个参数的问题：{result}"
    provider.shutdown()


def test_recall_tool_reports_reasons(home, client):
    """T-AL1-07 验收 5：召回要**说得出为什么**（六分量）。

    可解释性不是装饰：调试"它怎么又忘了"时，只有这一行能区分
    "根本没搜到"与"搜到了但被分数压下去"。
    """
    provider = client(home)
    call_tool(provider, "spirit_remember", {"content": "用户偏好深色主题"})

    items = call_tool(provider, "spirit_recall", {"query": "深色主题"})["data"]["items"]
    assert items, "记下来的东西找不回来"

    raw = items[0]["raw"]
    assert len(raw) >= 6, f"召回的原始分量不足六项：{raw}"
    assert all(isinstance(value, (int, float)) for value in raw.values())
    assert items[0]["score"] > 0
    provider.shutdown()


def test_prefetch_respects_token_budget(home, client):
    """T-AL1-05 验收 4：注入体必须**在预算内**。

    它是每轮都要塞进上下文的文本；没有上界意味着记忆越多、账单越大，
    还越可能把真正重要的指令挤出窗口。
    """
    provider = client(home)
    for index in range(20):
        call_tool(
            provider,
            "spirit_remember",
            {"content": f"第 {index} 条：用户偏好深色主题，夜里写代码刺眼"},
        )

    provider.services.config.recall["token_budget"] = 120
    small = provider.prefetch("深色主题", session_id="s1")
    provider.services.config.recall["token_budget"] = 4000
    large = provider.prefetch("深色主题", session_id="s1")

    assert small and large
    assert len(small) < len(large), "token_budget 被忽略了"
    provider.shutdown()


def test_correct_requires_a_real_patch(home, client):
    """空 patch / 非 dict 都要报字段级错误，而不是静默"改成功了"。"""
    provider = client(home)
    mem = call_tool(provider, "spirit_remember", {"content": "用户偏好深色主题"})["data"]["id"]

    assert call_tool(provider, "spirit_correct", {"mem_id": mem})["field"] == "patch"
    assert call_tool(provider, "spirit_correct", {"mem_id": mem, "patch": {}})["field"] == "patch"
    provider.shutdown()


def test_shutdown_is_idempotent(home, client):
    """宿主可能在任何时机收尾；收两次不能抛。"""
    provider = client(home)
    provider.shutdown()
    provider.shutdown()


# =========================================================================== #
# AL1 · M2 系统提示块与状态面（T-AL1-12）
# =========================================================================== #


def _seed_identity(provider, text: str) -> None:
    """通过宿主钩子写核心记忆——这是产品里**唯一**的核心记忆来源。"""
    provider.on_memory_write("add", "USER.md", text)
    assert provider.services.flush(10.0), "核心记忆的镜像写入应完成"


def test_system_prompt_block_contains_core_memory(home, client):
    """T-AL1-12 验收 1：核心记忆要**真的**出现在系统提示块里。

    "我是谁"就是这个块；它空着的时候，器灵的"人格"全靠模型自己猜。
    """
    provider = client(home, start_threads=True)
    _seed_identity(provider, "用户叫老张，做后端，偏好深色主题")

    block = provider.system_prompt_block(token_budget=4000)
    assert "老张" in block
    provider.shutdown()


def test_system_prompt_block_truncates_to_budget(home, client):
    """T-AL1-12 验收 2（P2-6 重写）：**token_budget 必须真的起作用**。

    修订前的写法是空库 + `len(block) < 1200`：空库时块长趋近 0，
    **把 token_budget 整个忽略掉也能通过**——"已验证"的错觉。
    现在先灌一条远超预算的核心记忆，再断言"小预算真的更短"。
    """
    provider = client(home, start_threads=True)
    _seed_identity(provider, "用户偏好深色主题。" + "补充细节" * 300)

    small = provider.system_prompt_block(token_budget=60)
    large = provider.system_prompt_block(token_budget=4000)

    assert small, "有核心记忆时不得返回空块"
    assert len(small) < len(large), "小预算必须真的更短——否则 token_budget 被忽略了"
    assert "补充细节" in large
    provider.shutdown()


def test_system_prompt_block_does_not_scan_long_term_memory(home, client):
    """T-AL1-12 验收 3：耗时**不随长期记忆总量增长**（宿主每轮都要拿它）。"""
    provider = client(home, start_threads=True)
    for index in range(200):
        call_tool(provider, "spirit_remember", {"content": f"第 {index} 条无关记忆"})
    assert provider.services.flush(20.0)

    started = time.perf_counter()
    provider.system_prompt_block(token_budget=400)
    elapsed = (time.perf_counter() - started) * 1000
    assert elapsed < 100, f"耗时 {elapsed:.1f}ms——核心记忆摘要不应随总量增长"
    provider.shutdown()


def test_recall_status_reflects_last_prefetch(home, client):
    """宿主要用它渲染"本轮注入了几条"——必须与最近一次 prefetch 一致。"""
    provider = client(home)
    assert provider.recall_status() is None, "这一轮什么都没注入 → None（宿主走通用渲染）"

    call_tool(provider, "spirit_remember", {"content": "用户偏好深色主题"})
    provider.prefetch("深色主题", session_id="s1")

    report = provider.recall_status()
    assert report is not None
    assert report.provider_label == "器灵"
    assert report.count == len(provider._last_recall) >= 1
    provider.shutdown()


def test_identity_signature_survives_restart_and_changes_with_the_store(home, client):
    """宿主拿它判断"还是不是同一个记忆主体"——所以它必须**跨重启稳定**。"""
    assert ArtifactSpiritProvider().identity_signature()["initialized"] is False

    first = client(home)
    signature = first.identity_signature()
    first.shutdown()

    assert signature["provider"] == "artifact-spirit"
    assert signature["spirit_id"], "空的身份指纹等于没有指纹"
    assert signature["schema_version"], "库结构版本要能暴露给宿主排障"
    assert Path(signature["db_path"]).parent == Path(home)

    second = client(home)
    assert second.identity_signature() == signature, "同一份库的指纹必须稳定"
    second.shutdown()


def test_backup_paths_cover_wal_and_config(home, client):
    """WAL 模式下 `-wal`/`-shm` **必须一起备份**：只拷主文件会丢掉最近的写入。"""
    assert ArtifactSpiritProvider().backup_paths() == [], "未初始化时没有可备份的东西"

    provider = client(home)
    paths = [Path(p) for p in provider.backup_paths()]
    db = Path(provider.services.backend.path)

    assert db in paths
    assert Path(config_path(home)) in paths, "配置丢了等于记忆的语义丢了"
    if Path(f"{db}-wal").exists():
        assert Path(f"{db}-wal") in paths
    assert all(path.exists() for path in paths), f"清单里不应有幻影文件：{paths}"
    assert all(Path(home) in path.parents for path in paths)
    provider.shutdown()


def test_queue_prefetch_is_non_blocking_and_does_no_io(home, client, monkeypatch):
    """`queue_prefetch` 是"排名第一的热路径"：**实时回答，不做任何 I/O 或起线程**。

    R8 规定线程只能由 runtime 铸造；在 AL1 里再起一个线程会让 profile 上下文丢失，
    而且召回本来就是毫秒级只读查询——为它排队是净开销。
    """
    provider = client(home)
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("queue_prefetch 不得做 I/O，也不得铸造线程")

    monkeypatch.setattr(threading, "Thread", boom)
    monkeypatch.setattr(provider.services.backend, "query", boom)

    started = time.perf_counter()
    provider.queue_prefetch("用户偏好", session_id="s1")
    elapsed = (time.perf_counter() - started) * 1000

    assert calls["n"] == 0
    assert elapsed < 5, f"耗时 {elapsed:.1f}ms——应当只是记下查询就返回"
    monkeypatch.undo()
    provider.shutdown()


# =========================================================================== #
# AL1 · M2 钩子行为（T-AL1-10）
# =========================================================================== #


def test_pre_compress_archives_working_memory(home, client):
    """T-AL1-10 验收 1：压缩前**必须把工作记忆抢救成情景记忆**。

    这是本次评审抓到的真实缺陷：`on_pre_compress` 投出的 commit 任务在 AL5 侧
    无人认领、被静默丢弃，于是它"永远成功返回、什么都没发生"。
    只断言"返回值是 str"的用例对这件事完全无感——压缩恰恰是最该抢救现场的时刻。
    """
    provider = client(home, start_threads=True)
    services = provider.services
    services.backend.wm_put("s1", "goal", "把器灵接进宿主", 0.9)
    assert services.backend.wm_list("s1"), "前置条件：工作记忆里有东西"

    text = provider.on_pre_compress([], session_id="s1")
    assert text and text.strip(), "返回值非空是宿主契约的一部分"

    counts: dict = {}
    deadline = time.time() + 10
    while time.time() < deadline:
        services.maintenance.trigger_now()
        counts = services.backend.count_by_layer()
        if counts.get("episodic", {}).get("active", 0) >= 1:
            break
        time.sleep(0.05)
    provider.shutdown()

    assert counts.get("episodic", {}).get("active", 0) >= 1, f"工作记忆未被归档：{counts}"


def test_memory_write_mirrors_into_the_right_layers(home, client):
    """T-AL1-10 验收 2：宿主 `MEMORY.md` → 语义记忆，`USER.md` → 核心记忆。

    这是"与宿主内置记忆**并存**而非取代"的落点。写错层不会报错，
    只是核心层永远空着、人格注入永远为空——**静默失效**。
    """
    provider = client(home, start_threads=True)
    _seed_identity(provider, "用户叫老张，做后端")
    provider.on_memory_write("add", "MEMORY.md", "用户在做 artifact-spirit 这个项目")
    assert provider.services.flush(10.0)

    semantic = [
        record
        for record in provider.services.backend.query(layer="semantic", status=None)
        if "artifact-spirit" in (record.content or "")
    ]
    assert semantic, "MEMORY.md 的镜像没进语义层"
    assert "老张" in provider.system_prompt_block(token_budget=4000), "USER.md 的镜像没进核心层"
    provider.shutdown()


def test_memory_write_and_pre_compress_do_not_block(home, client):
    """T-AL1-10 验收 3：两者都在宿主热路径上，必须"投递即返回"。"""
    provider = client(home, start_threads=True)

    started = time.perf_counter()
    provider.on_memory_write("add", "USER.md", "用户叫老张" * 50)
    provider.on_pre_compress([], session_id="s1")
    elapsed = (time.perf_counter() - started) * 1000

    assert elapsed < 50, f"耗时 {elapsed:.1f}ms——它们只该把活投出去"
    provider.shutdown()


def test_empty_memory_write_is_ignored(home, client):
    """空内容是宿主的正常噪声（文件被清空），不该在库里留一条空记忆。"""
    provider = client(home, start_threads=True)
    provider.on_memory_write("add", "MEMORY.md", "   ")
    assert provider.services.flush(10.0)

    assert provider.services.backend.query(layer="semantic", status=None) == []
    provider.shutdown()


def test_delegation_is_archived_as_a_turn(home, client):
    """T-AL1-12 的姊妹面：子代理做完的活儿也要留档，否则跨会话复盘看不到它做过什么。"""
    provider = client(home, start_threads=True)
    events: list = []
    real_submit = provider.services.submit

    def spy(event, *args, **kwargs):
        events.append(event)
        return real_submit(event, *args, **kwargs)

    provider.services.submit = spy  # type: ignore[method-assign]
    provider.on_delegation("调研 A 方案", "结论：B 更好", child_session_id="child-1")

    assert events, "委派结果必须进入写入管线"
    assert events[0].session_id == "child-1", "必须挂在子会话里，不能混进主会话"
    assert "调研 A 方案" in events[0].user
    assert events[0].assistant == "结论：B 更好"
    provider.shutdown()


# =========================================================================== #
# AL1 · INV-9 路径隔离
# =========================================================================== #


def test_inv9_outputs_stay_under_hermes_home(tmp_path):
    """INV-9：**所有产出路径都在宿主传入的 hermes_home 之下**。

    最小配置（连配置文件都没有）下尤其重要：此时路径全由默认值推导，推到别处
    就会写出一个"谁也不知道在哪"的库——多 profile 部署下等于把两个使用者的记忆
    混进同一个文件，而且**不会报错**。
    """
    home = tmp_path / "isolated_home"
    home.mkdir()

    provider = ArtifactSpiritProvider()
    provider.initialize(hermes_home=str(home), env={}, start_threads=False)

    root = home.resolve()
    outputs = [Path(provider.services.backend.path).resolve()]
    outputs += [Path(path).resolve() for path in provider.backup_paths()]
    stray = [str(path) for path in outputs if path != root and root not in path.parents]
    provider.shutdown()

    assert not stray, f"这些产出不在 hermes_home 之下（INV-9）：{stray}"


# =========================================================================== #
# AL1 · M0 打包与入口注册（T-AL1-01#R1）
# =========================================================================== #


def test_entry_point_is_declared_and_consistent():
    """T-AL1-01#R1：入口在 `pyproject.toml` 与 `plugin.yaml` **两处声明且必须一致**。

    入口写错的表现不是报错，而是"宿主根本发现不了这个插件"——`is_available` 没人调、
    工具列表为空、配置面板里没有这个 provider。而整包测试可以全绿：
    没有任何测试需要 import 它。
    """
    import tomllib

    root = PACKAGE_ROOT.parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = (PACKAGE_ROOT / "plugin.yaml").read_text(encoding="utf-8")

    providers = project["project"]["entry-points"]["hermes_agent.memory_providers"]
    assert providers == {"artifact-spirit": "artifact_spirit:register"}
    assert "entry_point: artifact_spirit:register" in manifest
    assert "hermes_agent.memory_providers" in manifest

    assert project["project"]["scripts"]["aspirit"] == "artifact_spirit.cli:main"
    assert "aspirit: artifact_spirit.cli:main" in manifest
    assert project["project"]["name"] == "artifact-spirit"

    from artifact_spirit import register

    assert callable(register), "入口必须指向一个可调用对象，否则宿主 import 时才发现"

