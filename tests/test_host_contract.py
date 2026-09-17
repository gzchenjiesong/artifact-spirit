"""宿主契约一致性（**对照真实外部代码，不是对照我们自己的注释**）。

这个文件存在的理由是一次真实的翻车：整包 347 项测试全绿、架构规则零违规，
但把 provider 拿到宿主契约前面一比，**8 个宿主钩子根本没实现、2 处返回值类型不符**。

问题不在"写错了"，而在**测试用的基准是自造的**：所有用例都只调用我们实现过的方法，
于是"没实现的方法"永远不会被调用、也永远不会失败。缺钩子的真实后果也不是崩溃——
宿主 `MemoryManager._each_provider` 会把 AttributeError 吞成一条日志，
表现为"每轮打一行错误 + 该能力静默失效"，比崩溃更难发现。

基准来自 `tests/vendor/host_memory_provider.py`（上游原样副本，见该目录 README）。
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from artifact_spirit.provider import ArtifactSpiritProvider

VENDOR = Path(__file__).resolve().parent / "vendor" / "host_memory_provider.py"


def _host_abc():
    spec = importlib.util.spec_from_file_location("host_memory_provider", VENDOR)
    module = importlib.util.module_from_spec(spec)
    # dataclass 装饰器会回查 sys.modules[cls.__module__]，不登记就会在 @dataclass 处炸
    sys.modules["host_memory_provider"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MemoryProvider


def _host_public_methods() -> dict[str, inspect.Signature]:
    host = _host_abc()
    return {
        name: inspect.signature(getattr(host, name))
        for name in dir(host)
        if not name.startswith("_") and callable(getattr(host, name, None))
    }


def test_all_host_methods_are_implemented():
    """宿主 ABC 的**每一个**公开方法都必须在 provider 上存在且可调用。"""
    missing = [
        name
        for name in _host_public_methods()
        if not callable(getattr(ArtifactSpiritProvider, name, None))
    ]
    assert not missing, (
        f"缺少宿主方法 {missing}——宿主不会因为缺方法而崩溃，"
        "而是每轮打一行日志并静默丢掉该能力（见本文件顶部说明）"
    )


@pytest.mark.parametrize("method_name", sorted(_host_public_methods()))
def test_signature_accepts_host_call(method_name):
    """宿主的调用方式必须能被接受。

    判据：宿主签名里的每个参数名，在实现里要么同名存在，要么被 ``**kwargs`` 吸收。
    **不比对顺序与返回值标注**——宿主只按关键字或位置传它自己那几个参数，
    多接受一些参数（如 ``session_id``）是允许的，也是必需的。
    """
    host_params = _host_public_methods()[method_name].parameters
    impl = getattr(ArtifactSpiritProvider, method_name)
    impl_params = inspect.signature(impl).parameters

    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in impl_params.values()
    )
    var_positional = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in impl_params.values()
    )

    for name, param in host_params.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if name in impl_params:
            continue
        # 位置参数可以被 *args 吸收；关键字参数只能被 **kwargs 吸收
        if param.kind is inspect.Parameter.KEYWORD_ONLY and has_var_kw:
            continue
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and (has_var_kw or var_positional):
            continue
        pytest.fail(
            f"{method_name} 无法接受宿主的参数 {name!r}"
            f"（宿主签名 {inspect.signature(getattr(_host_abc(), method_name))}）"
        )


def test_handle_tool_call_returns_json_string():
    """`handle_tool_call` 必须返回 **JSON 字符串**。

    宿主文档原文：*"Route a tool call to its provider; returns a JSON string
    (tool_error on failure)"*。返回 dict 不会报错——dict 会被当成模型看到的工具结果，
    于是模型读到的是 Python repr。这类错误不抛异常，只是"工具变得不好用"。
    """
    provider = ArtifactSpiritProvider()
    raw = provider.handle_tool_call("spirit_recall", {"query": "x"})
    assert isinstance(raw, str), f"期望 str，得到 {type(raw).__name__}"
    json.loads(raw)  # 必须是合法 JSON


def test_uninitialized_error_shape_matches_host_convention():
    """未初始化时也要给出 ``{"error": ...}``——与宿主 `tool_error` 同构。"""
    provider = ArtifactSpiritProvider()
    payload = json.loads(provider.handle_tool_call("spirit_recall", {"query": "x"}))
    assert "error" in payload


def test_get_config_schema_is_a_list_of_field_dicts():
    """配置面板要的是**扁平字段列表**；返回 dict 会让面板一个字段都不显示。"""
    provider = ArtifactSpiritProvider()
    schema = provider.get_config_schema()
    assert isinstance(schema, list)
    assert schema
    for field in schema:
        assert isinstance(field, dict)
        assert field.get("key")
        # 宿主 `_schema_field_kind` 只认这几种 type
        assert field.get("type") in {"text", "integer", "number", "boolean", None}


def test_save_config_returns_none_per_host_contract():
    """宿主契约里 `save_config(...) -> None`；返回值不被使用。"""
    provider = ArtifactSpiritProvider()
    assert provider.save_config({}, "") is None


def test_on_pre_compress_returns_string():
    """`on_pre_compress` 的返回值会被宿主拼进压缩提示词，必须是 str。"""
    provider = ArtifactSpiritProvider()
    assert isinstance(provider.on_pre_compress([]), str)


def test_pre_compress_does_not_claim_checkpoint_api():
    """器灵**不声明** v2 checkpoint（不设 `pre_compress_checkpoint_api_version`）。

    声明了就会进入宿主的 fail-closed 分支：`require_checkpoint=True` 时
    器灵必须保证 checkpoint 成功，否则宿主抛错并放弃压缩。
    "能不能保证"取决于背后的 LLM，器灵不该替它承诺。
    """
    assert not hasattr(ArtifactSpiritProvider, "pre_compress_checkpoint_api_version")


def test_background_threads_use_injectable_factory():
    """后台线程必须走可注入的铸造函数，否则拿不到宿主的 profile 上下文。

    宿主文档：*"Every memory-provider background job (prefetch, sync, writer loops)
    must go through this"* —— 一个用空 contextvars 启动的 worker 会**静默写到
    默认 profile 的数据库**。这类错误不报错，只写错地方。

    **探测宿主的是 AL1（`provider.py`），不是组合根**：R4 规定宿主耦合只允许
    出现在 provider.py，所以 `resolve_host_thread_factory` 从这里导出，
    组合根只接收注入结果。
    """
    from artifact_spirit.provider import resolve_host_thread_factory
    from artifact_spirit.runtime import default_thread_factory
    from artifact_spirit.runtime.maintenance import Maintenance
    from artifact_spirit.runtime.writer import Writer

    assert callable(default_thread_factory)
    assert callable(resolve_host_thread_factory())
    assert Writer.__dataclass_fields__["thread_factory"].default is None
    assert Maintenance.__dataclass_fields__["thread_factory"].default is None


def test_no_runtime_module_creates_threads_directly():
    """R8 收紧版：线程创建只允许出现在 `threading_.py` 一处。

    收成一处的意义是"用什么上下文起线程"成为一个可审查的策略。
    散落各处时，漏掉一处就是漏掉一个 profile。
    """
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT

    offenders = []
    for path in (PACKAGE_ROOT / "runtime").glob("*.py"):
        if path.name == "threading_.py":
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if "threading.Thread(" in line and not line.strip().startswith("#"):
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"这些位置直接创建了线程：{offenders}——应改用 threading_ 的铸造函数"


# --------------------------------------------------------------------------- #
# 无参 is_available —— 宿主就是这么调用它的
# --------------------------------------------------------------------------- #


def test_is_available_without_arguments_uses_resolved_home(tmp_path, monkeypatch):
    """`is_available()` **无参**调用也必须能工作。

    宿主在 `initialize` **之前**就探活，调用形式是
    `probe_availability(lambda: provider.is_available())` —— 没有任何参数。
    只认 `kwargs["hermes_home"]` 的实现会恒返回 False，宿主于是显示
    "not available"，用户以为插件装坏了。

    这条在真机部署时才暴露出来：本地测试都习惯显式传 hermes_home，
    等于绕过了"宿主其实不传"这个事实。
    """
    from artifact_spirit.provider import ArtifactSpiritProvider, _resolve_hermes_home

    # 造一个可用的 home
    home = tmp_path / "hermes_home"
    (home / "spirit").mkdir(parents=True)

    # 模拟"宿主不在场、只有环境变量"的场景
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _resolve_hermes_home() == str(home)

    provider = ArtifactSpiritProvider()
    assert provider.is_available() is True, "无参调用必须走解析链而不是直接返回 False"


def test_unavailable_reason_shares_home_resolution(tmp_path, monkeypatch):
    """`unavailable_reason` 与 `is_available` 必须同源。

    否则会出现"is_available 说可用、reason 说没收到 home"的自相矛盾输出。
    """
    from artifact_spirit.provider import ArtifactSpiritProvider

    home = tmp_path / "hermes_home"
    (home / "spirit").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    provider = ArtifactSpiritProvider()
    assert provider.unavailable_reason() == ""


# --------------------------------------------------------------------------- #
# 工具 schema 的形状 —— 宿主会把它**原样**交给模型
# --------------------------------------------------------------------------- #


def _host_normalize(schema):
    """走宿主自己的规范化函数（vendor 副本，见 tests/vendor/README.md）。"""
    spec = importlib.util.spec_from_file_location(
        "host_tool_schema", Path(__file__).resolve().parent / "vendor" / "host_tool_schema.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["host_tool_schema"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.normalize_tool_schema(schema)


def test_tool_schemas_survive_host_normalization():
    """每一个工具 schema 经宿主规范化后都应有可解析的 ``name``。

    宿主对**没有 name** 的 schema 会跳过并打一条 warning——不报错，
    只是那个工具从模型的可用工具集里消失。
    """
    provider = ArtifactSpiritProvider()
    raw = provider.get_tool_schemas()
    assert isinstance(raw, list) and raw, "get_tool_schemas 必须返回非空 list"

    for schema in raw:
        assert _host_normalize(schema) is not None, f"宿主无法解析：{schema!r}"


def test_tool_schemas_expose_parameters_not_input_schema():
    """工具的参数定义必须叫 ``parameters``（**不是** ``input_schema``）。

    宿主把它们原样塞进 ``{"type": "function", "function": schema}``
    （`agent_init` 与 `memory_manager` 两处），而 OpenAI 规范读的是
    ``function.parameters``；``input_schema`` 是 Anthropic 的叫法。

    写错的后果不是报错，而是**模型收到一个没有参数定义的函数**：工具照样能调，
    参数却传不进去。真机实测时模型的原话是"spirit_remember 参数传不进去"——
    整包单测与宿主集成验证都没有发现，因为集成验证是我自己构造 args 直接调
    `handle_tool_call` 的，**绕过了 schema**。
    """
    provider = ArtifactSpiritProvider()
    for raw in provider.get_tool_schemas():
        schema = _host_normalize(raw)
        assert schema is not None
        name = schema["name"]
        assert "input_schema" not in schema, (
            f"{name}: 用了 Anthropic 的 input_schema —— 模型会看不到参数"
        )
        params = schema.get("parameters")
        assert isinstance(params, dict), f"{name}: 缺少 parameters"
        assert params.get("type") == "object", f"{name}: parameters.type 必须是 object"
        assert isinstance(params.get("properties"), dict), f"{name}: 缺少 parameters.properties"


def test_wrapped_tools_have_callable_parameter_definitions():
    """模拟宿主最终交给模型的形状，并检查**必填参数真的可见**。

    宿主做的是 ``tools.append({"type": "function", "function": schema})``。
    只检查到这一层还不够：必须确认模型能在 ``properties`` 里找到必填参数的名字，
    否则它不知道该传什么——那正是"参数传不进去"的另一半原因。
    """
    provider = ArtifactSpiritProvider()
    tools = [
        {"type": "function", "function": _host_normalize(raw)}
        for raw in provider.get_tool_schemas()
    ]

    for tool in tools:
        fn = tool["function"]
        assert fn["name"] and fn["description"], f"{fn['name']}: name/description 不可为空"
        params = fn["parameters"]
        props = params["properties"]
        required = params.get("required")
        assert isinstance(required, list), f"{fn['name']}: required 必须是列表"

        for key in required:
            assert key in props, (
                f"{fn['name']}: 必填参数 {key!r} 没在 properties 里定义 —— 模型无从传值"
            )
            assert props[key].get("description"), f"{fn['name']}.{key}: 参数缺 description"


def test_tool_schema_names_match_handlers():
    """schema 里的工具名与 handler 分发表必须一一对应（不一致会静默失效）。"""
    from artifact_spirit.tools.schemas import TOOL_NAMES

    provider = ArtifactSpiritProvider()
    from_schemas = {_host_normalize(s)["name"] for s in provider.get_tool_schemas()}
    assert from_schemas == set(TOOL_NAMES), (
        f"schema 与实际工具名不一致：多={from_schemas - set(TOOL_NAMES)} "
        f"少={set(TOOL_NAMES) - from_schemas}"
    )
