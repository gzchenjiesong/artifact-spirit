"""AL4 模型层验收（T-AL4-01 ~ 12）。

全部测试基于 ``httpx.MockTransport``——**不连真实网关、不需要网络**（T-AL4-12）。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from artifact_spirit.model import (
    EmbeddingError,
    LLMError,
    ModelError,
    ModelResolver,
    OpenAICompatClient,
    OpenAICompatSettings,
    ProviderUnavailableError,
    SchemaValidationError,
    SchemaViolationError,
    validate_schema,
)

# --------------------------------------------------------------------------- #
# 测试替身：本地 mock 网关
# --------------------------------------------------------------------------- #


class Gateway:
    """可编程的 mock OpenAI 兼容网关。"""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.responses: list[dict | httpx.Response | Exception] = []
        self.default_chat = {"choices": [{"message": {"content": "好的"}}]}

    def push(self, item) -> None:
        self.responses.append(item)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if self.responses else None
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx.Response):
            return item
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content)
            payload = item or {
                "data": [
                    {"index": i, "embedding": [float(i)] * 4}
                    for i in range(len(body["input"]))
                ]
            }
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json=item or self.default_chat)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    @property
    def chat_calls(self) -> int:
        return sum(1 for r in self.requests if r.url.path.endswith("/chat/completions"))

    @property
    def embed_calls(self) -> int:
        return sum(1 for r in self.requests if r.url.path.endswith("/embeddings"))


@pytest.fixture
def gateway() -> Gateway:
    return Gateway()


def make_client(gateway: Gateway, **kwargs) -> OpenAICompatClient:
    settings = OpenAICompatSettings(
        base_url=kwargs.pop("base_url", "https://gw.example/v1"),
        api_key=kwargs.pop("api_key", "secret-key"),
        backoff_base=0.0,
        **kwargs,
    )
    return OpenAICompatClient(settings, transport=gateway.transport)


# --------------------------------------------------------------------------- #
# T-AL4-01 协议与异常族
# --------------------------------------------------------------------------- #


def test_exception_hierarchy():
    assert issubclass(LLMError, ModelError)
    assert issubclass(SchemaViolationError, LLMError)
    assert issubclass(ProviderUnavailableError, LLMError)
    assert issubclass(EmbeddingError, ModelError)
    assert not issubclass(EmbeddingError, LLMError), (
        "EmbeddingError **不得**是 LLMError 的子类——否则上层会误以为可以 fallback 到 LLM（INV-6）"
    )
    assert not issubclass(SchemaValidationError, ModelError), (
        "SchemaValidationError 表示'数据不合 schema'，不是模型层故障——不得并入 ModelError（AL4 LLD §2.2）"
    )


def test_protocols_defined():
    from artifact_spirit.model.base import EmbeddingProvider, LLMProvider, ModelRegistry

    assert hasattr(LLMProvider, "complete")
    assert hasattr(LLMProvider, "complete_json")
    assert hasattr(EmbeddingProvider, "embed")
    assert hasattr(ModelRegistry, "resolve_chain")


def test_protocol_module_has_no_io_imports():
    from artifact_spirit.compliance import resolved_imports
    from artifact_spirit.model.base import __file__ as base_file

    imports = resolved_imports(Path(base_file))
    assert "httpx" not in imports, "协议层必须零 I/O"


# --------------------------------------------------------------------------- #
# T-AL4-02 客户端骨架
# --------------------------------------------------------------------------- #


def test_complete_returns_text(gateway):
    gateway.push({"choices": [{"message": {"content": "你好，我是模型"}}]})
    client = make_client(gateway)
    assert client.complete(messages=[{"role": "user", "content": "hi"}], model="m") == "你好，我是模型"
    assert gateway.requests[0].headers["Authorization"] == "Bearer secret-key"


@pytest.mark.parametrize(
    "base_url", ["https://gw.example/v1", "https://gw.example/v1/", "https://gw.example/v1///"]
)
def test_base_url_trailing_slash_normalized(gateway, base_url):
    client = make_client(gateway, base_url=base_url)
    client.complete(messages=[{"role": "user", "content": "hi"}], model="m")
    assert gateway.requests[0].url.path == "/v1/chat/completions"
    assert "//chat" not in str(gateway.requests[0].url)


def test_timeout_is_passed_explicitly(gateway):
    client = make_client(gateway, timeout_llm=7.5)
    client.complete(messages=[{"role": "user", "content": "hi"}], model="m")
    timeout = gateway.requests[0].extensions.get("timeout")
    assert timeout, "必须显式传入超时——不得依赖 httpx 默认（C7）"
    assert any(abs(v - 7.5) < 0.01 for v in timeout.values() if isinstance(v, (int, float)))


def test_per_call_timeout_override(gateway):
    client = make_client(gateway, timeout_llm=60.0)
    client.complete(messages=[{"role": "user", "content": "hi"}], model="m", timeout=3.0)
    timeout = gateway.requests[0].extensions["timeout"]
    assert any(abs(v - 3.0) < 0.01 for v in timeout.values() if isinstance(v, (int, float)))


# --------------------------------------------------------------------------- #
# T-AL4-03 重试与异常分类
# --------------------------------------------------------------------------- #


def test_connection_failure_retries_exactly_twice(gateway):
    """连接类错误 → 共 3 次请求（首次 + 2 次重试）。"""
    for _ in range(3):
        gateway.push(httpx.ConnectError("network down"))
    client = make_client(gateway, max_retries=2)
    with pytest.raises(ProviderUnavailableError):
        client.complete(messages=[{"role": "user", "content": "x"}], model="m")
    assert gateway.chat_calls == 3


def test_http_429_is_not_retried(gateway):
    gateway.push(httpx.Response(429, text="slow down"))
    client = make_client(gateway)
    with pytest.raises(ProviderUnavailableError):
        client.complete(messages=[{"role": "user", "content": "x"}], model="m")
    assert gateway.chat_calls == 1, "4xx 重试没有意义（配额/配置问题）"


def test_http_401_is_not_retried_and_raises_unavailable(gateway):
    gateway.push(httpx.Response(401, text="bad key"))
    client = make_client(gateway)
    with pytest.raises(ProviderUnavailableError):
        client.complete(messages=[{"role": "user", "content": "x"}], model="m")
    assert gateway.chat_calls == 1


def test_server_error_retries_then_raises_unavailable(gateway):
    for _ in range(3):
        gateway.push(httpx.Response(503, text="oops"))
    client = make_client(gateway, max_retries=2)
    with pytest.raises(ProviderUnavailableError):
        client.complete(messages=[{"role": "user", "content": "x"}], model="m")
    assert gateway.chat_calls == 3


# --------------------------------------------------------------------------- #
# T-AL4-04 结构化输出
# --------------------------------------------------------------------------- #


def test_complete_json_returns_dict(gateway):
    gateway.push({"choices": [{"message": {"content": '{"memories": []}'}}]})
    client = make_client(gateway)
    result = client.complete_json(
        messages=[{"role": "user", "content": "x"}],
        model="m",
        schema={"type": "object", "required": ["memories"]},
    )
    assert result == {"memories": []}


def test_complete_json_retries_exactly_once_then_succeeds(gateway):
    gateway.push({"choices": [{"message": {"content": "这不是 JSON"}}]})
    gateway.push({"choices": [{"message": {"content": '{"ok": true}'}}]})
    client = make_client(gateway)
    assert client.complete_json(
        messages=[{"role": "user", "content": "x"}], model="m", schema={"type": "object"}
    ) == {"ok": True}
    assert gateway.chat_calls == 2


def test_complete_json_raises_after_two_failures(gateway):
    gateway.push({"choices": [{"message": {"content": "nope"}}]})
    gateway.push({"choices": [{"message": {"content": "still nope"}}]})
    client = make_client(gateway)
    with pytest.raises(SchemaViolationError):
        client.complete_json(
            messages=[{"role": "user", "content": "x"}], model="m", schema={"type": "object"}
        )
    assert gateway.chat_calls == 2, "重试恰好 1 次"


def test_complete_json_retry_appends_schema_hint(gateway):
    gateway.push({"choices": [{"message": {"content": "bad"}}]})
    gateway.push({"choices": [{"message": {"content": '{"memories": []}'}}]})
    client = make_client(gateway)
    client.complete_json(
        messages=[{"role": "user", "content": "x"}],
        model="m",
        schema={"type": "object", "required": ["memories"]},
    )
    second = json.loads(gateway.requests[1].content)
    assert "Schema" in second["messages"][-1]["content"]


def test_validator_detects_missing_field_and_type_error():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "tags": {"type": "array"}},
        "required": ["n"],
    }
    assert validate_schema({"n": 1, "tags": []}, schema) == []
    assert any("缺少必填字段" in e for e in validate_schema({"tags": []}, schema))
    assert any("期望类型" in e for e in validate_schema({"n": "1"}, schema))


def test_validator_supports_enum_and_anyof():
    schema = {"type": "object", "properties": {"k": {"enum": ["a", "b"]}}}
    assert validate_schema({"k": "a"}, schema) == []
    assert validate_schema({"k": "c"}, schema) != []

    any_of = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert validate_schema(3, any_of) == []
    assert validate_schema(None, any_of) == []
    assert validate_schema("x", any_of) != []


def test_validator_rejects_bool_as_integer():
    assert validate_schema(True, {"type": "integer"}) != []


def test_validator_has_no_third_party_dependency():
    from artifact_spirit.compliance import resolved_imports
    from artifact_spirit.model.base import __file__ as base_file

    imports = resolved_imports(Path(base_file))
    assert "jsonschema" not in imports


# --------------------------------------------------------------------------- #
# T-AL4-05 Embedding 批量与保序
# --------------------------------------------------------------------------- #


def test_embed_preserves_input_order(gateway):
    client = make_client(gateway)
    vectors = client.embed(["a", "b", "c"], model="emb")
    assert len(vectors) == 3
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


def test_embed_respects_server_index(gateway):
    """服务端乱序返回时，仍须按入参顺序归位（错配是灾难性 bug）。"""
    gateway.push(
        {
            "data": [
                {"index": 2, "embedding": [2.0]},
                {"index": 0, "embedding": [0.0]},
                {"index": 1, "embedding": [1.0]},
            ]
        }
    )
    client = make_client(gateway)
    vectors = client.embed(["a", "b", "c"], model="emb")
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


def test_embed_batch_size_config(gateway):
    client = make_client(gateway, embed_batch_size=2)
    client.embed(["a", "b", "c", "d", "e"], model="emb")
    assert gateway.embed_calls == 3


def test_embed_long_text_gets_own_batch(gateway):
    from artifact_spirit.model.openai_compat import SINGLE_TEXT_MAX_CHARS

    long_text = "很长的文本" * (SINGLE_TEXT_MAX_CHARS // 3)
    client = make_client(gateway)
    client.embed(["短", long_text, "短"], model="emb")
    sizes = [len(json.loads(r.content)["input"]) for r in gateway.requests]
    assert 1 in sizes, "超长文本必须单独成批"


def test_embed_empty_input_makes_no_request(gateway):
    client = make_client(gateway)
    assert client.embed([], model="emb") == []
    assert gateway.requests == []


def test_embed_count_mismatch_raises(gateway):
    gateway.push({"data": [{"index": 0, "embedding": [0.0]}]})  # 只回一条
    client = make_client(gateway)
    with pytest.raises(EmbeddingError):
        client.embed(["a", "b"], model="emb")


def test_embed_uses_embed_timeout(gateway):
    client = make_client(gateway, timeout_embed=4.5)
    client.embed(["a"], model="emb")
    timeout = gateway.requests[0].extensions["timeout"]
    assert any(abs(v - 4.5) < 0.01 for v in timeout.values() if isinstance(v, (int, float)))


# --------------------------------------------------------------------------- #
# T-AL4-06 三档解析与 fallback 链
# --------------------------------------------------------------------------- #


def test_spirit_config_wins(gateway, tmp_path):
    resolver = ModelResolver(
        llm={"base_url": "https://spirit/v1", "api_key_env": "SP_KEY", "extract": "glm-5.3-flash"},
        hermes_home=tmp_path,
        env={"SP_KEY": "k"},
        transport=gateway.transport,
    )
    chain = resolver.resolve_chain("extract")
    assert chain[0]["source"] == "spirit" and chain[0]["active"] is True
    assert chain[-1]["model"] == "glm-5.3-flash"


def test_host_fallback_when_spirit_unconfigured(gateway, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "# 宿主配置\n"
        "memory:\n"
        "  llm:\n"
        "    base_url: https://host-gw/v1\n"
        "    api_key_env: HOST_KEY\n"
        "    model: host-model\n",
        encoding="utf-8",
    )
    resolver = ModelResolver(
        hermes_home=tmp_path, env={"HOST_KEY": "hk"}, transport=gateway.transport
    )
    chain = resolver.resolve_chain("extract")
    host_segment = next(s for s in chain if s["source"] == "host")
    assert host_segment["active"] is True
    assert host_segment["base_url"] == "https://host-gw/v1"


def test_env_fallback_when_nothing_configured(gateway, tmp_path):
    resolver = ModelResolver(
        hermes_home=tmp_path,
        env={"OPENAI_API_KEY": "ek", "OPENAI_BASE_URL": "https://env-gw/v1"},
        transport=gateway.transport,
    )
    chain = resolver.resolve_chain("extract")
    assert next(s for s in chain if s["source"] == "env")["active"] is True
    assert resolver.llm("extract").route_source == "env"


def test_all_segments_down_raises(gateway, tmp_path):
    resolver = ModelResolver(hermes_home=tmp_path, env={}, transport=gateway.transport)
    with pytest.raises(ProviderUnavailableError):
        resolver.llm("extract")
    assert resolver.llm_available() is False


def test_spirit_segment_skipped_when_key_env_missing(gateway, tmp_path):
    """api_key_env 指向的环境变量缺失 → 该段不可用，应继续往下走。"""
    (tmp_path / "config.yaml").write_text(
        "memory:\n  llm:\n    base_url: https://host/v1\n    model: m\n", encoding="utf-8"
    )
    resolver = ModelResolver(
        llm={"base_url": "https://spirit/v1", "api_key_env": "MISSING_KEY"},
        hermes_home=tmp_path,
        env={},
        transport=gateway.transport,
    )
    chain = resolver.resolve_chain("extract")
    assert chain[0]["usable"] is False
    assert next(s for s in chain if s["source"] == "host")["active"] is True


def test_embedding_never_falls_back_to_llm(gateway, tmp_path):
    """INV-6：embedding 不可用时**不得**触碰 chat 接口。"""
    resolver = ModelResolver(
        llm={"base_url": "https://spirit/v1", "api_key_env": "SP_KEY"},
        hermes_home=tmp_path,
        env={"SP_KEY": "k"},
        transport=gateway.transport,
    )
    with pytest.raises(EmbeddingError) as excinfo:
        resolver.embedding()
    assert "INV-6" in str(excinfo.value) or "不会降级" in str(excinfo.value)
    assert gateway.chat_calls == 0, "embedding 缺失时绝不能调用 chat 接口"
    assert gateway.requests == []


def test_embedding_error_is_embedding_error_not_llm_error(gateway, tmp_path):
    resolver = ModelResolver(hermes_home=tmp_path, env={}, transport=gateway.transport)
    with pytest.raises(EmbeddingError):
        resolver.embedding()


# --------------------------------------------------------------------------- #
# T-AL4-06b 链路隔离（DES-REV-007 P0-10 回归）
# --------------------------------------------------------------------------- #


def test_llm_chain_failure_does_not_disable_embedding(gateway, tmp_path):
    """LLM 链解析失败**不得**污染 embedding 链（AL4 LLD §6 C11 / P0-10）。

    "链非空"被当作"该链路已解析完成"的判据，因此写错链的后果不是显示错，
    而是让另一条链**跳过解析**并误报不可用——上层随即静默降级 BM25
    （INV-6 的降级路径被误触发，且用户看到的是"embedding 没配"）。
    """
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  llm:\n"
        "    base_url: https://host-gw/v1\n"
        "    api_key_env: MISSING_HOST_KEY\n",
        encoding="utf-8",
    )
    resolver = ModelResolver(
        embedding={"base_url": "https://spirit/v1", "model": "kinfra-text-embedding-4b"},
        hermes_home=tmp_path,
        env={},
        transport=gateway.transport,
    )
    # 先解析 LLM —— 这一步曾经把 host 段写进 embedding 链
    assert resolver.llm_available() is False, "宿主 LLM 段应因环境变量缺失而不可用"
    assert [s["source"] for s in resolver.embedding_chain()] == ["spirit"], (
        "embedding 链里不得出现 LLM 解析产生的段"
    )
    assert resolver.embedding_available() is True, (
        "[models.embedding] 完全合法，不得因为 LLM 不可用而被判成不可用"
    )


def test_llm_host_segment_reports_which_key_env_is_missing(gateway, tmp_path):
    """宿主段被跳过的原因要能区分'没有'与'变量没设'（AL4 LLD §5 M3）。"""
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  llm:\n"
        "    base_url: https://host-gw/v1\n"
        "    api_key_env: MISSING_HOST_KEY\n",
        encoding="utf-8",
    )
    resolver = ModelResolver(hermes_home=tmp_path, env={}, transport=gateway.transport)
    reasons = [s.get("reason", "") for s in resolver.resolve_chain("extract")]
    assert any("MISSING_HOST_KEY" in r for r in reasons), (
        f"错误提示要指名**变量名**（这是可操作的）；实际 reasons={reasons}"
    )


def test_embedding_chain_never_contains_active_true_for_missing_layers(gateway, tmp_path):
    """embedding 链的每段都要带 `usable`，全断时不得出现"看起来生效"的段。"""
    resolver = ModelResolver(hermes_home=tmp_path, env={}, transport=gateway.transport)
    chain = resolver.embedding_chain()
    assert chain, "全断时也要给出链的诊断信息（否则 status 无法解释为什么没有向量检索）"
    assert all(s.get("usable") is not True for s in chain)


# --------------------------------------------------------------------------- #
# T-AL4-07 维度与模型属性
# --------------------------------------------------------------------------- #


def test_known_embedding_model_dim(gateway, tmp_path):
    resolver = ModelResolver(
        embedding={
            "base_url": "https://spirit/v1",
            "model": "kinfra-text-embedding-4b",
            "api_key_env": "SP_KEY",
        },
        hermes_home=tmp_path,
        env={"SP_KEY": "k"},
        transport=gateway.transport,
    )
    provider = resolver.embedding()
    assert provider.model == "kinfra-text-embedding-4b"
    assert provider.dim == 2560


def test_explicit_dim_overrides_known():
    dim = ModelResolver._embedding_dim("kinfra-text-embedding-4b", 1024)
    assert dim == 1024


def test_unknown_model_requires_explicit_dim():
    with pytest.raises(EmbeddingError):
        ModelResolver._embedding_dim("mystery-embed", None)
    assert ModelResolver._embedding_dim("mystery-embed", 768) == 768


def test_invalid_dim_rejected():
    with pytest.raises(EmbeddingError):
        ModelResolver._embedding_dim("m", 0)
    with pytest.raises(EmbeddingError):
        ModelResolver._embedding_dim("m", "abc")


# --------------------------------------------------------------------------- #
# T-AL4-08 配置对接与密钥纪律
# --------------------------------------------------------------------------- #


def test_api_key_read_from_env_var(gateway, tmp_path):
    resolver = ModelResolver(
        llm={"base_url": "https://s/v1", "api_key_env": "MY_KEY"},
        hermes_home=tmp_path,
        env={"MY_KEY": "resolved-value"},
        transport=gateway.transport,
    )
    handle = resolver.llm("extract")
    handle.complete(messages=[{"role": "user", "content": "x"}])
    assert gateway.requests[0].headers["Authorization"] == "Bearer resolved-value"


def test_missing_key_env_error_does_not_leak_key(gateway, tmp_path):
    resolver = ModelResolver(
        llm={"base_url": "https://s/v1", "api_key_env": "ABSENT_KEY"},
        hermes_home=tmp_path,
        env={},
        transport=gateway.transport,
    )
    chain = resolver.resolve_chain("extract")
    assert chain[0]["usable"] is False
    assert "ABSENT_KEY" in chain[0]["reason"], "错误提示要指明**变量名**（这是可操作的）"


def test_settings_repr_redacts_key():
    settings = OpenAICompatSettings(base_url="https://x/v1", api_key="super-secret-token")
    assert "super-secret-token" not in repr(settings)
    assert "super-secret-token" not in str(settings)


def test_exception_text_does_not_contain_key(gateway):
    gateway.push(httpx.Response(401, text="denied"))
    client = make_client(gateway, api_key="leak-me-please-1234567890")
    with pytest.raises(ProviderUnavailableError) as excinfo:
        client.complete(messages=[{"role": "user", "content": "x"}], model="m")
    assert "leak-me-please" not in str(excinfo.value)


# --------------------------------------------------------------------------- #
# T-AL4-09 降级路径可区分
# --------------------------------------------------------------------------- #


def test_f1_llm_unconfigured_raises_provider_unavailable(tmp_path, gateway):
    resolver = ModelResolver(hermes_home=tmp_path, env={}, transport=gateway.transport)
    with pytest.raises(ProviderUnavailableError):
        resolver.llm("extract")


def test_f2_embedding_unconfigured_raises_embedding_error(tmp_path, gateway):
    resolver = ModelResolver(hermes_home=tmp_path, env={}, transport=gateway.transport)
    with pytest.raises(EmbeddingError):
        resolver.embedding()


def test_f3_non_json_raises_schema_violation(gateway):
    gateway.push({"choices": [{"message": {"content": "x"}}]})
    gateway.push({"choices": [{"message": {"content": "y"}}]})
    client = make_client(gateway)
    with pytest.raises(SchemaViolationError):
        client.complete_json(
            messages=[{"role": "user", "content": "x"}], model="m", schema={"type": "object"}
        )


def test_degradation_exception_types_are_mutually_distinguishable(gateway, tmp_path):
    """AL2 靠异常类型做决策，因此三者必须互不包含。"""
    assert not issubclass(ProviderUnavailableError, EmbeddingError)
    assert not issubclass(EmbeddingError, ProviderUnavailableError)
    assert not issubclass(SchemaViolationError, EmbeddingError)
    assert not issubclass(SchemaViolationError, ProviderUnavailableError)


# --------------------------------------------------------------------------- #
# T-AL4-10 超时纪律（防过度设计回归）
# --------------------------------------------------------------------------- #


def test_no_rate_limiter_in_model_layer():
    """TPM 100 万 / RPM 300 极富余（ADR-003 §3.7）——不该出现限流器实现。"""
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT, iter_python_files

    banned = ("TokenBucket", "LeakyBucket", "rate_limit", "RateLimiter", "Semaphore")
    for path in iter_python_files(PACKAGE_ROOT / "model"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} 出现了限流器痕迹：{token}"


def test_timeouts_are_configurable():
    settings = OpenAICompatSettings(base_url="https://x/v1", timeout_llm=11.0, timeout_embed=22.0)
    assert settings.timeout_llm == 11.0
    assert settings.timeout_embed == 22.0


# --------------------------------------------------------------------------- #
# T-AL4-12 契约测试：分层与依赖
# --------------------------------------------------------------------------- #


def test_model_layer_respects_architecture():
    from artifact_spirit.compliance import check_architecture
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT

    violations = [v for v in check_architecture(PACKAGE_ROOT) if "/model/" in v["file"].replace("\\", "/")]
    assert violations == []


def test_model_layer_does_not_import_store_or_core():
    from artifact_spirit.compliance import resolved_imports
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT, iter_python_files

    for path in iter_python_files(PACKAGE_ROOT / "model"):
        imports = resolved_imports(path)
        assert not any(i.startswith("artifact_spirit.store") for i in imports), path
        assert not any(i.startswith("artifact_spirit.core") for i in imports), path


def test_no_openai_sdk_used():
    from artifact_spirit.compliance import resolved_imports
    from artifact_spirit.compliance.arch_rules import PACKAGE_ROOT, iter_python_files

    for path in iter_python_files(PACKAGE_ROOT / "model"):
        assert "openai" not in resolved_imports(path), path


def test_tests_need_no_network(gateway):
    """全部模型层测试基于 MockTransport——真实网络不可达也不影响。"""
    client = make_client(gateway)
    client.complete(messages=[{"role": "user", "content": "x"}], model="m")
    assert len(gateway.requests) == 1
