"""共享测试夹具。

各层的**独立验收**都靠替身完成（ENC-000 §5）：
AL3 用临时 SQLite 文件，AL4 用本地 mock HTTP，AL2 用内存 fake，AL1 用 fake 核心层。

从 2026-09-16 起，这里还承载**真实链路夹具**（`home` / `client` / `services_for`）：
AL1、AL5、CLI 与价值判决四组用例共用同一套"配置 → 装配"入口，
避免每个测试文件各写一份、各自漂移（设计评审 P1-6）。
"""

from __future__ import annotations

import faulthandler
import json
import os
from pathlib import Path

import httpx
import pytest

from artifact_spirit.config import save
from artifact_spirit.provider import ArtifactSpiritProvider
from artifact_spirit.runtime import start
from artifact_spirit.store import MemoryRecord, SQLiteBackend

# --------------------------------------------------------------------------- #
# 卡死看门狗（AS_TEST_TIMEOUT，单位秒；纯标准库，零依赖）
# --------------------------------------------------------------------------- #
#
# 症状驱动：合跑全量时曾出现"某条用例永久阻塞、整轮跑一小时不结束"，而该用例
# **单独跑 2 秒通过**——即进程内残留（线程 / 锁 / 连接）问题，且只有顺序执行才暴露。
# 没有看门狗时，这种缺陷的表现是"无限等待"，无法定位；有了它，超时即打印
# **全部线程栈**并让进程退出（exit=True），把"一小时"压缩成"一次超时窗口"。
#
# 默认关闭（避免慢机器误报）；需要时：`$env:AS_TEST_TIMEOUT="60"`。
#
# 落点必须**绕开 pytest 的 fd 级捕获**（`--capture=fd` 会把 fd 2 换成临时文件）：
# 超时是 `exit=True`——进程立刻退出，pytest 来不及回放被捕获的 stderr，
# 于是"栈打出来了但没人看得见"。这个坑本身就把一次定位拖成了两轮排查，
# 所以栈一律写进显式文件（`AS_TEST_DUMP_PATH` 可覆盖）。

_TEST_TIMEOUT = float(os.environ.get("AS_TEST_TIMEOUT") or 0)
_DUMP_SINK = None


def _dump_sink():
    global _DUMP_SINK
    if _DUMP_SINK is None:
        path = os.environ.get("AS_TEST_DUMP_PATH") or str(
            Path(__file__).resolve().parent.parent / ".codebuddy" / "hang_dump.txt"
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _DUMP_SINK = open(path, "ab", buffering=0)  # noqa: SIM115 - 进程级常驻，故意不关
    return _DUMP_SINK


@pytest.fixture(autouse=True)
def _hang_watchdog():
    if not _TEST_TIMEOUT:
        yield
        return
    faulthandler.dump_traceback_later(_TEST_TIMEOUT, file=_dump_sink(), exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def backend(tmp_path):
    """临时 SQLite 后端（维度 8，便于构造向量）。"""
    be = SQLiteBackend(
        str(tmp_path / "spirit.db"), embedding_dim=8, embedding_model="test-embed"
    )
    be.open()
    be.meta_set("spirit_name", "测试器灵")
    be.ensure_spirit_id()
    yield be
    be.close()


def vec(seed: float, dim: int = 8) -> list[float]:
    """构造一个可区分的确定性向量。"""
    return [(seed + i) / 100.0 for i in range(dim)]


def make_record(**kwargs) -> MemoryRecord:
    """构造记忆记录，带合理默认值。"""
    data = {
        "id": "",
        "layer": "semantic",
        "type": "fact",
        "content": "默认内容",
    }
    data.update(kwargs)
    return MemoryRecord(**data)


@pytest.fixture
def record_factory():
    return make_record


@pytest.fixture
def vec_factory():
    return vec


# --------------------------------------------------------------------------- #
# 把 WriteIntent 落到真实存储（AL2 测试用的"迷你 writer"）
# --------------------------------------------------------------------------- #
#
# AL2 本身不落库（R5），它的产出是意图。为了验证"意图在真实存储上产生正确结果"，
# 测试需要一个把意图翻译成 AL3 调用的执行器——这与 AL5 writer 的职责相同，
# 但在这里刻意保持极简、同步、无队列，以便断言。AL5 的 writer 会有独立的测试。


class RecordingEmbedding:
    """确定性的假 embedding——同样的文本永远得到同样的向量。"""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.model = "recording-embed"
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        seed = (sum(ord(c) for c in text) % 7) + 1
        return [(seed + i) / 100.0 for i in range(self.dim)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]


class ScriptedLLM:
    """按脚本返回结构化输出的假 LLM。

    ``script`` 是 ``(触发子串 → 返回值)`` 的有序列表；命中即返回。
    支持返回异常实例来表示"这次调用失败"。

    ``complete`` 与 ``complete_json`` 用**不同的默认值**：
    自然语言补全返回 ``default_text``，结构化补全返回 ``{"memories": []}``。
    混用会让"概览生成"意外拿到提取结果——那是测试替身的 bug，不是被测代码的。
    """

    def __init__(
        self,
        script: list[tuple[str, object]] | None = None,
        *,
        default_text: str = "（概览：要点摘要）",
    ) -> None:
        self.script = script or []
        self.default_text = default_text
        self.calls: list[str] = []

    def _lookup(self, text: str) -> object | None:
        for needle, value in self.script:
            if needle in text:
                return value
        return None

    def complete(self, *, messages, model=None, timeout=None, temperature=None) -> str:
        self.calls.append("complete")
        value = self._lookup(_last_content(messages))
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else self.default_text

    def complete_json(self, *, messages, schema, model=None, timeout=None) -> dict:
        self.calls.append("complete_json")
        value = self._lookup(_last_content(messages))
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, dict) else {"memories": []}  # type: ignore[return-value]


def _last_content(messages) -> str:
    last = messages[-1]
    return last["content"] if isinstance(last, dict) else last.content


def write_intents(backend, intents, embedding=None) -> None:
    """把写意图落到存储。**只做翻译，不做决策**（与 AL5 writer 同一契约）。"""
    for intent in intents:
        handle_intent(backend, intent, embedding)


def handle_intent(backend, intent, embedding=None) -> None:
    op = intent.op
    if op == "put" and intent.record is not None:
        vector = None
        if embedding is not None:
            vector = embedding.embed([intent.embed_text or intent.record.content])[0]
        backend.put(intent.record, vector, audit=intent.audit)
    elif op == "update":
        backend.update(intent.mem_id, intent.patch or {}, audit=intent.audit)
    elif op == "set_status":
        backend.set_status(
            intent.mem_id, intent.status, reason=intent.reason or "", actor=intent.actor
        )
    elif op == "touch":
        backend.touch(intent.mem_id, intent.ts, strength=intent.strength)
    elif op == "link":
        backend.link(
            intent.a_kind, intent.a_id, intent.b_kind, intent.b_id,
            intent.rel_type, intent.weight,
        )
    elif op == "reinforce":
        backend.reinforce(
            intent.a_id, intent.b_id, intent.delta, rel_type=intent.rel_type or "co_activation"
        )
    elif op == "forget":
        backend.hard_delete(
            intent.mem_id,
            reason=intent.reason or "",
            purge_snapshot=intent.purge_snapshot,
            actor=intent.actor,
        )
    elif op == "audit" and intent.audit is not None:
        backend.audit(intent.audit)
    elif op == "overview_invalidate":
        backend.overview_invalidate(scope_kind=intent.scope_kind, scope_id=intent.scope_id)
    elif op == "overview_put":
        backend.overview_put(
            intent.scope_kind, intent.scope_id, intent.overview_level,
            intent.overview_content, token_count=intent.token_count, model=intent.model,
        )
    elif op == "wm_put":
        backend.wm_put(
            intent.session_id, intent.chunk_key, intent.content, intent.salience
        )
    elif op == "wm_delete":
        backend.wm_clear(intent.session_id)
    elif op == "session_create":
        backend.session_create(intent.session_id, intent.ts)
    elif op == "session_end":
        backend.session_end(intent.session_id, intent.ts)
    elif op == "entity_upsert":
        backend.entity_upsert(intent.entity_name, intent.entity_type, aliases=list(intent.aliases))
    elif op == "restore":
        backend.restore_from_audit(intent.audit_id)
    else:  # pragma: no cover - 未知意图应当立刻暴露
        raise AssertionError(f"未处理的写意图：{op}")


@pytest.fixture
def embedding() -> RecordingEmbedding:
    return RecordingEmbedding()


@pytest.fixture
def write():
    return write_intents


# --------------------------------------------------------------------------- #
# 真实链路夹具：AL1 / AL5 / CLI / 价值判决共用的"配置 → 装配"入口
# --------------------------------------------------------------------------- #
#
# 唯一被替换的是**网络出口**（本地 mock 网关）：价值判决（V1/V2/V3）
# 必须在真实链路上做，替身只用来隔离单层（ENC-000 §5）。
# 所以这组夹具里没有假核心、假存储——只有假网关。


def _memory_payload(subject="用户", content="用户偏好深色主题", abstract="偏好深色主题"):
    return {
        "memories": [
            {
                "type": "preference",
                "layer": "semantic",
                "subject": subject,
                "predicate": "prefers",
                "object": "深色主题",
                "content": content,
                "abstract": abstract,
                "confidence": 0.92,
                "salience": 0.7,
            }
        ]
    }


def gateway_handler(request: httpx.Request) -> httpx.Response:
    """本地 mock 网关：`/chat/completions` 给提取结果，其余当 embedding 用。"""
    body = json.loads(request.content)
    if request.url.path.endswith("/chat/completions"):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(_memory_payload(), ensure_ascii=False)
                        }
                    }
                ]
            },
        )
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": index, "embedding": vec(1.0)}
                for index, _ in enumerate(body["input"])
            ]
        },
    )


def write_config(home: str, **overrides) -> None:
    values = {
        "spirit.name": "测试器灵",
        "backend.path": str(Path(home) / "spirit.db"),
        "models.embedding.base_url": "https://gw.example/v1",
        "models.embedding.model": "test-embed",
        "models.embedding.dim": 8,
        "models.embedding.api_key_env": "TEST_KEY",
        "models.llm.base_url": "https://gw.example/v1",
        "models.llm.api_key_env": "TEST_KEY",
        "models.llm.extract": "test-model",
    }
    values.update(overrides)
    save(home, values)


@pytest.fixture
def home(tmp_path) -> str:
    folder = tmp_path / "hermes_home"
    folder.mkdir()
    write_config(str(folder))
    return str(folder)


@pytest.fixture
def client():
    """带真实 HTTP 层配置的 provider（走 mock transport）。"""

    def factory(home: str, *, start_threads: bool = False, **kwargs) -> ArtifactSpiritProvider:
        provider = ArtifactSpiritProvider()
        provider.initialize(
            hermes_home=home,
            env={"TEST_KEY": "k"},
            start_threads=start_threads,
            # 出口必须在**装配时**给出：`start()` 会立刻建好 embedding/llm 客户端，
            # 之后再改 `resolver._transport` 是死代码——客户端已缓存，请求照样打真网络
            # （症状是单测静默变慢、偶发超时，而不是报错）。见 DES-REV-008 P1-49。
            transport=httpx.MockTransport(gateway_handler),
            **kwargs,
        )
        return provider

    return factory


def services_for(home: str, *, start_threads: bool = False):
    """装配一套真实服务（网络出口换成 mock 网关）。"""
    return start(
        home,
        env={"TEST_KEY": "k"},
        transport=httpx.MockTransport(gateway_handler),
        start_threads=start_threads,
    )


def call_tool(provider, name: str, args: dict | None = None, **kwargs):
    """按宿主方式调用工具并解析 JSON 串。

    宿主契约要求 `handle_tool_call` 返回 **JSON 字符串**（`MemoryManager` 的文档原文：
    "returns a JSON string (tool_error on failure)"）。测试也必须像宿主那样先解析——
    直接下标取值会 TypeError。这个助手让"契约"在测试里是显式的，而不是被默认掉。
    """
    raw = provider.handle_tool_call(name, args or {}, **kwargs)
    assert isinstance(raw, str), f"handle_tool_call 必须返回 str，实际 {type(raw).__name__}"
    return json.loads(raw)
