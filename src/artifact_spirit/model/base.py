"""AL4 模型层契约：三个 Protocol + 异常族 + **无依赖的 JSON Schema 轻量校验器**。

本模块**零 I/O**（不 import ``httpx``）——它是协议层。

> 为什么校验器放在协议模块里：它被 AL4（``complete_json``）与 AL2（``extract/schema.py``）
> 同时需要，而分层规则 R1 只允许 AL2 依赖 ``store/base.py`` 与 ``model/base.py``。
> 把纯计算工具放这里，避免了"两份校验器逐渐不一致"这种典型缺陷（C12）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

__all__ = [
    "KNOWN_EMBEDDING_DIMS",
    "TASKS",
    "ChatMessage",
    "EmbeddingError",
    "EmbeddingProvider",
    "HttpTransport",
    "LLMError",
    "LLMProvider",
    "ModelError",
    "ModelRegistry",
    "ProviderUnavailableError",
    "SchemaValidationError",
    "SchemaViolationError",
    "validate_schema",
]

TASKS = ("extract", "dedup", "summarize", "consolidate", "soul", "crosscheck")
"""按用途划分的模型档位（ADR-003 §3.2）。"""

HttpTransport = Any
"""HTTP 传输的**不透明注入点**（测试用 ``httpx.MockTransport`` 拦截网络）。

刻意声明为 ``Any``：AL5 的组合根要把宿主/测试给的 transport **原样透传**给 AL4。
若这里写成 ``httpx.BaseTransport``，AL5 的公开签名（``runtime.start``）就必须
``import httpx``，从而把 AL4 的技术栈泄漏进 AL5 的 API——而 AL4 自己的设计
（LLD-AL4 §3）又特意规定 ``model/base.py`` 不得 import httpx。二者不可能同时成立。
类型正确性由 AL4 内部真正构造 ``httpx.Client`` 的那一处兜住（见 ``openai_compat``）。
"""

KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "kinfra-text-embedding-4b": 2560,
    "kinfra-text-embedding-0.6b": 1024,
}
"""已知 embedding 模型的维度（实测确认，ADR-003 §3.4）。

未知模型必须在配置里显式给出 ``dim``——**维度猜错会导致全库向量失效**。

**为什么这张表在 ``base`` 而不在 ``resolver``**：它是 AL4 的**契约数据**
（"有哪些模型、各多少维"），不是实现细节。AL5 的配置层需要按模型名推断维度
（``Config.vec_dim``），若表留在 ``resolver``，AL5 就必须 import AL4 的实现模块
（且 ``resolver`` 会连带引入 ``httpx``）——这是 R9（层内实现模块不可跨层）要拦的事。
``resolver`` 仍然 re-export 本名字，历史上从 ``resolver`` 取用的调用方不受影响。
"""


# --------------------------------------------------------------------------- #
# 异常族
# --------------------------------------------------------------------------- #


class ModelError(Exception):
    """模型层异常基类。"""


class LLMError(ModelError):
    """LLM 调用失败。"""


class SchemaViolationError(LLMError):
    """输出不是合法 JSON，或不符合给定 schema。

    AL2 见到本异常时的行为：**仅存原文** + `audit(reason='extract_parse_failed')`——
    记忆不丢，只是未结构化（F3）。
    """


class ProviderUnavailableError(LLMError):
    """provider 不可达 / 未配置 / 鉴权失败（F1）。

    AL2 见到本异常时的行为：跳过提取，仅存原文 + 告警。
    """


class EmbeddingError(ModelError):
    """embedding 调用失败（F2）。

    **绝不允许降级为 LLM 调用**（INV-6）——embedding 与 chat 是两种模型，
    拿 chat 的输出当向量是灾难性的静默错误。
    AL2 见到本异常时的行为：显著性打分置 `w1=0`；召回跳过向量路，降级 BM25。
    """


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """一条对话消息。"""

    role: Literal["system", "user", "assistant"]
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# --------------------------------------------------------------------------- #
# 协议
# --------------------------------------------------------------------------- #


class LLMProvider(Protocol):
    """文本生成协议。

    ``model`` 可省略——实现通常携带一个**默认模型**（由任务档位决定），
    这在 AL2 里很关键：核心层不该知道 ``glm-5.3-flash`` 这种具体名字（C9）。
    """

    def complete(
        self,
        *,
        messages: list[ChatMessage] | list[dict],
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> str: ...

    def complete_json(
        self,
        *,
        messages: list[ChatMessage] | list[dict],
        schema: dict,
        model: str | None = None,
        timeout: float | None = None,
    ) -> dict: ...


class EmbeddingProvider(Protocol):
    """向量化协议。"""

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。**返回顺序必须与入参严格一致**。"""
        ...


class ModelRegistry(Protocol):
    """任务 → 模型的路由与 fallback 链。"""

    def llm(self, task: str = "extract") -> LLMProvider: ...

    def embedding(self) -> EmbeddingProvider: ...

    def resolve_chain(self, task: str = "extract") -> list[dict]:
        """返回**实际生效链**（供 `aspirit status` 展示是否走了 fallback）。"""
        ...


# --------------------------------------------------------------------------- #
# JSON Schema 轻量校验（不引入 jsonschema）
# --------------------------------------------------------------------------- #


class SchemaValidationError(ValueError):
    """数据结构不符合 schema。"""


_SUPPORTED_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def validate_schema(data: Any, schema: dict, *, path: str = "$") -> list[str]:
    """按 JSON Schema 的一个**常用子集**校验，返回错误列表（空 = 通过）。

    支持：``type`` / ``properties`` / ``required`` / ``items`` / ``enum`` /
    ``anyOf`` / ``additionalProperties: false``。

    刻意**手写**而非引入 `jsonschema`：器灵的依赖越少越好（ENC-000 §3.2 禁止清单），
    而提取 schema 用到的只是这个子集。
    """
    errors: list[str] = []
    _validate(data, schema, path, errors, depth=0)
    return errors


def _validate(data: Any, schema: dict, path: str, errors: list[str], depth: int) -> None:
    if depth > 32:  # pragma: no cover - 防御性
        errors.append(f"{path}: schema 嵌套过深")
        return

    if "anyOf" in schema:
        branch_errors = [
            validate_schema(data, branch, path=path) for branch in schema["anyOf"]
        ]
        if not any(not errs for errs in branch_errors):
            errors.append(f"{path}: 不满足 anyOf 的任何一支")
        return

    expected = schema.get("type")
    if expected is not None:
        types = [expected] if isinstance(expected, str) else list(expected)
        if not _matches_any(data, types):
            errors.append(
                f"{path}: 期望类型 {'/'.join(types)}，实际 {type(data).__name__}"
            )
            return

    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: 取值必须是 {schema['enum']}，实际 {data!r}")

    if isinstance(data, dict) and (expected == "object" or "properties" in schema):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: 缺少必填字段")
        for key, sub in props.items():
            if key in data:
                _validate(data[key], sub, f"{path}.{key}", errors, depth + 1)
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in props:
                    errors.append(f"{path}.{key}: 不允许的额外字段")

    if isinstance(data, list) and "items" in schema:
        for index, item in enumerate(data):
            _validate(item, schema["items"], f"{path}[{index}]", errors, depth + 1)


def _matches_any(data: Any, types: list[str]) -> bool:
    for name in types:
        if name == "integer" and isinstance(data, bool):
            continue  # bool 是 int 的子类，但语义上不是整数
        if name == "number" and isinstance(data, bool):
            continue
        py = _SUPPORTED_TYPES.get(name)
        if py is None:
            continue  # 未知类型名 → 视为通过（宽松）
        if isinstance(data, py):
            return True
    return False
