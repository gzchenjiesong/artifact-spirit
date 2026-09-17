"""OpenAI 兼容协议的 HTTP 客户端（T-AL4-02 / 03 / 04 / 05 / 10 / 11）。

**不引入 `openai` SDK**——直接用 `httpx`。理由（ADR-003 / ENC-000 §3.2）：
SDK 会带来版本漂移与额外的依赖面，而器灵用到的只是两个端点。

TokenHub（以及任何 OpenAI 兼容网关）的形态：

    POST {base_url}/chat/completions   → choices[0].message.content
    POST {base_url}/embeddings         → data[i].embedding

三条纪律：
1. **超时必须显式传入**（LLM 60s / embedding 30s）——不依赖 httpx 默认值
2. **连接类错误重试 2 次；4xx 一律不重试**（配置/配额问题，重试没有意义）
3. **密钥永不进日志或异常文本**（T-AL4-11）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Self

import httpx

from .base import (
    ChatMessage,
    EmbeddingError,
    LLMError,
    ProviderUnavailableError,
    SchemaViolationError,
    validate_schema,
)

__all__ = ["DEFAULT_TIMEOUT_EMBED", "DEFAULT_TIMEOUT_LLM", "OpenAICompatClient", "OpenAICompatSettings"]

DEFAULT_TIMEOUT_LLM = 60.0
DEFAULT_TIMEOUT_EMBED = 30.0
DEFAULT_EMBED_BATCH = 32
DEFAULT_MAX_RETRIES = 2
SINGLE_TEXT_MAX_CHARS = 2000
"""超过该长度的单条文本**单独成批**——避免拖累同批其他请求。"""


@dataclass(slots=True)
class OpenAICompatSettings:
    """客户端设置。

    ``api_key`` 是**已解析出的密钥值**（来自环境变量）。它的 ``repr`` 被刻意屏蔽，
    以免密钥通过日志/断言输出泄漏。
    """

    base_url: str
    api_key: str = ""
    timeout_llm: float = DEFAULT_TIMEOUT_LLM
    timeout_embed: float = DEFAULT_TIMEOUT_EMBED
    embed_batch_size: int = DEFAULT_EMBED_BATCH
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = 0.5
    """指数退避基数（秒）。测试传 0 以免拖慢。"""

    extra_headers: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - 简单分支
        return (
            f"OpenAICompatSettings(base_url={self.base_url!r}, "
            f"api_key={'***' if self.api_key else '<empty>'}, "
            f"timeout_llm={self.timeout_llm}, timeout_embed={self.timeout_embed}, "
            f"embed_batch_size={self.embed_batch_size})"
        )

    __str__ = __repr__


class OpenAICompatClient:
    """OpenAI 兼容客户端。

    Args:
        settings: 连接与超时设置。
        transport: 可注入的 httpx transport（测试用 ``httpx.MockTransport``）。
        client: 直接注入一个 ``httpx.Client``（优先级高于 ``transport``）。
    """

    def __init__(
        self,
        settings: OpenAICompatSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self._client = client or httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(
                max(settings.timeout_llm, settings.timeout_embed), connect=10.0
            ),
        )
        self._owns_client = client is None
        self._requests = 0

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def request_count(self) -> int:
        """已发出的 HTTP 请求数（测试断言重试次数用）。"""
        return self._requests

    # ------------------------------------------------------------------ #
    # URL 与请求
    # ------------------------------------------------------------------ #

    @property
    def _base(self) -> str:
        """归一化 base_url——去掉尾斜杠，避免出现 ``//chat/completions``（C10）。"""
        return self.settings.base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        headers.update(self.settings.extra_headers)
        return headers

    def _post(self, path: str, payload: dict, *, timeout: float) -> dict:
        """带重试策略的 POST。

        - 连接类错误（``TransportError`` / 超时）→ 指数退避重试 ``max_retries`` 次
        - 任何 4xx → **不重试**，按语义分类抛出
        - 5xx → 重试，耗尽后抛 ``ProviderUnavailableError``
        """
        url = f"{self._base}{path}"
        attempts = self.settings.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                self._requests += 1
                response = self._client.post(
                    url, json=payload, headers=self._headers(), timeout=timeout
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(self.settings.backoff_base * (2**attempt))
                    continue
                raise ProviderUnavailableError(
                    f"无法连接模型服务 {self._base}{path}（已重试 "
                    f"{self.settings.max_retries} 次）：{type(exc).__name__}"
                ) from exc

            if 400 <= response.status_code < 500:
                raise self._classify_4xx(response, path)

            if response.status_code >= 500:
                last_error = LLMError(f"服务端错误 {response.status_code}")
                if attempt < attempts - 1:
                    time.sleep(self.settings.backoff_base * (2**attempt))
                    continue
                raise ProviderUnavailableError(
                    f"模型服务返回 {response.status_code}（已重试 "
                    f"{self.settings.max_retries} 次）"
                ) from last_error

            try:
                return response.json()
            except ValueError as exc:
                raise LLMError(f"响应不是合法 JSON：{response.text[:200]}") from exc

        raise ProviderUnavailableError(  # pragma: no cover - 循环内已抛
            f"模型服务不可用：{last_error}"
        )

    @staticmethod
    def _classify_4xx(response: httpx.Response, path: str) -> Exception:
        """4xx 分类：鉴权/配额 → 不可用；其余 → 请求错误。**均不重试**。"""
        body = response.text[:200]
        if response.status_code in (401, 403):
            return ProviderUnavailableError(
                f"模型服务鉴权失败（{response.status_code}）。"
                "请检查 api_key_env 指向的环境变量是否已设置。"
            )
        if response.status_code == 429:
            return ProviderUnavailableError(
                f"模型服务配额或频率受限（429）。路径 {path}。"
            )
        return LLMError(f"模型服务返回 {response.status_code}：{body}")

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #

    def complete(
        self,
        *,
        messages: list[ChatMessage] | list[dict],
        model: str,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> str:
        """单轮补全，返回文本内容。"""
        payload: dict[str, Any] = {
            "model": model,
            "messages": _as_dicts(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature

        data = self._post(
            "/chat/completions",
            payload,
            timeout=timeout if timeout is not None else self.settings.timeout_llm,
        )
        return _extract_content(data)

    def complete_json(
        self,
        *,
        messages: list[ChatMessage] | list[dict],
        model: str,
        schema: dict,
        timeout: float | None = None,
    ) -> dict:
        """结构化输出：解析 → 校验 → **失败重试恰好 1 次** → 仍失败抛错。

        重试时追加"请严格按 schema 输出"的提示（T-AL4-04）。用 ``json.loads`` 解析，
        **禁止 `eval`**（C5）。
        """
        request_messages = list(messages)
        last_error = ""

        for attempt in range(2):
            payload: dict[str, Any] = {
                "model": model,
                "messages": _as_dicts(request_messages),
                "response_format": {"type": "json_object"},
            }
            data = self._post(
                "/chat/completions",
                payload,
                timeout=timeout if timeout is not None else self.settings.timeout_llm,
            )
            raw = _extract_content(data)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                last_error = f"不是合法 JSON：{exc}"
            else:
                errors = validate_schema(parsed, schema)
                if not errors:
                    return parsed
                last_error = "不符合 schema：" + "; ".join(errors[:5])

            if attempt == 0:
                request_messages = [
                    *_as_dicts(request_messages),
                    {
                        "role": "user",
                        "content": (
                            "上一次输出无法通过校验（"
                            + last_error
                            + "）。请严格按以下 JSON Schema 输出，"
                            "只输出 JSON，不要任何解释或代码块标记：\n"
                            + json.dumps(schema, ensure_ascii=False, indent=2)
                        ),
                    },
                ]

        raise SchemaViolationError(f"结构化输出两次均失败：{last_error}")

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #

    def embed(
        self, texts: list[str], *, model: str, timeout: float | None = None
    ) -> list[list[float]]:
        """批量向量化。

        **返回顺序与入参严格一致**——错配是灾难性的静默 bug（C3），
        故在函数末尾显式断言长度，并在拼装时按返回的 ``index`` 归位。
        """
        if not texts:
            return []

        vectors: list[list[float] | None] = [None] * len(texts)
        for start, end in self._batches(texts):
            chunk = texts[start:end]
            data = self._post(
                "/embeddings",
                {"model": model, "input": chunk},
                timeout=timeout if timeout is not None else self.settings.timeout_embed,
            )
            rows = data.get("data") or []
            if len(rows) != len(chunk):
                raise EmbeddingError(
                    f"embedding 返回数量与入参不符：期望 {len(chunk)}，收到 {len(rows)}"
                )
            for offset, row in enumerate(rows):
                # 优先按服务端 index 归位；缺失时退化为顺序归位
                index = row.get("index", offset)
                if not isinstance(index, int) or not 0 <= index < len(chunk):
                    index = offset
                vectors[start + index] = list(row["embedding"])

        if any(v is None for v in vectors):  # pragma: no cover - 防御性
            raise EmbeddingError("embedding 返回结果不完整")
        return [v for v in vectors if v is not None]

    def _batches(self, texts: list[str]) -> list[tuple[int, int]]:
        """切批：普通文本按 ``embed_batch_size`` 聚合；超长文本单独成批。"""
        size = max(1, self.settings.embed_batch_size)
        batches: list[tuple[int, int]] = []
        start = 0
        while start < len(texts):
            if len(texts[start]) > SINGLE_TEXT_MAX_CHARS:
                batches.append((start, start + 1))
                start += 1
                continue
            end = start
            while (
                end < len(texts)
                and end - start < size
                and len(texts[end]) <= SINGLE_TEXT_MAX_CHARS
            ):
                end += 1
            batches.append((start, end))
            start = end
        return batches


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


def _as_dicts(messages: list[ChatMessage] | list[dict]) -> list[dict]:
    out: list[dict] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            out.append(message.as_dict())
        else:
            out.append({"role": message["role"], "content": message["content"]})
    return out


def _extract_content(data: dict) -> str:
    try:
        choices = data["choices"]
    except (KeyError, TypeError) as exc:
        raise LLMError("响应缺少 choices 字段") from exc
    if not choices:
        raise LLMError("响应 choices 为空")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        # 部分网关把推理内容放在 reasoning_content，正文可能为空
        content = message.get("reasoning_content") or ""
    if not isinstance(content, str):
        raise LLMError(f"响应 content 类型异常：{type(content).__name__}")
    return content
