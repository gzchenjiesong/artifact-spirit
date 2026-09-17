"""AL4 模型层：协议（``base``）+ OpenAI 兼容客户端（``openai_compat``）+ 路由（``resolver``）。

分层规则 **R3**：本包**不得**依赖 ``core`` / ``store``，也**不得**引入 ``openai`` SDK
（直接用 httpx，见 ADR-003）。
"""

from __future__ import annotations

from .base import (
    KNOWN_EMBEDDING_DIMS,
    TASKS,
    ChatMessage,
    EmbeddingError,
    EmbeddingProvider,
    HttpTransport,
    LLMError,
    LLMProvider,
    ModelError,
    ModelRegistry,
    ProviderUnavailableError,
    SchemaValidationError,
    SchemaViolationError,
    validate_schema,
)
from .openai_compat import OpenAICompatClient, OpenAICompatSettings
from .resolver import (
    ModelResolver,
    ResolvedRoute,
    TaskEmbedding,
    TaskLLM,
    load_host_config_yaml,
)

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
    "ModelResolver",
    "OpenAICompatClient",
    "OpenAICompatSettings",
    "ProviderUnavailableError",
    "ResolvedRoute",
    "SchemaValidationError",
    "SchemaViolationError",
    "TaskEmbedding",
    "TaskLLM",
    "load_host_config_yaml",
    "validate_schema",
]
