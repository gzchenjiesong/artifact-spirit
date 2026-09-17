"""配置数据模型（LLD-AL5 §2.2 · T-AL5-01）。

**密钥纪律**：配置里只出现 ``api_key_env``（**变量名**），永不出现密钥本身。
这条纪律由 :func:`artifact_spirit.config.loader.save` 的白名单机制强制——
"不写密钥"不能只靠自觉，因为迟早会有人在图省事时把 key 贴进配置文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CONFIG_FILENAME",
    "DEFAULT_DB_RELPATH",
    "ENV_PREFIX",
    "BackendConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "SpiritConfig",
]

CONFIG_FILENAME = "artifact-spirit.toml"
DEFAULT_DB_RELPATH = "spirit/spirit.db"
ENV_PREFIX = "ARTIFACT_SPIRIT_"

DEFAULT_RECALL: dict = {
    "top_k": 8,
    "token_budget": 2000,
    "candidate_k": 24,
    "weights": {
        "semantic": 0.40,
        "importance": 0.20,
        "recency": 0.15,
        "entity": 0.10,
        "diffusion": 0.10,
        "core": 0.05,
    },
}

DEFAULT_SALIENCE: dict = {
    "threshold": 0.35,
    # 无向量时的门槛。None = 按 threshold×(1−novelty 权重) 自动推算。
    "degraded_threshold": None,
    "weights": {
        "novelty": 0.30,
        "instruction": 0.25,
        "entity": 0.20,
        "emotion": 0.15,
        "core_deviation": 0.10,
    },
}

DEFAULT_DECAY: dict = {
    "enabled": False,
    "w": 0.6,
    "tau_fast": 7.0,
    "beta": 0.5,
    # 注意：D-16/D-17 之后**不再有 θ_forget 删除阈值**——衰减只影响排序
}

DEFAULT_WORKER: dict = {
    "write_queue_max": 1000,
    "maintenance_interval_min": 30,
    "prefetch_timeout_ms": 300,
    "optimize_autonomous": False,
}


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """存储后端配置。"""

    kind: str = "sqlite"
    path: str = ""


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Embedding 配置。

    ``api_key_env`` 是**环境变量名**（如 ``ARTIFACT_SPIRIT_API_KEY``），不是密钥。
    """

    provider: str = "tokenhub"
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    dim: int | None = None

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "dim": self.dim,
        }


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM 配置（按任务档位映射模型名）。"""

    provider: str = "tokenhub"
    base_url: str = ""
    api_key_env: str = ""
    task_models: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            **self.task_models,
        }


@dataclass(frozen=True, slots=True)
class SpiritConfig:
    """器灵的完整配置。"""

    name: str = ""
    id: str = ""
    hermes_home: str = ""
    backend: BackendConfig = field(default_factory=BackendConfig)
    embedding: EmbeddingConfig | None = None
    llm: LLMConfig | None = None
    recall: dict = field(default_factory=lambda: dict(DEFAULT_RECALL))
    salience: dict = field(default_factory=lambda: dict(DEFAULT_SALIENCE))
    decay: dict = field(default_factory=lambda: dict(DEFAULT_DECAY))
    worker: dict = field(default_factory=lambda: dict(DEFAULT_WORKER))
    source: str = "defaults"
    """配置来源：``file`` / ``defaults``。让 `doctor` 能说清"读到了哪份配置"。"""

    # ------------------------------------------------------------------ #

    @property
    def db_path(self) -> str:
        """数据库路径（默认 ``{hermes_home}/spirit/spirit.db``，INV-9）。"""
        if self.backend.path:
            return self.backend.path
        return str(Path(self.hermes_home) / DEFAULT_DB_RELPATH)

    @property
    def embedding_dim(self) -> int:
        """向量维度。未显式配置时按已知模型推断（表由 AL4 的**协议层**给出）。

        只依赖 ``model.base``（契约数据），不碰 ``model.resolver``（实现，含 httpx）——
        AL5 的配置层不该把 AL4 的实现拖进来（R9：层内实现模块不可跨层）。
        """
        if self.embedding and self.embedding.dim:
            return int(self.embedding.dim)
        from ..model.base import KNOWN_EMBEDDING_DIMS

        model = self.embedding.model if self.embedding else ""
        return KNOWN_EMBEDDING_DIMS.get(model, 2560)

    def toml_payload(self) -> dict:
        """序列化为可写盘的 TOML 结构（**不含任何密钥**）。"""
        payload: dict = {
            "spirit": {"name": self.name, "id": self.id},
            "backend": {"kind": self.backend.kind, "path": self.backend.path},
            "recall": self.recall,
            "salience": self.salience,
            "decay": self.decay,
            "worker": self.worker,
        }
        models: dict = {}
        if self.embedding:
            models["embedding"] = self.embedding.as_dict()
        if self.llm:
            models["llm"] = self.llm.as_dict()
        if models:
            payload["models"] = models
        return payload
