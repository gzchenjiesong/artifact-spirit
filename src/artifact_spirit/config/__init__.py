"""AL5 配置：数据模型 + 装载 / 校验 / 落盘。

密钥纪律的核心在这层：**配置里只有 ``api_key_env``（变量名），没有密钥本身**，
且 :func:`save` 用白名单强制这一点。
"""

from __future__ import annotations

from .loader import (
    SAVE_WHITELIST,
    ConfigError,
    config_path,
    load,
    load_file,
    mirror_path,
    mirror_paths,
    save,
    toml_example,
    validate,
)
from .model import (
    CONFIG_FILENAME,
    DEFAULT_DECAY,
    DEFAULT_RECALL,
    DEFAULT_SALIENCE,
    DEFAULT_WORKER,
    ENV_PREFIX,
    BackendConfig,
    EmbeddingConfig,
    LLMConfig,
    SpiritConfig,
)

__all__ = [
    "CONFIG_FILENAME",
    "DEFAULT_DECAY",
    "DEFAULT_RECALL",
    "DEFAULT_SALIENCE",
    "DEFAULT_WORKER",
    "ENV_PREFIX",
    "SAVE_WHITELIST",
    "BackendConfig",
    "ConfigError",
    "EmbeddingConfig",
    "LLMConfig",
    "SpiritConfig",
    "config_path",
    "load",
    "load_file",
    "mirror_path",
    "mirror_paths",
    "save",
    "toml_example",
    "validate",
]
