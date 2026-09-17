"""配置装载、校验与落盘（T-AL5-01 / 02）。

装载顺序（LLD-AL5 §5 M4）：

```
1. {hermes_home}/artifact-spirit.toml       ← 主配置
2. 环境变量覆盖（ARTIFACT_SPIRIT_*）
3. LLM/Embedding 缺失时 → 读宿主 Hermes 配置（hermes_home/config.yaml）
4. 仍未 → 通用环境变量（OPENAI_API_KEY / OPENAI_BASE_URL）
```

第 3、4 段不在这里实现——它们属于**模型层的 fallback 链**（AL4 的 ``ModelResolver``）。
配置层的职责是如实地说出"器灵自己配了什么"，把"没配时怎么办"留给 AL4。
这样两条链路的职责不重叠，也不会互相打架。

用标准库 ``tomllib`` 读取，**不引第三方 TOML 库**（ENC-000 §3.2 依赖最小化）。
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path

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
    "SAVE_WHITELIST",
    "ConfigError",
    "config_path",
    "load",
    "load_file",
    "mirror_path",
    "mirror_paths",
    "save",
    "toml_example",
    "validate",
]


class ConfigError(Exception):
    """配置缺失或非法。**错误信息必须可操作**（C5）——指出具体字段与期望值。"""


SAVE_WHITELIST = frozenset(
    {
        "spirit.name",
        "spirit.id",
        "backend.kind",
        "backend.path",
        "models.embedding.provider",
        "models.embedding.base_url",
        "models.embedding.model",
        "models.embedding.api_key_env",
        "models.embedding.dim",
        "models.llm.provider",
        "models.llm.base_url",
        "models.llm.api_key_env",
        "models.llm.extract",
        "models.llm.dedup",
        "models.llm.summarize",
        "models.llm.consolidate",
        "models.llm.soul",
        "recall.top_k",
        "recall.token_budget",
        "recall.candidate_k",
        "salience.threshold",
        "salience.degraded_threshold",
        "decay.enabled",
        "decay.w",
        "decay.tau_fast",
        "decay.beta",
        "worker.write_queue_max",
        "worker.maintenance_interval_min",
        "worker.prefetch_timeout_ms",
        "worker.optimize_autonomous",
    }
)
"""``save`` 的**白名单**。

只有列在这里的路径会被写盘。任何形如 ``api_key`` / ``token`` / ``secret``
的字段天然不在白名单里，因此**不可能**被写进配置文件——这是"密钥不落盘"
的机械保证，而不是一句口头约定（C7 / ADR-001）。
"""

_SECRET_WORDS = frozenset({"key", "token", "secret", "password", "credential", "auth"})

# 这些后缀把"像密钥的词"变成了别的东西——变量名、预算、上限……
_NON_SECRET_SUFFIXES = frozenset(
    {"env", "var", "name", "budget", "limit", "max", "min", "count", "size", "seconds", "ms"}
)


def _looks_like_secret(key: str) -> bool:
    """判断某个配置键是不是"疑似密钥字段"。

    判据是**最后一个词**：``api_key`` → 命中；``api_key_env`` / ``token_budget`` → 放行。
    按整串做子串匹配会把"变量名"和"预算"这类完全无害的字段一起误伤——
    那会让"密钥纪律"变成"什么都不能配"。
    """
    parts = [part for part in re.split(r"[._\-]", key.casefold()) if part]
    if not parts:
        return False
    if parts[-1] in _NON_SECRET_SUFFIXES:
        return False
    return parts[-1] in _SECRET_WORDS


def config_path(hermes_home: str | Path) -> Path:
    """配置文件路径。**一律基于传入的 hermes_home**（INV-9）。"""
    return Path(hermes_home) / CONFIG_FILENAME


def mirror_paths(hermes_home: str | Path) -> tuple[Path, Path]:
    """**宿主面板可能读写的全部镜像路径**。

    宿主有**两套**配置面板约定，落到两个不同文件（都基于 ``hermes_home``）：

    | 约定 | 宿主读写位置 | 宿主入口 |
    |---|---|---|
    | 旧式（`get_config_schema` + `save_config`） | ``{home}/artifact-spirit.json`` | `_read_memory_provider_existing_values` |
    | 声明式（``config_schema.py`` 的 ``CONFIG_SCHEMA``） | ``{home}/artifact-spirit/config.json`` | `_read_flat_json` / `_write_provider_flat` |

    两套都不认 TOML。只写一个的后果不是报错，而是"面板里改的东西刷新后不见了"——
    而且是**只有用另一套约定的那个入口才会复现**，最难查的一类。
    所以这里把两个位置都当作镜像，读写都覆盖。
    """
    home = Path(hermes_home)
    return home / "artifact-spirit.json", home / "artifact-spirit" / "config.json"


def mirror_path(hermes_home: str | Path) -> Path:
    """主镜像路径（旧式约定）。保留此函数以免调用方破坏。"""
    return mirror_paths(hermes_home)[0]


# --------------------------------------------------------------------------- #
# 装载
# --------------------------------------------------------------------------- #


def load(hermes_home: str | Path, *, env: Mapping[str, str] | None = None) -> SpiritConfig:
    """装载配置。

    配置文件缺失**不是错误**——器灵可以在"什么都没配"的状态下启动，
    那时它会走 AL4 的 fallback 链去借宿主的模型。这是"零门槛接入"的前提。
    """
    home = str(hermes_home) if hermes_home else ""
    path = config_path(home)
    data: dict = {}
    source = "defaults"

    if path.exists():
        data = load_file(path)
        source = "file"

    # 镜像（宿主面板写下的，两套约定各一个文件）作为**低优先级**来源并入：
    # TOML 里显式写了就赢——手改 TOML 永远生效。
    mirrored = False
    for mirror_path_ in mirror_paths(home):
        mirror = _read_mirror(mirror_path_)
        if mirror:
            data = _merge_dotted(mirror, data)
            mirrored = True
    if mirrored and source == "defaults":
        source = "file"

    data = _reject_secrets(data, where=str(path))
    # **先并入默认值，再套环境变量覆盖**——否则「覆盖一个默认值」这件事
    # 会因为「配置文件里没写这一项」而静默失效。
    data = _with_defaults(data)
    data = _apply_env_overrides(data, env if env is not None else os.environ)

    return _build(data, hermes_home=home, source=source)


def _read_mirror(path: Path) -> dict:
    """读宿主面板镜像。**坏文件不阻断启动**——它只是投影，不是真相源。"""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _merge_dotted(base: dict, override: dict) -> dict:
    """把点号路径的扁平字典 ``base`` 并入嵌套字典 ``override``（``override`` 优先）。"""
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in override.items()}
    for dotted, value in base.items():
        if not isinstance(dotted, str) or not dotted:
            continue
        node = merged
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node.setdefault(parts[-1], value)
    return merged


def _with_defaults(data: dict) -> dict:
    """把默认值并入原始配置，让环境变量覆盖有东西可覆盖。"""
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    for key, defaults in (
        ("recall", DEFAULT_RECALL),
        ("salience", DEFAULT_SALIENCE),
        ("decay", DEFAULT_DECAY),
        ("worker", DEFAULT_WORKER),
    ):
        merged[key] = _merge(defaults, merged.get(key))
    return merged


def load_file(path: str | Path) -> dict:
    """读取并解析 TOML，语法错误翻译成可操作提示。"""
    target = Path(path)
    try:
        return tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"配置文件语法错误：{target}\n{exc}\n"
            "提示：TOML 的字符串需要引号，表头用 [方括号]，等号两侧不能省略。"
        ) from exc
    except OSError as exc:  # pragma: no cover
        raise ConfigError(f"无法读取配置文件 {target}：{exc}") from exc


def _reject_secrets(data: dict, *, where: str) -> dict:
    """发现疑似密钥写入配置文件 → **明确报错**，而不是默默忽略。

    静默忽略会让用户以为"配上了"，然后在调用时收到 401——那更难排查。
    """
    offenders = [key for key in _walk_keys(data) if _looks_like_secret(key)]
    if offenders:
        raise ConfigError(
            f"配置文件 {where} 中出现了疑似密钥字段：{sorted(set(offenders))}。\n"
            "器灵要求密钥**只经环境变量**注入：请改写为 api_key_env = \"你的环境变量名\"，"
            "并把密钥本身放到环境变量里。"
        )
    return data


def _walk_keys(node: object, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            keys.append(str(key))
            keys.extend(_walk_keys(value, f"{prefix}{key}."))
    return keys


def _apply_env_overrides(data: dict, env: Mapping[str, str]) -> dict:
    """环境变量覆盖。

    约定：``ARTIFACT_SPIRIT_<SECTION>_<FIELD>``，如
    ``ARTIFACT_SPIRIT_RECALL_TOP_K``、``ARTIFACT_SPIRIT_BACKEND_PATH``。
    只覆盖**已存在的键**——避免环境变量凭空造出未知配置项。
    """
    mut = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    overrides = {k[len(ENV_PREFIX) :]: v for k, v in env.items() if k.startswith(ENV_PREFIX)}

    for section in ("backend", "recall", "salience", "decay", "worker", "spirit"):
        target = mut.setdefault(section, {})
        if not isinstance(target, dict):  # pragma: no cover
            continue
        for key in list(target):
            env_key = f"{section.upper()}_{key.upper()}"
            if env_key in overrides:
                target[key] = _coerce(overrides[env_key], target[key])

    models = mut.get("models")
    if isinstance(models, dict):
        for kind in ("embedding", "llm"):
            node = models.get(kind)
            if not isinstance(node, dict):
                continue
            for key in list(node):
                env_key = f"{kind.upper()}_{key.upper()}"
                if env_key in overrides:
                    node[key] = _coerce(overrides[env_key], node[key])
    return mut


def _coerce(raw: str, template: object) -> object:
    """按原值的类型转换环境变量的字符串值。"""
    if isinstance(template, bool):
        return raw.strip().casefold() in {"1", "true", "yes", "on"}
    if isinstance(template, int):
        try:
            return int(raw)
        except ValueError:
            return raw
    if isinstance(template, float):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def _build(data: dict, *, hermes_home: str, source: str) -> SpiritConfig:
    spirit = data.get("spirit") or {}
    backend_raw = data.get("backend") or {}
    models = data.get("models") or {}

    embedding = None
    if isinstance(models.get("embedding"), dict):
        node = models["embedding"]
        embedding = EmbeddingConfig(
            provider=str(node.get("provider") or "tokenhub"),
            base_url=str(node.get("base_url") or ""),
            model=str(node.get("model") or ""),
            api_key_env=str(node.get("api_key_env") or ""),
            dim=int(node["dim"]) if node.get("dim") else None,
        )

    llm = None
    if isinstance(models.get("llm"), dict):
        node = models["llm"]
        llm = LLMConfig(
            provider=str(node.get("provider") or "tokenhub"),
            base_url=str(node.get("base_url") or ""),
            api_key_env=str(node.get("api_key_env") or ""),
            task_models={
                task: str(node[task]) for task in ("extract", "dedup", "summarize", "consolidate", "soul")
                if node.get(task)
            },
        )

    return SpiritConfig(
        name=str(spirit.get("name") or ""),
        id=str(spirit.get("id") or ""),
        hermes_home=hermes_home,
        backend=BackendConfig(
            kind=str(backend_raw.get("kind") or "sqlite"),
            path=str(backend_raw.get("path") or ""),
        ),
        embedding=embedding,
        llm=llm,
        recall=_merge(DEFAULT_RECALL, data.get("recall")),
        salience=_merge(DEFAULT_SALIENCE, data.get("salience")),
        decay=_merge(DEFAULT_DECAY, data.get("decay")),
        worker=_merge(DEFAULT_WORKER, data.get("worker")),
        source=source,
    )


def _merge(defaults: dict, override: object) -> dict:
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in defaults.items()}
    if isinstance(override, dict):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #


def validate(cfg: SpiritConfig) -> list[str]:
    """返回问题列表（空 = 通过）。

    **警告与错误分不清的时候，一律当作警告**——器灵宁可带着可疑配置跑起来并在
    `status` 里显示告警，也不要因为一个可调参数而拒绝启动。
    真正阻断启动的只有"跑不起来"的问题。
    """
    problems: list[str] = []

    queue_max = cfg.worker.get("write_queue_max")
    if not isinstance(queue_max, int) or queue_max <= 0:
        problems.append(
            f"worker.write_queue_max 必须是正整数，当前为 {queue_max!r}（C11：队列必须有安全上限）"
        )

    weights = cfg.recall.get("weights")
    if isinstance(weights, dict) and weights:
        total = sum(float(v) for v in weights.values())
        if abs(total - 1.0) > 0.05:
            problems.append(
                f"警告：recall.weights 合计为 {total:.3f}，偏离 1.0 较多——"
                "融合分数会被归一化，但仍建议校准（不影响运行）"
            )

    top_k = cfg.recall.get("top_k")
    if not isinstance(top_k, int) or top_k <= 0:
        problems.append(f"recall.top_k 必须是正整数，当前为 {top_k!r}")

    threshold = cfg.salience.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        problems.append(
            f"salience.threshold 必须落在 [0,1]，当前为 {threshold!r}"
        )

    interval = cfg.worker.get("maintenance_interval_min")
    if not isinstance(interval, (int, float)) or float(interval) <= 0:
        problems.append(
            f"worker.maintenance_interval_min 必须为正数（分钟），当前为 {interval!r}"
        )

    if cfg.embedding and cfg.embedding.model:
        dim = cfg.embedding.dim
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            problems.append(
                f"models.embedding.dim 必须是正整数，当前为 {dim!r}"
            )

    if cfg.backend.kind not in ("sqlite",):
        problems.append(
            f"backend.kind 目前仅支持 'sqlite'，当前为 {cfg.backend.kind!r}"
            "（后端可插拔；新增后端须先通过 ADR-001 的许可准入）"
        )

    return problems


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #


def save(hermes_home: str | Path, values: Mapping[str, object]) -> Path:
    """按**白名单**写入配置。非白名单字段一律忽略，密钥天然不可能落盘（C7）。"""
    path = config_path(hermes_home)
    existing = load_file(path) if path.exists() else {}

    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in existing.items()}
    for dotted, value in values.items():
        if dotted not in SAVE_WHITELIST:
            continue
        target = merged
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = target.get(part)
            if not isinstance(node, dict):
                node = {}
                target[part] = node
            target = node
        target[parts[-1]] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_toml(merged), encoding="utf-8")

    # 同步宿主面板镜像（两套约定各一个文件）：面板不认 TOML，只写 TOML 会导致
    # "保存后字段回弹"。只镜像白名单内、且确实被写过的键，保持投影最小。
    payload = json.dumps(
        {k: v for k, v in values.items() if k in SAVE_WHITELIST and v is not None},
        ensure_ascii=False,
        indent=2,
    )
    for mirror in mirror_paths(path.parent):
        try:
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.write_text(payload, encoding="utf-8")
        except OSError:  # pragma: no cover - 镜像写失败不影响真相源
            pass
    return path


def _dump_toml(data: Mapping[str, object]) -> str:
    """极简 TOML 序列化（只处理本配置用到的形态：表、标量、内联表）。"""
    lines: list[str] = ["# 器灵（Artifact Spirit）配置", "# 密钥不入此文件，只写环境变量名", ""]

    def _emit_table(name: str, node: Mapping[str, object]) -> None:
        scalars = {k: v for k, v in node.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in node.items() if isinstance(v, dict)}
        lines.append(f"[{name}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_fmt(value)}")
        lines.append("")
        for key, value in tables.items():
            _emit_table(f"{name}.{key}", value)

    for key, value in data.items():
        if isinstance(value, dict):
            _emit_table(key, value)
        else:
            lines.append(f"{key} = {_fmt(value)}")
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{k} = {_fmt(v)}" for k, v in value.items())
        return "{ " + inner + " }"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


# --------------------------------------------------------------------------- #
# 模板
# --------------------------------------------------------------------------- #


def toml_example() -> str:
    """``aspirit init`` 生成的注释完整模板。"""
    return '''# 器灵（Artifact Spirit）配置
# 生成自 `aspirit init`；可用环境变量覆盖任意一项：
#   ARTIFACT_SPIRIT_RECALL_TOP_K / ARTIFACT_SPIRIT_BACKEND_PATH / ...

[spirit]
name = ""            # 器灵名（aspirit init 写入）
id   = ""            # ULID，首次初始化时生成，之后不再变更

[backend]
kind = "sqlite"      # 目前仅支持 sqlite
path = ""            # 留空则用 {hermes_home}/spirit/spirit.db

[models.embedding]
provider    = "tokenhub"
base_url    = "https://tokenhub.tencentmaas.com/v1"
model       = "kinfra-text-embedding-4b"
api_key_env = "ARTIFACT_SPIRIT_API_KEY"   # 只写环境变量名，密钥永不入文件
dim         = 2560

[models.llm]
provider    = "tokenhub"
base_url    = "https://tokenhub.tencentmaas.com/v1"
api_key_env = "ARTIFACT_SPIRIT_API_KEY"
extract     = "glm-5.3-flash"
dedup       = "glm-5.3-flash"
summarize   = "glm-5.3-flash"
consolidate = "glm-5.3"
soul        = "kimi-k3"

# 若整段 [models] 留空或 api_key_env 指向的环境变量不存在，
# 器灵会依次尝试：宿主配置 {hermes_home}/config.yaml → 环境变量 OPENAI_API_KEY

[recall]
top_k        = 8
token_budget = 2000
[recall.weights]
semantic   = 0.40
importance = 0.20
recency    = 0.15
entity     = 0.10
diffusion  = 0.10
core       = 0.05

[salience]
threshold = 0.35

[decay]
enabled   = false    # 衰减只影响排序；删除仅由记忆优化任务发起
w         = 0.6
tau_fast  = 7.0
beta      = 0.5

[worker]
write_queue_max           = 1000
maintenance_interval_min  = 30
prefetch_timeout_ms       = 300
optimize_autonomous       = false   # 治理性删除默认需要确认
'''
