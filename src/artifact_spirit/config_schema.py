"""配置面板的**声明式**描述（T-AL1-04）。

只声明、不渲染——渲染是宿主的事。

**格式由宿主决定，不是审美选择。** 宿主 `_normalize_memory_provider_schema` 里写着
``if isinstance(raw, list)``；返回 dict 的后果不是"显示得难看"，而是**配置面板一个字段都不显示**，
用户在宿主里根本无法配置器灵。所以这里产出的是**扁平字段列表**。

字段 ``key`` 沿用配置文件的点号路径（``models.llm.extract``）：宿主会把 schema 的
``key`` 原样回传给 ``save_config(values)``，而 :func:`artifact_spirit.config.save`
本就按点号路径展开写入——两边天然对齐，中间不需要任何转换层。

**一个 secret 字段都没有**，这是有意的：器灵的密钥只走环境变量，
面板里只填"变量名"。所以"哪些字段是密钥"这个问题在器灵这里不存在。
"""

from __future__ import annotations

__all__ = ["CONFIG_FIELDS", "CONFIG_GROUPS", "DEFAULT_LLM_BASE_URL"]

DEFAULT_LLM_BASE_URL = "https://tokenhub.tencentmaas.com/v1"

CONFIG_FIELDS: list[dict] = [
    # ---------------- 身份 ----------------
    {
        "key": "spirit.name",
        "label": "器灵名称",
        "description": "器灵的名字。留空则用默认名。",
        "type": "text",
        "default": "",
    },
    {
        "key": "spirit.id",
        "label": "器灵 ID",
        "description": "首次初始化时生成（ULID），之后不再变更。一般不需要填。",
        "type": "text",
        "default": "",
    },
    # ---------------- 存储 ----------------
    {
        "key": "backend.kind",
        "label": "存储后端",
        "description": "目前仅支持 sqlite——单文件、零部署。",
        "type": "text",
        "default": "sqlite",
        "choices": ["sqlite"],
    },
    {
        "key": "backend.path",
        "label": "数据库路径",
        "description": "留空则使用 {hermes_home}/spirit/spirit.db",
        "type": "text",
        "default": "",
    },
    # ---------------- Embedding ----------------
    {
        "key": "models.embedding.base_url",
        "label": "Embedding 服务地址",
        "description": "OpenAI 兼容网关地址，例如 TokenHub。留空则召回降级为关键词（BM25）模式。",
        "type": "text",
        "default": DEFAULT_LLM_BASE_URL,
    },
    {
        "key": "models.embedding.model",
        "label": "Embedding 模型",
        "description": "写入与检索必须用同一个模型；换模型后需执行 aspirit reindex 全库重嵌入。",
        "type": "text",
        "default": "kinfra-text-embedding-4b",
    },
    {
        "key": "models.embedding.api_key_env",
        "label": "Embedding 密钥的变量名",
        "description": "这里填**环境变量名**，不是密钥本身——密钥永不写入配置文件。",
        "type": "text",
        "default": "ARTIFACT_SPIRIT_API_KEY",
    },
    {
        "key": "models.embedding.dim",
        "label": "Embedding 维度",
        "description": "已知模型可留空；未知模型必须显式填写——维度猜错会让全库向量失效。",
        "type": "integer",
        "default": 2560,
        "minimum": 1,
        "maximum": 65536,
    },
    # ---------------- LLM ----------------
    {
        "key": "models.llm.base_url",
        "label": "LLM 服务地址",
        "description": "留空则依次尝试宿主配置与通用环境变量。",
        "type": "text",
        "default": DEFAULT_LLM_BASE_URL,
    },
    {
        "key": "models.llm.api_key_env",
        "label": "LLM 密钥的变量名",
        "description": "同样是变量名，不是密钥。",
        "type": "text",
        "default": "ARTIFACT_SPIRIT_API_KEY",
    },
    {
        "key": "models.llm.extract",
        "label": "提取模型",
        "description": "高频任务（每轮对话都可能调用），建议用便宜的小模型。",
        "type": "text",
        "default": "glm-5.3-flash",
    },
    {
        "key": "models.llm.dedup",
        "label": "去重模型",
        "description": "判断这条记忆是不是已经记过了。",
        "type": "text",
        "default": "glm-5.3-flash",
    },
    {
        "key": "models.llm.summarize",
        "label": "摘要模型",
        "description": "生成 L1 概览。",
        "type": "text",
        "default": "glm-5.3-flash",
    },
    {
        "key": "models.llm.consolidate",
        "label": "巩固模型",
        "description": "会话结束后的离线整理，低频、可用大模型。",
        "type": "text",
        "default": "glm-5.3",
    },
    {
        "key": "models.llm.soul",
        "label": "核心记忆模型",
        "description": "身份 / soul 层更新，门槛最高。",
        "type": "text",
        "default": "kimi-k3",
    },
    # ---------------- 召回 ----------------
    {
        "key": "recall.top_k",
        "label": "最多召回条数",
        "description": "单次注入上下文的记忆上限。",
        "type": "integer",
        "default": 8,
        "minimum": 1,
        "maximum": 100,
    },
    {
        "key": "recall.token_budget",
        "label": "注入预算（token）",
        "description": "超过预算的候选会被裁掉。",
        "type": "integer",
        "default": 2000,
        "minimum": 100,
        "maximum": 100000,
    },
    {
        "key": "recall.candidate_k",
        "label": "候选池大小",
        "description": "融合排序前的候选数量。",
        "type": "integer",
        "default": 24,
        "minimum": 5,
        "maximum": 500,
    },
    # ---------------- 显著性 ----------------
    {
        "key": "salience.threshold",
        "label": "落库门槛",
        "description": "低于门槛的内容只进工作记忆、不写长期记忆。",
        "type": "number",
        "default": 0.35,
        "minimum": 0.0,
        "maximum": 1.0,
        "step": 0.01,
    },
    {
        "key": "salience.degraded_threshold",
        "label": "无向量时的落库门槛",
        "description": "embedding 不可用时使用的门槛。留空则按 门槛×(1−新颖度权重) 自动推算。",
        "type": "number",
        "default": 0.245,
        "minimum": 0.0,
        "maximum": 1.0,
        "step": 0.005,
    },
    # ---------------- 衰减 ----------------
    {
        "key": "decay.enabled",
        "label": "启用周期衰减",
        "description": "衰减**只影响排序**，不会因为「很久没想起」而删除任何记忆。",
        "type": "boolean",
        "default": False,
    },
    {
        "key": "decay.w",
        "label": "衰减快慢权重",
        "description": "Wixted 混合衰减的快衰成分权重。",
        "type": "number",
        "default": 0.6,
        "minimum": 0.0,
        "maximum": 1.0,
        "step": 0.05,
    },
    # ---------------- 运行时 ----------------
    {
        "key": "worker.write_queue_max",
        "label": "写队列上限",
        "description": "队列满时新写入会被丢弃并记日志（不阻塞宿主）。",
        "type": "integer",
        "default": 1000,
        "minimum": 10,
        "maximum": 1000000,
    },
    {
        "key": "worker.maintenance_interval_min",
        "label": "维护间隔（分钟）",
        "description": "后台维护线程的巡检周期。",
        "type": "integer",
        "default": 30,
        "minimum": 1,
        "maximum": 1440,
    },
    {
        "key": "worker.prefetch_timeout_ms",
        "label": "召回护栏（毫秒）",
        "description": "超过这个时间就返回保底内容——**绝不阻断宿主**。",
        "type": "integer",
        "default": 300,
        "minimum": 10,
        "maximum": 10000,
    },
    {
        "key": "worker.optimize_autonomous",
        "label": "允许系统自主删除",
        "description": "关闭时，治理性删除都需要人工确认。",
        "type": "boolean",
        "default": False,
    },
]

CONFIG_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("身份", ("spirit.name", "spirit.id")),
    ("存储", ("backend.kind", "backend.path")),
    (
        "Embedding",
        (
            "models.embedding.base_url",
            "models.embedding.model",
            "models.embedding.api_key_env",
            "models.embedding.dim",
        ),
    ),
    (
        "LLM",
        (
            "models.llm.base_url",
            "models.llm.api_key_env",
            "models.llm.extract",
            "models.llm.dedup",
            "models.llm.summarize",
            "models.llm.consolidate",
            "models.llm.soul",
        ),
    ),
    ("召回", ("recall.top_k", "recall.token_budget", "recall.candidate_k")),
    ("显著性", ("salience.threshold", "salience.degraded_threshold")),
    ("衰减（只影响排序）", ("decay.enabled", "decay.w")),
    (
        "运行时",
        (
            "worker.write_queue_max",
            "worker.maintenance_interval_min",
            "worker.prefetch_timeout_ms",
            "worker.optimize_autonomous",
        ),
    ),
]


# --------------------------------------------------------------------------- #
# 声明式 schema（宿主的**第二套**面板约定）
#
# 宿主有两套并存的约定，靠一个 `surface` 参数选：
#   1. 旧式：`provider.get_config_schema() -> list[dict]`（上面的 CONFIG_FIELDS），
#      保存走 `provider.save_config()`——**这条会落到我们的 TOML**。
#   2. 声明式：本文件里的 `CONFIG_SCHEMA`，宿主**按文件路径 exec 这个模块**再取
#      `getattr(module, "CONFIG_SCHEMA")`，字段值写到 `{home}/artifact-spirit/config.json`。
#
# 两套都提供，是因为前端请求哪一套不由我们决定：只提供一套时，另一套的面板会**空着**，
# 而"面板是空的"和"插件没装好"在使用者眼里没有区别。
#
# 值的一致性由 `config/loader.py` 保证：它把这两处 JSON 都当镜像读，TOML 始终是真相源。
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - 取决于宿主是否在场
    from plugins.memory.config_schema import (  # type: ignore[import-not-found]
        KIND_BOOL,
        KIND_NUMBER,
        KIND_SELECT,
        KIND_TEXT,
        ProviderConfigSchema,
        ProviderField,
        ProviderFieldOption,
    )

    _HOST_SCHEMA_AVAILABLE = True
except Exception:
    _HOST_SCHEMA_AVAILABLE = False

    KIND_TEXT, KIND_SELECT, KIND_NUMBER, KIND_BOOL = "text", "select", "number", "bool"

    class ProviderFieldOption:
        """宿主 ``ProviderFieldOption`` 的本地替身（属性名一致）。"""

        __slots__ = ("description", "label", "value")

        def __init__(self, value: str, label: str, description: str = "") -> None:
            self.value, self.label, self.description = value, label, description

    class ProviderField:
        """宿主 ``ProviderField`` 的本地替身。"""

        __slots__ = (
            "aliases",
            "default",
            "description",
            "env_fallbacks",
            "env_key",
            "group",
            "info",
            "inline",
            "key",
            "kind",
            "label",
            "options",
            "placeholder",
            "scope",
        )

        def __init__(
            self,
            key: str,
            label: str = "",
            kind: str = KIND_TEXT,
            default: object = "",
            description: str = "",
            placeholder: str = "",
            options: tuple = (),
            env_key: str | None = None,
            inline: bool = False,
            group: str = "",
        ) -> None:
            self.key, self.label, self.kind = key, label or key.replace(".", " ").title(), kind
            self.default, self.description, self.placeholder = default, description, placeholder
            self.options, self.env_key = options, env_key
            self.aliases, self.env_fallbacks = (), ()
            self.inline, self.group, self.info, self.scope = inline, group, "", "host"

        @property
        def is_secret(self) -> bool:
            return self.kind == "secret"

        def allowed_values(self) -> set[str]:
            return {opt.value for opt in self.options}

    class ProviderConfigSchema:
        """宿主 ``ProviderConfigSchema`` 的本地替身。"""

        __slots__ = ("docs_url", "fields", "label", "name", "storage")

        def __init__(self, name: str, label: str = "", storage: str = "flat_json",
                     docs_url: str = "", fields: tuple = ()) -> None:
            self.name, self.label, self.storage = name, label or name, storage
            self.docs_url, self.fields = docs_url, fields


_KIND_BY_TYPE = {
    "text": KIND_TEXT,
    "boolean": KIND_BOOL,
    "integer": KIND_NUMBER,
    "number": KIND_NUMBER,
}


def _to_declared_field(field: dict) -> object:
    """把 :data:`CONFIG_FIELDS` 的字段转成宿主声明式字段。

    只做形状转换，不改语义——两份 schema 描述的是同一批配置项，
    漂移会让"两套面板显示不一样"，所以这张表是**单向生成**的，不手写第二份。
    """
    choices = field.get("choices") or ()
    kind = KIND_SELECT if choices else _KIND_BY_TYPE.get(str(field.get("type")), KIND_TEXT)
    return ProviderField(
        key=str(field["key"]),
        label=str(field.get("label") or ""),
        kind=kind,
        default=field.get("default", ""),
        description=str(field.get("description") or ""),
        options=tuple(ProviderFieldOption(str(c), str(c)) for c in choices),
        inline=True,
    )


CONFIG_SCHEMA = ProviderConfigSchema(
    name="artifact-spirit",
    label="器灵（Artifact Spirit）",
    docs_url="",
    fields=tuple(_to_declared_field(f) for f in CONFIG_FIELDS),
)
"""宿主的**声明式**配置面板描述（由 :data:`CONFIG_FIELDS` 单向生成，不手写第二份）。"""


__all__ += ["CONFIG_SCHEMA"]
