"""任务 → 模型的路由与 fallback 链（T-AL4-06 / 07 / 08 / 09）。

四段 fallback（LLD-AL4 §5 M3）：

```
1. 器灵配置 [models.llm] 有值且 provider 可用 → 用之
2. 未配置 → 读宿主 Hermes 配置（{hermes_home}/config.yaml）
3. 仍未 → 通用环境变量（OPENAI_API_KEY / OPENAI_BASE_URL）
4. 全无 → 抛 ProviderUnavailableError
```

**为什么要有第 2、3 段**：器灵作为一个插件，最常见的使用方式是"宿主已经配好了模型，
我不想再配一遍"。若只认自己的配置，用户会被迫重复配置——这违背"零门槛接入"。

**embedding 不走这条链的 LLM 分支**（INV-6）：embedding 只能来自 embedding 配置。
拿 chat 模型冒充 embedding 是灾难性的静默错误，宁可失败。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import (
    KNOWN_EMBEDDING_DIMS,
    EmbeddingError,
    EmbeddingProvider,
    HttpTransport,
    ProviderUnavailableError,
)
from .openai_compat import OpenAICompatClient, OpenAICompatSettings

__all__ = [
    "KNOWN_EMBEDDING_DIMS",  # re-export：契约数据已上移到 base，此处保留旧取值路径
    "ModelResolver",
    "ResolvedRoute",
    "TaskEmbedding",
    "TaskLLM",
    "load_host_config_yaml",
]

DEFAULT_LLM_TASK_MODELS: dict[str, str] = {
    "extract": "glm-5.3-flash",
    "dedup": "glm-5.3-flash",
    "summarize": "glm-5.3-flash",
    "consolidate": "glm-5.3",
    "soul": "kimi-k3",
    # T-AL4-13：交叉验证走**轻量档**——它是"高门槛低频"的批量裁决（INV-11），
    # 一次处理一批冲突，用大模型属于浪费；判错的代价也被 KEEP_BOTH 兜底。
    "crosscheck": "glm-5.3-flash",
}

_ENV_LLM_BASE_URL = "OPENAI_BASE_URL"
_ENV_LLM_API_KEY = "OPENAI_API_KEY"
_DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"


# --------------------------------------------------------------------------- #
# 解析结果
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    """一条已解析的路由。``source`` 用于 `status` 展示走了哪一段 fallback。"""

    kind: str  # "llm" | "embedding"
    source: str  # "spirit" | "host" | "env"
    base_url: str
    api_key: str
    api_key_env: str
    model: str
    dim: int | None = None


# --------------------------------------------------------------------------- #
# 任务包装（把"模型名"从核心层手里拿走）
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TaskLLM:
    """绑定了默认模型的 LLM 句柄。**满足 ``LLMProvider`` 协议**。"""

    client: OpenAICompatClient
    default_model: str
    task: str
    route_source: str = "spirit"

    @property
    def model(self) -> str:
        return self.default_model

    def complete(
        self,
        *,
        messages,
        model: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> str:
        return self.client.complete(
            messages=messages,
            model=model or self.default_model,
            timeout=timeout,
            temperature=temperature,
        )

    def complete_json(
        self,
        *,
        messages,
        schema: dict,
        model: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        return self.client.complete_json(
            messages=messages,
            model=model or self.default_model,
            schema=schema,
            timeout=timeout,
        )


@dataclass(slots=True)
class TaskEmbedding:
    """绑定了模型的 embedding 句柄。**满足 ``EmbeddingProvider`` 协议**。"""

    client: OpenAICompatClient
    _model: str
    _dim: int
    route_source: str = "spirit"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts, model=self._model)


# --------------------------------------------------------------------------- #
# 解析器
# --------------------------------------------------------------------------- #


class ModelResolver:
    """模型路由解析器。

    Args:
        llm: 器灵配置里的 ``[models.llm]``（dict）；``None`` 表示未配置。
        embedding: 器灵配置里的 ``[models.embedding]``（dict）。
        hermes_home: 宿主目录——第 2 段 fallback 从这里读 ``config.yaml``（INV-9）。
        env: 环境变量映射（默认 ``os.environ``）；测试可注入。
        transport: 注入给 httpx 的 transport（测试用 MockTransport）。
    """

    def __init__(
        self,
        *,
        llm: Mapping[str, Any] | None = None,
        embedding: Mapping[str, Any] | None = None,
        hermes_home: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        self._llm_cfg = dict(llm) if llm else None
        self._embedding_cfg = dict(embedding) if embedding else None
        self._hermes_home = str(hermes_home) if hermes_home else None
        self._env: Mapping[str, str] = env if env is not None else os.environ
        self._transport = transport

        self._llm_route: ResolvedRoute | None = None
        self._llm_chain: list[dict] = []
        self._llm_client: OpenAICompatClient | None = None
        self._embedding_route: ResolvedRoute | None = None
        self._embedding_chain: list[dict] = []
        self._embedding_client: OpenAICompatClient | None = None

    # ------------------------------------------------------------------ #
    # 对外
    # ------------------------------------------------------------------ #

    def resolve_chain(self, task: str = "extract") -> list[dict]:
        """返回**实际生效链**（含被跳过段的跳过原因）。

        这是 `aspirit status` 的"生效模型链"区块——用户能一眼看出是否走了 fallback。

        **全部段不可用时也照常返回链**（而不是抛错）：恰恰是这种时候，
        用户最需要看到"是哪一段因为什么原因没用上"。
        """
        active = self._probe_llm()
        chain: list[dict] = []
        for segment in self._llm_chain:
            entry = dict(segment)
            entry["active"] = bool(active and segment.get("source") == active.source)
            chain.append(entry)
        if not chain:
            chain.append({"source": "none", "active": False, "reason": "四段链全部不可用"})
        chain.append(
            {
                "task": task,
                "model": self._task_model(task) if active else "",
                "source": active.source if active else "none",
                "active": True,
            }
        )
        return chain

    def embedding_chain(self) -> list[dict]:
        """embedding 的生效链（供 `status` 展示降级原因）。"""
        active = self._probe_embedding()
        chain: list[dict] = []
        for segment in self._embedding_chain:
            entry = dict(segment)
            entry["active"] = bool(active and segment.get("source") == active.source)
            chain.append(entry)
        if not chain:
            chain.append({"source": "none", "active": False, "reason": "无可用 embedding"})
        return chain

    def llm(self, task: str = "extract") -> TaskLLM:
        """取某任务档位的 LLM。未配置时抛 :class:`ProviderUnavailableError`（F1）。"""
        route = self._ensure_llm()
        client = self._llm_client
        assert client is not None  # _ensure_llm 成功时必然已建
        return TaskLLM(
            client=client, default_model=self._task_model(task), task=task, route_source=route.source
        )

    def embedding(self) -> EmbeddingProvider:
        """取 embedding provider。未配置时抛 :class:`EmbeddingError`（F2，不 fallback 到 LLM）。"""
        route = self._ensure_embedding()
        client = self._embedding_client
        assert client is not None
        assert route.dim is not None
        return TaskEmbedding(
            client=client, _model=route.model, _dim=route.dim, route_source=route.source
        )

    def embedding_available(self) -> bool:
        """embedding 是否可用（**不做网络调用**——只看配置能否解析）。"""
        try:
            self._ensure_embedding()
        except EmbeddingError:
            return False
        return True

    def llm_available(self) -> bool:
        try:
            self._ensure_llm()
        except ProviderUnavailableError:
            return False
        return True

    def close(self) -> None:
        for client in (self._llm_client, self._embedding_client):
            if client is not None:
                client.close()

    # ------------------------------------------------------------------ #
    # LLM 四段链
    # ------------------------------------------------------------------ #

    def _ensure_llm(self) -> ResolvedRoute:
        route = self._probe_llm()
        if route is None:
            raise ProviderUnavailableError(
                "没有可用的 LLM：器灵配置、宿主配置（"
                f"{self._hermes_home or '<未提供 hermes_home>'}/config.yaml）"
                f"与环境变量（{_ENV_LLM_API_KEY}）三段都没能提供模型服务。"
                "记忆仍会保存为原文，只是不做结构化提取。"
            )
        return route

    def _probe_llm(self) -> ResolvedRoute | None:
        """解析 LLM 路由并**填充链信息**，但不抛错（供 status 与 llm() 共用）。"""
        if self._llm_route is None and not self._llm_chain:
            self._llm_route = self._resolve_llm()
            if self._llm_route is not None:
                self._llm_client = self._make_client(self._llm_route)
        return self._llm_route

    def _resolve_llm(self) -> ResolvedRoute | None:
        cfg = self._llm_cfg
        if cfg and cfg.get("base_url"):
            key_env = str(cfg.get("api_key_env") or "")
            key = str(self._env.get(key_env, "")) if key_env else ""
            usable = bool(key) or not key_env
            self._llm_chain.append(
                {
                    "source": "spirit",
                    "base_url": cfg["base_url"],
                    "model": self._task_model("extract", cfg),
                    "api_key_env": key_env,
                    "usable": usable,
                    "reason": "" if usable else f"环境变量 {key_env} 未设置",
                }
            )
            if usable:
                return ResolvedRoute(
                    kind="llm",
                    source="spirit",
                    base_url=str(cfg["base_url"]),
                    api_key=key,
                    api_key_env=key_env,
                    model=self._task_model("extract", cfg),
                )
        else:
            self._llm_chain.append(
                {"source": "spirit", "usable": False, "reason": "器灵未配置 [models.llm]"}
            )

        host = self._host_llm()
        if host:
            self._llm_chain.append(
                {
                    "source": "host",
                    "base_url": host["base_url"],
                    "model": host.get("model") or "(宿主默认)",
                    "api_key_env": host.get("api_key_env", ""),
                    "usable": True,
                    "reason": "",
                }
            )
            return ResolvedRoute(
                kind="llm",
                source="host",
                base_url=host["base_url"],
                api_key=host.get("api_key", ""),
                api_key_env=host.get("api_key_env", ""),
                model=host.get("model") or self._task_model("extract"),
            )
        self._llm_chain.append(
            {
                "source": "host",
                "usable": False,
                "reason": "宿主配置中未找到可用模型服务",
            }
        )

        api_key = str(self._env.get(_ENV_LLM_API_KEY, ""))
        base_url = str(self._env.get(_ENV_LLM_BASE_URL, _DEFAULT_OPENAI_BASE))
        if api_key:
            self._llm_chain.append(
                {
                    "source": "env",
                    "base_url": base_url,
                    "model": self._task_model("extract"),
                    "api_key_env": _ENV_LLM_API_KEY,
                    "usable": True,
                    "reason": "",
                }
            )
            return ResolvedRoute(
                kind="llm",
                source="env",
                base_url=base_url,
                api_key=api_key,
                api_key_env=_ENV_LLM_API_KEY,
                model=self._task_model("extract"),
            )
        self._llm_chain.append(
            {
                "source": "env",
                "usable": False,
                "reason": f"环境变量 {_ENV_LLM_API_KEY} 未设置",
            }
        )
        return None

    def _task_model(self, task: str, cfg: Mapping[str, Any] | None = None) -> str:
        source = cfg if cfg is not None else (self._llm_cfg or {})
        explicit = source.get(task)
        if explicit:
            return str(explicit)
        if task in DEFAULT_LLM_TASK_MODELS:
            return DEFAULT_LLM_TASK_MODELS[task]
        # 未在链上找到具体模型名时，退回任意一个已配置的
        for candidate in ("extract", "consolidate", "soul"):
            if source.get(candidate):
                return str(source[candidate])
        return str(source.get("model") or "glm-5.3-flash")

    # ------------------------------------------------------------------ #
    # Embedding 链（**不含 LLM 分支** —— INV-6）
    # ------------------------------------------------------------------ #

    def _ensure_embedding(self) -> ResolvedRoute:
        route = self._probe_embedding()
        if route is None:
            raise EmbeddingError(
                "没有可用的 embedding 服务。注意：embedding **不会**降级为 LLM 调用"
                f"（INV-6）。请在 [models.embedding] 配置 provider/base_url/model/dim，"
                f"或设置环境变量 {_ENV_LLM_API_KEY} 与模型名。"
                "在配置完成前，召回会自动降级为关键词（BM25）模式。"
            )
        return route

    def _probe_embedding(self) -> ResolvedRoute | None:
        """解析 embedding 路由并填充链信息，但不抛错。"""
        if self._embedding_route is None and not self._embedding_chain:
            self._embedding_route = self._resolve_embedding()
            if self._embedding_route is not None:
                self._embedding_client = self._make_client(self._embedding_route)
        return self._embedding_route

    def _resolve_embedding(self) -> ResolvedRoute | None:
        cfg = self._embedding_cfg
        if cfg and cfg.get("base_url") and cfg.get("model"):
            key_env = str(cfg.get("api_key_env") or "")
            key = str(self._env.get(key_env, "")) if key_env else ""
            if not key and key_env:
                self._embedding_chain.append(
                    {
                        "source": "spirit",
                        "usable": False,
                        "reason": f"环境变量 {key_env} 未设置",
                    }
                )
            else:
                dim = self._embedding_dim(str(cfg["model"]), cfg.get("dim"))
                self._embedding_chain.append(
                    {
                        "source": "spirit",
                        "base_url": cfg["base_url"],
                        "model": cfg["model"],
                        "dim": dim,
                        "api_key_env": key_env,
                        "usable": True,
                        "reason": "",
                    }
                )
                return ResolvedRoute(
                    kind="embedding",
                    source="spirit",
                    base_url=str(cfg["base_url"]),
                    api_key=key,
                    api_key_env=key_env,
                    model=str(cfg["model"]),
                    dim=dim,
                )
        else:
            self._embedding_chain.append(
                {"source": "spirit", "usable": False, "reason": "器灵未配置 [models.embedding]"}
            )

        host = self._host_embedding()
        if host:
            dim = self._embedding_dim(host["model"], host.get("dim"))
            self._embedding_chain.append(
                {
                    "source": "host",
                    "base_url": host["base_url"],
                    "model": host["model"],
                    "dim": dim,
                    "usable": True,
                    "reason": "",
                }
            )
            return ResolvedRoute(
                kind="embedding",
                source="host",
                base_url=host["base_url"],
                api_key=host.get("api_key", ""),
                api_key_env=host.get("api_key_env", ""),
                model=host["model"],
                dim=dim,
            )
        self._embedding_chain.append(
            {"source": "host", "usable": False, "reason": "宿主未配置 embedding"}
        )

        api_key = str(self._env.get(_ENV_LLM_API_KEY, ""))
        model = str(self._env.get("OPENAI_EMBEDDING_MODEL", ""))
        if api_key and model:
            base_url = str(self._env.get(_ENV_LLM_BASE_URL, _DEFAULT_OPENAI_BASE))
            dim = self._embedding_dim(model, None)
            self._embedding_chain.append(
                {
                    "source": "env",
                    "base_url": base_url,
                    "model": model,
                    "dim": dim,
                    "usable": True,
                    "reason": "",
                }
            )
            return ResolvedRoute(
                kind="embedding",
                source="env",
                base_url=base_url,
                api_key=api_key,
                api_key_env=_ENV_LLM_API_KEY,
                model=model,
                dim=dim,
            )
        self._embedding_chain.append(
            {"source": "env", "usable": False, "reason": "缺少 OPENAI_EMBEDDING_MODEL"}
        )
        return None

    @staticmethod
    def _embedding_dim(model: str, configured: Any) -> int:
        if configured:
            try:
                dim = int(configured)
            except (TypeError, ValueError) as exc:
                raise EmbeddingError(
                    f"embedding dim 配置非法：{configured!r}，应为正整数"
                ) from exc
            if dim <= 0:
                raise EmbeddingError(f"embedding dim 必须为正整数，收到 {dim}")
            return dim
        if model in KNOWN_EMBEDDING_DIMS:
            return KNOWN_EMBEDDING_DIMS[model]
        raise EmbeddingError(
            f"未知 embedding 模型 {model!r}，必须在配置中显式给出 dim。"
            f"已知模型：{sorted(KNOWN_EMBEDDING_DIMS)}"
        )

    # ------------------------------------------------------------------ #
    # 宿主配置（第 2 段 fallback）
    # ------------------------------------------------------------------ #

    def _host_llm(self) -> dict | None:
        data = self._host_config()
        if not data:
            return None
        for path in (("memory", "llm"), ("llm",), ("openai",), ("providers", "openai")):
            node = _dig(data, path)
            if isinstance(node, dict) and node.get("base_url"):
                return self._materialize_host_node(node, self._llm_chain)
        # 扁平写法
        if data.get("llm_base_url"):
            return self._materialize_host_node(
                {
                    "base_url": data["llm_base_url"],
                    "model": data.get("llm_model"),
                    "api_key_env": data.get("llm_api_key_env"),
                },
                self._llm_chain,
            )
        return None

    def _materialize_host_node(
        self, node: Mapping[str, Any], chain: list[dict]
    ) -> dict | None:
        """把宿主配置节点解析成可用路由。

        **声明的 ``api_key_env`` 必须真的存在**——否则这段 fallback 是不可用的，
        应当继续往下走，而不是拿一个空密钥去请求（会得到 401 而非清晰提示）。

        ``chain`` 是**调用方所属**的那条链（LLM 调用传 ``_llm_chain``、embedding 传
        ``_embedding_chain``）。这个参数不是可选的装饰：链非空被当作"该链路已解析完成"
        的判据，写错链会让另一条链**跳过解析**并误报不可用（见 C11）。
        """
        key_env = str(node.get("api_key_env") or "")
        if key_env and not self._env.get(key_env):
            chain.append(
                {"source": "host", "usable": False, "reason": f"环境变量 {key_env} 未设置"}
            )
            return None
        return {
            "base_url": str(node["base_url"]),
            "model": node.get("model"),
            "dim": node.get("dim"),
            "api_key_env": key_env,
            "api_key": self._env.get(key_env, "") if key_env else "",
        }

    def _host_embedding(self) -> dict | None:
        data = self._host_config()
        if not data:
            return None
        for path in (("memory", "embedding"), ("embedding",)):
            node = _dig(data, path)
            if isinstance(node, dict) and node.get("base_url") and node.get("model"):
                resolved = self._materialize_host_node(node, self._embedding_chain)
                if resolved is not None:
                    resolved["model"] = str(node["model"])
                return resolved
        return None

    def _host_config(self) -> dict:
        if not self._hermes_home:
            return {}
        path = Path(self._hermes_home) / "config.yaml"
        return load_host_config_yaml(path)

    # ------------------------------------------------------------------ #
    # 客户端
    # ------------------------------------------------------------------ #

    def _make_client(self, route: ResolvedRoute) -> OpenAICompatClient:
        return OpenAICompatClient(
            OpenAICompatSettings(base_url=route.base_url, api_key=route.api_key),
            transport=self._transport,
        )


# --------------------------------------------------------------------------- #
# 极简 YAML 读取
# --------------------------------------------------------------------------- #


def load_host_config_yaml(path: str | Path) -> dict:
    """读取宿主 ``config.yaml``——**只支持映射与标量**的极小子集。

    刻意不引入 PyYAML：器灵的依赖面越小越好，而这里只需要读几个已知字段。
    解析失败一律返回空 dict（读不到宿主配置不是错误，只是少一段 fallback）。
    """
    target = Path(path)
    if not target.exists():
        return {}
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):  # pragma: no cover
        return {}
    return _parse_minimal_yaml(lines)


def _parse_minimal_yaml(lines: list[str]) -> dict:
    root: dict[str, Any] = {}
    # 栈：(缩进, 容器)
    stack: list[tuple[int, dict]] = [(-1, root)]
    pending_list: tuple[int, dict, str] | None = None

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        if line.startswith("- "):
            if pending_list is not None:
                _, container, key = pending_list
                container.setdefault(key, [])
                if isinstance(container[key], list):
                    container[key].append(_scalar(line[2:].strip()))
            continue

        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1] if stack else root

        if value == "":
            child: dict[str, Any] = {}
            container[key] = child
            stack.append((indent, child))
            pending_list = (indent, container, key)
        else:
            container[key] = _scalar(value)
            pending_list = None

    return root


def _scalar(text: str) -> Any:
    text = text.split(" #")[0].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    lowered = text.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~", ""}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _dig(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node
