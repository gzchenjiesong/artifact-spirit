"""器灵 / Artifact Spirit —— 为 AI Agent 设计的借鉴脑科学研究的记忆系统。

本包是 Hermes Agent 的 MemoryProvider 插件（entry point: ``hermes_agent.memory_providers``）。

约束：本文件**只做注册，零业务逻辑**（R6）。
"""

from __future__ import annotations

__version__ = "1.0.0"
"""本包版本。**必须与 `pyproject.toml` 的 `[project].version` 一致**——
前者给打包器、后者给运行时自报，不一致会让「装的是哪个版本」有两个都说得通的答案。
由 `tests/test_release.py` 钉住（R6 只约束"不放业务逻辑"，版本号属于元数据）。
"""

__all__ = ["__version__", "register"]


def register(ctx) -> None:
    """Hermes 插件入口。

    延迟导入 provider，避免加载宿主时把整包依赖一次性拉起来（R6 / C1）。
    """
    from .provider import ArtifactSpiritProvider

    ctx.register_memory_provider(ArtifactSpiritProvider())
