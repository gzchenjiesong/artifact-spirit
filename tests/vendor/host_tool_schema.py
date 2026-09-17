"""宿主 ``normalize_tool_schema`` 的原样副本（Hermes Agent，MIT）。

来源：``NousResearch/hermes-agent`` → ``agent/memory_manager.py``。
刷新：``python scripts/realtest/contract_diff.py --fetch``。

为什么要把别人的代码抄进来：这条函数定义了**工具 schema 的正确形状**。
把它记在注释里不够——下一次改代码的人没有东西可以比对，形状会慢慢漂。
这里抄一份，测试就能机械地断言"宿主 normalize 之后，我的每个工具都真有 parameters"。

抄的是**行为契约**，不是实现细节：即使上游重构，只要它对合法 schema 的判定不变，
这份副本就仍然有效（``tests/test_host_contract.py`` 会跑它）。
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["normalize_tool_schema"]

# 上游 docstring 的关键部分（保留原文，因为它解释了为什么必须有这条检查）：
#
#   Context engines and memory providers expose tool schemas via
#   ``get_tool_schemas()``. The expected shape is a bare function schema
#   (``{"name": ..., "description": ..., "parameters": ...}``) which callers
#   wrap as ``{"type": "function", "function": schema}``.
#
#   Some providers instead return an entry that is *already* in OpenAI tool
#   form (``{"type": "function", "function": {"name": ...}}``). Wrapping that
#   a second time produces a ``function`` with no top-level ``name``. Strict
#   providers (e.g. DeepSeek) reject the *entire* request with
#   ``tools[N].function: missing field name`` (HTTP 400), so one bad schema
#   disables the whole toolset and breaks every turn (#47707).


def normalize_tool_schema(schema: Any) -> Optional[dict]:
    """Return a function-tool dict with a resolvable top-level ``name``."""
    if not isinstance(schema, dict):
        return None
    # Unwrap an already-wrapped OpenAI tool entry.
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        schema = schema["function"]
        if not isinstance(schema, dict):
            return None
    name = schema.get("name", "")
    if not name or not isinstance(name, str):
        return None
    return schema
