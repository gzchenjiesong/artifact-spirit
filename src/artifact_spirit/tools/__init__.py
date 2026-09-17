"""``spirit_*`` 工具：schema 声明与调用分发（**严格分离**）。

分离的收益是可验证的：``tests/test_host_contract.py::test_tool_schema_names_match_handlers``
会断言**schema 里的每个工具名都能被 handler 处理，反之亦然**——
悬空工具（schema 有、实现没有）是这类插件最常见的低级故障。
"""

from __future__ import annotations

from .handlers import ToolError, ToolHandlers, ToolResult, translate_fault
from .schemas import TOOL_NAMES, TOOL_SCHEMAS, schema_names, tool_schema

__all__ = [
    "TOOL_NAMES",
    "TOOL_SCHEMAS",
    "ToolError",
    "ToolHandlers",
    "ToolResult",
    "schema_names",
    "tool_schema",
    "translate_fault",
]
