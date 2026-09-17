"""工具 schema 声明（T-AL1-07 · 与 handler **严格分离**）。

分离的理由很实际：schema 是"给模型看的契约"，handler 是"给系统执行的逻辑"。
混在一起时，改一个参数名要同时改两处、还容易漏；分开之后，
**工具名与 schema 的一致性可以被机械地断言**（见
``tests/test_host_contract.py::test_tool_schema_names_match_handlers``）。
"""

from __future__ import annotations

__all__ = ["TOOL_NAMES", "TOOL_SCHEMAS", "schema_names", "tool_schema"]

_LEVELS = ["L0", "L1", "L2"]
_LAYERS = ["episodic", "semantic", "procedural", "core"]


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """构造一条 **bare function schema**。

    键名是**宿主的接口**，不是我们的命名偏好：宿主把这里返回的 dict
    **原样**塞进 ``{"type": "function", "function": schema}`` 再交给模型
    （`agent_init` 与 `memory_manager` 两处都是这么做的），而 OpenAI 规范读的是
    ``function.parameters``。

    曾经这里写的是 ``input_schema``——那是 Anthropic 的叫法。后果不是报错，
    而是**模型收到一个没有参数定义的函数**：它照样能被调用，只是参数传不进去，
    于是模型会告诉你"这个工具的参数似乎没用"。同一类形状错误在本项目里
    出现过三次（``get_config_schema`` 返回 dict 而非 list、``handle_tool_call``
    返回 dict 而非 JSON 串、以及这一处），共同点是**宿主对形状错误静默容忍**。
    所以 ``tests/test_host_contract.py`` 里有一条机械断言，用宿主自己的
    规范化函数把这条契约钉住。

    ``enum`` / ``type`` / ``description`` 都是标准 JSON Schema 关键字，宿主会原样透传。
    """
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _tool(
        "spirit_recall",
        "跨层召回相关记忆。**返回体包含召回原因（六因子分量）**，可据此判断结果为什么被选中。",
        {
            "query": {"type": "string", "description": "要查找的内容"},
            "layers": {
                "type": "array",
                "items": {"type": "string", "enum": _LAYERS},
                "description": "限定层（可选）",
            },
            "top_k": {"type": "integer", "description": "最多返回条数，默认 8"},
            "token_budget": {"type": "integer", "description": "返回内容预算（token），默认 2000"},
        },
        ["query"],
    ),
    _tool(
        "spirit_expand",
        "**逐级展开**一条记忆：L0=一句话摘要，L1=主题级概览，L2=原文+来源+关联。"
        "只返回所请求的级别——需要细节时再显式展开下一级。",
        {
            "ref": {"type": "string", "description": "记忆 id"},
            "level": {"type": "string", "enum": _LEVELS, "description": "展开级别，默认 L0"},
        },
        ["ref"],
    ),
    _tool(
        "spirit_asof",
        "**按时间点查询**：返回某条记忆在 `ts` 那一刻有效的版本。"
        "记忆被新事实取代后**不会被删除**，所以这个工具能回答"
        "「我上个月填的地址是什么」——那是「当前值」答不了的。",
        {
            "ref": {"type": "string", "description": "记忆 id"},
            "ts": {"type": "string", "description": "时间点（ISO8601）"},
        },
        ["ref", "ts"],
    ),
    _tool(
        "spirit_remember",
        "显式记住一条内容（带层提示与强度）。适合用户明确要求「记住」的场景。",
        {
            "content": {"type": "string", "description": "要记住的内容"},
            "layer": {"type": "string", "enum": _LAYERS, "description": "建议归属层，默认 semantic"},
            "confidence": {"type": "number", "description": "置信度 0~1，默认 0.9"},
        },
        ["content"],
    ),
    _tool(
        "spirit_forget",
        "物理删除一条记忆。**默认只做预演（dry-run）**，需显式 `confirm=true` 才真正执行。"
        "删除会保留可恢复快照（可用 spirit_restore 找回），`reason` 必填。",
        {
            "mem_id": {"type": "string", "description": "要删除的记忆 id"},
            "reason": {"type": "string", "description": "删除理由（必填，会写入审计）"},
            "confirm": {"type": "boolean", "description": "确认真正执行；缺省为 false（仅预演）"},
            "purge_snapshot": {
                "type": "boolean",
                "description": "合规删除：连快照一并清除，**不可恢复**。默认 false",
            },
        },
        ["mem_id", "reason"],
    ),
    _tool(
        "spirit_restore",
        "从审计快照恢复被删除的记忆（删除的安全网）。",
        {"audit_id": {"type": "integer", "description": "删除事件对应的审计 id"}},
        ["audit_id"],
    ),
    _tool(
        "spirit_review",
        "逐条审查「器灵记住了什么」：层 / 内容 / 摘要 / 时间 / 置信度 / 来源 / 状态 / 重要度构成。"
        "输出为人类可读文本。",
        {
            "layer": {"type": "string", "enum": _LAYERS, "description": "只看某一层（可选）"},
            "since": {"type": "string", "description": "起始时间（ISO8601，可选）"},
            "limit": {"type": "integer", "description": "最多条数，默认 50"},
        },
        [],
    ),
    _tool(
        "spirit_trace",
        "溯源：某条记忆的完整变更史（谁、何时、因何改的）。",
        {"mem_id": {"type": "string", "description": "记忆 id"}},
        ["mem_id"],
    ),
    _tool(
        "spirit_import",
        "**传承重建**：把一个档案导进本器灵。**不重新提取**——档案里每条都是"
        "已提取过的结论，再提一遍等于用模型的偶然行为覆盖你的资产。"
        "按内容指纹判重，故**重复导入零新增**。",
        {"path": {"type": "string", "description": "档案路径（.md / .json）"}},
        ["path"],
    ),
    _tool(
        "spirit_ingest",
        "**批量素材导入**：把外部素材喂进来**做提取**。与 `spirit_import` 语义不重叠——"
        "那个是重建（不提取），这个是提取（素材尚未结构化）。",
        {
            "text": {"type": "string", "description": "要导入的素材正文"},
            "session_id": {"type": "string", "description": "归属会话 id（可选）"},
        },
        ["text"],
    ),
    _tool(
        "spirit_core",
        "**核心记忆干预**：把一条记忆升格为人格的一部分（`promote`），"
        "或降回普通记忆（`demote`）。核心记忆是 `system_prompt_block` 读的东西——"
        "它错了污染的不是一条记忆，而是人格。**每次干预都留审计**，`reason` 必填。",
        {
            "mem_id": {"type": "string", "description": "记忆 id"},
            "action": {
                "type": "string",
                "enum": ["promote", "demote"],
                "description": "升格或降格",
            },
            "reason": {"type": "string", "description": "为什么这么做（必填）"},
            "as_type": {
                "type": "string",
                "enum": ["identity", "soul"],
                "description": "升格后的类型（仅 promote 用）：identity=我是谁，soul=我的底色",
            },
            "target_layer": {
                "type": "string",
                "enum": ["episodic", "semantic", "procedural"],
                "description": "降格的目标层（仅 demote 用）",
            },
        },
        ["mem_id", "action", "reason"],
    ),
    _tool(
        "spirit_correct",
        "主动干预：修正某条记忆的内容 / 层 / 置信度等。**留痕可溯**，`reason` 必填。",
        {
            "mem_id": {"type": "string", "description": "记忆 id"},
            "patch": {"type": "object", "description": "要修改的字段"},
            "reason": {"type": "string", "description": "修改理由（必填）"},
        },
        ["mem_id", "patch", "reason"],
    ),
    _tool(
        "spirit_consolidate",
        "手动触发巩固（工作记忆 → 情景记忆 → 语义记忆）。",
        {"session_id": {"type": "string", "description": "要巩固的会话 id"}},
        ["session_id"],
    ),
    _tool(
        "spirit_reflect",
        "记忆健康度报告：层分布、遗忘候选、召回权重、正在成为候选信念的记忆。",
        {},
        [],
    ),
    _tool(
        "spirit_export",
        "**导出人类可读的记忆档案**（或完整记忆包）。产物用纯文本编辑器即可读懂，"
        "不依赖器灵运行——这是「记忆属于使用者」的兑现方式。",
        {
            "path": {
                "type": "string",
                "description": "导出路径（.md 或 .json）。省略则写到 hermes_home/memories.md",
            },
            "fmt": {"type": "string", "enum": ["markdown", "json"], "description": "格式，默认 markdown"},
        },
        [],
    ),
]

TOOL_NAMES: tuple[str, ...] = tuple(schema["name"] for schema in TOOL_SCHEMAS)


def tool_schema(name: str) -> dict | None:
    for schema in TOOL_SCHEMAS:
        if schema["name"] == name:
            return schema
    return None


def schema_names() -> tuple[str, ...]:
    """给测试用的别名——断言"schema 与 handler 一一对应"。"""
    return TOOL_NAMES
