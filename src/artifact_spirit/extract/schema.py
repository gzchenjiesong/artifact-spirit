"""提取契约：记忆对象 schema + 轻量校验器 + **同源的提示词**（LLD-AL2 §2.5 · C12）。

> **C12 是这一节存在的理由**：提示词里说"字段是 A"，而校验器要求"字段是 B"，
> 是最典型也最难查的一类缺陷——模型严格照提示词输出，然后被校验器判为非法，
> 表现为"提取质量莫名很差"。把两者放在同一文件、由同一组常量生成，
> 这个坑就不存在了。
"""

from __future__ import annotations

from ..common import MEMORY_TYPES
from ..model.base import validate_schema

__all__ = [
    "DEDUP_SCHEMA",
    "EXTRACTION_SCHEMA",
    "MEMORY_ITEM_SCHEMA",
    "MEMORY_LAYERS",
    "MEMORY_TYPES",
    "SCOPE_TYPES",
    "SYSTEM_PROMPT",
    "parse_memories",
    "validation_errors",
]

MEMORY_LAYERS = ("episodic", "semantic", "procedural", "core")
SCOPE_TYPES = ("global", "project", "session")

# 字段说明**单一来源**：schema 与提示词都从这里取，保证永不漂移（C12）
FIELD_HINTS: dict[str, str] = {
    "type": f"记忆类型，取值之一：{'|'.join(MEMORY_TYPES)}",
    "layer": f"归属层，取值之一：{'|'.join(MEMORY_LAYERS)}",
    "subject": "主体（谁/什么）",
    "predicate": "关系或属性（做什么/是什么）",
    "object": "客体或取值",
    "content": "一句自然语言陈述——这是**事实来源**，必须完整、自含",
    "abstract": "一句话摘要（L0），不超过 40 字",
    "scope": f'作用域，形如 {{"type": "{"|".join(SCOPE_TYPES)}", "id": "..."}}',
    "confidence": "置信度 0~1",
    "salience": "显著性 0~1",
    "valid_from": "生效时间（ISO8601 或 null）",
    "valid_to": "失效时间（ISO8601 或 null）",
}

MEMORY_ITEM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "type": {"enum": list(MEMORY_TYPES)},
        "layer": {"enum": list(MEMORY_LAYERS)},
        "subject": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "predicate": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "object": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "content": {"type": "string"},
        "abstract": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "scope": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "type": {"enum": list(SCOPE_TYPES)},
                        "id": {"type": "string"},
                    },
                    "required": ["type"],
                },
                {"type": "null"},
            ]
        },
        "confidence": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "salience": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "valid_from": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "valid_to": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["type", "layer", "content"],
}

EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "memories": {"type": "array", "items": MEMORY_ITEM_SCHEMA},
    },
    "required": ["memories"],
}

DEDUP_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision": {"enum": ["ADD", "UPDATE", "IGNORE", "MERGE"]},
        "target_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["decision"],
}


def _field_lines() -> str:
    return "\n".join(f"- `{name}`：{hint}" for name, hint in FIELD_HINTS.items())


SYSTEM_PROMPT = f"""你是一个记忆提取器。从对话中抽取**值得长期记住**的信息，输出严格 JSON。

## 只抽取这些
- 关于使用者的事实、偏好、身份、习惯（`fact` / `preference` / `identity`）
- 项目、组织、人物的稳定属性（`entity`）
- 发生了什么的整体事件（`event`）
- 待办与计划（`intent`）

## 不要抽取
- 寒暄、客套、纯情绪表达
- 临时的、一次性的操作细节
- 任何你不确定的推断（宁可不记）

## 字段
{_field_lines()}

## 硬规则
1. `content` 必须**自含**——脱离上下文也能读懂，不要用"它""那个"
2. 一条记忆只表达**一件事**，不要把多个事实挤进一条
3. 三元组（subject / predicate / object）能填就填，填不了留 null
4. `confidence` 反映你对这条的把握，不要一律给 1.0
5. 没有值得记的内容时，`memories` 返回**空数组**——这完全正常
6. **时间一律换算成绝对日期**：把「昨天」「上周五」「刚才」按对话发生的时间
   换算成 `YYYY-MM-DD` 填进 `valid_from` / `object` / `content`；
   实在换算不出来就留 null，**不要照抄相对说法**——那等于没记下时间
7. **输出语言跟随对话原文**：原文是英文就用英文、是中文就用中文。
   换语言会让同一条记忆与它自己的检索线索对不上（问题是英文、记忆是中文，
   两边的词一个都对不上）

只输出 JSON，不要任何解释、不要代码块标记。
"""


def validation_errors(payload: object) -> list[str]:
    """校验提取输出结构，返回错误列表（空 = 通过）。"""
    return validate_schema(payload, EXTRACTION_SCHEMA)


def parse_memories(payload: dict, *, default_scope: dict | None = None) -> list[dict]:
    """把校验通过的 payload 整理成候选列表。

    **调用方必须先跑 :func:`validation_errors`**——本函数假定结构合法
    （校验失败的记忆不得进入 AL3 是硬约束 C5）。
    """
    out: list[dict] = []
    for item in payload.get("memories", []):
        candidate = {
            "type": item["type"],
            "layer": item["layer"],
            "content": str(item["content"]).strip(),
            "subject": _clean(item.get("subject")),
            "predicate": _clean(item.get("predicate")),
            "object": _clean(item.get("object")),
            "abstract": _clean(item.get("abstract")),
            "scope": item.get("scope") or default_scope,
            "confidence": _clamp(item.get("confidence"), 0.7),
            "salience": _clamp(item.get("salience"), 0.0),
            "valid_from": _clean(item.get("valid_from")),
            "valid_to": _clean(item.get("valid_to")),
        }
        if candidate["content"]:
            out.append(candidate)
    return out


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clamp(value: object, default: float) -> float:
    try:
        number = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
