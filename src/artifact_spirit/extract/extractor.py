"""结构化提取（LLD-AL2 §5 M8 · Mem0 范式）。

**ADD-only 单遍**：提取器只负责"从这段对话里能抽出什么"，
"要不要真写进去"是去重器的判断（见 :mod:`artifact_spirit.extract.dedup`）。
把两件事分开，是因为它们的失败模式完全不同——提取怕漏、去重怕重。

两条降级路径在这里落地：

| 故障 | 行为 |
|---|---|
| **F1** LLM 不可用 | **跳过提取，仅存原文** + 告警。记忆不丢，只是未结构化 |
| **F3** 输出非 JSON / 不合 schema | 由 AL4 重试 1 次；仍失败 → 仅存原文 + ``audit(reason='extract_parse_failed')`` |

**校验失败的记忆绝不进入 AL3**（C5）——不接受"部分写入"，那会让记忆库出现
一半结构化一半没有的中间态，之后无从判断哪条可信。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..common import first_sentence
from ..model.base import (
    LLMError,
    LLMProvider,
    ProviderUnavailableError,
    SchemaViolationError,
)
from ..store.base import MemoryRecord
from .schema import SYSTEM_PROMPT, parse_memories, validation_errors

__all__ = ["MAX_CONTENT_CHARS", "ExtractResult", "Extractor"]

MAX_CONTENT_CHARS = 6000
"""单轮对话送入提取器的长度上限——超长事件本身就该被巩固处理，不是提取。"""


@dataclass(slots=True)
class ExtractResult:
    """提取结果。

    ``degraded`` 非空表示走了降级路径，值即原因——
    **必须被上层记录**（F1/F3 的 audit 留痕），否则降级会变成静默的数据形态差异。
    """

    candidates: list[dict] = field(default_factory=list)
    degraded: str | None = None
    raw_text: str = ""
    fallback_only: bool = False
    """``True`` 表示"只存原文"——上层应据此产出保底记忆而不是丢弃这一轮。"""


@dataclass(slots=True)
class Extractor:
    """结构化提取器。"""

    llm: LLMProvider | None = None
    max_chars: int = MAX_CONTENT_CHARS

    def extract(
        self, text: str, *, scope: dict | None = None, now: str | None = None
    ) -> ExtractResult:
        """从一段文本抽取记忆候选。**任何失败都不抛错**——降级是设计的一部分。

        ``now`` 是**这段对话发生的时间**，会随正文一起交给模型。
        它不是装饰：没有它，``昨天``/``上周五``/``刚才``这些说法**无从换算**——
        模型不知道"今天"是哪天，只能照抄一个相对说法，
        于是事实里的时间点在库里彻底消失（实测 LoCoMo 的标准答案是绝对日期，
        而库里只有一句"参加了支持小组"）。
        """
        text = (text or "").strip()
        if not text:
            return ExtractResult(raw_text="", degraded="empty_input", fallback_only=True)

        text = text[: self.max_chars]

        if self.llm is None:
            return ExtractResult(
                raw_text=text, degraded="llm_unconfigured", fallback_only=True
            )

        try:
            payload = self.llm.complete_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _with_time(text, now)},
                ],
                schema=_schema_for_llm(),
            )
        except ProviderUnavailableError:
            return ExtractResult(
                raw_text=text, degraded="provider_unavailable", fallback_only=True
            )
        except SchemaViolationError:
            # AL4 已经重试过一次；到这里说明两次都没拿到合法 JSON
            return ExtractResult(
                raw_text=text, degraded="extract_parse_failed", fallback_only=True
            )
        except LLMError:
            return ExtractResult(raw_text=text, degraded="llm_error", fallback_only=True)

        errors = validation_errors(payload)
        if errors:
            # 结构非法 → **不产出任何部分候选**（C5）
            return ExtractResult(
                raw_text=text,
                degraded="schema_invalid:" + "; ".join(errors[:3]),
                fallback_only=True,
            )

        candidates = parse_memories(payload, default_scope=scope)
        return ExtractResult(candidates=candidates, raw_text=text)

    # ------------------------------------------------------------------ #

    def to_records(self, candidates: list[dict], *, id_gen, now: str) -> list[MemoryRecord]:
        """把候选转成记录（**尚未落库**，交去重器决策后才写）。"""
        records: list[MemoryRecord] = []
        for candidate in candidates:
            content = candidate["content"]
            records.append(
                MemoryRecord(
                    id=id_gen(candidate["layer"]),
                    layer=candidate["layer"],
                    type=candidate["type"],
                    content=content,
                    abstract=candidate.get("abstract") or first_sentence(content),
                    subject=candidate.get("subject"),
                    predicate=candidate.get("predicate"),
                    object=candidate.get("object"),
                    scope=candidate.get("scope"),
                    confidence=float(candidate.get("confidence", 0.7)),
                    salience=float(candidate.get("salience", 0.0)),
                    valid_from=_sane_valid_from(candidate.get("valid_from"), now),
                    valid_to=_normalise_ts(candidate.get("valid_to")),
                    created_at=now,
                    updated_at=now,
                )
            )
        return records


def _with_time(text: str, now: str | None) -> str:
    """把"这段对话发生在什么时候"随正文交给模型。

    单独成一行、放在正文之前：模型要把 `yesterday` 换算成绝对日期，
    就必须先知道"今天"是哪天——而这个信息**只有调用方有**。
    """
    if not now:
        return text
    return f"对话发生时间：{now}\n\n{text}"


def _parse_ts(value: object) -> datetime | None:
    """解析 ISO8601。**解析不了就返回 `None`**——不猜、不兜底成"现在"。"""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        # 不必把 `Z` 换成 `+00:00`：Python 3.11 起 `fromisoformat` 直接吃 `Z`
        # （本项目 `requires-python = ">=3.11"`）。多一次替换就多一个出错的地方。
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def _normalise_ts(value: object) -> str | None:
    """规范化时间戳格式，解析不了则**丢弃**。

    丢弃而不是原样留着：`Z` 与 `+00:00` 两种写法混在库里时，
    字符串比较会给出与时间顺序无关的结果——而本仓库的时态比较正是基于字符串。
    留一个格式怪异的字段，等于给后面埋一个"偶尔顺序不对"的坑。
    """
    parsed = _parse_ts(value)
    return parsed.isoformat() if parsed else None


def _sane_valid_from(raw: object, now: str) -> str | None:
    """收敛模型给的 `valid_from`：**不能晚于本轮时间**。

    模型会从对话里读到日期（「我 2023 年搬来北京」）并填进 `valid_from`——那是对的，
    一条事实确实可能从上个月开始成立。但它也会**编**：
    实测出现过 `2025-03-07T00:00:00Z`，而那段对话发生在 2023 年。

    采信一个晚于本轮时间的 `valid_from`，会立刻炸在**后面**：
    该条被更新时 `valid_to`（本轮时间）小于它的 `valid_from`，
    撞上「有效期不能是空区间」的守卫——而那个报错长得像时态逻辑的错，
    会把人引去修错地方（这正是它第一次被发现时的样子）。

    **一条事实不可能从未来开始成立**，所以这是输入校验，不是业务规则。
    校验不过就退回「本轮时间」——宁可把"从什么时候开始"记宽一点，
    也不能让一条自相矛盾的记录进库。
    """
    parsed = _parse_ts(raw)
    if parsed is None:
        return None
    limit = _parse_ts(now)
    if limit is not None and parsed > limit:
        return limit.isoformat()
    return parsed.isoformat()


def _schema_for_llm() -> dict:
    """送给模型的结构描述。

    与 :data:`~artifact_spirit.extract.schema.EXTRACTION_SCHEMA` **同源**（C12）——
    校验器与提示词描述同一份结构，不会出现"提示说 A、校验要 B"。
    """
    from .schema import EXTRACTION_SCHEMA

    return EXTRACTION_SCHEMA
