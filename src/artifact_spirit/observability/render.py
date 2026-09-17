"""AL5 人可读渲染：把 AL2 的**数据结构**渲染成文本。

## 为什么这些函数在 AL5 而不在 AL2

INV-1 规定"**文本是投影**"：`Reviewer.review()` / `Reviewer.trace()` 的返回值是
数据结构（`list[dict]`），而"人怎么读它"属于**投影关注点**，不是领域逻辑。
它们原先住在 `core/review.py`，结果是 AL5 的 `status` 必须 `import core.review`
再原样转发（`# noqa: F401 - 对外转发`）——**AL5 反向持有了 AL2 的具体模块**，
违反 DES-000 §4.1「AL5 ⟂ 业务」。

收敛到这里之后：

- AL2 只产出数据，不再关心人类可读性（`core/review.py` 不再定义 `format_*`）；
- AL5 不再依赖 AL2 的任何具体模块；
- AL1 的两个消费者（`tools/handlers.py`、`cli.py`）改成 `from ..observability import ...`，
  与它们已有的 `from ..observability import reflect` 保持一致。

## 纪律

本模块是**纯函数**：输入数据结构、输出 str，不读时钟以外的任何外部状态
（`now` 可注入，便于测试与快照）。
"""

from __future__ import annotations

from ..common import now_iso

__all__ = ["format_review_text", "format_trace_text"]


def format_review_text(rows: list[dict], *, now: str | None = None) -> str:
    """把审查结果渲染成**人可以直接读**的文本（V2 的兑现方式）。

    刻意不用 JSON：`spirit_review` 的读者是人，不是程序。
    """
    if not rows:
        return "（还没有任何记忆）"
    lines = [f"器灵记忆审查 · 共 {len(rows)} 条 · {now or now_iso()}", ""]
    for index, row in enumerate(rows, start=1):
        lines.append(f"{index}. [{row['layer']}/{row['type']}] {row['id']}")
        lines.append(f"   内容：{row['content']}")
        if row.get("abstract"):
            lines.append(f"   摘要：{row['abstract']}")
        lines.append(
            f"   状态：{row['status']} · 置信度 {row['confidence']:.2f}"
            f" · 重要度 {row['importance']:.2f}"
            f" · 被引用 {row.get('inbound_refs', 0)} 次"
        )
        if row.get("source_session"):
            lines.append(f"   来源：会话 {row['source_session']}")
        lines.append(f"   记录于：{row['created_at']}（最后更新 {row['updated_at']}）")
        lines.append("")
    return "\n".join(lines)


def format_trace_text(events: list[dict]) -> str:
    """把溯源结果渲染成人可读文本。"""
    if not events:
        return "（没有找到这条记忆的变更记录）"
    lines = [f"变更史 · 共 {len(events)} 条事件", ""]
    for event in events:
        actor = event["actor"]
        reason = f" · 理由：{event['reason']}" if event.get("reason") else ""
        lines.append(f"[{event['ts']}] {event['op']}（{actor}）{reason}")
        if event.get("before"):
            lines.append(f"    变更前：{event['before']}")
        if event.get("after"):
            lines.append(f"    变更后：{event['after']}")
    return "\n".join(lines)
