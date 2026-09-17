"""工具调用分发（T-AL1-07）。

三条纪律：

1. **参数校验前置**：非法参数返回**结构化错误**，不抛异常到宿主（F10）
2. **危险操作默认安全**：`spirit_forget` 默认 dry-run，须显式 `confirm`
3. **可解释性**：`spirit_recall` 必须给出召回原因（来自六因子分量）

还有一个容易被忽略的分寸：**工具返回的是给人/给模型看的东西**。
`spirit_expand` 只返回所请求的级别，`spirit_review` 返回可读文本而不是 JSON 转储——
否则 V1 的注意力收益与 V2 的可读性都只是口号。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import ConfigError
from ..core.base import RecallQuery
from ..model.base import ProviderUnavailableError, SchemaViolationError
from ..observability import format_review_text, format_trace_text
from ..store.base import AuditEvent, MemoryRecord, StorageFatalError
from .schemas import TOOL_NAMES

if TYPE_CHECKING:  # pragma: no cover
    from ..core.facade import ArtifactSpiritCore

__all__ = ["ToolError", "ToolHandlers", "ToolResult", "translate_fault"]

_LAYERS = ("episodic", "semantic", "procedural", "core")

REMEMBER_ABSTRACT_BUDGET = 24
"""`spirit_remember` 的 L0 摘要预算（token）。

摘要**必须比正文短**——否则分级加载没有意义（V1）：L0 就等于全文时，
"先给印象、再给细节"这件事根本无法发生。
"""


@dataclass(slots=True)
class ToolResult:
    """工具返回。``ok=False`` 时是**结构化错误**（不是异常）。"""

    ok: bool
    text: str = ""
    data: dict | None = None
    error: str | None = None
    field: str | None = None

    def as_dict(self) -> dict:
        payload: dict[str, Any] = {"ok": self.ok}
        if self.text:
            payload["text"] = self.text
        if self.data:
            payload["data"] = self.data
        if self.error:
            payload["error"] = self.error
        if self.field:
            payload["field"] = self.field
        return payload


class ToolError(ValueError):
    """参数校验失败。**只在 handler 内部抛出**，边界会转成 :class:`ToolResult`。"""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


_FAULT_TEMPLATES: tuple[tuple[type[BaseException], str], ...] = (
    (
        StorageFatalError,
        "存储不可用：{detail}。本次写入已暂缓，请稍后重试；持续失败可运行 `aspirit doctor` 体检。",
    ),
    (
        SchemaViolationError,
        "提取结果未通过校验，已保留原文（以未结构化形式存储）：{detail}",
    ),
    (
        ProviderUnavailableError,
        "模型不可用，记忆将以未结构化形式保存：{detail}",
    ),
    (
        ConfigError,
        "配置有误：{detail}。可运行 `aspirit config` 检查。",
    ),
)


def translate_fault(exc: BaseException) -> str:
    """把下层异常翻译成**上层能懂的语义**（LLD-AL1 §5 M8 / T-AL1-09 验收 1）。

    为什么值得单列一张表：这几个异常族的 docstring（`store/base.py`、
    `model/base.py`）都白纸黑字写着"**由 AL1 转换为工具级错误**"——
    契约被写了三遍，实现却一次都没兑现，于是用户看到的是
    `内部错误：StorageFatalError: database is locked`，
    而不是"存储不可用，请稍后重试"。**分层翻译的意图全丢了**。

    兜底分支**不改安全行为**（仍不崩宿主、仍返回结构化错误），只是把
    "哪一层出的错"翻译成"用户该做什么"。四个异常族互不同源，匹配顺序无歧义。
    """
    detail = str(exc) or type(exc).__name__
    for exc_type, template in _FAULT_TEMPLATES:
        if isinstance(exc, exc_type):
            return template.format(detail=detail)
    return f"内部错误：{type(exc).__name__}: {detail}"


class ToolHandlers:
    """`spirit_*` 工具的调用分发。

    它**不持有业务逻辑**——所有决策都在 AL2。这里只做三件事：
    校验参数、调用核心层、把结果整理成人能读的形态。
    """

    def __init__(self, services) -> None:
        self.services = services

    # ------------------------------------------------------------------ #

    @property
    def core(self) -> ArtifactSpiritCore:
        return self.services.core

    def names(self) -> tuple[str, ...]:
        return TOOL_NAMES

    def handle(self, name: str, args: Mapping[str, Any] | None = None) -> ToolResult:
        """分发入口。**任何异常都在这里被转成结构化错误**（F10）。"""
        args = dict(args or {})
        handler: Callable[[dict], ToolResult] | None = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return ToolResult(ok=False, error=f"未知工具：{name}", field="name")
        try:
            return handler(args)
        except ToolError as exc:
            return ToolResult(ok=False, error=str(exc), field=exc.field)
        except KeyError as exc:
            return ToolResult(ok=False, error=f"目标不存在：{exc}", field="mem_id")
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:
            # 不崩宿主 ≠ 只丢一句类型名：先按 M8 翻译成可操作文案（T-AL1-09 验收 1）。
            return ToolResult(ok=False, error=translate_fault(exc))

    # ------------------------------------------------------------------ #
    # 各工具
    # ------------------------------------------------------------------ #

    def _tool_spirit_recall(self, args: dict) -> ToolResult:
        query = _require_str(args, "query")
        layers = args.get("layers")
        if layers is not None and not isinstance(layers, list):
            raise ToolError("layers 必须是数组", field="layers")
        if layers and any(layer not in _LAYERS for layer in layers):
            raise ToolError(f"layers 只能取 {_LAYERS}", field="layers")

        top_k = _optional_int(args, "top_k", 8)
        budget = _optional_int(args, "token_budget", 2000)
        vec = self._vectorize(query)

        hits = self.core.recall(
            RecallQuery(
                text=query,
                vec=vec,
                session_id=args.get("session_id", ""),
                layers=list(layers) if layers else None,
                token_budget=budget,
                top_k=top_k,
            )
        )
        if not hits:
            return ToolResult(ok=True, text="（没有找到相关记忆）", data={"items": []})

        lines = [f"找到 {len(hits)} 条相关记忆：", ""]
        items = []
        for index, hit in enumerate(hits, start=1):
            record = hit.record
            lines.append(f"{index}. [{record.layer}/{record.type}] {record.id}")
            lines.append(f"   {record.abstract or record.content}")
            # 可解释性：这条**为什么**被召回
            reasons = " · ".join(
                f"{name} {value:.2f}" for name, value in sorted(hit.raw.items()) if value > 0
            )
            lines.append(f"   召回原因：{reasons or '（无显著信号）'}")
            items.append(
                {
                    "id": record.id,
                    "layer": record.layer,
                    "content": record.abstract or record.content,
                    "raw": {k: round(v, 4) for k, v in hit.raw.items()},
                    "score": round(hit.score, 4),
                }
            )
        return ToolResult(ok=True, text="\n".join(lines), data={"items": items})

    def _tool_spirit_expand(self, args: dict) -> ToolResult:
        ref = _require_str(args, "ref")
        level = args.get("level", "L0")
        if level not in ("L0", "L1", "L2"):
            raise ToolError("level 只能是 L0 / L1 / L2", field="level")

        text = self.core.expand(ref, level, hot_path=False)
        self.services.write_now(self.core.drain_pending())
        if not text:
            return ToolResult(ok=False, error=f"没有找到记忆：{ref}", field="ref")
        # **只返回所请求的级别**——不顺手把 L2 一并给出（V1）
        return ToolResult(ok=True, text=text, data={"ref": ref, "level": level})

    def _tool_spirit_asof(self, args: dict) -> ToolResult:
        """按时间点取该时刻有效的版本（T-AL1-14 · M3）。

        **"那时没有有效版本"必须是结构化结果**（`found: False` + 说明），
        不能返回空字符串——用户得分得清"那时不存在"与"取到了但内容是空的"。
        这也是把 `None` 从 AL2 一路翻到工具面时最容易丢掉的一层信息。
        """
        ref = _require_str(args, "ref")
        ts = _require_str(args, "ts")

        record = self.core.asof(ref, ts)
        if record is None:
            return ToolResult(
                ok=True,
                text=(
                    f"在 {ts} 时刻没有有效版本：{ref}——它当时还不存在，"
                    "或已被取代且没有后继"
                ),
                data={"ref": ref, "ts": ts, "found": False},
            )
        return ToolResult(
            ok=True,
            text=record.content,
            data={
                "ref": ref,
                "ts": ts,
                "found": True,
                "mem_id": record.id,
                "layer": record.layer,
                "valid_from": record.valid_from,
                "valid_to": record.valid_to,
                "superseded_by": record.superseded_by,
            },
        )

    def _tool_spirit_remember(self, args: dict) -> ToolResult:
        content = _require_str(args, "content")
        layer = args.get("layer", "semantic")
        if layer not in _LAYERS:
            raise ToolError(f"layer 只能取 {_LAYERS}", field="layer")
        confidence = _optional_float(args, "confidence", 0.9)

        from ..common import summarize
        from ..core.base import WriteIntent

        # 显式记忆走正常写入路径（**不绕过审计**）——只是来源标记为 user
        now = self.core.clock()
        record = MemoryRecord(
            id=self.core.id_gen(layer),
            layer=layer,
            type="fact",
            content=content,
            abstract=summarize(content, budget=REMEMBER_ABSTRACT_BUDGET),
            confidence=confidence,
            salience=1.0,
            created_at=now,
            updated_at=now,
        )
        intents = [
            WriteIntent(
                op="put",
                record=record,
                embed_text=content,
                actor="user",
                audit=AuditEvent(
                    op="add",
                    actor="user",
                    target_kind="memory",
                    target_id=record.id,
                    reason="用户显式要求记住",
                ),
            )
        ]
        self.services.write_now(intents)
        return ToolResult(
            ok=True, text=f"已记住（{record.id}）：{content}", data={"id": record.id}
        )

    def _tool_spirit_forget(self, args: dict) -> ToolResult:
        mem_id = _require_str(args, "mem_id")
        reason = _require_str(args, "reason")
        confirm = bool(args.get("confirm", False))
        purge = bool(args.get("purge_snapshot", False))

        if not confirm:
            record = self.services.backend.get(mem_id)
            if record is None:
                return ToolResult(ok=False, error=f"没有找到记忆：{mem_id}", field="mem_id")
            return ToolResult(
                ok=True,
                text=(
                    f"【预演】将要删除 {mem_id}：{record.abstract or record.content}\n"
                    f"理由：{reason}\n"
                    "删除会保留快照（可用 spirit_restore 找回）。"
                    "确认执行请带 confirm=true。"
                ),
                data={"dry_run": True, "mem_id": mem_id},
            )

        intents = self.core.forget(
            mem_id, reason=reason, source="tool:spirit_forget", purge_snapshot=purge
        )
        self.services.write_now(intents)
        tail = "（快照已清除，不可恢复）" if purge else "（可经 spirit_restore 恢复）"
        return ToolResult(ok=True, text=f"已删除 {mem_id}{tail}", data={"mem_id": mem_id})

    def _tool_spirit_restore(self, args: dict) -> ToolResult:
        audit_id = args.get("audit_id")
        if not isinstance(audit_id, int):
            raise ToolError("audit_id 必须是整数", field="audit_id")
        intents = self.core.restore(audit_id)
        self.services.write_now(intents)
        return ToolResult(
            ok=True,
            text=f"已从审计 {audit_id} 的快照恢复记忆（沿用原 id）",
            data={"audit_id": audit_id},
        )

    def _tool_spirit_review(self, args: dict) -> ToolResult:
        layer = args.get("layer")
        if layer is not None and layer not in _LAYERS:
            raise ToolError(f"layer 只能取 {_LAYERS}", field="layer")
        rows = self.core.review(
            layer=layer, since=args.get("since"), limit=_optional_int(args, "limit", 50)
        )
        return ToolResult(ok=True, text=format_review_text(rows), data={"items": rows})

    def _tool_spirit_trace(self, args: dict) -> ToolResult:
        mem_id = _require_str(args, "mem_id")
        events = self.core.trace(mem_id)
        return ToolResult(ok=True, text=format_trace_text(events), data={"items": events})

    def _tool_spirit_correct(self, args: dict) -> ToolResult:
        mem_id = _require_str(args, "mem_id")
        patch = args.get("patch")
        if not isinstance(patch, dict) or not patch:
            raise ToolError("patch 必须是非空对象", field="patch")
        reason = _require_str(args, "reason")
        intents = self.core.correct(mem_id, patch, reason=reason)
        self.services.write_now(intents)
        return ToolResult(
            ok=True,
            text=f"已修正 {mem_id}：{sorted(patch)}",
            data={"mem_id": mem_id, "changed": sorted(patch)},
        )

    def _tool_spirit_core(self, args: dict) -> ToolResult:
        """核心记忆干预（T-AL1-15）。

        工具面**只做翻译**：门槛、审计、层归属全在 AL2（R4 的精神）。
        `reason` 在这里先校验一次，是为了让模型收到"缺参数"，
        而不是一条从深处冒上来的异常——**错误越早越像人话**。
        """
        mem_id = _require_str(args, "mem_id")
        action = _require_str(args, "action")
        reason = _require_str(args, "reason")

        if action == "promote":
            as_type = str(args.get("as_type") or "identity")
            intents = self.core.promote(mem_id, reason=reason, as_type=as_type)
            text = f"已升格 {mem_id} 为核心记忆——它现在会出现在系统提示里"
        elif action == "demote":
            target = str(args.get("target_layer") or "semantic")
            intents = self.core.demote(mem_id, reason=reason, target_layer=target)
            text = f"已将 {mem_id} 降格为 {target}——内容仍在，只是不再注入人格"
        else:
            raise ToolError("action 只能是 promote / demote", field="action")

        self.services.write_now(intents)
        return ToolResult(
            ok=True,
            text=text,
            data={"mem_id": mem_id, "action": action, "changed": len(intents)},
        )

    def _tool_spirit_consolidate(self, args: dict) -> ToolResult:
        session_id = _require_str(args, "session_id")
        report = self.core.consolidate(session_id=session_id)
        self.services.write_now(report.intents)
        if report.skipped:
            return ToolResult(
                ok=True,
                text=f"未执行巩固（{report.skipped}）",
                data={"skipped": report.skipped},
            )
        return ToolResult(
            ok=True,
            text=(
                f"已巩固会话 {session_id}：生成情景记忆 {report.episode_id}，"
                f"提升为语义记忆 {len(report.promoted)} 条"
            ),
            data={"episode_id": report.episode_id, "promoted": report.promoted},
        )

    def _tool_spirit_reflect(self, args: dict) -> ToolResult:
        from ..observability import reflect

        report = reflect(self.services)
        lines = [
            f"记忆总量 {report['total']} · 平均强度 {report['average_strength']:.3f}",
            "",
            "为什么遗忘：" + report["why_forget"]["explain"],
            (
                f"  降级候选 {len(report['why_forget']['downgrade_candidates'])} 条"
                f" · 不可达候选 {len(report['why_forget']['unreachable_candidates'])} 条"
            ),
            "",
            "为什么召回："
            + " / ".join(f"{k} {v}" for k, v in report["why_recall"]["weights"].items()),
            f"正在成为候选信念 {len(report['becoming_beliefs'])} 条",
        ]
        return ToolResult(ok=True, text="\n".join(lines), data=report)

    def _tool_spirit_export(self, args: dict) -> ToolResult:
        """导出档案。

        ``path`` **可选**：让模型去编一个绝对路径是没必要的负担——它不知道宿主把
        hermes_home 放在哪，编出来的路径多半写不进去。省略时落到
        ``{hermes_home}/memories.md``，需要时再显式指定。
        """
        fmt = args.get("fmt", "markdown")
        if fmt not in ("markdown", "json"):
            raise ToolError("fmt 只能是 markdown / json", field="fmt")
        raw_path = args.get("path")
        if raw_path in (None, ""):
            suffix = "json" if fmt == "json" else "md"
            path = str(Path(self.services.config.hermes_home or ".") / f"memories.{suffix}")
        else:
            path = _require_str(args, "path")
        self.services.backend.export_archive(path, fmt=fmt)
        return ToolResult(
            ok=True,
            text=f"已导出到 {path}（{fmt}）——纯文本即可阅读，不依赖器灵运行",
            data={"path": path, "fmt": fmt},
        )

    def _tool_spirit_import(self, args: dict) -> ToolResult:
        """传承重建（T-AL1-17）。

        与 `spirit_ingest` 的分工是**硬分工**：这里**一次 LLM 调用都不发生**——
        档案里每条都是已经提取过的结论。所以本地图把它做成可断言的性质：
        `test_value.py` 的 V3 段会数模型调用次数。
        """
        from ..store.archive import load_archive

        path = _require_str(args, "path")
        pack = load_archive(path)  # 解析在 AL3，翻译在 AL2，落库在 AL5 —— 各归其位
        report = self.core.restore_pack(pack)
        self.services.write_now(report.intents)

        parts = [f"已从 {path} 重建：新增 {report.imported} 条，跳过 {report.skipped} 条（内容已存在）"]
        if report.relations or report.entities:
            parts.append(f"关联 {report.relations} 条、实体 {report.entities} 个")
        if report.restored_superseded:
            parts.append(f"回填「被取代」关系 {report.restored_superseded} 条")
        if report.errors:
            # **错误要显式说出来**：合并进"跳过"会把数据问题伪装成"幂等生效"
            parts.append(f"⚠ {len(report.errors)} 条未能重建：{report.errors[:3]}")
        return ToolResult(
            ok=True,
            text="；".join(parts),
            data={
                "path": path,
                "imported": report.imported,
                "skipped": report.skipped,
                "relations": report.relations,
                "entities": report.entities,
                "restored_superseded": report.restored_superseded,
                "errors": report.errors,
            },
        )

    def _tool_spirit_ingest(self, args: dict) -> ToolResult:
        """批量素材导入（T-AL1-17）——**做提取**。

        与 `spirit_import` 的边界：把**档案**喂给 ingest 会重复提取一遍
        （用模型的偶然行为覆盖已确定的结论）；把**素材**交给 import 则会解析失败。
        两个入口不能互相替代，用例各自钉住。
        """
        from ..common import now_iso
        from ..core.base import TurnEvent

        text = _require_str(args, "text")
        session_id = str(args.get("session_id") or "ingest")
        ts = now_iso()
        intents = self.core.ingest_turn(
            TurnEvent(session_id=session_id, user=text, assistant="", ts=ts)
        )
        self.services.write_now(intents)
        return ToolResult(
            ok=True,
            text=f"已导入素材（会话 {session_id}）：产出 {len(intents)} 条写意图",
            data={"session_id": session_id, "intents": len(intents)},
        )

    # ------------------------------------------------------------------ #

    def _vectorize(self, text: str) -> list[float] | None:
        provider = self.services.core.embedding
        if provider is None:
            return None
        try:
            return provider.embed([text])[0]
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #


def _require_str(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{key} 是必填的非空字符串", field=key)
    return value


def _optional_int(args: dict, key: str, default: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{key} 必须是整数", field=key)
    return value


def _optional_float(args: dict, key: str, default: float) -> float:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{key} 必须是数字", field=key)
    return float(value)
