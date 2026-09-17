"""``aspirit`` 命令行（T-AL1-08 / 10）。

**CLI 是薄壳**：解析参数 → 调服务 → 格式化输出。业务逻辑一行都不在这里。

两处与 provider 一致的约定：

- **共用同一套装配**（走 AL5 ``start``），不重复实现（C10）
- **破坏性命令默认 dry-run**：`decay` / `forget` 不加 ``--apply`` 就只预演

`--json` 让所有命令可被脚本消费——这也是集成测试观察 CLI 的入口。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from .config import toml_example
from .config.loader import ConfigError, config_path
from .observability import (
    AuditView,
    doctor,
    doctor_text,
    format_audit_text,
    format_restorable_text,
    layers,
    reflect,
    status,
    status_text,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aspirit",
        description="器灵 —— 为 AI Agent 设计的记忆系统",
    )
    parser.add_argument(
        "--home",
        default="",
        help="hermes_home（默认取环境变量 ARTIFACT_SPIRIT_HOME 或当前目录）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本化）")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="初始化：生成配置、命名器灵")
    sub.add_parser("status", help="状态：层计数 / 健康度 / 生效模型链 / 降级告警")
    sub.add_parser("layers", help="五类记忆分布")
    p_review = sub.add_parser("review", help="逐条审查：器灵记住了什么（人类可读）")
    p_review.add_argument("--limit", type=int, default=20, help="最多显示多少条（默认 20）")
    sub.add_parser("reflect", help="记忆健康度报告")

    p_trace = sub.add_parser("trace", help="溯源：某条记忆的完整变更史")
    p_trace.add_argument("mem_id")

    p_exp = sub.add_parser("export", help="导出人类可读档案 / 完整记忆包")
    p_exp.add_argument("path")
    p_exp.add_argument("--fmt", choices=["markdown", "json"], default="markdown")

    p_imp = sub.add_parser("import", help="从档案 / 记忆包**重建**（不提取）")
    p_imp.add_argument("path")

    p_ingest = sub.add_parser("ingest", help="批量素材导入（**做提取**，与 import 不重叠）")
    p_ingest.add_argument("text")
    p_ingest.add_argument("--session-id", default="ingest", help="归属会话 id")

    p_cons = sub.add_parser("consolidate", help="手动触发巩固")
    p_cons.add_argument("session_id")

    p_decay = sub.add_parser("decay", help="衰减排序（**只影响排序，不删除**）")
    p_decay.add_argument("--apply", action="store_true", help="真正写回强度（默认只预演）")

    p_forget = sub.add_parser("forget", help="物理删除（**白名单操作**；默认预演）")
    p_forget.add_argument("mem_id")
    p_forget.add_argument("--reason", required=True, help="删除理由（会写入审计）")
    p_forget.add_argument("--apply", action="store_true", help="真正执行（默认只预演）")
    p_forget.add_argument(
        "--purge-snapshot",
        action="store_true",
        help="合规删除：连快照一并清除，**不可恢复**",
    )

    p_restore = sub.add_parser("restore", help="从审计快照恢复被删记忆")
    p_restore.add_argument("audit_id", type=int)

    p_correct = sub.add_parser("correct", help="主动干预：修正某条记忆")
    p_correct.add_argument("mem_id")
    p_correct.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p_correct.add_argument("--reason", required=True)

    p_audit = sub.add_parser("audit", help="审计日志：谁、何时、以何因做了什么")
    p_audit.add_argument("--limit", type=int, default=50)
    p_audit.add_argument("--since", default=None)
    p_audit.add_argument("--forgetting", action="store_true", help="只看删除相关事件")

    sub.add_parser("doctor", help="自检：配置 / DB / 模型配置 / schema（纯本地，不联网）")
    sub.add_parser("reindex", help="全库重嵌入（换 embedding 模型后）")
    sub.add_parser("replay", help="从审计重放重建派生字段")

    p_opt = sub.add_parser("optimize", help="记忆优化：治理性删除与降级（默认只报告）")
    p_opt.add_argument("--apply", action="store_true", help="真正执行（默认只报告）")
    p_opt.add_argument("--autonomous", action="store_true", help="系统自主，无需确认")

    # `aspirit <cmd> --json` 与 `aspirit --json <cmd>` **都要能用**。
    # 文档与集成脚本一律写前者（T-AL1-08 的验收原文就是 `aspirit status --json`），
    # 而 argparse 的顶层 flag 只认后者——`status --json` 会直接报
    # "unrecognized arguments" 并以 2 退出。**这是被验收条件抓到的一处偏差**，
    # 修法是让两种位置都成立，而不是把文档改成实现的样子：
    # 文档全篇都在用前一种写法，改文档等于把偏差扩散到所有调用方。
    # 集中在这里给每个子命令补上，新增子命令不会漏（下面有测试钉住）。
    #
    # `default=SUPPRESS` **不能省**：`_SubParsersAction` 是先把子命令解析进一个
    # 全新命名空间、再整体回写到父命名空间，所以子解析器的默认值会**覆盖**
    # 顶层已解析出的值——写成默认 False 会让 `aspirit --json status` 反而走人类可读分支。
    for command_parser in sub.choices.values():
        command_parser.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help="以 JSON 输出（便于脚本化）",
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    home = args.home or _default_home()

    if args.command == "init":
        return _cmd_init(parser, home, args)
    if args.command == "doctor" and not Path(home).exists():
        # 没装配也能给出"配置在哪、缺什么"
        print(f"hermes_home 不存在：{home}")
        return 2

    try:
        services = _open_services(home)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"启动失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        return _dispatch(services, args)
    finally:
        # 只收线程，**不清 `applier.failures`**：那里记的是"向量化失败、已降级存储"
        # 这类信号，本次调用没来得及上报就丢弃，等于把降级悄悄吞掉。
        services.stop(drain=False)


# --------------------------------------------------------------------------- #
# 各子命令
# --------------------------------------------------------------------------- #


def _cmd_init(parser: argparse.ArgumentParser, home: str, args) -> int:
    target = config_path(home)
    if target.exists():
        print(f"配置已存在：{target}（未覆盖）")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(toml_example(), encoding="utf-8")
    print(f"已生成配置模板：{target}")
    print("下一步：填好 [models] 的 api_key_env 指向的环境变量，然后运行 aspirit status")
    return 0


def _open_services(home: str):
    from .runtime import start

    return start(home, start_threads=False)


def _dispatch(services, args) -> int:
    command = args.command
    as_json = args.json

    if command == "status":
        report = status(services)
        _emit(report, as_json, lambda: status_text(report))
        return 0

    if command == "layers":
        report = layers(services)
        _emit(
            report,
            as_json,
            lambda: "\n".join(
                f"{name:<11} 活跃 {bucket['active']:>4} · 休眠 {bucket['dormant']:>4}"
                f" · 已遗忘 {bucket['forgotten']:>3}"
                for name, bucket in report["layers"].items()
                if isinstance(bucket, dict)
            ),
        )
        return 0

    if command == "review":
        from .observability import format_review_text

        rows = services.core.review(limit=int(getattr(args, "limit", 20) or 20))
        _emit({"items": rows}, as_json, lambda: format_review_text(rows))
        return 0

    if command == "trace":
        from .observability import format_trace_text

        events = services.core.trace(args.mem_id)
        _emit({"items": events}, as_json, lambda: format_trace_text(events))
        return 0

    if command == "reflect":
        report = reflect(services)
        _emit(report, as_json, lambda: _render_reflect(report))
        return 0

    if command == "export":
        services.backend.export_archive(args.path, fmt=args.fmt)
        print(f"已导出到 {args.path}（{args.fmt}）")
        return 0

    if command == "import":
        # **走与在线写入同一条路径**（T-AL2-23）：解析在 AL3、翻译在 AL2、
        # 落库经 `write_now`（CLI 无后台线程时的单写者入口）。
        # 原先直调 `backend.import_archive` 会**绕过写队列与审计**——
        # 而传承导入恰恰是最需要留痕的操作："这批是什么时候、从哪进来的"。
        from .store.archive import load_archive

        pack = load_archive(args.path)
        report = services.core.restore_pack(pack)
        services.write_now(report.intents)
        data = {
            "imported": report.imported,
            "skipped": report.skipped,
            "relations": report.relations,
            "entities": report.entities,
            "restored_superseded": report.restored_superseded,
            "errors": report.errors,
        }
        _emit(
            data,
            as_json,
            f"重建完成：新增 {report.imported} 条、跳过 {report.skipped} 条"
            + (f"、{len(report.errors)} 条未能重建" if report.errors else ""),
        )
        return 0

    if command == "ingest":
        from .common import now_iso
        from .core.base import TurnEvent

        intents = services.core.ingest_turn(
            TurnEvent(
                session_id=args.session_id, user=args.text, assistant="", ts=now_iso()
            )
        )
        services.write_now(intents)
        print(f"已导入素材（会话 {args.session_id}），产出 {len(intents)} 条写意图")
        return 0

    if command == "consolidate":
        report = services.core.consolidate(session_id=args.session_id)
        services.write_now(report.intents)
        text = (
            f"未执行（{report.skipped}）"
            if report.skipped
            else f"已巩固：情景 {report.episode_id}，提升 {len(report.promoted)} 条"
        )
        _emit(
            {"episode_id": report.episode_id, "promoted": report.promoted, "skipped": report.skipped},
            as_json,
            text,
        )
        return 0

    if command == "decay":
        report = services.core.decay(dry_run=not args.apply)
        if args.apply:
            services.write_now(report.intents)
        lines = [
            (
                f"扫描 {report.total} 条，强度变化 {report.recomputed} 条"
                f"（{'已写回' if args.apply else '**预演，未写回**'}）"
            ),
            f"低强度候选 {len(report.low_strength)} 条",
            "降级/删除：**本命令不做**（治理性删除由 aspirit optimize 发起）",
        ]
        _emit(
            {
                "total": report.total,
                "recomputed": report.recomputed,
                "low_strength": report.low_strength,
                "dry_run": report.dry_run,
            },
            as_json,
            "\n".join(lines),
        )
        return 0

    if command == "forget":
        if not args.apply:
            record = services.backend.get(args.mem_id)
            if record is None:
                print(f"没有找到记忆：{args.mem_id}")
                return 1
            print(
                f"【预演】将要删除 {args.mem_id}：{record.abstract or record.content}\n"
                f"理由：{args.reason}\n"
                + ("⚠️ 合规删除：快照将被清除，**不可恢复**" if args.purge_snapshot else "（将保留快照，可用 aspirit restore 找回）")
                + "\n确认执行请追加 --apply"
            )
            return 0
        intents = services.core.forget(
            args.mem_id,
            reason=args.reason,
            source="cli:aspirit forget",
            purge_snapshot=args.purge_snapshot,
        )
        services.write_now(intents)
        print(f"已删除 {args.mem_id}")
        return 0

    if command == "restore":
        services.write_now(services.core.restore(args.audit_id))
        print(f"已从审计 {args.audit_id} 恢复")
        return 0

    if command == "correct":
        patch = _parse_set(args.set)
        services.write_now(services.core.correct(args.mem_id, patch, reason=args.reason))
        print(f"已修正 {args.mem_id}：{sorted(patch)}")
        return 0

    if command == "audit":
        view = AuditView(services.backend)
        if args.forgetting:
            entries = view.forgetting(limit=args.limit)
            # 删除账本后面**接上救援面板**：看到"删了什么"的同时就能看到"还能救什么、
            # 怎么救"。这两者本来就是同一个问题的两面，分成两个命令反而会让人漏掉后半句。
            restorable = view.restorable()
            _emit(
                {"items": entries, "restorable": restorable},
                as_json,
                lambda: format_audit_text(entries) + "\n\n" + format_restorable_text(restorable),
            )
            return 0
        entries = view.entries(since=args.since, limit=args.limit)
        _emit({"items": entries}, as_json, lambda: format_audit_text(entries))
        return 0

    if command == "doctor":
        report = doctor(services)
        _emit(report, as_json, lambda: doctor_text(report))
        return 0 if report["ok"] else 1

    if command == "reindex":
        from .store.reindex import reindex

        provider = services.core.embedding
        if provider is None:
            print("embedding 不可用，无法重嵌入。请先配置 [models.embedding]。")
            return 2
        result = reindex(
            services.backend,
            lambda texts: provider.embed(texts),
            model=provider.model,
            dim=provider.dim,
        )
        _emit(result, as_json, f"重嵌入完成：{result}")
        return 0

    if command == "replay":
        result = services.backend.replay_derived()
        _emit(result, as_json, f"派生字段已从审计重建：{result}")
        return 0

    if command == "optimize":
        report = services.core.optimize(
            dry_run=not args.apply, autonomous=args.autonomous
        )
        if args.apply:
            services.write_now(report.intents)
        lines = [
            f"扫描 {report.scanned} 条",
            f"不可达候选 {len(report.unreachable)} 条 · 降级候选 {len(report.downgrades)} 条",
            f"处置意图 {len(report.intents)} 条（{'已执行' if args.apply else '**仅报告**'}）",
        ]
        lines.extend(f"  · {note}" for note in report.notes)
        _emit(
            {
                "scanned": report.scanned,
                "unreachable": report.unreachable,
                "downgrades": report.downgrades,
                "notes": report.notes,
                "intents": len(report.intents),
            },
            as_json,
            "\n".join(lines),
        )
        return 0

    print(f"未知命令：{command}", file=sys.stderr)  # pragma: no cover
    return 2


# --------------------------------------------------------------------------- #
# 内部
# --------------------------------------------------------------------------- #


def _default_home() -> str:
    import os

    return os.environ.get("ARTIFACT_SPIRIT_HOME") or os.getcwd()


def _emit(payload: dict, as_json: bool, text: str | Callable[[], str]) -> None:
    """输出结果。**`--json` 时绝不渲染人类文本**（P2-5）。

    `text` 接受可调用对象：调用点写的是 `status_text(report)` 这样的**实参**，
    在 JSON 分支里同样会被求值。而人类渲染要遍历整份报告，是命令里最容易抛异常的
    一段——让它和 `--json` 共享命运，就是"脚本的输出被人类渲染的异常绑架"，
    可脚本恰恰比人更依赖 `--json`（人还能靠肉眼读输出，脚本只能靠退出码）。
    所以凡是要遍历数据/调 `format_*` 的文本一律传 lambda；
    纯 f-string 拼接没有抛异常的风险，直接传字符串即可。
    """
    if as_json:
        _print_out(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        _print_out(text() if callable(text) else text)


def _print_out(text: str) -> None:
    """打印，但**不让控制台编码把命令打崩**。

    Windows 默认控制台是 GBK，装不下 `✅` / `⚠️` 这类符号——
    而 `status` / `doctor` 的输出里全是它们。不兜底的话，**排障入口自己先抛异常退出**
    （真机上复现过：`UnicodeEncodeError` → traceback → exit 1），
    对着报错的人却看不到任何体检结果。

    做法是先试编码、不行就把不可表示的字符替换掉：输出变形可以接受，
    命令崩掉不行。UTF-8 环境下这段是零开销的直通。
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = text.encode(encoding, "replace").decode(encoding, "replace")
    print(text)


def _render_reflect(report: dict) -> str:
    lines = [
        f"记忆总量 {report['total']} · 平均强度 {report['average_strength']:.3f}",
        "",
        "【为什么遗忘】" + report["why_forget"]["explain"],
        f"  降级候选 {len(report['why_forget']['downgrade_candidates'])} 条",
        f"  不可达候选 {len(report['why_forget']['unreachable_candidates'])} 条",
        "",
        "【为什么召回】" + report["why_recall"]["explain"],
        "  权重：" + " / ".join(f"{k} {v}" for k, v in report["why_recall"]["weights"].items()),
        "",
        f"【正在成为候选信念】{len(report['becoming_beliefs'])} 条",
    ]
    for mem_id in report["becoming_beliefs"][:10]:
        lines.append(f"  · {mem_id}")
    return "\n".join(lines)


def _parse_set(pairs: list[str]) -> dict:
    patch: dict = {}
    for item in pairs:
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            continue
        patch[key] = _coerce(raw.strip())
    return patch


def _coerce(raw: str):
    if raw.casefold() in ("true", "false"):
        return raw.casefold() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
