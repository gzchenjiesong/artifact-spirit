"""可观测性：``status`` / ``layers`` / ``reflect`` / ``audit_view`` / ``doctor``
（LLD-AL5 §5 M5 · T-AL5-06 / 07 / 12）。

**全部是 AL3 之上的只读投影**，无副作用（INV-1）。

一个刻意的设计选择：**默认给 L0 级信息，细节按需下钻**。
`status` 一次糊出几千行等于没给信息——这和分级加载是同一个道理，
只是作用在"人读界面"而不是"模型上下文"上。

**可读 ≠ 可溯**：查当前状态看投影；查历史看 `audit` 事件表。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from ..common import now_iso
from .audit_view import AuditView

if TYPE_CHECKING:  # pragma: no cover
    from ..runtime.lifecycle import Services

__all__ = ["audit_view", "doctor", "doctor_text", "layers", "reflect", "status", "status_text"]


def status(services: Services) -> dict:
    """五区块状态报告——**排障的主要入口**。

    | 区块 | 内容 |
    |---|---|
    | 身份 | spirit_name / spirit_id / schema 版本 |
    | 层计数 | 各层 active·dormant·forgotten |
    | 健康度 | 存储规模、平均强度、待重算概览 |
    | **生效模型链** | `resolve_chain` 的实际结果（看清是否走了 fallback） |
    | **降级告警** | 当前处于降级状态的子系统 |
    """
    cfg = services.config
    backend = services.backend
    core = services.core

    health = core.health()
    counts = health.layer_counts

    return {
        "identity": {
            "name": backend.meta_get("spirit_name") or cfg.name or "(未命名)",
            "spirit_id": backend.meta_get("spirit_id") or "",
            "schema_version": backend.meta_get("schema_version") or "",
            "hermes_home": cfg.hermes_home,
            "db_path": cfg.db_path,
            "config_source": cfg.source,
        },
        "layers": {
            "episodic": _counts_for(counts, "episodic"),
            "semantic": _counts_for(counts, "semantic"),
            "procedural": _counts_for(counts, "procedural"),
            "core": _counts_for(counts, "core"),
            "total": health.total,
        },
        "health": {
            "total": health.total,
            "core_count": health.core_count,
            "avg_strength": health.avg_strength,
            "stale_overviews": health.stale_overviews,
            "write_queue_depth": services.writer.queue.depth(),
            "write_queue_max": services.writer.queue.maxsize,
            "queue": asdict(services.writer.queue.stats),
            "writer_running": services.writer.running,
            "maintenance_running": services.maintenance.running,
        },
        "models": {
            "llm_chain": services.resolver.resolve_chain("extract"),
            "embedding_chain": services.resolver.embedding_chain(),
        },
        "degradations": _degradations(services),
    }


def _counts_for(counts: dict, layer: str) -> dict:
    bucket = counts.get(layer, {})
    return {
        "active": int(bucket.get("active", 0)),
        "dormant": int(bucket.get("dormant", 0)),
        "forgotten": int(bucket.get("forgotten", 0)),
    }


def _degradations(services: Services) -> list[str]:
    """降级告警。

    这是 `status` 里最容易被写空、却最有价值的一块——
    用户遇到"记忆好像不太灵"时，第一件事就是看这里有没有告警。
    """
    alerts: list[str] = []
    if not services.resolver.embedding_available():
        alerts.append("BM25 模式：embedding 不可用，召回已降级为纯关键词")
    if not services.resolver.llm_available():
        alerts.append("仅存原文：LLM 不可用，提取已降级（记忆不丢，只是未结构化）")
    stats = services.writer.queue.stats
    if stats.dropped:
        alerts.append(f"写队列已丢弃 {stats.dropped} 个任务（低优先级先行）")
    if stats.failed:
        alerts.append(f"写任务失败 {stats.failed} 次（详见 writer.errors）")
    if services.writer.errors:
        alerts.append(f"最近一次写失败：{services.writer.errors[-1][1]}")
    if services.maintenance.errors:
        alerts.append(f"最近一次维护失败：{services.maintenance.errors[-1]}")
    # 配置校验的**告警**必须在这里可见：T-AL5-04 的验收是"不阻断，但要可见"，
    # 而 warnings 此前只被写进 Services、没有任何出口（DES-REV-008 P1-45）。
    alerts.extend(f"配置告警：{warning}" for warning in services.warnings)
    alerts.extend(services.notes)
    return alerts


def layers(services: Services) -> dict:
    """各层记忆分布（`layers` 视图）。

    **是四层不是五层**：感觉记忆与工作记忆不落 ``memories`` 表
    （``store.base.Layer`` 的注释），LLD-AL5 §2.3 原先写"五类"是错的。
    """
    report = status(services)
    return {
        "layers": report["layers"],
        "total": report["health"]["total"],
    }


def reflect(services: Services) -> dict:
    """记忆健康度报告：**必须能解释"为什么遗忘 / 为什么召回 / 哪些在成为候选信念"**。"""
    core = services.core
    health = core.health()
    decay = core.decay(dry_run=True)
    optimization = core.optimize(dry_run=True)
    promote_candidates = [
        record.id
        for record in services.backend.query(status="active", limit=500)
        if record.access_count >= 2 and record.layer == "episodic"
    ]

    return {
        "layer_counts": health.layer_counts,
        "total": health.total,
        "average_strength": health.avg_strength,
        "why_forget": {
            "explain": (
                "衰减只影响排序（D-16）；物理删除仅由记忆优化任务发起（D-17），"
                "判据是**图结构不可达**，与时间、访问频率无关（D-23）"
            ),
            "downgrade_candidates": optimization.downgrades,
            "unreachable_candidates": optimization.unreachable,
            "notes": optimization.notes,
        },
        "why_recall": {
            "explain": "六因子加权融合：语义 / 重要度 / 时间邻近 / 实体 / 扩散 / 核心一致性",
            "weights": services.core.settings.recall.as_dict(),
        },
        "becoming_beliefs": promote_candidates[:20],
        "low_strength": decay.low_strength[:20],
        "stale_overviews": health.stale_overviews,
    }


def audit_view(
    services: Services, *, since: str | None = None, limit: int = 50
) -> list[dict]:
    """审计视图（只读投影，按时间倒序）。

    **委托给 :class:`AuditView`，不另建一份投影。**

    这里曾经自己拼一遍字典，而且编号用的是 ``enumerate(..., start=1)``——
    也就是"**过滤结果里的第几条**"。``AuditView.entries`` 的注释早就点名了这个
    陷阱：``--since`` 下位置编号会与真实 ``audit_id`` 错位，而 ``aspirit restore``
    只认真实 id，用户照着错位的编号去恢复，**恢复的就是另一条记忆**。
    同一个账本两份投影、两种编号语义，迟早会有一处被接上 CLI 而没带注释
    （DES-REV-008 P1-47）。删掉重复投影比修正它更省事，也更难再错。
    """
    return AuditView(services.backend).entries(since=since, limit=limit)


def doctor(services: Services) -> dict:
    """自检：逐项体检，**每项独立报告通过/失败**。

    输出刻意保持"逐项 + 可操作建议"的形态——`doctor` 的读者是遇到问题的人，
    不是日志分析系统。

    **全程本地，不联网**（INV-5）：模型相关项检查的是"**配置**是否可用"
    （`api_key_env` 有没有配、生效模型链解析到谁），不是"网关连不连得通"。
    探活会把一个本可离线跑通的命令变成依赖网络与额度的命令，
    真实连通性属于集成验收（LLD-AL4 §9）。
    """
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, hint: str = "") -> None:
        checks.append({"check": name, "ok": ok, "detail": detail, "hint": hint})

    cfg = services.config
    add(
        "配置装载",
        True,
        f"来源：{cfg.source}；hermes_home={cfg.hermes_home or '(未提供)'}",
        "" if cfg.source == "file" else "未找到配置文件——可在 hermes_home 下运行 aspirit init 生成",
    )

    try:
        meta = services.backend.meta_get("schema_version")
        add("数据库", True, f"路径 {cfg.db_path}；schema 版本 {meta}")
    except Exception as exc:
        add("数据库", False, f"{type(exc).__name__}: {exc}", "检查磁盘空间与文件权限")

    dim = services.backend.embedding_dim
    configured = cfg.embedding_dim
    add(
        "向量维度一致性",
        dim == configured,
        f"库中 {dim} / 配置 {configured}",
        "" if dim == configured else "维度不一致需执行 aspirit reindex",
    )

    # ---- M3–M5 新增能力的体检（T-AL5-13）----
    temporal_ok, temporal_detail = _check_temporal(services)
    add(
        "双时态字段",
        temporal_ok,
        temporal_detail,
        "" if temporal_ok else "缺 valid_from / valid_to / superseded_by——库过旧，需重建",
    )

    add(
        "档案格式",
        True,
        f"archive_version={services.backend.archive_version()}"
        "（与 schema_version 分开演化）",
    )

    emb_ok = services.resolver.embedding_available()
    add(
        "Embedding 配置",
        emb_ok,
        "已配置" if emb_ok else "未配置或密钥缺失",
        "" if emb_ok else "配置 [models.embedding] 后召回将启用向量路",
    )
    llm_ok = services.resolver.llm_available()
    add(
        "LLM 配置",
        llm_ok,
        "已配置" if llm_ok else "未配置",
        "" if llm_ok else "配置 [models.llm]，或让宿主提供模型服务",
    )

    add(
        "写队列",
        services.writer.queue.depth() < services.writer.queue.maxsize,
        f"深度 {services.writer.queue.depth()}/{services.writer.queue.maxsize}"
        f"；已处理 {services.writer.queue.stats.processed}",
    )
    add(
        "线程状态",
        True,
        f"writer={'运行' if services.writer.running else '未启动'}"
        f" · maintenance={'运行' if services.maintenance.running else '未启动'}",
    )

    try:
        report = services.backend.reconcile(dry_run=True)
        add(
            "向量对账",
            not report["missing_vectors"] and not report["orphan_vectors"],
            f"缺失 {len(report['missing_vectors'])} · 孤儿 {len(report['orphan_vectors'])}",
            "运行 aspirit reindex 或重启以自动对账",
        )
    except Exception as exc:
        add("向量对账", False, f"{type(exc).__name__}: {exc}")

    return {
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
        "generated_at": now_iso(),
    }


# --------------------------------------------------------------------------- #
# 人类可读渲染
# --------------------------------------------------------------------------- #


def status_text(report: dict) -> str:
    """把 status 渲染成人能直接读的文本（默认视图，`--json` 用于脚本化）。"""
    identity = report["identity"]
    lines = [
        f"器灵 {identity['name']}",
        f"  身份：{identity['spirit_id'] or '(未初始化)'} · schema v{identity['schema_version']}",
        f"  Hermes 目录：{identity['hermes_home'] or '(未提供)'}",
        f"  配置来源：{identity['config_source']}",
        "",
        "层计数",
    ]
    for layer in ("episodic", "semantic", "procedural", "core"):
        bucket = report["layers"][layer]
        lines.append(
            f"  {layer:<11} 活跃 {bucket['active']:>4} · 休眠 {bucket['dormant']:>4}"
            f" · 已遗忘 {bucket['forgotten']:>3}"
        )
    lines.append(f"  {'合计':<11} {report['layers']['total']}")

    health = report["health"]
    lines.extend(
        [
            "",
            "健康度",
            (
                f"  平均强度 {health['avg_strength']:.3f}"
                f" · 待重算概览 {health['stale_overviews']}"
            ),
            (
                f"  写队列 {health['write_queue_depth']}/{health['write_queue_max']}"
                f" · 线程 writer={'运行' if health['writer_running'] else '停'}"
                f" / maintenance={'运行' if health['maintenance_running'] else '停'}"
            ),
            "",
            "生效模型链",
        ]
    )
    for segment in report["models"]["llm_chain"]:
        mark = "✅" if segment.get("active") else "  "
        reason = segment.get("reason") or segment.get("model") or ""
        lines.append(f"  {mark} {segment.get('source', '?'):<8} {reason}")
    lines.append("  embedding:")
    for segment in report["models"]["embedding_chain"]:
        mark = "✅" if segment.get("active") else "  "
        reason = segment.get("reason") or segment.get("model") or ""
        lines.append(f"    {mark} {segment.get('source', '?'):<8} {reason}")

    lines.extend(["", "降级告警"])
    alerts = report.get("degradations") or []
    lines.extend([f"  ⚠️ {alert}" for alert in alerts] or ["  （无）"])
    return "\n".join(lines)


def _check_temporal(services: Services) -> tuple[bool, str]:
    """双时态三列是否可用（T-AL5-13）。

    用 ``SELECT ... LIMIT 0`` 而不是"读一条看字段在不在"：前者**不依赖库里有没有数据**
    ——空库也应该能通过这项体检，而"读一条"在空库上会假报"字段缺失"。
    体检工具误报的代价是：用户会去修一个根本不存在的问题。
    """
    try:
        services.backend.conn.execute(
            "SELECT valid_from, valid_to, superseded_by FROM memories LIMIT 0"
        ).fetchall()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "valid_from / valid_to / superseded_by 均可用"


def doctor_text(report: dict) -> str:
    lines = [f"器灵自检 · {'全部通过' if report['ok'] else '存在问题'}", ""]
    for check in report["checks"]:
        mark = "✅" if check["ok"] else "❌"
        lines.append(f"{mark} {check['check']}：{check['detail']}")
        if not check["ok"] and check.get("hint"):
            lines.append(f"    → {check['hint']}")
    return "\n".join(lines)
