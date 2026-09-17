"""组合根：装配依赖、启停顺序（LLD-AL5 §5 M8 · T-AL5-03）。

**这是唯一允许 import 各层具体实现的位置**——它是依赖注入容器，
因此它是 AL5 边界规则的**单文件例外**（R10；AL5 其余实现不得依赖业务层实现），
而不是"R1 不约束组合根"（那是把 AL5 的边界说成别人的边界）。

## 启停顺序不可颠倒

```
start:  装载配置 → 校验 → 打开 AL3 + migrate → 校验 embedding 一致性（INV-2）
        → 起 writer → 起 maintenance → 启动对账
stop:  停止接收 → flush 写队列 → 停 maintenance → 停 writer → 关 AL3
```

**writer 必须先于 maintenance 停止**——反过来，maintenance 最后一批写意图会
投进一个已经停掉的队列，那些工作静默消失。顺序写在这里，也断言在测试里。
这条顺序只有在维护产出**确实经写队列**时才是承重的（`enqueue`）。
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..common import estimate_tokens
from ..config import ConfigError, SpiritConfig, load, validate
from ..core import ArtifactSpiritCore, CoreSettings, WriteIntent
from ..core.base import RecallWeights
from ..core.recall import DecayParams
from ..core.salience import SalienceConfig, SalienceWeights
from ..model import EmbeddingProvider, HttpTransport, LLMProvider, ModelResolver
from ..store.base import MemoryBackend
from ..store.sqlite_backend import SQLiteBackend
from .maintenance import Maintenance
from .threading_ import ThreadFactory, default_thread_factory
from .timeout import TimeoutRunner
from .writer import WriteQueue, Writer, intent_priority

__all__ = ["IntentApplier", "Services", "start", "stop"]


class IntentApplier:
    """把写意图翻译成 AL3 调用。

    **只做翻译，不做决策**——任何"这条记忆重不重要"的判断都不在这里
    （那是 AL2 的事，见 LLD-AL5 §6 C8）。
    """

    def __init__(self, *, backend: MemoryBackend, embedding: EmbeddingProvider | None = None) -> None:
        self.backend = backend
        self.embedding = embedding
        self.failures: list[str] = []

    def apply(self, intent: WriteIntent) -> None:
        handler = getattr(self, f"_op_{intent.op}", None)
        if handler is None:  # pragma: no cover - 未知意图要立刻暴露
            raise ValueError(f"未知写意图：{intent.op}")
        handler(intent)

    # ------------------------------------------------------------------ #
    # 各 op 的落地
    # ------------------------------------------------------------------ #

    def _op_put(self, intent: WriteIntent) -> None:
        record = intent.record
        assert record is not None
        vector = self._vectorize(intent.embed_text or record.content)
        self.backend.put(record, vector, audit=intent.audit)

    def _op_update(self, intent: WriteIntent) -> None:
        patch = dict(intent.patch or {})
        self._refresh_vector_if_needed(intent, patch)
        self.backend.update(intent.mem_id, patch, audit=intent.audit)

    def _op_set_status(self, intent: WriteIntent) -> None:
        self.backend.set_status(
            intent.mem_id, intent.status, reason=intent.reason or "", actor=intent.actor
        )

    def _op_touch(self, intent: WriteIntent) -> None:
        self.backend.touch(intent.mem_id, intent.ts or "", strength=intent.strength)

    def _op_link(self, intent: WriteIntent) -> None:
        self.backend.link(
            intent.a_kind, intent.a_id, intent.b_kind, intent.b_id,
            intent.rel_type, intent.weight,
        )

    def _op_reinforce(self, intent: WriteIntent) -> None:
        self.backend.reinforce(
            intent.a_id, intent.b_id, intent.delta,
            rel_type=intent.rel_type or "co_activation",
        )

    def _op_forget(self, intent: WriteIntent) -> None:
        if not intent.reason:
            raise ValueError("删除必须带 reason（D-17）")
        self.backend.hard_delete(
            intent.mem_id,
            reason=intent.reason,
            purge_snapshot=intent.purge_snapshot,
            actor=intent.actor,
        )

    def _op_restore(self, intent: WriteIntent) -> None:
        self.backend.restore_from_audit(intent.audit_id)

    def _op_audit(self, intent: WriteIntent) -> None:
        if intent.audit is not None:
            self.backend.audit(intent.audit)

    def _op_overview_invalidate(self, intent: WriteIntent) -> None:
        self.backend.overview_invalidate(
            scope_kind=intent.scope_kind, scope_id=intent.scope_id
        )

    def _op_overview_put(self, intent: WriteIntent) -> None:
        content = intent.overview_content or ""
        self.backend.overview_put(
            intent.scope_kind, intent.scope_id, intent.overview_level, content,
            token_count=intent.token_count or estimate_tokens(content),
            model=intent.model,
        )

    def _op_wm_put(self, intent: WriteIntent) -> None:
        self.backend.wm_put(
            intent.session_id, intent.chunk_key, intent.content, intent.salience
        )

    def _op_wm_delete(self, intent: WriteIntent) -> None:
        self.backend.wm_clear(intent.session_id)

    def _op_session_create(self, intent: WriteIntent) -> None:
        self.backend.session_create(intent.session_id or "", intent.ts or "")

    def _op_session_end(self, intent: WriteIntent) -> None:
        self.backend.session_end(intent.session_id or "", intent.ts or "")

    def _op_entity_upsert(self, intent: WriteIntent) -> None:
        self.backend.entity_upsert(
            intent.entity_name, intent.entity_type, aliases=list(intent.aliases)
        )

    # ------------------------------------------------------------------ #

    def _vectorize(self, text: str) -> list[float] | None:
        if self.embedding is None:
            return None
        try:
            return self.embedding.embed([text or ""])[0]
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            return None

    def _refresh_vector_if_needed(self, intent: WriteIntent, patch: dict) -> None:
        """内容变了就重算向量——否则检索会命中"旧意义"的向量。"""
        if not ({"content", "abstract", "subject", "object"} & set(patch)):
            return
        if self.embedding is None:
            return
        record = self.backend.get(intent.mem_id)
        if record is None:
            return
        merged = {**{k: getattr(record, k) for k in patch}, **patch}
        text = str(merged.get("abstract") or merged.get("content") or "")
        vector = self._vectorize(text)
        if vector is None:
            return
        try:
            self.backend.set_vector(intent.mem_id, vector)
        except Exception as exc:
            self.failures.append(f"vector_refresh: {type(exc).__name__}: {exc}")


@dataclass
class Services:
    """装配完成的运行时服务集合。`aspirit` CLI 与 provider 共用这一份。"""

    config: SpiritConfig
    backend: SQLiteBackend
    resolver: ModelResolver
    core: ArtifactSpiritCore
    writer: Writer
    maintenance: Maintenance
    applier: IntentApplier
    timeout_runner: TimeoutRunner = field(default_factory=TimeoutRunner)
    warnings: list[str] = field(default_factory=list)
    started: bool = False
    notes: list[str] = field(default_factory=list)
    sync_write_timeout: float = 5.0
    """:meth:`write_now` 等待 writer 落库的超时（有 writer 线程时走这条路，见该方法）。"""
    """装配过程中的可读说明（如"未配置 embedding，已降级为关键词模式"）。"""

    _stop_sequence: list[str] = field(default_factory=list)
    _stopped: bool = False

    # ------------------------------------------------------------------ #
    # 写路径
    # ------------------------------------------------------------------ #

    def submit(self, payload: object, *, kind: str = "intent.memory", priority: int | None = None) -> bool:
        """非阻塞投递（宿主主线程用）。"""
        return self.writer.submit(payload, kind=kind, priority=priority)

    def enqueue(self, intents: list[WriteIntent]) -> None:
        """把意图投递给单写者（非阻塞）。**所有写路径都走这里。**

        热路径（宿主主线程）与**离线产出（维护线程）**共用它：维护线程直接
        ``write_now`` 会绕开队列，让 AL3 出现第二条写连接——``SQLiteBackend``
        的写连接按线程 id 分配，换线程写会先关掉 writer 的写连接再建一条，
        于是"单写者"（D-03 / C10）当场作废，而且是**静默**的
        （T-AL5-09 要求 3"其写操作投递到同一 write_queue，不持有独立写连接"；
        DES-REV-008 P0-12）。
        """
        for intent in intents:
            self.writer.submit(
                intent, kind=f"intent.{intent.op}", priority=intent_priority(intent.op)
            )

    def write(self, intents: list[WriteIntent]) -> None:
        """热路径写入口（宿主主线程用）——与 :meth:`enqueue` 同义。"""
        self.enqueue(intents)

    def write_now(self, intents: list[WriteIntent]) -> None:
        """**同步**落库——返回时已经写进库（CLI / 工具 / 测试需要确定的时序）。

        语义是"同步"，但**写者仍然唯一**：只要 writer 线程在跑，这里就把意图投队列、
        再等它落库。**不这样做会踩一个静默的坑**——``SQLiteBackend._write_connection()``
        按**线程 id** 分配写连接，换线程写会先 `close()` 掉对方那条再新建一条：

        - "单写者"（D-03）在**最热的那条路径**上作废：`tools/handlers.py` 的六个工具
          （remember / forget / restore / correct / expand / consolidate）都调本方法，
          而它们跑在**宿主主线程**上；
        - 每次调用还要额外付一次建连接的代价（PRAGMA + 函数注册 + 扩展加载）。

        （DES-REV-008 P0-15。原实现**无条件直接应用**，只在 docstring 里叮嘱"调用方必须
        满足此刻没有 writer 线程"——那是把正确性托付给自觉。）

        两种情况仍直接应用：

        - 没有 writer 线程（CLI、``start_threads=False`` 的确定性测试）；
        - **本方法恰好在 writer 线程内被调用**——再投队列会自己等自己，直接死锁。
        """
        if not intents:
            return
        if self.writer.running and self.writer.thread_id != threading.get_ident():
            self.enqueue(intents)
            if not self.writer.queue.flush(timeout=self.sync_write_timeout):
                self.notes.append(
                    f"同步写入未在 {self.sync_write_timeout:g}s 内完成（意图已入队，落库会继续）"
                )
            return
        for intent in intents:
            self.applier.apply(intent)

    def drain_pending(self) -> list[WriteIntent]:
        """取走核心层的读路径意图（如 L1 概览重算请求）并落库。"""
        intents = self.core.drain_pending()
        self.write_now(intents)
        return intents

    def flush(self, timeout: float = 5.0) -> bool:
        return self.writer.queue.flush(timeout=timeout)

    def run_with_timeout(self, action, timeout: float, fallback):
        """带护栏地执行（AL1 的 ``prefetch`` 用它）——线程只在本层创建（R8）。"""
        return self.timeout_runner(action, timeout, fallback)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def stop(self, *, drain: bool = True) -> list[str]:
        """按约定顺序收尾。**幂等**——可能被调用多次。

        即使从未起线程（CLI / 测试的确定性场景），也照常给出完整序列：
        "关掉了什么"应当如实反映，而不是因为没起线程就变成空。
        """
        if self._stopped:
            return self._stop_sequence

        sequence: list[str] = ["stop_accepting", "flush_write_queue"]
        if drain and self.writer.running:
            # ``drain=False`` 就该是"不等"：以前这里无条件 flush，于是"不 drain"
            # 的收尾仍要白等满 5 秒，而残留任务数又照报 0。
            #
            # 同理，**没有消费者时等待没有任何意义**：任务只减不增，但没人来取，
            # ``flush`` 只能等满超时（Windows 上实测约 7.8 秒，因为
            # ``Condition.wait(0.01)`` 的实际粒度粗得多）。宿主没起线程
            # （CLI / 确定性测试）时如实报残留即可（DES-REV-008 P1-51）。
            self.writer.queue.flush(timeout=5.0)

        if self.started:
            # **先停 maintenance 再停 writer**（顺序颠倒会让维护产出投进已停的队列）。
            # 注意 drain 里的维护产出是 ``enqueue`` 进写队列的，因此下一条
            # ``writer.stop(drain=True)`` 才是它们真正落库的时刻。
            sequence.append("stop_maintenance")
            self.maintenance.stop(drain=drain)
            sequence.append("stop_writer")
            remaining = self.writer.stop(drain=drain)
        else:
            # 没有消费者：等待不可能成功，直接如实计数（同样是"丢了多少"的报案）。
            remaining = self.writer.queue.pending()

        if remaining:
            # LLD-AL5 §7 F5：写队列不持久化（M6 的取舍），但"丢了多少"
            # 必须报出来——报 0 会让 C3 的验收变成恒真（DES-REV-008 P0-12）。
            sequence.append(f"undrained:{remaining}")
            self.notes.append(
                f"收尾时仍有 {remaining} 个写任务未完成（flush 超时、未 drain 或无消费者）"
                "——写队列不持久化，这些是可重建的加工任务，需要重建而非恢复"
            )

        sequence.append("close_backend")
        self.backend.close()

        self.started = False
        self._stopped = True
        self.resolver.close()
        self._stop_sequence = sequence
        return sequence

    @property
    def stop_sequence(self) -> list[str]:
        """上一次 stop 的调用序列（测试用它断言顺序）。"""
        return list(self._stop_sequence)


# --------------------------------------------------------------------------- #
# 装配
# --------------------------------------------------------------------------- #


def start(
    hermes_home: str,
    *,
    config: SpiritConfig | None = None,
    env: Mapping[str, str] | None = None,
    transport: HttpTransport | None = None,
    embedding: EmbeddingProvider | None = None,
    llm: LLMProvider | None = None,
    start_threads: bool = True,
    reconcile: bool = True,
    thread_factory: ThreadFactory | None = None,
) -> Services:
    """装配并启动。

    ``start_threads=False`` 时不创建任何线程——测试与 CLI 的确定性场景用这个。

    ``thread_factory`` 是**宿主 profile 隔离的注入点**：宿主 Hermes 的
    ``spawn_context_thread`` 会在调用方的 contextvars 里起线程，而一个用空上下文
    启动的 worker 会静默写到默认 profile 的库里去。

    探测宿主的动作**不在这里**：组合根不 import 宿主（R4），由 AL1 的
    ``provider.resolve_host_thread_factory()`` 探测后传入。传 ``None`` 表示
    "没有宿主"（CLI、独立测试），退回标准库实现。
    """
    cfg = config or load(hermes_home, env=env)
    problems = validate(cfg)
    fatal = [p for p in problems if not p.startswith("警告")]
    warnings = [p for p in problems if p.startswith("警告")]
    if fatal:
        raise ConfigError("配置校验未通过：\n- " + "\n- ".join(fatal))

    backend = SQLiteBackend(
        cfg.db_path,
        embedding_dim=cfg.embedding_dim,
        embedding_model=(cfg.embedding.model if cfg.embedding and cfg.embedding.model else None),
    )
    backend.open()

    # INV-2：embedding 模型变更必须拒绝启动，否则写入与检索的向量空间不一致
    backend.assert_embedding_model(cfg.embedding.model if cfg.embedding else None)

    resolver = ModelResolver(
        llm=cfg.llm.as_dict() if cfg.llm else None,
        embedding=cfg.embedding.as_dict() if cfg.embedding else None,
        hermes_home=cfg.hermes_home or hermes_home,
        env=env,
        transport=transport,
    )

    notes: list[str] = []
    embed_provider = embedding
    if embed_provider is None:
        try:
            embed_provider = resolver.embedding()
        except Exception as exc:
            notes.append(
                f"embedding 不可用（{type(exc).__name__}）→ 召回降级为关键词（BM25）模式；"
                "配置好 [models.embedding] 后自动恢复"
            )

    llm_provider = llm
    if llm_provider is None:
        try:
            llm_provider = resolver.llm("extract")
        except Exception as exc:
            notes.append(
                f"LLM 不可用（{type(exc).__name__}）→ 提取降级为'仅存原文'；"
                "配置好 [models.llm] 或宿主的模型服务后自动恢复"
            )

    core = ArtifactSpiritCore(
        backend=backend,
        embedding=embed_provider,
        llm=llm_provider,
        settings=_core_settings(cfg),
    )
    applier = IntentApplier(backend=backend, embedding=embed_provider)

    # 线程铸造策略：由 AL1 探测宿主后注入（继承 profile 上下文），
    # 未注入则用标准库。**这是 profile 隔离能否成立的关键注入点**。
    make_thread = thread_factory or default_thread_factory

    writer = Writer(
        backend=backend,
        apply=lambda task: _apply_task(task, core=core, applier=applier, services_holder=holder),
        queue=WriteQueue(maxsize=int(cfg.worker.get("write_queue_max", 1000))),
        thread_factory=make_thread,
    )
    maintenance = Maintenance(
        writer=writer,
        handler=lambda payload: _maintenance_task(payload, core=core, holder=holder),
        on_cycle=lambda: _maintenance_cycle(core=core, holder=holder),
        interval_seconds=float(cfg.worker.get("maintenance_interval_min", 30)) * 60.0,
        thread_factory=make_thread,
    )

    services = Services(
        config=cfg,
        backend=backend,
        resolver=resolver,
        core=core,
        writer=writer,
        maintenance=maintenance,
        applier=applier,
        timeout_runner=TimeoutRunner(thread_factory=make_thread),
        warnings=warnings,
        notes=notes,
    )
    holder: list[Services] = [services]

    if start_threads:
        writer.start()          # 顺序：writer 先起
        maintenance.start()     # maintenance 后起
        services.started = True

    if reconcile:
        _safe_reconcile(services, embed_provider)

    return services


def _apply_task(task, *, core: ArtifactSpiritCore, applier: IntentApplier, services_holder: list) -> None:
    """writer 的任务分派。**所有写都发生在这一个线程里**（D-03）。"""
    kind = task.kind
    payload = task.payload
    if kind.startswith("intent."):
        applier.apply(payload)
        return
    if kind == "turn":
        for intent in core.ingest_turn(payload):
            applier.apply(intent)
        return
    if kind == "commit":
        report = core.consolidate(session_id=payload)
        for intent in report.intents:
            applier.apply(intent)
        return
    if kind == "maintenance":
        _maintenance_task(payload, core=core, holder=services_holder)
        return
    raise ValueError(f"未知任务类型：{kind}")


def _maintenance_task(payload: object, *, core: ArtifactSpiritCore, holder: list) -> None:
    """维护任务：payload 是 ``{"action": ...}`` 形式的字典。

    本函数跑在 **maintenance 线程**（或 writer 线程转发的 ``kind="maintenance"``），
    因此它产生的写意图**一律 ``enqueue`` 回写队列**，绝不 ``write_now``
    （DES-REV-008 P0-12：直接落库会让 AL3 多出一条写连接，"单写者"静默作废）。
    """
    if not isinstance(payload, dict):  # pragma: no cover
        return
    action = payload.get("action")
    if action == "refresh_overviews":
        services = holder[0]
        services.enqueue(core.refresh_overviews(limit=int(payload.get("limit", 20))))
    elif action == "decay":
        report = core.decay(dry_run=bool(payload.get("dry_run", True)))
        holder[0].enqueue(report.intents)
    elif action == "optimize":
        report = core.optimize(
            dry_run=bool(payload.get("dry_run", True)),
            autonomous=bool(payload.get("autonomous", False)),
        )
        holder[0].enqueue(report.intents)
    elif action == "consolidate":
        # 巩固：`on_session_end` / `on_pre_compress` / `on_session_switch` 都把
        # 巩固投到**维护队列**（AL5 LLD："巩固在 maintenance 线程执行"，
        # 别把它挪到 writer——长跑的 LLM 工作会堵住热路径的写）。
        # 这里曾漏掉这个分支：任务入队后只认 refresh_overviews/decay/optimize，
        # 于是"会话结束 / 压缩前该归档的"被**静默丢弃**（P1-5）。
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            return
        report = core.consolidate(session_id=session_id)
        holder[0].enqueue(report.intents)


def _maintenance_cycle(*, core: ArtifactSpiritCore, holder: list) -> None:
    """周期性整理：重算 L1 概览 + （可选）衰减排序。

    **不做删除**——治理性删除只在显式授权下由 optimize 执行。
    产出同样经 :meth:`Services.enqueue`（见 ``_maintenance_task``）。
    """
    services = holder[0]
    services.enqueue(core.refresh_overviews(limit=20))
    if services.config.decay.get("enabled"):
        report = core.decay(dry_run=False)
        services.enqueue(report.intents)


def _safe_reconcile(services: Services, embed_provider: EmbeddingProvider | None) -> None:
    """启动对账。**失败不影响启动**——对账是修复动作，不是前提条件。"""
    try:
        embed_fn = None
        if embed_provider is not None:
            embed_fn = lambda texts: embed_provider.embed(texts)
        report = services.backend.reconcile(dry_run=False, embed_fn=embed_fn)
        if report["missing_vectors"] or report["orphan_vectors"]:
            services.notes.append(
                f"启动对账：补算 {report['repaired']} 条向量，"
                f"清理 {len(report['orphan_vectors'])} 条孤儿向量"
            )
    except Exception as exc:
        services.notes.append(f"启动对账失败（不影响使用）：{type(exc).__name__}: {exc}")


def _core_settings(cfg: SpiritConfig) -> CoreSettings:
    weights_raw = cfg.recall.get("weights") or {}
    decay_raw = cfg.decay
    salience_raw = cfg.salience
    return CoreSettings(
        recall=RecallWeights(
            **{
                key: float(weights_raw.get(key, getattr(RecallWeights(), key)))
                for key in RecallWeights().as_dict()
            }
        ),
        decay=DecayParams(
            w=float(decay_raw.get("w", 0.6)),
            tau_fast=float(decay_raw.get("tau_fast", 7.0)),
            beta=float(decay_raw.get("beta", 0.5)),
        ),
        salience=SalienceConfig(
            threshold=float(salience_raw.get("threshold", 0.35)),
            degraded_threshold=(
                float(salience_raw["degraded_threshold"])
                if salience_raw.get("degraded_threshold") not in (None, "")
                else None
            ),
            weights=SalienceWeights(
                **{
                    key: float((salience_raw.get("weights") or {}).get(key, getattr(SalienceWeights(), key)))
                    for key in SalienceWeights().as_dict()
                }
            ),
        ),
        candidate_k=int(cfg.recall.get("candidate_k", 24)),
    )


def stop(services: Services, *, drain: bool = True) -> list[str]:
    """``Services.stop`` 的函数式入口（与 ``start`` 对称）。"""
    return services.stop(drain=drain)
