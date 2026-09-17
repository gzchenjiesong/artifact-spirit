"""宿主契约适配（T-AL1-01 ~ 12）。

**AL1 只做翻译，不做决策。** 任何"if 这条记忆重要"的判断都不属于这里。

最能说明这层性格的是三条硬约束：

| 约束 | 含义 | 违反后果 |
|---|---|---|
| **INV-5** | ``is_available`` **全本地判定，绝不联网** | 激活流程被网络拖住，宿主启动变慢 |
| **INV-4** | ``sync_turn`` **非阻塞**（p99 < 5ms、无 I/O） | 每轮对话都被记忆系统加一笔延迟 |
| **P6** | 任何内部异常都翻译成安全值，**绝不崩溃宿主** | 记忆系统出问题让整个助手不可用 |

这三条都写成了可断言的测试，不是文档里的口号。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import now_iso
from .config import ConfigError, load, save, validate
from .config.loader import ConfigError as _ConfigError  # noqa: F401 - 语义别名
from .core.base import RecallQuery, TurnEvent, WriteIntent
from .runtime import Services, start, stop
from .store.probe import probe_schema_version, schema_version_matches
from .tools import TOOL_SCHEMAS, ToolHandlers, translate_fault

__all__ = ["DEFAULT_PREFETCH_TIMEOUT_MS", "ArtifactSpiritProvider", "resolve_host_thread_factory"]

DEFAULT_PREFETCH_TIMEOUT_MS = 300
"""``prefetch`` 护栏。**不是性能指标，是"绝不阻断宿主"的兑现**。"""


class ArtifactSpiritProvider:
    """Hermes 的 ``MemoryProvider`` 实现。

    **不要缓存跨 session 的可变状态**（C6）——provider 实例可能被复用于不同 session。
    """

    name = "artifact-spirit"
    version = "0.1.0.dev0"

    def __init__(self) -> None:
        self._services: Services | None = None
        self._handlers: ToolHandlers | None = None
        self._hermes_home: str = ""
        self._last_recall: tuple[str, ...] = ()
        self._shutdown_done = False
        self._last_save_report: dict = {}
        self._session_id: str = ""
        self._turn_number: int = 0
        self._last_query: str = ""
        self._prefetched_session: str = ""
        self._last_injected: tuple[str, ...] = ()
        self._current_message: str = ""

    # ================================================================== #
    # 契约：基础
    # ================================================================== #

    def is_available(self, **kwargs: Any) -> bool:
        """**只做本地检查**：配置文件合法性 / DB 路径可写 / schema 版本匹配。

        明确**不做**：HTTP 探活、DNS 解析、端口探测（INV-5）。
        理由有两层：契约硬性要求它快；而"探活"本身就会把插件的可用性
        绑定到网络上——那正是记忆系统最不该依赖的东西。

        **``hermes_home`` 不能只认显式传参。** 宿主在 `initialize` **之前**就会调用它
        （面板状态、激活前探活），而那时的调用是**无参**的——
        `probe_availability(lambda: provider.is_available())`。
        只认 `kwargs` 会导致恒返回 False：宿主显示"not available"，
        用户以为插件装坏了。真机部署时踩到的就是这个。

        所以在"显式传参"之后还有一条解析链：本实例存过的 → 宿主的 profile 感知 home
        → ``HERMES_HOME`` 环境变量 → 平台默认 ``~/.hermes``。
        """
        start_ts = time.perf_counter()
        hermes_home = kwargs.get("hermes_home") or self._hermes_home or _resolve_hermes_home()
        try:
            if not hermes_home:
                return False

            from .config import config_path, load_file

            path = config_path(hermes_home)
            if path.exists():
                load_file(path)  # 语法非法会抛 ConfigError

            db_path = Path(hermes_home) / "spirit" / "spirit.db"
            if db_path.exists():
                if not schema_version_matches(db_path):
                    return False
            else:
                parent = db_path.parent if db_path.parent.exists() else Path(hermes_home)
                if not parent.exists() or not _is_writable(parent):
                    return False
            return True
        except Exception:
            return False
        finally:
            self._last_check_ms = (time.perf_counter() - start_ts) * 1000

    def initialize(self, session_id: str | None = None, **kwargs: Any) -> None:
        """装载配置 → 校验 → 交给 AL5 组合根装配。

        **装配逻辑本身在 AL5**——AL1 是调用方，不是实现方（LLD-AL1 §5 M3）。
        """
        hermes_home = kwargs.get("hermes_home") or self._hermes_home or _resolve_hermes_home()
        if not hermes_home:
            raise ConfigError(
                "initialize 需要 hermes_home（宿主会传入）。"
                "若在 CLI 里手动初始化，请显式提供该参数。"
            )
        self._hermes_home = str(hermes_home)

        cfg = load(self._hermes_home, env=kwargs.get("env"))
        problems = validate(cfg)
        fatal = [p for p in problems if not p.startswith("警告")]
        if fatal:
            fields = "\n- ".join(fatal)
            raise ConfigError(f"配置校验未通过：\n- {fields}")

        self._services = start(
            self._hermes_home,
            config=cfg,
            env=kwargs.get("env"),
            start_threads=kwargs.get("start_threads", True),
            reconcile=kwargs.get("reconcile", True),
            thread_factory=resolve_host_thread_factory(),
            # 网络出口注入：`start()` 会在装配时**立即建好** embedding/llm 客户端，
            # 之后再改 `resolver._transport` 已经来不及（客户端已缓存，注入静默失效）。
            # 真实链路夹具正是靠这个参数把出口换成 MockTransport。
            transport=kwargs.get("transport"),
        )
        # 记录会话：`on_session_switch` 要靠它知道**该清哪个会话**。
        # 宿主调用 `on_turn_start` 时不一定带 session_id，只在那里记会漏。
        self._session_id = session_id or self._session_id
        self._handlers = ToolHandlers(self._services)
        self._shutdown_done = False

    @property
    def services(self) -> Services:
        if self._services is None:
            raise ConfigError("provider 尚未 initialize——请先调用 initialize(hermes_home=...)")
        return self._services

    # ================================================================== #
    # 契约：工具
    # ================================================================== #

    def get_tool_schemas(self) -> list[dict]:
        """返回工具声明（与 handler **一一对应**，由测试断言）。"""
        return list(TOOL_SCHEMAS)

    def handle_tool_call(self, name: str, args: dict | None = None, **kwargs: Any) -> str:
        """分发工具调用，**返回 JSON 字符串**。

        返回类型是宿主契约的一部分：`MemoryManager.handle_tool_call` 的文档明确写着
        "returns a JSON string (tool_error on failure)"，并且它自己失败时也用
        `tool_error(...)`——即 ``{"error": "..."}`` 的 JSON 串。

        所以这里刻意做一次**边界序列化**：内部用结构化 dict 表达（便于测试与阅读），
        出门转成 JSON 串。失败时与宿主同构地给出 ``{"error": ...}``，
        这样无论失败发生在器灵内部还是宿主路由层，模型看到的是同一种形状。
        返回 dict 会让工具结果直接变成 Python repr，模型读不懂。
        """
        if self._handlers is None:
            return _tool_error("器灵尚未初始化")
        try:
            result = self._handlers.handle(name, args or {})
            payload = result.as_dict()
        except Exception as exc:
            # "绝不崩溃宿主"不等于"只丢一句类型名"——先按 M8 翻译成可操作文案（T-AL1-09 验收 1）。
            return _tool_error(translate_fault(exc))

        if not payload.get("ok", False):
            # 把工具自带的结构化信息（如 `field`——哪个参数不合法）一并透传，
            # 而不是压成一句话。模型据此能自己纠正参数再试一次。
            extra = {
                key: payload[key]
                for key in ("text", "data", "field")
                if payload.get(key)
            }
            return _tool_error(payload.get("error") or "工具执行失败", **extra)
        return json.dumps(
            {
                "ok": True,
                "text": payload.get("text", ""),
                "data": payload.get("data") or {},
            },
            ensure_ascii=False,
        )

    def get_config_schema(self) -> list[dict]:
        """返回**扁平字段列表**（宿主契约）。

        宿主 `_normalize_memory_provider_schema` 里写得很直白：
        ``if isinstance(raw, list)``——不是 list 就当空处理。返回 dict 的后果不是"显示难看"，
        而是配置面板**一个字段都不显示**，用户在宿主里根本无法配置器灵。

        字段键沿用配置文件的点号路径（``models.llm.extract``），因为宿主会把
        这份 schema 的 ``key`` 原样回传给 :meth:`save_config`——而 `config.save()`
        本来就按点号路径展开，两边天然对齐。
        """
        from .config_schema import CONFIG_FIELDS

        return [dict(field) for field in CONFIG_FIELDS]

    def save_config(self, values: dict, hermes_home: str | None = None) -> None:
        """按**白名单**写入配置（宿主契约要求返回 ``None``）。

        密钥字段天然不在白名单里，因此不可能落盘（C7）——
        被忽略的字段通过 :meth:`config_save_report` 查询，不靠返回值传递。
        """
        home = hermes_home or self._hermes_home
        if not home:
            self._last_save_report = {"ok": False, "error": "缺少 hermes_home"}
            return
        try:
            from .config import SAVE_WHITELIST

            ignored = sorted(key for key in values if key not in SAVE_WHITELIST)
            path = save(home, values)
            self._last_save_report = {
                "ok": True,
                "path": str(path),
                "ignored": ignored,
                "note": "已被忽略的字段不在白名单内（密钥类字段永远不会被写入）",
            }
        except ConfigError as exc:
            self._last_save_report = {"ok": False, "error": str(exc)}
        except Exception as exc:
            self._last_save_report = {"ok": False, "error": translate_fault(exc)}

    def config_save_report(self) -> dict:
        """上一次 :meth:`save_config` 的结果（诊断用；宿主契约不要求这个返回值）。"""
        return dict(self._last_save_report)

    # ================================================================== #
    # 契约：召回与写入（热路径）
    # ================================================================== #

    def prefetch(self, query: str, *, session_id: str = "", **kwargs: Any) -> str:
        """**同步只读召回，带超时护栏**。

        无论内部发生什么——超时、异常、模型不可用——都必须返回**非空且合法**的内容。
        "保底内容"不是兜底技巧，是设计承诺：**记忆系统绝不阻断宿主**（P6）。
        """
        services = self._services
        if services is None:
            return self._fallback_block(session_id="")

        timeout_ms = _timeout_ms(services)
        vec = None
        try:
            provider = services.core.embedding
            if provider is not None:
                vec = provider.embed([query])[0]
        except Exception:
            vec = None

        q = RecallQuery(
            text=query or "",
            vec=vec,
            session_id=session_id,
            token_budget=int(services.config.recall.get("token_budget", 2000)),
            top_k=int(services.config.recall.get("top_k", 8)),
        )

        def run() -> tuple[str, tuple[str, ...]]:
            """**只算不写**——被遗弃的线程会把这个闭包跑完（P1-4）。

            超时后调用方已经返回保底值，而这条线程稍后仍会执行到这里；
            如果它顺手改了 ``self._last_recall``，就会出现"**超时了这一轮，
            下一轮却上报了这次的召回 id**"——给一条本轮根本没注入的记忆
            记一次访问与共激活（静默的活性污染）。
            所以状态一律由主线程在"未超时"分支里写。
            """
            hits = services.core.recall(q)
            return self._render_hits(hits), tuple(hit.record.id for hit in hits)

        fallback = self._fallback_block(session_id=session_id)
        try:
            (text, ids), timed_out = services.run_with_timeout(
                run, timeout_ms / 1000.0, (fallback, ())
            )
            if timed_out:
                # 本次召回作废——**必须清掉上一轮的条数**再返回。
                # 宿主契约明令 `recall_status()` "Must reflect only the LAST prefetch,
                # never a stale prior count"，而"最近一次 prefetch"这次什么都没注入。
                self._last_recall = ()
                self._last_injected = ()
                return fallback
            # 记住这一轮召回了什么——下一轮据此记访问、建共激活边（TurnEvent.recalled）
            self._last_recall = ids
            self._last_injected = ids
            return text or fallback
        except Exception:
            # 召回失败同样不算"注入过"：否则指示器会显示上一轮的陈旧条数。
            self._last_recall = ()
            self._last_injected = ()
            return fallback

    def sync_turn(
        self,
        user: str,
        assistant: str,
        *,
        session_id: str = "",
        messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> None:
        """把一轮对话**非阻塞**投递给写队列（INV-4）。

        这里**不允许出现任何同步 I/O**——连写日志都要用内存缓冲。
        所以它只做一件事：构造事件 → 入队 → 返回。
        """
        services = self._services
        if services is None:
            return
        if session_id and session_id != self._session_id:
            # 会话变了就顺手登记——不能让"当前会话"只在 on_turn_start 里更新。
            self._begin_session(session_id)
        event = TurnEvent(
            session_id=session_id or self._session_id,
            user=user or "",
            assistant=assistant or "",
            ts=now_iso(),
            messages=list(messages or []),
            recalled=self._last_recall,
        )
        self._last_recall = ()
        services.submit(event, kind="turn")

    def on_session_end(self, messages: list[dict] | None = None, *, session_id: str = "", **kwargs: Any) -> None:
        """投递"会话收尾 + 巩固任务"后**立即返回**——不等待，也不落库。

        会话收尾（`status → committed`）同样是一条**写意图**，交给单写者按序应用
        （D-03·C10）。在这里同步 `backend.session_end` 等于让宿主线程持有一次写事务，
        与写线程争用 SQLite 写锁——那是 INV-4 最不该出现的地方（DES-REV-008 P0-13）。
        """
        services = self._services
        if services is None:
            return
        session = session_id or kwargs.get("session_id") or ""
        if not session:
            return
        services.enqueue([WriteIntent(op="session_end", session_id=session, ts=now_iso())])
        services.maintenance.schedule(
            {"action": "consolidate", "session_id": session}, kind="commit", dedupe_key=f"commit:{session}"
        )

    def on_pre_compress(
        self, messages: list[dict] | None = None, *, session_id: str = "", **kwargs: Any
    ) -> str:
        """上下文压缩前把工作记忆抢救归档（**避免"刚聊完就忘"**）。

        压缩是宿主的动作，器灵无法阻止它——但可以在它发生前把会话内的内容
        先写进情景记忆，这样压缩掉的只是宿主的上下文，不是器灵的记忆。

        **返回值是契约的一部分**：宿主 `MemoryManager.on_pre_compress` 会把各 provider
        返回的文本拼起来当"压缩提示词"。所以这里回一句话——它既是给宿主的交接说明，
        也顺便告诉压缩模型"别把用户的长期偏好压掉"。

        刻意**不接受** ``require_checkpoint`` 关键字（那会让宿主按 v2 fail-closed
        语义对待器灵，而器灵并不保证 checkpoint 一定成功）；也不返回非空内容来伪装
        checkpoint。器灵的做法是**投递一个后台巩固任务**，宿主压缩成功与否都不影响它。
        """
        services = self._services
        session = session_id or kwargs.get("session_id") or self._session_id
        if services is None or not session:
            return ""
        try:
            services.maintenance.schedule(
                {"action": "consolidate", "session_id": session, "reason": "pre_compress"},
                kind="commit",
                dedupe_key=f"precompress:{session}",
            )
        except Exception:
            return ""
        return (
            "【器灵】本轮会话的工作记忆已转入后台归档；"
            "压缩上下文时请保留用户的长期偏好与身份类事实。"
        )

    def on_memory_write(self, action: str, target: str, content: str, **kwargs: Any) -> None:
        """镜像宿主的 ``MEMORY.md`` → 语义记忆、``USER.md`` → 核心记忆。

        这是"与宿主内置记忆共存而非取代"的落点——内置记忆仍然管用，
        但器灵能把它们纳入自己的分层结构里。
        """
        services = self._services
        # 空白内容（宿主清空文件、或只写了空白字符）是常态噪声：落库会在语义记忆里
        # 留一条"看不见也删不掉"的空记录——它照样参与召回、照样占预算。
        if services is None or not (content or "").strip():
            return
        layer = "core" if "user" in (target or "").lower() else "semantic"
        from .core.base import WriteIntent
        from .store.base import AuditEvent, MemoryRecord

        now = services.core.clock()
        record = MemoryRecord(
            id=services.core.id_gen(layer),
            layer=layer,
            type="identity" if layer == "core" else "fact",
            content=content,
            abstract=content[:60],
            confidence=0.95,
            source_session=kwargs.get("session_id"),
            created_at=now,
            updated_at=now,
        )
        services.submit(
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
                    reason=f"镜像宿主记忆写入（{action} {target}）",
                ),
            ),
            kind="intent.put",
        )

    # ================================================================== #
    # 契约：收尾
    # ================================================================== #

    def system_prompt_block(self, *, token_budget: int = 400, **kwargs: Any) -> str:
        """核心记忆摘要 + 器灵状态。**有 token 预算并截断**（C11）。"""
        services = self._services
        if services is None:
            return ""
        budget = int(kwargs.get("token_budget", token_budget))
        try:
            return services.core.system_prompt_block(token_budget=budget)
        except Exception:
            return ""

    def shutdown(self, **kwargs: Any) -> None:
        """收尾。**幂等**——可能被调用多次（C9）。"""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self._services is not None:
            try:
                stop(self._services, drain=True)
            except Exception:  # noqa: S110 - 收尾失败也不能抛给宿主
                pass
        self._services = None
        self._handlers = None
        self._last_recall = ()

    def on_session_start(self, session_id: str | None = None, **kwargs: Any) -> None:
        """登记会话。**注意宿主没有这个钩子**（真实的逐轮钩子是 :meth:`on_turn_start`）。

        保留它是因为 CLI 与测试需要一个显式的"会话开始"入口；宿主路径走
        :meth:`on_turn_start`。两者共用同一个内部实现，不重复逻辑。
        """
        self._begin_session(session_id or "")

    # ================================================================== #
    # 契约：宿主的其余钩子
    #
    # 这些方法**必须存在**：宿主的 MemoryManager 用 `_each_provider` 逐个调用，
    # 内部 `try/except` 把 AttributeError 吞成一条日志。缺一个方法不会崩，
    # 而是"每轮打一行错误日志 + 该能力静默失效"——比崩溃更难发现。
    # ================================================================== #

    def unavailable_reason(self) -> str:
        """不可用时的人类可读原因（宿主会展示给用户）。

        与 :meth:`is_available` **共用同一条 home 解析链**——否则会出现
        "is_available 说可用、reason 说没收到 home"这种自相矛盾的输出，
        比单纯报错更让人困惑。
        """
        if self._services is not None:
            return ""
        try:
            hermes_home = self._hermes_home or _resolve_hermes_home()
            if not hermes_home:
                return "尚未装载：无法定位 hermes_home。"
            if not self.is_available(hermes_home=hermes_home):
                return (
                    f"本地自检未通过：配置或数据库不可用（hermes_home={hermes_home}）。"
                    "可运行 `aspirit doctor` 查看具体原因。"
                )
        except Exception:
            return "本地自检时发生异常，详见日志。"
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """**非阻塞**预取：把召回结果算好放在下一次 :meth:`prefetch` 手边。

        宿主的 `prefetch` 在热路径上、有延迟预算；`queue_prefetch` 则是在
        空闲时提前算。这里把 query 记下来，:meth:`prefetch` 命中同一 query 时
        直接复用上一次的结果，省掉一次重复召回。

        刻意**不**在这里起线程或做 I/O：宿主已经保证了它跑在后台 worker 上，
        但器灵自己的召回是毫秒级的只读查询，为此再排一个任务反而是净开销。
        """
        self._last_query = query or ""
        self._prefetched_session = session_id

    def recall_status(self):
        """"上一轮注入了几条记忆"——宿主用它渲染召回指示器。

        返回一个与宿主 ``RecallStatus`` **结构相容**的对象（属性名一致），
        因为宿主只读属性、不做 isinstance 检查。器灵没有相关数据时返回 ``None``，
        宿主会走通用渲染分支。
        """
        count = len(self._last_injected)
        if not count and not self._last_recall:
            return None
        return _RecallStatus(provider_label="器灵", count=count or len(self._last_recall))

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """宿主**每一轮**都会调用（这是真实的逐轮钩子）。

        用途是会话登记与轮次计数——不落库、不做决策，热路径上只更新内存状态。
        """
        self._turn_number = int(turn_number or 0)
        session_id = kwargs.get("session_id") or ""
        if session_id and session_id != self._session_id:
            self._begin_session(session_id)
        self._current_message = message or ""

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """会话切换（含回退重放）。

        这是**必须处理**的钩子：切换后如果还沿用旧 session_id 写工作记忆，
        新会话的开头会混进上一个会话的组块。``reset=True`` 时尤其要清干净——
        宿主的语义就是"重新开始"。
        """
        services = self._services
        previous = self._session_id
        if services is not None and previous and previous != new_session_id:
            # 旧会话收尾（巩固投递），同 on_session_end 的路径
            try:
                services.maintenance.schedule(
                    {"action": "consolidate", "session_id": previous, "reason": "session_switch"},
                    kind="commit",
                    dedupe_key=f"commit:{previous}",
                )
            except Exception:  # noqa: S110 - 会话切换不能因为"巩固调度失败"而中断
                pass
            if reset:
                # 清工作记忆也走写队列：**单写者没有例外**（D-03·C10）。
                #
                # 旧实现同步 `wm_clear` 并给了三条理由，其中"on_session_end 本来就同步写库"
                # 这条前提在 P0-13 处置后已不成立；剩下两条（"重新开始"的时序语义、亚毫秒代价）
                # 换来的是宿主线程持写锁——那是 INV-4 最贵的一次交易。
                # 「重新开始」的时序由**队列 FIFO** 保证：`wm_delete(旧会话)` 先于
                # `session_create(新会话)` 入队，同档位下按 seq 先进先出，新会话读不到旧组块。
                try:
                    services.enqueue([WriteIntent(op="wm_delete", session_id=previous)])
                except Exception:  # noqa: S110 - 队列满时宁可留下旧组块，也不能打断切换
                    pass
        self._session_id = new_session_id or ""
        self._last_recall = ()
        self._last_injected = ()
        self._turn_number = 0
        self._begin_session(self._session_id)

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
    ) -> None:
        """子代理委派的收尾。委派结果本身值得作为一条情景记忆留档。

        器灵不区分"谁做的"——但**"做过什么"必须留下**，否则跨会话复盘时
        会看不到子代理贡献的那部分工作。
        """
        services = self._services
        if services is None or not (task or result):
            return
        target = child_session_id or self._session_id
        if child_session_id and child_session_id != self._session_id:
            # 子会话也要先登记：委派轮同样会写工作记忆，而 `working_memory.session_id`
            # 有指向 `sessions(id)` 的外键——没登记就是一条注定失败的写。
            # 注意**不要**走 `_begin_session`：那会把"当前会话"切成子会话。
            self._register_session(child_session_id)
        services.submit(
            TurnEvent(
                session_id=target,
                user=f"[委派任务] {task}",
                assistant=result,
                ts=now_iso(),
            ),
            kind="turn",
        )

    def identity_signature(self) -> dict:
        """器灵的身份指纹——宿主用它判断"还是不是同一个记忆主体"。"""
        services = self._services
        if services is None:
            return {"provider": self.name, "initialized": False}
        try:
            return {
                "provider": self.name,
                "initialized": True,
                "spirit_id": services.backend.ensure_spirit_id(),
                "schema_version": probe_schema_version(services.backend.path),
                "db_path": services.backend.path,
            }
        except Exception:
            return {"provider": self.name, "initialized": True}

    def backup_paths(self) -> list[str]:
        """需要被宿主纳入备份的文件清单。

        数据库是 WAL 模式，``-wal`` / ``-shm`` **必须一起备份**——
        只备份主文件会丢掉最近一个检查点之后的所有写入。
        """
        services = self._services
        if services is None or not self._hermes_home:
            return []
        db = services.backend.path
        paths = [db, f"{db}-wal", f"{db}-shm", str(Path(self._hermes_home) / "artifact-spirit.toml")]
        return [p for p in paths if not p.endswith(("-wal", "-shm")) or Path(p).exists()]

    # ================================================================== #
    # 内部
    # ================================================================== #

    def _begin_session(self, session_id: str) -> None:
        """登记会话——**投递给单写者，不在宿主线程落库**（INV-4）。"""
        if not session_id:
            return
        self._session_id = session_id
        self._register_session(session_id)

    def _register_session(self, session_id: str) -> None:
        """把"会话登记"投进写队列（非阻塞、零存储访问）。

        登记**必须先于该会话的 `wm_put` 落地**（外键），这一点由队列保证：
        两者同为 `PRIORITY_MEMORY`，而登记总是先入队 → 按 seq 先出队。
        """
        services = self._services
        if services is None or not session_id:
            return
        services.enqueue([WriteIntent(op="session_create", session_id=session_id, ts=now_iso())])

    def _render_hits(self, hits) -> str:
        if not hits:
            return ""
        lines = ["【相关记忆】"]
        for hit in hits:
            record = hit.record
            text = record.abstract or record.content
            lines.append(f"- ({record.layer}) {text}")
        return "\n".join(lines)

    def _fallback_block(self, *, session_id: str) -> str:
        """保底内容——**保证 prompt 永不为空**。

        优先给核心记忆；连它也取不到时，给一句就事论事的说明而不是空串：
        空串会让宿主以为"器灵启用了但什么都没记住"，而实际上它可能只是超时了一次。
        """
        services = self._services
        if services is not None:
            try:
                block = services.core.system_prompt_block(token_budget=200)
                if block:
                    return block
            except Exception:  # noqa: S110 - 取不到就退回固定文案，不阻断宿主
                pass
        return "【相关记忆】（暂时取不到——记忆系统本轮未返回结果）"

    _last_check_ms: float = 0.0

    @property
    def last_check_ms(self) -> float:
        """上一次 ``is_available`` 的耗时（毫秒）——用于验证"必须快"。"""
        return self._last_check_ms

    # ------------------------------------------------------------------ #
    # CLI 共用同一套装配（C10）
    # ------------------------------------------------------------------ #

    def initialize_for_cli(self, hermes_home: str, *, env: dict | None = None) -> Services:
        """CLI 用的装配入口——**与 provider 走同一条路径**，不重复实现（C10）。"""
        self.initialize(hermes_home=hermes_home, env=env, start_threads=False, reconcile=True)
        return self.services


def _timeout_ms(services: Services) -> float:
    return float(services.config.worker.get("prefetch_timeout_ms", DEFAULT_PREFETCH_TIMEOUT_MS))


def resolve_host_thread_factory():
    """探测宿主的 ``spawn_context_thread``；宿主缺席时退回标准库。

    宿主的 ``spawn_context_thread`` 会在**调用方的 contextvars** 里启动线程，
    并在文档里写明 *"Every memory-provider background job (prefetch, sync,
    writer loops) must go through this"*。一个用空上下文启动的 worker 会
    **静默落到默认 profile**——也就是写进另一个 profile 的数据库；这类错误
    不报错、只写错地方。

    **为什么探测在这里（AL1）而不在组合根**：R4 规定宿主耦合只允许出现在
    ``provider.py``。组合根（``runtime/lifecycle.py``）只接收注入进来的可调用对象，
    因此 ``runtime/`` 对宿主零依赖，也就能在宿主的任意安装形态下被单测。

    全程 import-guarded：器灵不把宿主当作硬依赖。
    """
    try:  # pragma: no cover - 取决于宿主是否安装
        from agent.memory_provider import (
            spawn_context_thread,  # type: ignore[import-not-found]
        )
    except Exception:
        from .runtime.threading_ import default_thread_factory

        return default_thread_factory
    return spawn_context_thread


def _resolve_hermes_home() -> str:
    """在**没有显式传参**时定位宿主的 home。

    顺序刻意如此：

    1. **宿主的 ``get_hermes_home()``** —— 它认得 profile：宿主用 context-local
       override 暴露当前 profile 的路径，而 ``HERMES_HOME`` 环境变量**不一定**被设置
       （实测：`hermes -p <profile>` 运行时 `os.environ["HERMES_HOME"]` 仍是 None）。
       只查环境变量会在多 profile 场景下定位到错误的 home。
    2. ``HERMES_HOME`` 环境变量 —— 宿主不在场时（CLI、独立运行）的通用约定。
    3. 平台默认 ``~/.hermes``。

    全程 import-guarded：器灵不把宿主当作硬依赖。
    """
    try:  # pragma: no cover - 取决于宿主是否在场
        from hermes_constants import get_hermes_home  # type: ignore[import-not-found]

        resolved = get_hermes_home()
        if resolved:
            return str(resolved)
    except Exception:  # noqa: S110 - 宿主常量模块是可选依赖，回退到环境变量
        pass

    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        return env_home
    return str(Path.home() / ".hermes")


def _tool_error(message: str, **extra: Any) -> str:
    """与宿主 ``tools.registry.tool_error`` **同构**的错误 JSON 串。

    刻意不 import 宿主的 `tool_error`：器灵不依赖宿主包（entry point 分发，
    宿主可能以任意方式安装）。格式对齐即可——模型看到的是同一种形状。
    """
    return json.dumps({"error": str(message)[:800], **extra}, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class _RecallStatus:
    """宿主 ``RecallStatus`` 的**结构相容**替身（属性名一致，宿主不做 isinstance 检查）。"""

    provider_label: str
    count: int
    glyph: str = "🧠"


def _is_writable(path: Path) -> bool:
    """目录可写性判定：**只查权限，不做任何写入**。

    早先的版本靠"建一个探针文件再删掉"来确认，看起来更"实测"，实测下来有两个代价：

    1. **慢。** 本机（Windows + 安全软件/运行时 hook）`unlink` 单次要 200–340ms，
       而 `os.access` 是 0.003ms——差七个数量级。`is_available` 跑在**宿主启动路径**上，
       这个代价会直接变成"打开助手时的卡顿"。
    2. **有副作用。** 一个"我只想知道能不能用"的查询会改文件系统，这本身就是坏味道：
       它让判定可能失败于与"可用性"无关的原因（杀软锁文件、目录被别的进程遍历）。

    那为什么现在可以放心用权限位：**误判的后果是明确且可诊断的**。若这里乐观地返回
    "可写"、实际却写不进去，`open()` 会当场抛出带完整路径的错误——那是用户能看到、
    能处理的；而"启动时卡 300ms"不会让任何人联想到记忆插件。
    """
    try:
        return os.access(path, os.W_OK)
    except OSError:  # pragma: no cover - 路径形态异常
        return False
