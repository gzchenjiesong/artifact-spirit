"""维护线程（LLD-AL5 §5 M1 / M3 · T-AL5-09）。

承担**低频长跑**的工作：巩固 / 衰减排序 / **L1 概览重算** / 摘要生成。

## 为什么这些必须离线

它们是"回头整理"，不是"当下要答"。放进在线热路径有三个代价：
① 拖慢 ``prefetch``（300ms 护栏）；② 抢 writer 的写带宽；
③ 让一次对话的延迟取决于积压了多少待整理的东西。

## 一条容易踩的坑

**维护线程的写操作必须投递到同一个 write_queue**，不能持有独立写连接。
否则"单写者"（D-03）就不成立了，``SQLITE_BUSY`` 会重新出现——
而且是那种"平时没事、压力大才偶发"的难查类型。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .threading_ import ThreadFactory, default_thread_factory
from .writer import WriteQueue

if TYPE_CHECKING:  # pragma: no cover
    from .writer import Writer

__all__ = ["Maintenance", "TaggedTask"]


@dataclass(frozen=True, slots=True)
class TaggedTask:
    """带去重键的任务包装。"""

    payload: object
    key: str = ""


@dataclass(slots=True)
class Maintenance:
    """维护调度器。

    Args:
        writer: 单写者。维护产出的写意图**一律经它**投递（D-03）。
        handler: 处理一个被投递的维护任务。
        on_cycle: 每个周期额外要跑的事（可空）。
        interval_seconds: 调度间隔。
    """

    writer: Writer
    handler: Callable[[object], None]
    on_cycle: Callable[[], None] | None = None
    interval_seconds: float = 1800.0
    queue: WriteQueue = field(default_factory=lambda: WriteQueue(maxsize=256))
    cycles: int = 0
    errors: list[str] = field(default_factory=list)
    thread_factory: ThreadFactory | None = None
    """线程铸造策略（同 Writer）——宿主 profile 隔离要求后台线程继承调用方上下文。"""

    _thread: threading.Thread | None = None
    _thread_id: int | None = None
    _stop: threading.Event | None = None
    _wake: threading.Event | None = None
    _keys: set[str] | None = None
    _keys_lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._keys = set()
        self._keys_lock = threading.Lock()

    def _make_thread(self, **kwargs: Any) -> threading.Thread:
        return (self.thread_factory or default_thread_factory)(**kwargs)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread is not None:
            return
        assert self._stop is not None
        self._stop.clear()
        self._thread = self._make_thread(
            target=self._loop, name="aspirit-maintenance", daemon=True
        )
        self._thread.start()

    def stop(self, *, drain: bool = True, timeout: float = 5.0) -> None:
        """停止维护线程。

        组合根必须**先停 maintenance 再停 writer**，否则维护产出会投进一个
        已经停掉的队列（LLD-AL5 §5 M8 的启停顺序）。
        """
        if self._thread is None:
            return
        if drain:
            self.queue.flush(timeout=timeout)
        assert self._stop is not None
        assert self._wake is not None
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def thread_id(self) -> int | None:
        """执行维护任务的那个线程 id。

        与 :attr:`Writer.thread_id` 一样**缓存**下来：``Thread.ident`` 在
        ``stop()`` 之后会变回 ``None``，于是"维护任务跑在维护线程上"这条断言
        在收尾之后就问不出答案了——上一版用例正是因此被写成
        ``all(...) or len(set(seen)) == 1``（半恒真的兜底，见 DES-REV-008 P1-44）。
        """
        return self._thread_id

    # ------------------------------------------------------------------ #
    # 投递
    # ------------------------------------------------------------------ #

    def schedule(
        self, payload: object, *, kind: str = "maintenance", dedupe_key: str | None = None
    ) -> bool:
        """投递一个维护任务。

        ``dedupe_key`` 相同的任务**只保留一个**——同一会话被 `on_session_end`
        重复触发时，没必要巩固两遍（巩固本身幂等，但没必要做无用功）。
        """
        if dedupe_key:
            assert self._keys is not None and self._keys_lock is not None
            with self._keys_lock:
                if dedupe_key in self._keys:
                    return False
                self._keys.add(dedupe_key)
        accepted = self.queue.submit(TaggedTask(payload=payload, key=dedupe_key or ""), kind=kind)
        assert self._wake is not None
        self._wake.set()
        return accepted

    def trigger_now(self) -> None:
        """立即唤醒一次（不必等下一个周期）。"""
        assert self._wake is not None
        self._wake.set()

    def run_cycle_now(self) -> None:
        """同步跑一次周期任务（`aspirit consolidate` 之类的显式触发会用它）。"""
        self.cycles += 1
        self._safe(self.on_cycle)
        self.queue.flush(timeout=3.0)

    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        assert self._stop is not None
        assert self._wake is not None
        self._thread_id = threading.get_ident()
        while not self._stop.is_set():
            while True:
                task = self.queue.get(timeout=0)
                if task is None:
                    break
                self._consume(task)
                # 与 ``Writer._loop`` 对称：``get()`` 会把任务记成"在途"，
                # 消费完必须销账，否则 ``queue.flush()`` 永远等不满
                # （``_inflight`` 只增不减）——`stop(drain=True)` 于是每次白等
                # 满超时，Windows 上约 7.8 秒，而"已 drain 干净"这个承诺是假的
                # （DES-REV-008 P1-50）。
                self.queue.notify_consumed()

            self._wake.wait(self.interval_seconds)
            self._wake.clear()
            if self._stop.is_set():
                break
            if self.on_cycle is not None:
                self.cycles += 1
                self._safe(self.on_cycle)

    def _consume(self, task) -> None:
        tagged = task.payload if isinstance(task.payload, TaggedTask) else None
        payload = tagged.payload if tagged else task.payload
        if tagged and tagged.key:
            assert self._keys is not None and self._keys_lock is not None
            with self._keys_lock:
                self._keys.discard(tagged.key)
        self._safe(lambda: self.handler(payload))

    def _safe(self, action: Callable[[], None] | None) -> None:
        """C4：单任务异常不得杀死线程。"""
        if action is None:
            return
        try:
            action()
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
