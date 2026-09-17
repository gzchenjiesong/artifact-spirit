"""单写者线程与写队列（LLD-AL5 §5 M2 / M3 · D-03）。

**所有写操作经这一个队列、由这一个线程串行执行**。

- **收益**：从根本上消除 `SQLITE_BUSY` 与写竞争——不需要复杂重试，
  也不需要担心两个线程交错产生半写状态
- **代价**：写吞吐有上限（本场景远未触及）

三条纪律（都是可断言的）：

| # | 纪律 | 为什么 |
|---|---|---|
| 1 | ``submit()`` **非阻塞** | 宿主主线程绝不能被记忆系统挂住（INV-4） |
| 2 | 队列满时**按优先级丢弃**并计数 | 记忆写入 > 关联更新 > 提取 > 摘要 |
| 3 | 单任务异常**不得杀死线程** | 一条坏记忆不该让整个器灵停摆 |

> **队列不持久化**（LLD-AL5 §5 M6 的取舍）：成员都是可重建的加工任务，
> 崩溃丢失只影响及时性，不影响真相源。持久化队列会引入复杂度与新的失败面。
"""

from __future__ import annotations

import heapq
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..common import now_iso
from .events import (
    PRIORITY,
    PRIORITY_EXTRACT,
    PRIORITY_MEMORY,
    PRIORITY_RELATION,
    PRIORITY_SUMMARY,
    QueueStats,
    Task,
)
from .threading_ import ThreadFactory, default_thread_factory

if TYPE_CHECKING:  # pragma: no cover
    from ..store.base import MemoryBackend

__all__ = ["INTENT_PRIORITY", "WriteQueue", "Writer", "intent_priority", "kind_priority"]


class WriteQueue:
    """有界优先队列。**本类不做任何业务判断**（C8）。"""

    def __init__(self, maxsize: int = 1000) -> None:
        if maxsize <= 0:
            raise ValueError("write_queue_max 必须为正整数（C11）")
        self.maxsize = maxsize
        self._heap: list[Task] = []
        self._seq = 0
        self._inflight = 0
        """已出队、**尚未处理完**的任务数。

        ``flush`` 必须连它一起等：任务一出队就不在 ``_heap`` 里了，只看堆深
        会让 ``flush`` 在"消费者正拿着最后一条还没落库"时提前返回 ``True``
        ——而调用方拿这个返回值当"写完了"用（`on_memory_write` 的镜像、
        关闭前的 drain）。这是"静默丢写"最容易藏身的地方。
        """
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self.stats = QueueStats(max_depth=0)

    # ------------------------------------------------------------------ #

    def submit(self, payload: object, *, kind: str = "intent.memory", priority: int | None = None) -> bool:
        """非阻塞投递。返回 ``True`` 表示已入队。

        队满时**丢弃优先级最低的那个**并计数——绝不阻塞调用方（纪律 1、2）。
        """
        with self._lock:
            if len(self._heap) >= self.maxsize:
                evicted = self._evict_lowest()
                if evicted is None:
                    self.stats.dropped += 1
                    self.stats.dropped_by_kind[kind] = (
                        self.stats.dropped_by_kind.get(kind, 0) + 1
                    )
                    return False
                self.stats.dropped += 1
                self.stats.dropped_by_kind[evicted.kind] = (
                    self.stats.dropped_by_kind.get(evicted.kind, 0) + 1
                )

            self._seq += 1
            task = Task(
                kind=kind,
                payload=payload,
                priority=priority if priority is not None else kind_priority(kind),
                seq=self._seq,
                submitted_at=now_iso(),
            )
            heapq.heappush(self._heap, task)
            self.stats.accepted += 1
            self.stats.depth = len(self._heap)
            self.stats.max_depth = max(self.stats.max_depth, self.stats.depth)
            self._not_empty.notify()
            return True

    def _evict_lowest(self) -> Task | None:
        """丢掉优先级最低（数字最大）的任务；同优先级丢最早入队的。"""
        if not self._heap:
            return None
        worst_index = 0
        for index, task in enumerate(self._heap):
            current = self._heap[worst_index]
            if (task.priority, -task.seq) > (current.priority, -current.seq):
                worst_index = index
        victim = self._heap[worst_index]
        self._heap[worst_index] = self._heap[-1]
        self._heap.pop()
        heapq.heapify(self._heap)
        return victim

    def get(self, timeout: float | None = None) -> Task | None:
        """取一个任务；``timeout=0`` 表示不等待。"""
        with self._lock:
            if not self._heap:
                if timeout is None:  # pragma: no cover
                    self._not_empty.wait(1.0)
                elif timeout > 0:
                    self._not_empty.wait(timeout)
                if not self._heap:
                    return None
            task = heapq.heappop(self._heap)
            self._inflight += 1
            self.stats.depth = len(self._heap)
            return task

    def flush(self, timeout: float = 5.0) -> bool:
        """等待**队列清空且无在途任务**。返回 ``True`` 表示真的写完了。

        ``Condition.wait`` 会释放锁，因此 writer 线程能在等待期间继续消费。

        判据是 ``_heap or _inflight``：只等堆会漏掉"已被取走、正在落库"的那一条，
        于是 `flush()` 返回 ``True`` 之后读到的却是旧数据（测试与 `stop()` 的
        drain 都以它为准，见 C3）。
        """
        with self._lock:
            waited = 0.0
            step = 0.01
            while (self._heap or self._inflight) and waited < timeout:
                self._not_empty.wait(step)
                waited += step
            return not self._heap and self._inflight == 0

    def depth(self) -> int:
        with self._lock:
            return len(self._heap)

    def pending(self) -> int:
        """**未完成**的任务数 = 堆里的 + 已出队未处理完的。

        与 :meth:`depth` 的区别就是 ``_inflight``：报"残留任务数"时只看堆会
        把正在落库的那一条算作已完成（DES-REV-008 P0-12）。``stop()`` 的返回值
        与"是否真的 drain 干净"都必须用它。
        """
        with self._lock:
            return len(self._heap) + self._inflight

    def drain(self) -> list[Task]:
        """取走全部任务（关闭时用；返回的仍然按优先级排序）。"""
        with self._lock:
            tasks = sorted(self._heap, key=lambda t: t.sort_key)
            self._heap.clear()
            self.stats.depth = 0
            return tasks

    def notify_consumed(self) -> None:
        """消费者处理完一个任务后唤醒等待者。"""
        with self._lock:
            if self._inflight > 0:
                self._inflight -= 1
            self._not_empty.notify_all()

    def notify_waiters(self) -> None:
        """只唤醒等待者，**不动在途计数**。

        ``notify_consumed()`` 同时承担"出队计数 -1"和"唤醒"两件事，停止线程时
        借它来唤醒会**凭空减掉一个在途任务**——``flush()`` 于是可能在最后一条
        还没落库时就返回 ``True``（DES-REV-008 P1-52）。
        """
        with self._lock:
            self._not_empty.notify_all()


@dataclass(slots=True)
class Writer:
    """唯一写者。"""

    backend: MemoryBackend
    apply: Callable[[object], None]
    """把任务落到存储的可调用对象。

    由组合根注入——writer 自己**不 import AL2/AL3 的具体实现**（保持它是纯运行时组件）。
    """

    queue: WriteQueue
    clock: Callable[[], str] = now_iso
    on_error: Callable[[Task, BaseException], None] | None = None
    thread_factory: ThreadFactory | None = None
    """线程铸造策略。宿主注入 ``spawn_context_thread`` 以继承其 profile 上下文（见 threading_）。"""

    _thread: threading.Thread | None = None
    _stop: threading.Event | None = None
    _thread_id: int | None = None
    errors: list[tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self.errors = []

    def _make_thread(self, **kwargs: Any) -> threading.Thread:
        return (self.thread_factory or default_thread_factory)(**kwargs)

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread is not None:
            return
        assert self._stop is not None
        self._stop.clear()
        self._thread = self._make_thread(target=self._loop, name="aspirit-writer", daemon=True)
        self._thread.start()

    def stop(self, *, drain: bool = True, timeout: float = 5.0) -> int:
        """停止写者。

        ``drain=True``（默认）时**先冲刷队列再停线程**——顺序颠倒会丢写（C3）。
        返回未处理完的任务数（0 表示干净收尾）。

        计数用 :meth:`WriteQueue.pending`（堆 + 在途）而不是堆深：``flush``
        超时时任务已经被取走、只是还没落库，只看堆会**把丢写报成 0**——
        恰好在最需要报案的时候（LLD-AL5 §7 F5"记录未完成的任务数"）。
        """
        if self._thread is None:
            return self.queue.pending()
        if drain:
            self.queue.flush(timeout=timeout)
        remaining = self.queue.pending()
        assert self._stop is not None
        self._stop.set()
        self.queue.notify_waiters()
        self._thread.join(timeout=timeout)
        self._thread = None
        return remaining

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def thread_id(self) -> int | None:
        """执行写操作的那个线程 id（测试用它断言"单写者"）。"""
        return self._thread_id

    def submit(self, payload: object, *, kind: str = "intent.memory", priority: int | None = None) -> bool:
        """非阻塞投递。"""
        return self.queue.submit(payload, kind=kind, priority=priority)

    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        assert self._stop is not None
        self._thread_id = threading.get_ident()
        while True:
            task = self.queue.get(timeout=0.05)
            if task is not None:
                self._run_one(task)
                self.queue.notify_consumed()
                continue
            if self._stop.is_set() and self.queue.depth() == 0:
                break

    def _run_one(self, task: Task) -> None:
        """执行单个任务。

        **异常必须被吞掉并记录**（纪律 3）：一条坏记忆不该让整个器灵停摆。
        """
        try:
            self.apply(task)
            self.queue.stats.processed += 1
        except Exception as exc:
            self.queue.stats.failed += 1
            assert self.errors is not None
            self.errors.append((task.kind, f"{type(exc).__name__}: {exc}"))
            if self.on_error is not None:
                self.on_error(task, exc)


INTENT_PRIORITY: dict[str, int] = {
    # 真相源写入——丢了不可重建
    "put": PRIORITY_MEMORY,
    "update": PRIORITY_MEMORY,
    "set_status": PRIORITY_MEMORY,
    "forget": PRIORITY_MEMORY,
    "restore": PRIORITY_MEMORY,
    "wm_put": PRIORITY_MEMORY,
    "wm_delete": PRIORITY_MEMORY,
    # 会话登记 / 收尾——工作记忆的外键前提，必须先于 ``wm_put`` 应用
    "session_create": PRIORITY_MEMORY,
    "session_end": PRIORITY_MEMORY,
    # 关联与实体——可由记忆重建，但重建成本高
    "link": PRIORITY_RELATION,
    "reinforce": PRIORITY_RELATION,
    "touch": PRIORITY_RELATION,
    "entity_upsert": PRIORITY_RELATION,
    # 审计——可重跑，但审计链断裂会伤可追溯性
    "audit": PRIORITY_EXTRACT,
    # 概览投影——纯派生物，可随时重算
    "overview_put": PRIORITY_SUMMARY,
    "overview_invalidate": PRIORITY_SUMMARY,
}
"""``IntentOp`` → 优先级。分档依据是**能否从已有数据重建**；溢出时从大数字开始丢。

**必须与 AL2 的 ``IntentOp`` 穷举一致**——
``tests/test_al5_runtime.py::test_intent_priority_covers_every_intent_op`` 守着。
"""


def intent_priority(op: str) -> int:
    """把写意图的 ``op`` 映射到优先级。

    与 LLD-AL5 §5 M3 的约定一致：**记忆写入 > 关联更新 > 提取/审计 > 摘要**。

    未知 ``op`` **抛错**而不是静默退回最低优先级——静默退回意味着
    "AL2 新加的写入在队列溢出时第一个被丢"，这是最难排查的一类退化
    （DES-REV-003 P1-6：旧实现让 ``audit`` / ``overview_put`` / ``wm_delete``
    静默落到 SUMMARY）。
    """
    try:
        return INTENT_PRIORITY[op]
    except KeyError:
        raise ValueError(
            f"未知的写意图 op={op!r}：请把它登记进 runtime.writer.INTENT_PRIORITY"
            f"（当前已知：{sorted(INTENT_PRIORITY)}）"
        ) from None


def kind_priority(kind: str) -> int:
    """把**任务类型**映射到优先级（LLD-AL5 §5 M3）。

    **穷举**：未知类型抛错，不用"默认档兜底"——那等于让 AL5 替业务层猜优先级
    （C8；DES-REV-008 P1-43）。

    ``kind="intent.<op>"`` 的形态直接委托给 :func:`intent_priority`：任务类型
    有两套命名（队列统计用 ``intent.put``，档位表用 ``intent.memory``），
    再长第三张表就会漂移。
    """
    if kind in PRIORITY:
        return PRIORITY[kind]
    if kind.startswith("intent."):
        return intent_priority(kind[len("intent.") :])
    raise ValueError(
        f"未登记优先级的任务类型 kind={kind!r}——新增任务类型必须显式登记"
        f"（LLD-AL5 §6 C8；当前已知：{sorted(PRIORITY)}）"
    )
