"""超时护栏（R8：线程只在本层创建）。

AL1 的 ``prefetch`` 有 300ms 护栏，但**不能自己起线程**（R8）。
所以护栏能力放在这里，由组合根注入给适配层——需要线程的地方集中在一处，
既守住了规则，也没有把"护栏"降级成口号。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .threading_ import ThreadFactory, default_thread_factory

__all__ = ["TimeoutRunner", "run_with_timeout"]

T = TypeVar("T")


def run_with_timeout(
    action: Callable[[], T],
    timeout: float,
    fallback: T,
    *,
    on_timeout: Callable[[], None] | None = None,
    thread_factory: ThreadFactory | None = None,
) -> tuple[T, bool]:
    """在护栏内执行 ``action``。

    Returns:
        ``(结果, 是否超时)``。超时返回 ``fallback``。

    实现方式是**守护线程 + join(timeout)**：超时后我们放弃等待（不阻塞调用方），
    被遗弃的线程会自然结束、它的结果被丢弃。

    **"结果没人要"不等于"没有影响"**：``action`` 仍会在那条被遗弃的线程里**跑完**。
    因此 **``action`` 必须无副作用**——尤其不得写调用方可见的共享状态，否则会出现
    "超时已经返回保底值、迟到线程随后把状态改了"这类**静默污染**（P1-4 在
    ``prefetch`` 上实测到：迟到的召回 id 被算进了下一轮）。
    需要落状态的，**由调用方在未超时分支里写**，不要在 ``action`` 内写。

    这比"让宿主多等 2 秒"要好得多（P6：记忆系统绝不阻断宿主）。
    """
    if timeout <= 0:
        return action(), False

    box: list[T] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            box.append(action())
        except BaseException as exc:
            error.append(exc)

    make_thread = thread_factory or default_thread_factory
    thread = make_thread(target=worker, name="aspirit-timeout", daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        if on_timeout is not None:
            on_timeout()
        return fallback, True
    if error:
        raise error[0]
    return (box[0] if box else fallback), False


class TimeoutRunner:
    """可注入的超时执行器（单测可换成"直接调用"的确定性替身）。"""

    def __init__(self, *, enabled: bool = True, thread_factory: ThreadFactory | None = None) -> None:
        self.enabled = enabled
        self.thread_factory = thread_factory
        self.timeouts = 0

    def __call__(
        self, action: Callable[[], T], timeout: float, fallback: T, **kwargs
    ) -> tuple[T, bool]:
        if not self.enabled:
            return action(), False
        result, timed_out = run_with_timeout(
            action, timeout, fallback, thread_factory=self.thread_factory, **kwargs
        )
        if timed_out:
            self.timeouts += 1
        return result, timed_out
