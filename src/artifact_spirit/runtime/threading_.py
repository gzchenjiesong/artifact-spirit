"""线程铸造（**唯一的线程创建出口**）。

R8 规定只有 ``runtime/`` 可以创建线程；这个模块把"创建"这件事再收紧一层：
**任何需要线程的地方都走这里**，于是"用什么上下文创建线程"成为一个可注入的策略。

为什么需要注入：宿主 Hermes 的多 profile 隔离靠 ``contextvars``
（``HERMES_HOME`` 覆盖 + 每轮密钥作用域）。它的 ``spawn_context_thread`` 会在**调用方的
contextvars** 里启动线程，并在文档里写明 *"Every memory-provider background job
(prefetch, sync, writer loops) must go through this"*。

一个用空上下文启动的 worker 会**静默落到默认 profile**——也就是写进另一个 profile 的数据库。
这类错误不会报错、只会写错地方，所以不能靠"记得传对不对"，而要把创建口收成一处、
由**组合根**把宿主实现注入进来（宿主缺席时用这里的标准库实现）。

**本模块不 import 宿主**（R4：宿主耦合只允许出现在 ``provider.py``）。
宿主的探测在 AL1——``provider.resolve_host_thread_factory()``——
组合根只接收结果，因此 ``runtime/`` 对宿主零依赖。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

__all__ = ["ThreadFactory", "default_thread_factory"]

ThreadFactory = Callable[..., Any]
"""``(target=..., name=..., daemon=...) -> Thread``。"""


def default_thread_factory(**kwargs: Any) -> threading.Thread:
    """标准库实现——组合根未注入宿主实现时的默认。

    宿主不在场（CLI、独立测试）时这就是正确答案：不是"降级"，
    而是"单独运行也可以"这条需求本身。
    """
    kwargs.setdefault("daemon", True)
    return threading.Thread(**kwargs)
