"""AL5 运行时：**唯一允许创建线程的层**（R8）。

| 线程 | 职责 | 读写 |
|---|---|---|
| Hermes 主线程 | `prefetch` 只读召回、工具调用 | 只读 |
| **writer** | 消费写队列，串行执行**所有**写操作 | 写 |
| **maintenance** | 巩固 / 衰减 / 概览重算（低频长跑） | 写（**经同一队列**） |

组合根是 :func:`lifecycle.start`——它是唯一允许 import 各层具体实现的位置。
"""

from __future__ import annotations

from .events import (
    PRIORITY,
    PRIORITY_EXTRACT,
    PRIORITY_MEMORY,
    PRIORITY_RELATION,
    PRIORITY_SUMMARY,
    QueueStats,
    Task,
    TaskKind,
)
from .lifecycle import IntentApplier, Services, start, stop
from .maintenance import Maintenance, TaggedTask
from .threading_ import (
    ThreadFactory,
    default_thread_factory,
)
from .timeout import TimeoutRunner, run_with_timeout
from .writer import WriteQueue, Writer, intent_priority

__all__ = [
    "PRIORITY",
    "PRIORITY_EXTRACT",
    "PRIORITY_MEMORY",
    "PRIORITY_RELATION",
    "PRIORITY_SUMMARY",
    "IntentApplier",
    "Maintenance",
    "QueueStats",
    "Services",
    "TaggedTask",
    "Task",
    "TaskKind",
    "ThreadFactory",
    "TimeoutRunner",
    "WriteQueue",
    "Writer",
    "default_thread_factory",
    "intent_priority",
    "run_with_timeout",
    "start",
    "stop",
]
