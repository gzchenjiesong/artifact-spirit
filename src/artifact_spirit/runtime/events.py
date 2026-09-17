"""运行时事件与任务（LLD-AL5 §2.1）。

任务只有四种，且**都只携带数据不携带行为**——队列里放的是"要做什么"，
不是"怎么做"。这保证了队列不需要理解业务（C8）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "PRIORITY",
    "PRIORITY_EXTRACT",
    "PRIORITY_MEMORY",
    "PRIORITY_RELATION",
    "PRIORITY_SUMMARY",
    "Task",
    "TaskKind",
]

TaskKind = str

# 优先级：数字越小越先执行。溢出时**从最大的数字开始丢**。
PRIORITY_MEMORY = 0
PRIORITY_RELATION = 1
PRIORITY_EXTRACT = 2
PRIORITY_SUMMARY = 3

PRIORITY: dict[str, int] = {
    "intent.memory": PRIORITY_MEMORY,
    "intent.relation": PRIORITY_RELATION,
    "turn": PRIORITY_EXTRACT,
    "commit": PRIORITY_SUMMARY,
    "maintenance": PRIORITY_SUMMARY,
}


@dataclass(frozen=True, slots=True)
class Task:
    """一个待执行的任务。

    ``seq`` 由队列分配，用于同优先级下的 FIFO 稳定排序。
    """

    kind: TaskKind
    payload: object
    priority: int = PRIORITY_MEMORY
    seq: int = 0
    submitted_at: str = ""

    @property
    def sort_key(self) -> tuple[int, int]:
        """越小越先执行（优先级，然后入队序）。"""
        return (self.priority, self.seq)

    def __lt__(self, other: object) -> bool:
        """让 ``heapq`` 能直接比较 ``Task``（否则 ``heappush`` 会抛 TypeError）。"""
        if not isinstance(other, Task):  # pragma: no cover
            return NotImplemented
        return self.sort_key < other.sort_key


@dataclass(slots=True)
class QueueStats:
    """队列观测数据（`status` 与 `doctor` 的输入）。"""

    depth: int = 0
    max_depth: int = 0
    accepted: int = 0
    dropped: int = 0
    dropped_by_kind: dict[str, int] = field(default_factory=dict)
    processed: int = 0
    failed: int = 0
