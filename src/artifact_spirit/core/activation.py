"""扩散激活与 Hebbian 学习（LLD-AL2 §5 M4）。

```
on_co_access(a, b):  w_ab ← w_ab + η · (1 − w_ab)      # 共激活增强，趋近 1
召回时:              activated = seeds ∪ { x | w(seed, x) > θ }
```

**边权与 ``co_count`` 双写**：权用于打分，计数用于巩固判定（"被反复引用"是
``importance`` 的分量之一）。

MVP 只做**一拍扩散**；多跳（``hops > 1``）留到 M4 之后——组合爆炸的代价
远大于收益。

**依据**：**持久联结与瞬时激活必须分开**——共现统计是持久结构（边权），
扩散激活是瞬时的检索促进（不落库），两者不得混用同一字段
（见 ``docs/design/10-神经科学依据与机制映射.md`` §4.4，DES-RES-003）。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ..store.base import AuditEvent, MemoryBackend
from .base import Clock, WriteIntent

__all__ = ["DEFAULT_ETA", "MAX_PAIRS", "Activator"]

DEFAULT_ETA = 0.3
"""Hebbian 学习率。"""

MAX_PAIRS = 12
"""单次共激活最多强化多少对——防止一轮对话产出 O(n²) 条写意图。"""


@dataclass(slots=True)
class Activator:
    """共激活建边与强化。"""

    backend: MemoryBackend
    clock: Clock
    eta: float = DEFAULT_ETA
    max_pairs: int = MAX_PAIRS

    def co_activate(
        self, mem_ids: list[str], *, rel_type: str = "co_activation"
    ) -> list[WriteIntent]:
        """把同一次召回中出现的记忆两两强化。

        幂等：AL3 的 ``link``/``reinforce`` 对同一对记忆是幂等的，
        因此重复处理同一批不会产生重复边，只会让权重向 1 收敛。
        """
        unique = list(dict.fromkeys(m for m in mem_ids if m))
        if len(unique) < 2:
            return []

        pairs = list(combinations(unique, 2))[: self.max_pairs]
        existing = {self._pair_key(edge) for edge in self.backend.all_relations()}

        intents: list[WriteIntent] = []
        for left, right in pairs:
            key = self._pair_key({"src_id": left, "dst_id": right})
            op = "reinforce" if key in existing else "link"
            weight = self._current_weight(left, right)
            intents.append(
                WriteIntent(
                    op=op,
                    a_id=left,
                    b_id=right,
                    rel_type=rel_type,
                    weight=weight if op == "link" else 0.0,
                    delta=self.eta if op == "reinforce" else 0.0,
                    actor="system",
                    audit=AuditEvent(
                        op=op,
                        actor="system",
                        target_kind="relation",
                        target_id=f"{left}->{right}",
                        after={"rel_type": rel_type, "eta": self.eta},
                    ),
                )
            )
        return intents

    def activate(self, seeds: list[str], *, hops: int = 1, threshold: float = 0.2) -> dict[str, float]:
        """一拍扩散：返回 ``{mem_id: 激活值}``。

        ``hops > 1`` 暂不实现——见模块头的取舍说明。
        """
        if hops > 1:  # pragma: no cover - MVP 明确只做一拍
            raise NotImplementedError("MVP 只支持一跳扩散；多跳留待 M4 之后")

        activated: dict[str, float] = dict.fromkeys(seeds, 1.0)
        for kind, node_id, weight in self._all_neighbors(seeds):
            if kind != "memory" or weight < threshold:
                continue
            activated[node_id] = max(activated.get(node_id, 0.0), float(weight))
        return activated

    # ------------------------------------------------------------------ #

    def _all_neighbors(self, seeds: list[str]) -> list[tuple[str, str, float]]:
        out: list[tuple[str, str, float]] = []
        for seed in seeds:
            out.extend(self.backend.neighbors("memory", seed, limit=20))
        return out

    @staticmethod
    def _pair_key(edge: dict) -> tuple[str, str]:
        left, right = edge["src_id"], edge["dst_id"]
        return (left, right) if left <= right else (right, left)

    def _current_weight(self, left: str, right: str) -> float:
        for kind, node_id, weight in self.backend.neighbors("memory", left, limit=50):
            if node_id == right:
                return float(weight)
        return self.eta
