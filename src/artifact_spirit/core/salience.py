"""显著性过滤（LLD-AL2 §5 M1 · 感觉记忆门口）。

决定"值不值得记"。

**依据**：重要性是**编码期**就打上的标记（情绪唤醒 / 自我相关会加强编码），
与访问频率是两个独立维度——所以它在门口打分，而不是靠后续访问次数累积
（见 ``docs/design/10-神经科学依据与机制映射.md`` §4.1，DES-RES-003）。

**为什么必须零 LLM**：它在**每一轮对话**都跑。若每轮都要一次模型调用，
成本会随对话轮数线性增长，而收益只是一句"要不要记"——这是一笔明显亏本的买卖。

唯一的模型依赖是**新颖度**需要一次 embedding（与已有记忆比对相似度）。
embedding 不可用时**置 ``w1=0`` 并重新归一化**，退化为纯规则打分，绝不抛错（F2）。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..common import estimate_tokens
from ..model.base import EmbeddingError, EmbeddingProvider
from ..store.base import MemoryBackend

__all__ = [
    "EMOTION_PATTERNS",
    "INSTRUCTION_PATTERNS",
    "SalienceConfig",
    "SalienceResult",
    "SalienceScorer",
    "SalienceWeights",
]

INSTRUCTION_PATTERNS = (
    re.compile(r"记住|记下|别忘|不要忘|以后都|以后请|从此|总是|永远"),
    re.compile(r"我不喜欢|我最喜欢|我讨厌|我偏好|我更愿意|我习惯"),
    re.compile(r"\bremember\b|\bnote that\b|\bfrom now on\b|\balways\b|\bnever\b|\bprefer\b", re.IGNORECASE),
)

EMOTION_PATTERNS = (
    re.compile(r"[！!]{1,}"),
    re.compile(r"[？?]{2,}"),
    re.compile(r"太(棒|好|差|糟)了|简直|非常(重要|关键)|一定要"),
    re.compile(r"\b(amazing|terrible|critical|urgent|hate|love)\b", re.IGNORECASE),
)

_ENTITY_HINT = re.compile(r"[A-Z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,4}(?:项目|系统|平台|公司|团队)")


@dataclass(frozen=True, slots=True)
class SalienceWeights:
    """五因子权重。**集中配置，不散落魔法数字**（C9）。"""

    novelty: float = 0.30
    instruction: float = 0.25
    entity: float = 0.20
    emotion: float = 0.15
    core_deviation: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "novelty": self.novelty,
            "instruction": self.instruction,
            "entity": self.entity,
            "emotion": self.emotion,
            "core_deviation": self.core_deviation,
        }


@dataclass(frozen=True, slots=True)
class SalienceConfig:
    threshold: float = 0.35
    weights: SalienceWeights = field(default_factory=SalienceWeights)
    novelty_top_k: int = 5
    degraded_threshold: float | None = None
    """**无向量时**使用的落库门槛。``None`` 表示按权重自动推算。

    为什么必须有这个字段：门槛是**分数的函数**，而新颖度缺席时分数换了一套量纲。
    没有它，两个数字就会互相错位——

    | 情形 | 分数怎么算 | 门槛该是多少 |
    |---|---|---|
    | 五因子齐全 | ``Σ(w·v) / 1.0`` | ``threshold`` |
    | 新颖度缺席 | ``Σ_剩余(w·v) / 0.70``（重归一化） | ``threshold × 0.70`` |

    若缺席时仍用原门槛，等于要求**规则因子独自凑满原本五因子的全部证据量**：
    实测下 ``请记住：我对花生过敏`` 只有 0.298，永远过不了 0.35 这条线——
    门槛事实上永久关闭，器灵一句都记不住。这不是"更严格的筛选"，
    而是把「降级打分」偷偷变成了「关闭写入」。
    """

    def effective_threshold(self, *, novelty_available: bool = True) -> float:
        """按**当前可用的因子集**给出等价门槛（让门槛与分数在同一量纲上比较）。"""
        if novelty_available:
            return self.threshold
        if self.degraded_threshold is not None:
            return self.degraded_threshold
        # 缺席因子占的权重比，就是量纲缩小的比例
        scale = 1.0 - self.weights.as_dict().get("novelty", 0.0)
        return max(0.0, self.threshold * scale)


@dataclass(slots=True)
class SalienceResult:
    """打分结果。

    ``embedding`` 会被写入路径**复用**——既然为了新颖度已经算过一次向量，
    没必要在落库时再算一次（高频路径上的双重开销）。
    """

    score: float
    factors: dict[str, float] = field(default_factory=dict)
    embedding: list[float] | None = None
    degraded: str | None = None
    novelty_available: bool = True
    """新颖度**是否真的算出来了**。

    单独记一个布尔值而不是去解析 ``degraded`` 字符串：门槛判定要读它，
    而字符串是给人看的、语义会随文案变化。判据与展示解耦。
    """

    def passes_threshold(self, config: SalienceConfig) -> bool:
        """是否达到落库门槛。**低于阈值只进工作记忆，不提交长期记忆。**

        门槛取自 :meth:`SalienceConfig.effective_threshold`——它知道新颖度缺席时
        分数换了量纲，因此拿同一量纲上的数字来比。
        """
        return self.score >= config.effective_threshold(
            novelty_available=self.novelty_available
        )


class SalienceScorer:
    """零 LLM 的显著性打分器。"""

    def __init__(
        self,
        *,
        backend: MemoryBackend,
        embedding: EmbeddingProvider | None = None,
        config: SalienceConfig | None = None,
    ) -> None:
        self.backend = backend
        self.embedding = embedding
        self.config = config or SalienceConfig()

    def score(self, text: str, *, session_id: str | None = None) -> SalienceResult:
        """对一段文本打分。**任何内部失败都不得抛错**。"""
        text = (text or "").strip()
        if not text:
            return SalienceResult(score=0.0, factors={}, novelty_available=False)

        degraded: str | None = None
        embedding: list[float] | None = None
        novelty = 0.5  # 无向量时的中性值（下面会置权重为 0）

        if self.embedding is not None:
            try:
                embedding = self.embedding.embed([text])[0]
            except (EmbeddingError, IndexError, TypeError):
                degraded = "embedding_unavailable"
                embedding = None

        weights = dict(self.config.weights.as_dict())
        if embedding is not None:
            novelty = self._novelty(embedding)
        else:
            weights["novelty"] = 0.0
            degraded = degraded or "embedding_unavailable"

        factors = {
            "novelty": novelty,
            "instruction": self._instruction(text),
            "entity": self._entity_density(text),
            "emotion": self._emotion(text),
            "core_deviation": self._core_deviation(text),
        }

        total_weight = sum(weights.values())
        if total_weight <= 0:  # pragma: no cover - 配置异常时的兜底
            return SalienceResult(
                score=0.0,
                factors=factors,
                embedding=embedding,
                degraded=degraded,
                novelty_available=embedding is not None,
            )

        value = sum(weights[k] * v for k, v in factors.items()) / total_weight
        return SalienceResult(
            score=max(0.0, min(1.0, value)),
            factors=factors,
            embedding=embedding,
            degraded=degraded,
            novelty_available=embedding is not None,
        )

    # ------------------------------------------------------------------ #
    # 五因子
    # ------------------------------------------------------------------ #

    def _novelty(self, embedding: list[float]) -> float:
        """``1 − max_sim``——与已有记忆越像，越不值得再记一条。"""
        try:
            hits = self.backend.vector_search(embedding, top_k=self.config.novelty_top_k)
        except Exception:
            return 0.5
        if not hits:
            return 1.0
        best = max(h.score for h in hits)
        return max(0.0, min(1.0, 1.0 - best))

    @staticmethod
    def _instruction(text: str) -> float:
        """显式指令信号——"记住 X" 远远比一句闲聊值得记。"""
        hits = sum(1 for pattern in INSTRUCTION_PATTERNS if pattern.search(text))
        return min(1.0, hits / len(INSTRUCTION_PATTERNS) + (0.5 if hits else 0.0))

    @staticmethod
    def _entity_density(text: str) -> float:
        """专有名词密度。零 LLM——用正则粗识别（大写词 / "X项目"式命名）。"""
        tokens = max(1, estimate_tokens(text))
        entities = len(_ENTITY_HINT.findall(text))
        return min(1.0, entities / max(1.0, tokens / 12.0))

    @staticmethod
    def _emotion(text: str) -> float:
        hits = sum(1 for pattern in EMOTION_PATTERNS if pattern.search(text))
        return min(1.0, hits / len(EMOTION_PATTERNS) + (0.4 if hits else 0.0))

    def _core_deviation(self, text: str) -> float:
        """与核心记忆的偏离度。**提到"我是谁 / 我偏好什么"的，更值得记**。"""
        terms = self.backend.core_memory_terms()
        if not terms:
            return 0.0
        lowered = text.casefold()
        return 0.7 if any(term in lowered for term in terms) else 0.0


def factor_breakdown(result: SalienceResult, config: SalienceConfig) -> Mapping[str, float]:
    """按权重展开的贡献明细（供 `review` 与调试展示"为什么没记住这句"）。"""
    weights = config.weights.as_dict()
    total = sum(weights.values()) or 1.0
    return {k: weights.get(k, 0.0) * v / total for k, v in result.factors.items()}
