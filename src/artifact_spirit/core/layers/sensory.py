"""感觉记忆（LLD-AL2 §5 M1）。

**门口，不是仓库。**

它只做一件事：给这一轮的输入打分，决定"值不值得记"。
**低于阈值的内容不会产出任何落库候选**——噪声不落盘，这是记忆质量的第一道闸门。

注意它是唯一一个 ``candidates()`` **恒返回空**的层服务：感觉记忆本身不持久化，
达标的候选由提取器产出（见 :mod:`artifact_spirit.extract.extractor`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..base import RecallQuery, Scored, TurnContext
from ..salience import SalienceConfig, SalienceResult, SalienceScorer
from .base import BaseLayerService

__all__ = ["SensoryLayer"]


@dataclass(slots=True)
class SensoryLayer(BaseLayerService):
    """显著性过滤。"""

    layer: str = "sensory"
    config: SalienceConfig = field(default_factory=SalienceConfig)
    scorer: SalienceScorer | None = None

    def __post_init__(self) -> None:
        if self.scorer is None:
            self.scorer = SalienceScorer(backend=self.backend, config=self.config)

    def filter(self, ctx: TurnContext) -> SalienceResult:
        """对一轮输入打分。"""
        assert self.scorer is not None
        return self.scorer.score(ctx.text, session_id=ctx.session_id)

    def candidates(self, ctx: TurnContext) -> list[dict]:
        """**恒为空**——感觉记忆不落盘（设计如此，不是未实现）。"""
        return []

    def recall(self, q: RecallQuery) -> list[Scored]:
        """感觉记忆不参与跨会话召回——它只活在当前这一轮。"""
        return []
