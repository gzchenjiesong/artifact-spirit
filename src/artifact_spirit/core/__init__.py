"""AL2 核心层：领域逻辑（**纯逻辑，零 I/O**）。

分层规则 **R5**：本包不得出现 ``httpx`` / ``sqlite3`` / ``openai``。
所有外部能力（backend、llm、embedding、clock、id 生成器）**通过构造函数注入**——
禁止模块级单例或全局变量（LLD-AL2 §4）。
"""

from __future__ import annotations

from .base import (
    Clock,
    ConsolidationReport,
    CoreFacade,
    DecayReport,
    HealthReport,
    IdGen,
    OptimizationReport,
    RecallQuery,
    RecallWeights,
    Scored,
    TurnContext,
    TurnEvent,
    WriteIntent,
)
from .consolidation import Consolidator, Optimizer
from .decay import Decayer
from .facade import ArtifactSpiritCore, CoreSettings
from .progressive import LEVELS, ExpandResult, ProgressiveLoader
from .recall import Recaller, score_importance
from .review import Reviewer
from .salience import SalienceConfig, SalienceResult, SalienceScorer

__all__ = [
    "LEVELS",
    "ArtifactSpiritCore",
    "Clock",
    "ConsolidationReport",
    "Consolidator",
    "CoreFacade",
    "CoreSettings",
    "DecayReport",
    "Decayer",
    "ExpandResult",
    "HealthReport",
    "IdGen",
    "OptimizationReport",
    "Optimizer",
    "ProgressiveLoader",
    "RecallQuery",
    "RecallWeights",
    "Recaller",
    "Reviewer",
    "SalienceConfig",
    "SalienceResult",
    "SalienceScorer",
    "Scored",
    "TurnContext",
    "TurnEvent",
    "WriteIntent",
    "score_importance",
]
