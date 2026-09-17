"""AL2 核心层的六个层服务。

五类记忆 + 核心记忆（M6/M7/M9/M11 的机制分别落在各服务与 ``core/`` 顶层模块）。
"""

from __future__ import annotations

from .base import BaseLayerService, LayerService, derive_chunk_key
from .core_memory import CORE_TYPES, CORE_UPDATE_THRESHOLD, CoreMemoryLayer
from .episodic import EpisodicLayer
from .procedural import SOLIDIFY_THRESHOLD_PLACEHOLDER, ProceduralLayer
from .semantic import SemanticLayer
from .sensory import SensoryLayer
from .working import WORKING_CAPACITY, WorkingLayer

__all__ = [
    "CORE_TYPES",
    "CORE_UPDATE_THRESHOLD",
    "SOLIDIFY_THRESHOLD_PLACEHOLDER",
    "WORKING_CAPACITY",
    "BaseLayerService",
    "CoreMemoryLayer",
    "EpisodicLayer",
    "LayerService",
    "ProceduralLayer",
    "SemanticLayer",
    "SensoryLayer",
    "WorkingLayer",
    "derive_chunk_key",
]
