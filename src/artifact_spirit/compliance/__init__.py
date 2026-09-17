"""合规门禁（T-AL5-11 / T-AL3-23）：依赖扫描、架构规则、密钥扫描。

这是 CI 六项门禁中的 `deps` / `arch` / `secrets` 三项的实现。
设计要点：**扫描器自身必须可被验证**——因此每个扫描器都接受一个 root 参数，
测试可以往临时目录里注入一行违规代码，断言扫描器确实报错。
"""

from __future__ import annotations

from .arch_rules import (
    COMPOSITION_ROOT,
    LAYER_IDS,
    LAYER_MAP,
    PACKAGE_ROOT,
    check_architecture,
    iter_python_files,
    module_imports,
    resolved_imports,
)
from .dep_scan import (
    FORBIDDEN_MODULES,
    SECRET_PATTERNS,
    declared_requirements,
    scan_dependencies,
    scan_forbidden_modules,
    scan_secrets,
)

__all__ = [
    "COMPOSITION_ROOT",
    "FORBIDDEN_MODULES",
    "LAYER_IDS",
    "LAYER_MAP",
    "PACKAGE_ROOT",
    "SECRET_PATTERNS",
    "check_architecture",
    "declared_requirements",
    "iter_python_files",
    "module_imports",
    "resolved_imports",
    "scan_dependencies",
    "scan_forbidden_modules",
    "scan_secrets",
]
