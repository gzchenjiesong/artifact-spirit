"""依赖与密钥扫描（R7 / INV-10 / CI secrets 门禁）。

**为什么要有这个**：器灵的默认栈必须保持全 MIT / public-domain，不得混入传染性许可
（AGPL / SSPL）或许可不明的依赖——否则某天顺手引入一个包，整个项目的许可形态就被
改写了。所以这里把"许可准入"变成 CI 会失败的检查。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

__all__ = [
    "FORBIDDEN_MODULES",
    "SECRET_PATTERNS",
    "declared_requirements",
    "imported_roots",
    "scan_dependencies",
    "scan_forbidden_modules",
    "scan_secrets",
]

FORBIDDEN_MODULES = frozenset({"openviking", "openviking_sdk"})
"""INV-10 / R7：任何位置都不得引入。

清单按**许可类别**维护：凡判定为传染性许可（AGPL / SSPL）或许可不明的第三方包一律
加入此集合。集合内的包名只是这条规则的已识别实例，规则本身与具体产品无关。
"""

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("bearer", re.compile(r"Authorization['\"]?\s*[:=]\s*['\"]Bearer\s+[A-Za-z0-9\-_]{16,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
)
"""密钥模式。**不匹配测试夹具里的占位串**（它们长度不足或不含 `sk-` 前缀）。"""


def imported_roots(path: Path) -> set[str]:
    """收集该文件导入的**顶层模块名**（含相对导入解析后的首段）。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return set()

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module is None:
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def scan_forbidden_modules(root: Path) -> list[dict]:
    """扫描源码树中的禁用模块导入（R7）。"""
    violations: list[dict] = []
    for path in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for module in sorted(imported_roots(path) & FORBIDDEN_MODULES):
            violations.append(
                {
                    "rule": "R7",
                    "file": str(path),
                    "module": module,
                    "message": f"禁止引入 {module}（INV-10 / AGPL 合规）",
                }
            )
    return violations


def scan_dependencies(root: Path, *, pyproject: Path | None = None) -> list[dict]:
    """依赖扫描 = 源码扫描 + `pyproject.toml` 的**已声明依赖**扫描。

    两者都要查：只在源码里禁、却在依赖声明里放行，等于没禁。

    注意是**解析 TOML 取依赖名**，而不是全文子串匹配——否则"禁止引入某依赖"
    这句注释本身就会被误判为违规。
    """
    root = Path(root)
    violations = scan_forbidden_modules(root)

    candidate = pyproject
    if candidate is None:
        for probe in (root.parent / "pyproject.toml", root.parent.parent / "pyproject.toml"):
            if probe.exists():
                candidate = probe
                break

    if candidate is not None and Path(candidate).exists():
        for name in declared_requirements(Path(candidate)):
            if name.split(".")[0].split("-")[0].split("[")[0] in FORBIDDEN_MODULES:
                violations.append(
                    {
                        "rule": "R7",
                        "file": str(candidate),
                        "module": name,
                        "message": f"依赖声明中出现 {name}（INV-10）",
                    }
                )
    return violations


def declared_requirements(pyproject: Path) -> list[str]:
    """从 ``pyproject.toml`` 中取出依赖名（去掉版本约束）。"""
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError):  # pragma: no cover
        return []

    project = data.get("project", {})
    raw: list[str] = list(project.get("dependencies", []) or [])
    for group in (project.get("optional-dependencies") or {}).values():
        raw.extend(group or [])

    names: list[str] = []
    for spec in raw:
        token = spec.split(";")[0].split(">=")[0].split("<=")[0]
        token = token.split("==")[0].split("~=")[0].split(">")[0].split("<")[0]
        token = token.split("[")[0].strip()
        if token:
            names.append(token)
    return names


_NOT_SHIPPED = frozenset(
    {
        "__pycache__",
        ".git",
        ".codebuddy",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
    }
)
"""**不会随仓库发布**的目录——密钥扫描跳过它们。

判据是「它**进不进版本控制**」（`.gitignore` 已排除），而不是「它可不可疑」。
这样扫描器的**范围**才与它的**目的**（别把密钥提交进仓库）对齐：
对一个不入库的文件报「疑似密钥」，是**报错了对象**。

更要紧的是**误报的代价**：本地笔记、工具数据一旦被扫进来，人会开始加
`# noqa`，或者干脆把密钥挪到"扫不到的地方"——**门禁会因此被绕过，
而不是被满足**。

`extra_files` 不受本集合影响——那是调用方**显式**要求检查的文件
（如 `pyproject.toml`），显式优先于默认排除。
"""


def scan_secrets(root: Path, *, extra_files: tuple[Path, ...] = ()) -> list[dict]:
    """扫描源码与给定文件中的疑似真实密钥。

    跳过明显的占位/示例串（``example``、``your-key``、``xxx`` 等），
    以及**不进版本控制**的目录（见 `_NOT_SHIPPED`）。
    """
    root = Path(root)
    violations: list[dict] = []
    targets = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and not any(part in _NOT_SHIPPED for part in p.parts)
    ]
    targets.extend(extra_files)

    for path in targets:
        if path.suffix not in {".py", ".toml", ".yaml", ".yml", ".json", ".md", ".cfg", ".ini"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # pragma: no cover
            continue
        for name, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                snippet = match.group(0)
                if _is_placeholder(snippet):
                    continue
                violations.append(
                    {
                        "rule": "secrets",
                        "file": str(path),
                        "kind": name,
                        "message": f"疑似真实密钥：{snippet[:12]}…",
                    }
                )
    return violations


_PLACEHOLDER_HINTS = ("example", "your-", "your_", "xxx", "placeholder", "dummy", "fake", "test")


def _is_placeholder(snippet: str) -> bool:
    lowered = snippet.casefold()
    return any(hint in lowered for hint in _PLACEHOLDER_HINTS)
