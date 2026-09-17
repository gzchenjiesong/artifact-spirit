"""分层架构规则 R1–R10（CI 门禁 `arch`）。

**为什么用代码而不是靠自觉**：分层文档里写了 R1–R8，但文档不会阻止任何一次 import。
这些规则真正的价值在于**可 CI 固化**——违反即红灯（DES-000 §4.2）。

规则采用**白名单**而非黑名单：只有列出的依赖方向被允许。
新增层时若不更新规则，检查会失败——这是刻意的，逼迫架构变更被显式记录。

## 覆盖面对齐（DES-REV-003 修订）

修订前有四个洞，都在这里补掉：

| 洞 | 现象 | 修法 |
|---|---|---|
| 层识别不全 | `_layer_of` 只认 6 个包，`provider.py`/`cli.py`/`tools/`/`runtime/`/`compliance/` 与所有顶层文件**无任何规则** | 引入 :data:`LAYER_MAP`，把包名映射到 AL1–AL5，**没有映射的包会被检查报错**（不再静默放行） |
| R4 是死代码 | 宿主标记含 `agent.memory_provider`，但只比对 ``name.split(".")[0]``，永远得到 `"agent"` | 宿主匹配改为**点分边界**匹配，并显式豁免唯一允许依赖宿主的 `provider.py` |
| R1/R4/R5 串台 | 三条规则 `layers`+`allowed` 完全相同，core 里一处越界依赖同时报 3 条 | 拆职责：**R1** 只做白名单、**R5** 只做 I/O 库黑名单、**R4** 只做宿主黑名单（且不再带白名单） |
| AL5 无边界 | AL5 可任意 import 业务层，"横向支撑"只是口号 | 新增 **R10**：AL5 只准依赖协议 + `common`，唯一例外是**组合根** `runtime/lifecycle.py`（显式写出，不靠"整层放行"） |

补的两条新规则之所以是"新"而不是"加严旧规则"：

- **R9（实现模块不可跨层）**——白名单只能约束"往哪去"，约束不了"从哪来"。
  `store.sqlite_backend` 一旦被 AL1 导入，连 sqlite-vec 一起拖进最外层，
  把 AL3 的内部结构变成事实上的公开 API。这类"实现模块当接口用"的退化
  需要**按模块点名**，所以单独成条（并让 `store/base.py` 里那句
  "正是 R9 要拦的事"有对应的代码）。
- **R10（AL5 边界）**——AL5 是"横向支撑"，如果不画边界，
  `config/` 直接 import `core.facade` 也能全绿，"横向支撑"就只是口号。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "COMPOSITION_ROOT",
    "LAYER_IDS",
    "LAYER_MAP",
    "LAYER_RULES",
    "PACKAGE_ROOT",
    "LayerRule",
    "check_architecture",
    "iter_python_files",
    "module_imports",
    "resolved_imports",
]

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
"""``src/artifact_spirit``。"""


# --------------------------------------------------------------------------- #
# 层映射：包名 → 层编号
# --------------------------------------------------------------------------- #

LAYER_IDS: tuple[str, ...] = ("AL1", "AL2", "AL3", "AL4", "AL5")

LAYER_MAP: dict[str, str] = {
    # AL1 适配层：三张对外契约（宿主 provider / 工具面 / 运维 CLI）
    "provider": "AL1",
    "tools": "AL1",
    "cli": "AL1",
    # AL2 核心层：core 与 extract 是同一逻辑层的两个子包
    "core": "AL2",
    "extract": "AL2",
    # AL3 存储层
    "store": "AL3",
    # AL4 模型层
    "model": "AL4",
    # AL5 辅助机制
    "runtime": "AL5",
    "config": "AL5",
    "observability": "AL5",
    "compliance": "AL5",
}
"""包名（相对 ``artifact_spirit``）到层编号的**唯一**映射。

``_layer_of`` 只认这张表：**新增包而忘记登记会被报为"未归层"**，
而不是像修订前那样静默逃过所有规则。顶层的 ``common`` / ``config_schema``
是共享内核（零包内依赖），刻意保持不归层。
"""

COMPOSITION_ROOT = "runtime/lifecycle.py"
"""组合根：**唯一**允许 import 各层具体实现的位置（AL5）。

它同时是 R9 的显式例外——把"整层放行"换成"只放行一个文件"，
AL5 其余部分（writer / maintenance / config / observability）就必须守边界。
"""

_SHARED = "artifact_spirit.common"
"""共享内核（无 I/O、无包内依赖），任何层都可以依赖。"""


@dataclass(frozen=True, slots=True)
class LayerRule:
    """一条分层规则。

    ``layers`` 中的 ``"*"`` 表示**全包生效**（含未归层的顶层文件）。
    ``exempt`` 是**相对于包根**的 posix 路径前缀：命中即整条规则对该文件失效。
    """

    code: str
    layers: tuple[str, ...]
    allowed: frozenset[str]
    banned_modules: frozenset[str]
    description: str
    exempt: frozenset[str] = frozenset()


# AL2 = core + extract，二者是**同一个逻辑层**的两个子包，互相依赖必须放行。
# 把它们写进白名单而不是靠"同层豁免"，是为了让这条判断显式可见。
_L2_PACKAGES = frozenset({"artifact_spirit.core", "artifact_spirit.extract"})

_PROTOCOLS = frozenset(
    {
        "artifact_spirit.store.base",
        "artifact_spirit.model.base",
        _SHARED,
        # 纯标识符生成器：零 I/O、零存储依赖，AL2 需要它来生成记忆 ID。
        # 显式列进白名单而非放宽规则——白名单的价值就在于"每一条放行都看得见"。
        "artifact_spirit.store.ids",
        *_L2_PACKAGES,
    }
)

_IO_MODULES = frozenset({"httpx", "sqlite3", "openai"})

# 宿主标识。**按点分边界匹配**（见 `_banned_hit`）：
# 命中 `hermes` 即同时命中 `hermes_agent` / `hermes_constants` / `hermes_cli`——
# 修订前只比对首段，`agent.memory_provider` 这种点分路径根本抓不到。
_HOST_MARKERS = frozenset(
    {"hermes", "hermes_agent", "hermes_cli", "hermes_constants", "agent.memory_provider"}
)

# 并发/异步原语只在 runtime/ 出现（线程**创建**另由 `_check_thread_creation` 守）。
_CONCURRENCY_MODULES = frozenset({"concurrent", "multiprocessing", "asyncio"})

# R9：**层内实现模块**。它们只能在所属包内部被导入。
# 判据是"它是不是这一层的实现细节"：`sqlite_backend` 会连带拉入 sqlite-vec，
# `resolver` / `openai_compat` 绑定 httpx 与具体厂商协议——都是实现，不是接口。
# 跨层需要它们的能力时，正确做法是往 `*.base` 协议层加一个抽象，而不是直接 import。
_INTERNAL_ONLY: dict[str, str] = {
    "artifact_spirit.store.sqlite_backend": "store",
    "artifact_spirit.model.resolver": "model",
    "artifact_spirit.model.openai_compat": "model",
}

LAYER_RULES: tuple[LayerRule, ...] = (
    LayerRule(
        code="R1",
        layers=("AL2",),
        allowed=_PROTOCOLS,
        banned_modules=frozenset(),
        description=(
            "core/ 与 extract/ 只能依赖 store/base.py、model/base.py 的协议与数据类，"
            "以及共享内核 common（白名单外一律越界）"
        ),
    ),
    LayerRule(
        code="R2",
        layers=("AL3",),
        allowed=frozenset({"artifact_spirit.store", _SHARED}),
        banned_modules=frozenset({"httpx", "openai"}),
        description="store/ 禁止依赖 core/ / extract/ / model/（sqlite3 是它的本职，不在此列）",
    ),
    LayerRule(
        code="R3",
        layers=("AL4",),
        allowed=frozenset({"artifact_spirit.model", _SHARED}),
        banned_modules=frozenset({"sqlite3", "openai"}),
        description="model/ 禁止依赖 core/ / store/；且**不得引入 openai SDK**（直接用 httpx）",
    ),
    # R4 刻意**不带白名单**（allowed 为空 = 不做越界检查）：它只负责一件事——
    # 宿主耦合收敛到 provider.py 一处。白名单归属 R1，避免一条违规报三遍。
    LayerRule(
        code="R4",
        layers=("*",),
        allowed=frozenset(),
        banned_modules=_HOST_MARKERS,
        description="只有 provider.py 可依赖宿主（Hermes）模块，其余位置一律禁止",
        exempt=frozenset({"provider.py"}),
    ),
    LayerRule(
        code="R5",
        layers=("AL2",),
        allowed=frozenset(),
        banned_modules=_IO_MODULES,
        description="core/ 与 extract/ 内不得出现 httpx / sqlite3 / openai 等 I/O 库",
    ),
    LayerRule(
        code="R8",
        layers=("*",),
        allowed=frozenset(),
        banned_modules=_CONCURRENCY_MODULES,
        description="只有 runtime/ 可使用并发 / 异步原语，且只有它能创建线程",
        exempt=frozenset({"runtime/"}),
    ),
    LayerRule(
        code="R10",
        layers=("AL5",),
        allowed=frozenset(
            {
                "artifact_spirit.runtime",
                "artifact_spirit.config",
                "artifact_spirit.observability",
                "artifact_spirit.compliance",
                # AL5 只认**协议**，不认实现：跨层引用止步于这两个文件。
                "artifact_spirit.store.base",
                "artifact_spirit.model.base",
                _SHARED,
            }
        ),
        banned_modules=frozenset(),
        description=(
            "AL5 不得依赖业务层实现（核心/存储/模型的具体模块）——"
            f"唯一例外是组合根 {COMPOSITION_ROOT}"
        ),
        exempt=frozenset({COMPOSITION_ROOT}),
    ),
)


def iter_python_files(root: Path = PACKAGE_ROOT):
    """遍历包内的 Python 文件（跳过缓存）。"""
    for path in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _pkg_parts(path: Path, package_root: Path = PACKAGE_ROOT) -> list[str]:
    rel = path.resolve().relative_to(package_root.resolve())
    return ["artifact_spirit", *rel.parts[:-1]]


def resolved_imports(
    path: Path, package_root: Path = PACKAGE_ROOT, *, expand_names: bool = False
) -> set[str]:
    """把文件中的 import 语句解析为**绝对点分路径**集合。

    相对导入（``from ..common import x``）会被解析到 ``artifact_spirit.common``——
    否则 R2 之类的规则会被相对导入轻易绕过。

    ``expand_names=True`` 时额外把 ``from X import a, b`` 展开成 ``X.a`` / ``X.b``：
    白名单只需要包级目标（``from ..store import base`` 与 ``from ..store.base import X``
    等价），但 R9 必须看到具体模块名，否则 ``from ..store import sqlite_backend``
    会伪装成一次对包根的合法依赖。
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return set()

    pkg = _pkg_parts(path, package_root)
    out: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(pkg) - (node.level - 1)
                base = pkg[: max(keep, 0)]
                target = ".".join([*base, node.module]) if node.module else ".".join(base)
            else:
                target = node.module or ""
            if not target:
                continue
            out.add(target)
            if expand_names:
                out.update(
                    f"{target}.{alias.name}" for alias in node.names if alias.name != "*"
                )
    return out


def module_imports(path: Path) -> set[str]:
    """收集顶层模块名（供依赖扫描复用）。"""
    return {name.split(".")[0] for name in resolved_imports(path)}


def _package_of(path: Path, package_root: Path) -> str | None:
    """直接子包名（``core`` / ``runtime`` …）；顶层文件返回 ``None``。"""
    parts = _pkg_parts(path, package_root)
    return parts[1] if len(parts) > 1 else None


def _layer_of(path: Path, package_root: Path) -> str | None:
    """文件所属的层编号（AL1–AL5）；未登记则 ``None``。"""
    package = _package_of(path, package_root)
    return LAYER_MAP.get(package) if package else None


def _layer_id_of_module(target: str) -> str | None:
    """点分模块路径 → 层编号（用于同层豁免判断）。"""
    parts = target.split(".")
    if len(parts) < 2 or parts[0] != "artifact_spirit":
        return None
    return LAYER_MAP.get(parts[1])


def _matches(rule: LayerRule, layer: str | None) -> bool:
    return "*" in rule.layers or (layer is not None and layer in rule.layers)


def _is_exempt(path: Path, exempt: frozenset[str], root: Path) -> bool:
    """``exempt`` 是相对于包根的 posix 路径或前缀（``"runtime/"``）。"""
    if not exempt:
        return False
    rel = path.resolve().relative_to(Path(root).resolve()).as_posix()
    return any(rel == entry or rel.startswith(entry) for entry in exempt)


def _banned_hit(target: str, banned: frozenset[str]) -> str | None:
    """按**点分边界**判断 ``target`` 是否命中黑名单。

    边界感知是必需的：

    * ``httpx`` 命中 ``httpx``（顶层同名），也命中 ``httpx._client``；
    * ``hermes`` 命中 ``hermes_constants``——宿主的模块名带后缀，
      只比对首段会漏掉；
    * 但 ``agent`` **不**命中 ``agent.memory_provider``（后者是完整路径）。
    """
    top = target.split(".")[0]
    for entry in banned:
        if target == entry or target.startswith(entry + "."):
            return entry
        if "." not in entry and (top == entry or top.startswith(entry + "_")):
            return entry
    return None


def _is_allowed(target: str, allowed: frozenset[str]) -> bool:
    """``allowed`` 里既可以是模块也可以整层前缀；空集合 = 不做白名单检查。"""
    if not allowed:
        return True
    for entry in allowed:
        if target == entry or target.startswith(entry + "."):
            return True
    return False


def _same_layer(target: str, layer: str) -> bool:
    """目标是否与本文件同层。

    没有这条，``core/__init__.py`` 汇总导出自己的子模块都会被判为"越界依赖"，
    ``runtime/status.py`` 引用 ``runtime/lifecycle.py`` 同理——那是把规则用错了地方。
    """
    return _layer_id_of_module(target) == layer


def check_architecture(root: Path = PACKAGE_ROOT) -> list[dict]:
    """执行 R1–R10 检查，返回违规列表（空 = 全绿）。"""
    root = Path(root)
    violations: list[dict] = []

    for path in iter_python_files(root):
        layer = _layer_of(path, root)
        imports = resolved_imports(path, root)

        for rule in LAYER_RULES:
            if not _matches(rule, layer):
                continue
            if _is_exempt(path, rule.exempt, root):
                continue
            for target in sorted(imports):
                hit = _banned_hit(target, rule.banned_modules)
                if hit is not None:
                    violations.append(
                        {
                            "rule": rule.code,
                            "file": str(path),
                            "message": f"{rule.description}（命中：{hit}）",
                        }
                    )
                # 白名单检查：只对"包内跨层引用"生效
                if not target.startswith("artifact_spirit."):
                    continue
                if layer is not None and _same_layer(target, layer):
                    continue  # 同层内部依赖（含 __init__ 汇总导出）永远放行
                if _is_allowed(target, rule.allowed):
                    continue
                violations.append(
                    {
                        "rule": rule.code,
                        "file": str(path),
                        "message": f"{rule.description}（越界依赖：{target}）",
                    }
                )

    violations.extend(_check_implementation_leaks(root))
    violations.extend(_check_package_init(root))
    violations.extend(_check_thread_creation(root))
    return violations


def _check_implementation_leaks(root: Path) -> list[dict]:
    """R9：层内实现模块只在层内可见。

    与白名单互补：白名单管"能往哪去"，这条管"实现模块不能被当成接口用"。
    组合根同样豁免——它存在的全部理由就是装配具体实现。
    """
    violations: list[dict] = []
    for path in iter_python_files(root):
        if _is_exempt(path, frozenset({COMPOSITION_ROOT}), root):
            continue
        owner_package = _package_of(path, root)
        targets = resolved_imports(path, root, expand_names=True)
        for target, owner in _INTERNAL_ONLY.items():
            if owner_package == owner:
                continue
            if not any(
                target == mid or mid.startswith(target + ".") for mid in targets
            ):
                continue
            violations.append(
                {
                    "rule": "R9",
                    "file": str(path),
                    "message": (
                        f"实现模块 {target} 只能在 {owner}/ 内部导入（R9）——"
                        "跨层请依赖 *.base 协议"
                    ),
                }
            )
    return violations


_THREAD_SYMBOLS = frozenset({"Thread", "ThreadPoolExecutor", "Process", "ProcessPoolExecutor"})


def _check_thread_creation(root: Path) -> list[dict]:
    """R8：只有 ``runtime/`` 可以**创建**线程/进程。

    注意规则只针对"创建"——``threading.Lock``（如 ULID 生成器里的互斥）不违反 R8，
    因此不能简单地把 ``threading`` 整个模块列为禁用。
    """
    violations: list[dict] = []
    for path in iter_python_files(root):
        if _package_of(path, root) == "runtime":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _THREAD_SYMBOLS:
                violations.append(
                    {
                        "rule": "R8",
                        "file": str(path),
                        "message": f"仅 runtime/ 可创建线程或进程（命中：{name}）",
                    }
                )
    return violations


def _check_package_init(root: Path) -> list[dict]:
    """R6：``artifact_spirit/__init__.py`` 只做 register，零业务逻辑。

    判据是"模块顶层不得 import 包内业务模块"——延迟导入（函数体内）是允许的，
    这正是 R6 想要的效果：宿主加载插件时不付出重量级导入的代价。
    """
    init = root / "__init__.py"
    if not init.exists():
        return []
    try:
        tree = ast.parse(init.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return []

    pkg = _pkg_parts(init, root)
    violations: list[dict] = []

    for node in tree.body:
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(pkg) - (node.level - 1)
                base = pkg[: max(keep, 0)]
                targets = [
                    ".".join([*base, node.module]) if node.module else ".".join(base)
                ]
            elif node.module:
                targets = [node.module]

        for target in targets:
            if target.startswith("artifact_spirit.") and target != _SHARED:
                violations.append(
                    {
                        "rule": "R6",
                        "file": str(init),
                        "message": (
                            f"包 __init__ 顶层不得导入业务模块（{target}）——"
                            "应延迟到函数体内（R6）"
                        ),
                    }
                )
    return violations
