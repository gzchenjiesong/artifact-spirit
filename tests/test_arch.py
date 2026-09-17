"""架构规则 R1–R10 与依赖扫描（T-AL3-23 / T-AL5-11）。

**本文件的价值在于"扫描器自身有效"**：不仅要断言当前代码全绿，
还要往临时目录里注入违规代码，断言扫描器确实报错——否则一个永远返回
"通过"的检查器会给人虚假的安全感。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from artifact_spirit.compliance import (
    FORBIDDEN_MODULES,
    PACKAGE_ROOT,
    check_architecture,
    scan_dependencies,
    scan_forbidden_modules,
    scan_secrets,
)

# --------------------------------------------------------------------------- #
# 当前代码库全绿
# --------------------------------------------------------------------------- #


def test_architecture_rules_pass_on_package():
    violations = check_architecture(PACKAGE_ROOT)
    assert violations == [], f"架构违规：{violations}"


def test_no_forbidden_modules_currently():
    assert scan_forbidden_modules(PACKAGE_ROOT) == []


def test_no_secrets_currently():
    project_root = PACKAGE_ROOT.parent.parent
    assert scan_secrets(PACKAGE_ROOT) == []
    assert scan_secrets(project_root, extra_files=(project_root / "pyproject.toml",)) == []


def test_pyproject_has_no_forbidden_dependency():
    project_root = PACKAGE_ROOT.parent.parent
    assert scan_dependencies(PACKAGE_ROOT, pyproject=project_root / "pyproject.toml") == []


def test_forbidden_module_policy_is_not_empty():
    """许可黑名单为空 = R7 门禁静默失效。

    这条断言把"别忘了维护许可策略"变成 CI 会失败的检查——否则某次"清理词典"
    的改动会让所有注入测试一并失去意义（它们只能测到空集合）。
    """
    assert FORBIDDEN_MODULES, "许可黑名单为空，R7 依赖扫描将永远通过"


# --------------------------------------------------------------------------- #
# 扫描器有效性自证（注入 → 必须报错）
# --------------------------------------------------------------------------- #


@pytest.fixture
def injected_package(tmp_path: Path) -> Path:
    """把包复制到临时目录，供注入测试用。"""
    target = tmp_path / "artifact_spirit"
    shutil.copytree(PACKAGE_ROOT, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def test_dep_scan_catches_injected_forbidden_module(injected_package: Path):
    """`import X` 形式必须被抓到。

    注入用的是黑名单里的**已识别实例**（而不是写死某个产品名）：
    要验证的是"扫描器有效"，规则本身按许可类别维护。
    """
    forbidden = min(FORBIDDEN_MODULES)
    (injected_package / "evil.py").write_text(f"import {forbidden}\n", encoding="utf-8")
    violations = scan_forbidden_modules(injected_package)
    assert any(v["module"] == forbidden for v in violations)


def test_dep_scan_catches_injected_forbidden_submodule(injected_package: Path):
    """`from X import Y` 形式同样必须被抓到（按顶层模块名匹配）。"""
    forbidden = max(FORBIDDEN_MODULES)
    (injected_package / "evil.py").write_text(
        f"from {forbidden} import Client\n", encoding="utf-8"
    )
    violations = scan_forbidden_modules(injected_package)
    assert any(v["module"] == forbidden for v in violations)


def test_arch_rule_r2_catches_store_importing_core(injected_package: Path):
    evil = injected_package / "store" / "evil.py"
    evil.write_text("from ..core import facade\n", encoding="utf-8")
    violations = check_architecture(injected_package)
    assert any(v["rule"] == "R2" for v in violations), violations


def test_arch_rule_r5_catches_core_importing_sqlite3(injected_package: Path):
    """R5 独立生效——**且只报 R5**。

    DES-REV-003 修订前 R1/R4/R5 三条规则的 ``layers`` + ``allowed`` 完全一样，
    core 里一处越界依赖会同时报三条（文案还串台）。修订后职责拆分：
    R5 只管 I/O 库，命中就只报 R5。
    """
    evil = injected_package / "core" / "evil.py"
    evil.parent.mkdir(exist_ok=True)
    evil.write_text("import sqlite3\n", encoding="utf-8")
    violations = check_architecture(injected_package)
    rules = {v["rule"] for v in violations if "evil.py" in v["file"]}
    assert rules == {"R5"}, violations


def test_arch_rule_r1_reports_boundary_violation_exactly_once(injected_package: Path):
    """core 越界依赖**只报一条**（R1），不再被 R4/R5 重复报三遍。

    修订前 R1/R4/R5 三条规则的 ``layers`` + ``allowed`` 完全相同，
    同一行越界 import 会产出三条文案还串台的告警——告警一多就会被整体忽略。
    """
    evil = injected_package / "core" / "evil.py"
    evil.parent.mkdir(exist_ok=True)
    evil.write_text("from ..config import loader\n", encoding="utf-8")
    violations = [v for v in check_architecture(injected_package) if "evil.py" in v["file"]]
    assert [v["rule"] for v in violations] == ["R1"], violations


def test_arch_rule_r3_catches_model_importing_store(injected_package: Path):
    evil = injected_package / "model" / "evil.py"
    evil.parent.mkdir(exist_ok=True)
    evil.write_text("from ..store import sqlite_backend\n", encoding="utf-8")
    violations = check_architecture(injected_package)
    assert any(v["rule"] == "R3" for v in violations), violations


def test_arch_rule_r6_catches_heavyweight_package_init(injected_package: Path):
    (injected_package / "__init__.py").write_text(
        "from .provider import ArtifactSpiritProvider\n\n"
        "def register(ctx):\n"
        "    ctx.register_memory_provider(ArtifactSpiritProvider())\n",
        encoding="utf-8",
    )
    violations = check_architecture(injected_package)
    assert any(v["rule"] == "R6" for v in violations), violations


def test_arch_rule_r8_catches_thread_creation_outside_runtime(injected_package: Path):
    evil = injected_package / "core" / "evil.py"
    evil.parent.mkdir(exist_ok=True)
    evil.write_text(
        "import threading\n\n"
        "def spawn():\n"
        "    return threading.Thread(target=lambda: None)\n",
        encoding="utf-8",
    )
    violations = check_architecture(injected_package)
    assert any(v["rule"] == "R8" for v in violations), violations


def test_arch_rule_r4_catches_dotted_host_module(injected_package: Path):
    """R4 必须抓到**点分路径**形式的宿主模块。

    修订前 R4 只比对 ``name.split(".")[0]``，对 ``agent.memory_provider``
    永远得到 ``"agent"``——标记写在那里，却是死代码。
    """
    evil = injected_package / "runtime" / "evil.py"
    evil.write_text("from agent.memory_provider import spawn_context_thread\n", encoding="utf-8")
    violations = check_architecture(injected_package)
    assert any(v["rule"] == "R4" for v in violations), violations


def test_arch_rule_r4_catches_host_module_with_suffix(injected_package: Path):
    """宿主的模块名带后缀（``hermes_constants``），首段比对同样会漏。

    provider 里真实存在的就是这个形式，所以按**点分边界 + 下划线后缀**匹配：
    ``hermes`` 命中 ``hermes_constants``。
    """
    evil = injected_package / "tools" / "evil.py"
    evil.write_text("from hermes_constants import get_hermes_home\n", encoding="utf-8")
    violations = check_architecture(injected_package)
    assert any(v["rule"] == "R4" for v in violations), violations


def test_arch_rule_r4_exempts_provider_only(injected_package: Path):
    """唯一允许依赖宿主的是 ``provider.py``——这正是 R4 的**目的**。

    如果忘了豁免它，规则会把"规定允许的事"报成红灯，然后被人删掉。
    """
    provider = injected_package / "provider.py"
    provider.write_text(
        provider.read_text(encoding="utf-8")
        + "\nfrom agent.memory_provider import spawn_context_thread\n",
        encoding="utf-8",
    )
    violations = [v for v in check_architecture(injected_package) if v["rule"] == "R4"]
    assert violations == [], violations


def test_arch_rule_r8_catches_asyncio_outside_runtime(injected_package: Path):
    """并发/异步原语只在 ``runtime/`` 出现（模块级黑名单，与线程创建互补）。"""
    evil = injected_package / "store" / "evil.py"
    evil.write_text("import asyncio\n", encoding="utf-8")
    violations = check_architecture(injected_package)
    assert any(v["rule"] == "R8" for v in violations), violations


def test_arch_rule_r9_catches_implementation_module_leaking_across_layers(
    injected_package: Path,
):
    """``store.sqlite_backend`` 不能被层外 import——它会把 sqlite-vec 一起拖出去。

    这里刻意用 ``from ..store import sqlite_backend`` 这种**包根 + 名字**的写法：
    白名单只看包级目标，会被它绕过；R9 必须展开 imported names 才抓得住。
    """
    evil = injected_package / "core" / "evil.py"
    evil.parent.mkdir(exist_ok=True)
    evil.write_text("from ..store import sqlite_backend\n", encoding="utf-8")
    violations = check_architecture(injected_package)
    assert any(v["rule"] == "R9" for v in violations), violations


def test_arch_rule_r9_allows_composition_root(injected_package: Path):
    """组合根装配具体实现是它的职责，不构成违规。"""
    from artifact_spirit.compliance import COMPOSITION_ROOT

    root_file = injected_package / COMPOSITION_ROOT
    source = root_file.read_text(encoding="utf-8")
    assert "sqlite_backend" in source, "组合根已不再装配 sqlite_backend？规则例外的前提变了"
    violations = [v for v in check_architecture(injected_package) if v["rule"] == "R9"]
    assert violations == [], violations


def test_arch_rule_r10_catches_al5_importing_business_layer(injected_package: Path):
    """AL5 是"横向支撑"，不是"可以随手拿业务层"。

    ``config/`` 直接 import ``core.facade`` 修不出任何东西，只会让 AL5 变成
    第二套业务实现的入口。
    """
    evil = injected_package / "config" / "evil.py"
    evil.write_text("from ..core import facade\n", encoding="utf-8")
    violations = check_architecture(injected_package)
    assert any(v["rule"] == "R10" for v in violations), violations


def test_arch_rule_r10_exempts_composition_root_only(injected_package: Path):
    """例外必须**精确到一个文件**：同样的 import 放在 AL5 别处就要红灯。"""
    from artifact_spirit.compliance import COMPOSITION_ROOT

    exempt_target = injected_package / COMPOSITION_ROOT
    exempt_target.write_text(
        exempt_target.read_text(encoding="utf-8") + "\nfrom ..extract import dedup\n",
        encoding="utf-8",
    )
    evil = injected_package / "observability" / "evil.py"
    evil.write_text("from ..extract import dedup\n", encoding="utf-8")

    flagged = {v["file"] for v in check_architecture(injected_package) if v["rule"] == "R10"}
    assert str(evil) in flagged, flagged
    assert str(exempt_target) not in flagged, flagged


def test_layer_map_covers_every_package():
    """**未登记的包 = 逃过所有分层规则的包**，所以登记表必须与目录同步。

    修订前 ``_layer_of`` 只认 6 个包，``tools/`` / ``runtime/`` / ``compliance/``
    与全部顶层文件一条规则都不受——这是"规则看起来在跑、其实没覆盖"的典型形态。
    """
    from artifact_spirit.compliance import LAYER_IDS, LAYER_MAP

    assert set(LAYER_MAP.values()) <= set(LAYER_IDS)

    packages = {
        p.name
        for p in PACKAGE_ROOT.iterdir()
        if p.is_dir() and p.name != "__pycache__" and (p / "__init__.py").exists()
    }
    modules = {p.stem for p in PACKAGE_ROOT.glob("*.py") if p.name != "__init__.py"}
    entries = packages | modules
    # 顶层 `common.py` / `config_schema.py` 是零包内依赖的共享内核，
    # 刻意不进层表（见 arch_rules 模块文档）。
    shared_kernel = {"common", "config_schema"}

    assert entries - set(LAYER_MAP) - shared_kernel == set(), (
        "有包未登记到 LAYER_MAP，将逃过全部分层规则"
    )
    assert set(LAYER_MAP) - entries == set(), "LAYER_MAP 里有已不存在的包"


def test_secret_scan_catches_injected_key(tmp_path: Path):
    # 刻意用拼接构造，避免本文件自身命中密钥扫描
    fake_key = "sk-" + "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
    (tmp_path / "leak.py").write_text(f'API_KEY = "{fake_key}"\n', encoding="utf-8")
    violations = scan_secrets(tmp_path)
    assert any(v["kind"] == "openai_key" for v in violations)


def test_secret_scan_ignores_placeholders(tmp_path: Path):
    (tmp_path / "config.py").write_text(
        'API_KEY_ENV = "ARTIFACT_SPIRIT_API_KEY"\n'
        "EXAMPLE_KEY = \"sk-\" + \"your-key-example-placeholder\"\n",
        encoding="utf-8",
    )
    assert scan_secrets(tmp_path) == []


# --------------------------------------------------------------------------- #
# 包内**不得存在导入环**——且与导入顺序无关
#
# 刻意不占 R 编号：这是正确性约束（违反即 ImportError），不是分层规则。
# --------------------------------------------------------------------------- #


def test_no_circular_import_regardless_of_order():
    """每个内部模块都必须能被**单独第一个**导入。

    这是实现期真实踩到的坑：`extract/schema.py` 曾为取一个领域常量
    （`MEMORY_TYPES`）反向依赖 `core` 包，而 `core/__init__` 又会加载
    facade → extractor。于是"先 import extract 的调用方"直接 ImportError。

    单测当时全绿，因为测试文件固定先 import core——**测试的导入顺序
    掩盖了缺陷**。所以这条用例必须用子进程、逐个模块单独拉起，
    才能真实反映"宿主可能以任何顺序导入"这一事实。
    """
    import subprocess
    import sys

    modules = [
        "artifact_spirit",
        "artifact_spirit.common",
        "artifact_spirit.store",
        "artifact_spirit.model",
        "artifact_spirit.core",
        "artifact_spirit.extract",
        "artifact_spirit.config",
        "artifact_spirit.runtime",
        "artifact_spirit.observability",
        "artifact_spirit.tools",
        "artifact_spirit.provider",
        "artifact_spirit.cli",
        "artifact_spirit.compliance",
    ]
    src = str(PACKAGE_ROOT.parent)
    failures = []
    for module in modules:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": src},
        )
        if proc.returncode != 0:
            last = (proc.stderr or "").strip().splitlines()[-1:]
            failures.append(f"{module}: {last[0] if last else '?'}")
    assert not failures, "存在导入环（与顺序相关）：\n" + "\n".join(failures)


def test_extract_does_not_depend_on_core_package():
    """`extract/` 不得依赖 `core/` **包**（只可依赖其叶模块或包根）。

    依赖包会触发 `core/__init__` → facade → extractor 的环。
    这条断言把"为什么不能这么写"固定在代码里，而不是留在某人的记忆里。
    """
    import ast

    offenders = []
    for path in (PACKAGE_ROOT / "extract").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module in {"core", "..core"}:
                offenders.append(f"{path.name}: from {node.module} import ...")

            # 合并成一个 if 会丢掉这层区分，可读性反而更差。
            elif isinstance(node, ast.ImportFrom) and (node.level or 0) >= 2:  # noqa: SIM102
                # 相对层级 2 且模块名首段为 core → 越界
                if (node.module or "").split(".")[0] == "core":
                    offenders.append(f"{path.name}: from {(node.module or '')} ...")
    assert not offenders, "extract 反向依赖了 core 包：\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# vec0 KNN 的写法必须跨 SQLite 版本可移植
# --------------------------------------------------------------------------- #


def test_vec0_knn_uses_k_constraint_not_limit():
    """vec0 的 KNN 必须写成 ``AND k = ?``，不能用 ``LIMIT``。

    这不是风格偏好，是**可移植性**：实测 sqlite-vec 0.1.9 在两个 SQLite 版本上行为不同

    | SQLite | ``ORDER BY distance LIMIT ?`` | ``AND k = ?`` |
    |---|---|---|
    | 3.53（开发机） | ✅ | ✅ |
    | 3.40（Debian 12 / Python 3.11） | ❌ 报 "A LIMIT or 'k = ?' constraint is required on vec0 knn queries" | ✅ |

    部署目标机器正是后者。本地全绿、上线后向量召回整体不可用——这类缺陷只有在
    真实环境里才暴露，所以这里用源码断言把它钉住。
    """
    backend = (PACKAGE_ROOT / "store" / "sqlite_backend.py").read_text(encoding="utf-8")
    knn_lines = [ln for ln in backend.splitlines() if "embedding MATCH" in ln]
    assert knn_lines, "找不到 KNN 查询——实现变了的话这条断言要跟着改"
    for line in knn_lines:
        assert "k = ?" in line, f"KNN 查询必须用 `AND k = ?`：{line.strip()}"
        assert "LIMIT" not in line.upper(), f"KNN 查询不得使用 LIMIT：{line.strip()}"
