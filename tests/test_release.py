"""发布契约（v1.0 收口）：版本号、许可、命令行入口。

这三样都**只在"用户拿到包之后"才生效**，所以本地跑得再熟也不会暴露问题——
它红的时候，用户已经在自己的机器上了。因此它们必须由用例钉住，而不是靠"应该没问题"。
"""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _project() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]


def test_version_is_a_release_and_both_copies_agree():
    """版本号**两处定义必须一致**，且**不带 dev 后缀**。

    两处定义是现实：打包要 `pyproject.toml`，运行时自报要 `__init__.py`。
    它们不一致的后果很隐蔽——**装出来的包自报的版本，与包管理器记录的不是同一个**，
    于是"线上跑的是哪个版本"这个问题有了两个都说得通的答案。

    只改一处是这类收口最常见的漏法，所以这里查的是**两处相等**，不是"某一处对"。
    """
    packaged = _project()["version"]
    from artifact_spirit import __version__

    assert packaged == __version__, (
        f"pyproject 写 {packaged!r}，而 __init__ 自报 {__version__!r}——发布前必须一致"
    )
    assert "dev" not in packaged, f"发布版不该带 dev 后缀：{packaged!r}"
    assert re.match(r"^\d+\.\d+\.\d+$", packaged), f"应当是 x.y.z 形式：{packaged!r}"


def test_license_is_declared():
    """许可必须**显式声明**（R7）——它是可再分发的先决条件。"""
    assert _project().get("license"), "pyproject 里必须声明 license"


def test_license_text_exists_and_is_shipped():
    """`LICENSE` 正文必须存在，且**被声明为要打包进去**。

    只写 `license = "MIT"` 是不够的：GitHub 靠**文件**识别协议，
    而"声明了协议却没有任何正文"正是发布前最常见的一种漏法——
    两边各自看都自洽，合起来却是"声称 MIT 的包没有任何许可文本"。
    """
    path = ROOT / "LICENSE"
    assert path.exists(), "缺少 LICENSE 正文"
    text = path.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "WITHOUT WARRANTY" in text.upper(), "正文必须含免责声明——那才是 MIT 的实质条款"

    files = _project().get("license-files") or []
    assert any("LICENSE" in str(f) for f in files), (
        "应当在 `license-files` 里带上 LICENSE，否则发布件不带许可正文"
    )


def test_mypy_is_configured_not_left_implicit():
    """`mypy` 必须有**显式配置**——不能"从没跑过"却出现在门禁表里。

    v1.0 时它一条配置都没有（仓库里只有 `[tool.ruff]`），却在内外部被当作 CI 门禁提。
    那是典型的**纸面门禁**：给人已经检查过的错觉，比没有门禁更误导。

    这里要求的不是"清零"，而是**配置在案、剩余项可解释**：
    第三方无 stub 与宿主模块按模块声明豁免，条件定义（`try/except` 同名）按模块豁免，
    其余一类在实现记录里写明。
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.mypy]" in text, "mypy 必须有显式配置，不能只靠默认值"
    assert "[[tool.mypy.overrides]]" in text, (
        "豁免要**按模块**声明；全局放行会把'依赖真的装错了'也一起放过"
    )


def test_ci_runs_exactly_the_gates_the_readme_claims():
    """CI 配置要存在，且**跑的就是 README 写的门禁**。

    两边不一致时，"本地绿了、CI 红"会成为常态——而人很快会开始**忽略 CI**，
    于是它退化成一个装饰。这里只查"存在且包含那两条"，不模拟 CI 运行时。

    同时钉住**反向**的一件事：`mypy` **不该**出现在 CI 里。
    它还有 41 项未清零，放进去只会常年红着——**常年红的检查等于没有检查**。
    """
    ci = ROOT / ".github" / "workflows" / "ci.yml"
    assert ci.exists(), "缺少 CI 配置"
    text = ci.read_text(encoding="utf-8")
    assert "ruff check ." in text, "CI 必须跑 ruff"
    assert "pytest -q" in text, "CI 必须跑 pytest"

    # **只查命令行，不查注释**：CI 文件里正解释着"为什么不放 mypy"，
    # 那是文档而不是一条会红的检查。断言扫全文，就会把说明当成违规——
    # 这正是"断言比意图宽一格"的典型。
    offenders = [
        line.strip()
        for line in text.splitlines()
        if "mypy" in line and not line.strip().startswith("#")
    ]
    assert not offenders, f"mypy 未清零，刻意不进 CI：{offenders}"


def test_console_scripts_point_at_callables_that_exist():
    """声明了命令行入口，就必须**真的能导到**。

    模块路径写错时，问题出现在**用户安装之后**：报的是一句
    "command not found"，看起来像安装失败，而不像配置写错。
    """
    scripts = _project().get("scripts") or {}
    assert scripts, "应当声明命令行入口（aspirit / artifact-spirit）"
    for name, target in scripts.items():
        module, _, func = target.partition(":")
        assert module and func, f"{name} 的入口写法应当是「模块:函数」，实得 {target!r}"
        loaded = importlib.import_module(module)
        assert callable(getattr(loaded, func, None)), f"{name}：{module} 里没有可调用的 {func}"


def test_requires_python_matches_what_the_code_actually_uses():
    """`requires-python` 必须覆盖代码真正用到的语法/标准库。

    钉住 `>=3.11`：`datetime.fromisoformat` 吃 `Z` 后缀（3.11 起）与
    标准库 `tomllib`（3.11 起）都在被用——把它们写进声明里，
    "在 3.10 上装得上却跑不起来"才不会发生。
    """
    requires = _project().get("requires-python") or ""
    assert "3.11" in requires, f"代码用到 3.11 的语法与标准库，声明却是 {requires!r}"


@pytest.mark.parametrize("module", ["artifact_spirit", "artifact_spirit.cli"])
def test_release_artifacts_are_importable(module: str):
    """发布件至少得**导得进来**——这是"可安装"的最低证据。"""
    assert importlib.import_module(module) is not None
