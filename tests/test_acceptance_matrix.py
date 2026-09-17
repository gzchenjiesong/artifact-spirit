"""校验五层验收追溯矩阵（设计纲领 **E7** 第 ⑤ 条：规则本身必须可执行）。

E7 说的是"用例错了就改用例、不许改代码去迁就错的用例"。但这句话如果**只写在文档里**，
它自己就违反了 E7——**规则也必须是可执行的检查**。

这里八条断言各堵一个具体失效模式。**除第一条外，每条都对 AL1–AL5 五层各跑一遍**
（2026-09-16 从"仅 AL1"扩到五层：矩阵不覆盖某层，等于那层的验收条款从未被对照过）：

| 断言 | 堵的失效模式 |
|---|---|
| `test_every_layer_is_declared` | 又漏了某一层（矩阵必须五层齐全） |
| `test_every_task_in_task_book_has_matrix_entry` | 任务书新增验收项，追溯表没跟上——新要求悄悄没人管 |
| `test_matrix_keys_exist_in_design_docs` | 追溯链指向**已废止**的条款 |
| `test_referenced_tests_exist` | **主力**：用例被删 / 改名后，追溯表不再静默失效 |
| `test_statuses_are_known` | 状态写错（如 `covred`）被当成有效档位 |
| `test_covered_items_actually_cite_tests` | 把"测过了"写成口头承诺 |
| `test_gaps_and_stale_items_are_acknowledged` | 把缺口偷偷记成"通过"（违反 E4） |
| `test_invariant_tests_are_all_referenced` | 反方向：不变量用例存在，却没挂到任何验收项（野用例） |

判据取自逐层复核的教训：**"过时"必须变成红灯，不能是沉默**。

> **矩阵自身的失效方式有两种，方向相反、性质相同**（见 07-设计评审.md §12.9）：
> ① 引用**不存在**的用例名（靠本文件的 `test_referenced_tests_exist` 抓）；
> ② 把**已覆盖**的条目继续记成 `gap`（**没有任何自动信号**，只能靠与用例清单**双向比对**）。
> 所以本文件既查"引用存不存在"，也查"缺口有没有留痕"。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from acceptance_matrix import (
    COVERED,
    LAYERS,
    REVIEW_LOG,
    STATUSES,
    TASK_BOOKS,
    Item,
    Layer,
)

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"

SOLUTION_DESIGN = "docs/design/00-方案设计.md"

ALL_ITEMS: tuple[Item, ...] = tuple(item for layer in LAYERS for item in layer.all_items)
"""五层全部条目（含各层的 `INV-` 不变量）——供跨层断言使用。"""


def _read(relative: str) -> str:
    """读权威文档；**文件不在原位就报红**（否则追溯矩阵会静默指向空气）。"""
    path = ROOT / relative
    assert path.exists(), f"权威文档不在原位：{relative}"
    return path.read_text(encoding="utf-8")


def _test_names() -> set[str]:
    """扫出 `tests/` 下所有测试函数名。

    用 AST 扫而不是导入模块：导入会执行模块级代码（建库、起线程），
    而这里只需要名字。
    """
    names: set[str] = set()
    for path in sorted(TESTS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_test_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_test_func and node.name.startswith("test_"):
                names.add(node.name)
    return names


def _dangling_tests(layer: Layer, known: set[str]) -> list[tuple[str, str]]:
    return [
        (item.key, name)
        for item in layer.all_items
        for name in item.tests
        if name not in known
    ]


def test_every_layer_is_declared() -> None:
    """五层都必须在矩阵里——**漏一层等于那层的验收从未被对照过**。"""
    declared = {layer.name for layer in LAYERS}
    assert declared == set(TASK_BOOKS), (
        f"矩阵层集与任务书层集不一致：矩阵有 {sorted(declared)}，"
        f"任务书有 {sorted(TASK_BOOKS)}"
    )
    empty = [layer.name for layer in LAYERS if not layer.items]
    assert not empty, f"这些层的矩阵是空的（等于没覆盖）：{empty}"


def test_every_task_in_task_book_has_matrix_entry() -> None:
    """任务书里的每个 `T-{层}-xx` 都要在追溯矩阵里有条目。

    这是"设计改了用例没跟上"的正面堵口：**新增一条验收要求而追溯表没跟，
    门禁立刻报红**——它逼着人在加要求的同时交代"怎么验、谁来验"。
    """
    problems: list[str] = []
    for layer in LAYERS:
        text = _read(layer.task_book)
        declared = set(re.findall(rf"{layer.task_prefix}-\d\d", text))
        assert declared, f"{layer.name} 的任务书里扫不到 {layer.task_prefix}-xx——文档结构变了？"
        recorded = {item.key.split("#")[0] for item in layer.items}
        missing = sorted(declared - recorded)
        if missing:
            problems.append(f"{layer.name} 缺追溯条目：{missing}")
    assert not problems, "这些验收项没有追溯条目（新增条款请同步建表）：\n" + "\n".join(problems)


def test_matrix_keys_exist_in_design_docs() -> None:
    """追溯矩阵引用的编号必须真实存在。

    防的是**条款已废止而追溯链还挂着**——那种情况下矩阵会让人误以为
    "这条要求有人管"，实际它早就不在设计要求里了。
    """
    design_docs = _read(SOLUTION_DESIGN)
    problems: list[str] = []

    for layer in LAYERS:
        task_book = _read(layer.task_book)
        lld_and_design = _read(layer.lld) + design_docs

        for item in layer.items:
            base = item.key.split("#")[0]
            if base not in task_book:
                problems.append(f"{layer.name}: {item.key} 在编码任务书里不存在（条款已废止？）")

        for item in layer.invariants:
            if item.key not in lld_and_design:
                problems.append(f"{layer.name}: {item.key} 在 LLD / 方案设计里不存在")

    assert not problems, "\n".join(problems)


def test_referenced_tests_exist() -> None:
    """矩阵引用的用例必须真实存在——**这是防"用例过时"的主力**。

    教训：**一份没人校验的对照表比没有更危险**。用例被删除或改了名字，
    这条断言立刻报红，追溯表不可能"静静地失效"。
    """
    known = _test_names()
    assert known, "没扫到任何测试用例——扫描路径错了？"

    dangling = [(layer.name,) + pair for layer in LAYERS for pair in _dangling_tests(layer, known)]
    assert not dangling, f"矩阵引用了不存在的用例（已被删除或改名）：{dangling}"


def test_statuses_are_known() -> None:
    """状态必须是四档之一——否则拼错的档位会被当成有效结论。"""
    unknown = [(item.key, item.status) for item in ALL_ITEMS if item.status not in STATUSES]
    assert not unknown, f"这些条目的状态不在 {STATUSES} 之内：{unknown}"


def test_covered_items_actually_cite_tests() -> None:
    """`covered` 却不引用用例 = 把"测过了"写成口头承诺。"""
    bare = [item.key for item in ALL_ITEMS if item.status == COVERED and not item.tests]
    assert not bare, f"这些条目声称 covered 却没有用例：{bare}"


def test_gaps_and_stale_items_are_acknowledged() -> None:
    """非 `covered` 的条目必须写明裁决编号，且该编号能在评审日志里找到。

    矩阵**允许**记缺口（`weak` / `gap` / `stale`），但**必须留痕**（E4）。
    编号指向 `docs/design/07-设计评审.md` 的逐层复核部分；找不到编号或找不到对应条目，
    就说明这个缺口从来没被裁决过——那正是"缺口能活很久"的原因。
    """
    review = _read(REVIEW_LOG)
    problems: list[str] = []

    for item in ALL_ITEMS:
        if item.status == COVERED:
            continue
        if not item.note:
            problems.append(f"{item.key}（{item.status}）没有任何说明")
            continue
        rulings = re.findall(r"P\d+-\d+", item.note)
        if not rulings:
            problems.append(f"{item.key}（{item.status}）未指明裁决编号（形如 P1-5）")
            continue
        problems.extend(
            f"{item.key} 引用的 {ruling} 在评审日志里找不到" for ruling in rulings if ruling not in review
        )

    assert not problems, "缺口必须留痕：\n" + "\n".join(problems)


def test_invariant_tests_are_all_referenced() -> None:
    """反方向检查：每条不变量用例都要挂到某个验收项上。

    新加了一条 `test_invariant_inv*` 却没挂上去，说明它守的约束
    **在设计里没有定位**——要么补挂到对应条款，要么这条用例是野的。
    两个方向都查，才叫追溯；只查一个方向就还是"看起来有人维护"。
    """
    referenced = {name for item in ALL_ITEMS for name in item.tests}
    orphans = sorted(
        name
        for name in _test_names()
        if name.startswith("test_invariant_inv") and name not in referenced
    )
    assert not orphans, f"这些不变量用例没有挂到任何验收项：{orphans}"
