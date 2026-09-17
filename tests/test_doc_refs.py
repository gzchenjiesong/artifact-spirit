"""校验"注释里引用的评审编号"必须真实存在。

## 为什么需要这个文件

这与 `test_acceptance_matrix.py::test_referenced_tests_exist` 是**同一种病的两个层级**：

| | 矩阵 → 用例名 | 注释 → 评审编号 |
|---|---|---|
| 引用方 | `acceptance_matrix.py` | `src/**`、`tests/**` 的 docstring / 注释 |
| 被引用方 | 测试函数名 | `07-设计评审.md` 的轮次与 P0/P1/P2 编号 |
| 有校验吗 | **有** | **此前零校验** |

2026-09-16 实测：`DES-REV-008` 在 `src/` 与 `tests/` 里被引用 **23 处**，而评审日志里
**一次都没出现过**——第八部分当时还没写。更隐蔽的是编号本身也错了（AL5 用了
`P1-13 ~ P1-21`，与 AL2 / AL3 撞号），而**注释不参与执行，所以不会让任何测试变红**。

**引用一个不存在的裁判，等于没引用。**（AL4 P2-26 的原话，这次翻到了上一层。）
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW_LOG = "docs/design/07-设计评审.md"

# 扫描范围：源码与测试（文档由评审日志自身保证，不在这里扫）
SCAN_DIRS = ("src", "tests")

_ROUND = re.compile(r"DES-REV-(\d{3})")
_ITEM = re.compile(r"P([012])-(\d+)")
_ITEM_GROUP = re.compile(r"P[012]-\d+(?:\s*/\s*P[012]-\d+)*")
_SECTION = re.compile(r"^# 第.+?部分.+?DES-REV-(\d{3})", re.MULTILINE)


def _review_text() -> str:
    path = ROOT / REVIEW_LOG
    assert path.exists(), f"评审日志不在原位：{REVIEW_LOG}"
    return path.read_text(encoding="utf-8")


def _round_sections(review: str) -> dict[str, str]:
    """按 `# 第N部分 …（DES-REV-00x …）` 把日志切成"每个轮次自己的正文"。

    **必须按轮次限定范围**：`P1-6` 在 DES-REV-001、DES-REV-003、DES-REV-004 里都存在，
    不限定范围的话任何编号都能"碰巧通过"——那等于没校验。
    """
    marks = list(_SECTION.finditer(review))
    sections: dict[str, str] = {}
    for index, match in enumerate(marks):
        start = match.start()
        end = marks[index + 1].start() if index + 1 < len(marks) else len(review)
        sections[match.group(1)] = review[start:end]
    return sections


def _source_files() -> Iterator[Path]:
    for directory in SCAN_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def test_review_log_has_round_sections() -> None:
    """自检：能被切分，否则下面两条断言会因为"扫不到东西"而假绿。"""
    sections = _round_sections(_review_text())
    assert sections, "评审日志里扫不到 `# 第N部分 … DES-REV-xxx` 标题——文档结构变了？"
    assert len(sections) >= 3, f"只扫到 {len(sections)} 个轮次，切分规则可能失效了"


def test_referenced_review_rounds_exist() -> None:
    """注释里提到的 `DES-REV-00x` 必须是日志里真实存在的轮次。"""
    known = {f"DES-REV-{key}" for key in _round_sections(_review_text())}

    dangling = sorted(
        {
            f"{_relative(path)} → DES-REV-{match}"
            for path in _source_files()
            for match in _ROUND.findall(path.read_text(encoding="utf-8"))
            if f"DES-REV-{match}" not in known
        }
    )
    assert not dangling, "注释引用了不存在的评审轮次：\n" + "\n".join(dangling)


def test_referenced_review_items_exist() -> None:
    """`DES-REV-00x P?-nn` 里的编号必须落在**该轮次自己的正文**里。

    这是真正咬人的那条：轮次名对、编号错（或撞到别层的同号条目）都逃不过。
    """
    sections = _round_sections(_review_text())
    problems: list[str] = []

    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"DES-REV-(\d{3})\s*(" + _ITEM_GROUP.pattern + r")", text):
            round_id = match.group(1)
            section = sections.get(round_id)
            if section is None:
                continue  # 轮次本身不存在，由上面那条报红
            for item in _ITEM.finditer(match.group(2)):
                key = f"P{item.group(1)}-{item.group(2)}"
                if key not in section:
                    problems.append(f"{_relative(path)} → DES-REV-{round_id} {key}")

    assert not problems, "注释引用了该轮次里不存在的编号：\n" + "\n".join(sorted(set(problems)))
