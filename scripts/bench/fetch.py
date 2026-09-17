"""公开数据集的获取（D-13 的前置步骤）。

## 为什么"下载"也要写成脚本

因为"我是从网上拉的"不是一个**可复现**的说明。写进脚本的必须有三样：
**从哪拉、拉到哪、怎么确认拉对了**。

第三样最要紧。数据集文件被截断、或者上游换了版本，跑出来的分数会**偏低或偏高**，
而你不会知道——你只会以为是系统的问题，然后去修一个不存在的问题。
所以本脚本在下载之后**立刻用真实加载器解析一遍**，把"读出了多少题"打出来：
这一步过了，才谈得上信后面的分数。

## 用法

    python scripts/bench/fetch.py                 # 列出可用数据集
    python scripts/bench/fetch.py locomo          # 下载 + 自检
    python scripts/bench/fetch.py locomo --mirror # 只打印 URL（手动下载用）

数据落在 `scripts/bench/data/`（已在 `.gitignore` 里——几十 MB 的文件不该进仓库）。
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.datasets import DatasetError, load

DATA_DIR = Path(__file__).resolve().parent / "data"

# 多个候选 URL：**镜像的存在本身就是"上游会变"的证据**。
# 只写一个地址的脚本，会在上游改名的那天变成一个说不清哪里错的报错。
SOURCES: dict[str, tuple[str, list[str]]] = {
    "locomo": (
        "locomo10.json",
        [
            "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
            "https://raw.githubusercontent.com/snap-research/locomo/master/data/locomo10.json",
        ],
    ),
    "longmemeval": (
        "longmemeval_oracle.json",
        [
            # **cleaned 版是当前推荐的**。原 `longmemeval` 仓库已被作者标注弃用，
            # 理由是里面有"会干扰答案正确性的噪声历史会话"——
            # 用原版跑出来的分会**偏低**，而偏低的原因不是系统不行，是数据里塞了干扰项。
            (
                "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
                "longmemeval_oracle.json"
            ),
            # 原仓库的对应文件（结构相同，作兜底）。
            # **注意它没有 `.json` 后缀**——HF 上那个仓库的文件名就是不带的。
            (
                "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/"
                "longmemeval_oracle"
            ),
        ],
    ),
}


def _download(urls: list[str], target: Path) -> str:
    """依次尝试候选地址。**全部失败时把每个错误都报出来**——
    只报最后一个会让"为什么那三个都不行"重新变成一次排查。"""
    errors: list[str] = []
    for url in urls:
        try:
            print(f"  尝试 {url}")
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                payload = response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            errors.append(f"{url} → {type(exc).__name__}: {exc}")
            continue
        if not payload:
            errors.append(f"{url} → 返回了空内容")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        print(f"  已下载 {len(payload) / 1024:.0f} KB → {target}")
        return url
    raise SystemExit("全部候选地址都失败：\n  " + "\n  ".join(errors))


def _verify(name: str, path: Path) -> bool:
    """**下载后立刻用真实加载器解析一遍。**

    这一步是整个脚本存在的理由：如果只检查"文件存在且非空"，
    一个被截断到一半的 JSON 会**通过检查**，然后在跑分时才炸——
    或者更糟：`json.loads` 成功了但题数少了一半，而分数只是"看起来偏低"。
    """
    try:
        questions = load(name, path)
    except DatasetError as exc:
        print(f"  ✗ 自检失败：{exc}")
        return False
    except Exception as exc:
        print(f"  ✗ 自检失败（{type(exc).__name__}）：{exc}")
        return False

    categories: dict[str, int] = {}
    for question in questions:
        categories[question.category] = categories.get(question.category, 0) + 1
    print(f"  ✓ 自检通过：读出 {len(questions)} 题")
    print(f"    类别分布：{categories}")

    # **时间戳解析率**：它是"双时态能不能工作"的先决条件。
    # 两个数据集给的都是自由文本（`1:56 pm on 8 May, 2023`）。解析不出来就只能留空，
    # 而那是**静默的性质退化**：时态类的题会从"考时态"悄悄变成"考运气"，
    # 分数照样打得出来，只是它不再衡量它声称衡量的东西。
    total_turns = sum(len(q.turns) for q in questions)
    undated = sum(1 for q in questions for turn in q.turns if not turn.ts)
    if undated:
        print(f"  ⚠ 时间戳：{undated}/{total_turns} 段没解析出来——时态类会因此失真")
    else:
        print(f"  ✓ 时间戳：{total_turns} 段全部解析成功")
    print(f"    首题：{questions[0].id} — {questions[0].question[:60]}")
    print("\n  接下来（**先 inspect，再跑分**）：")
    print(f"    python scripts/bench/run.py --inspect --dataset {name} --path {path}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="获取公开评测数据集（D-13 前置）")
    parser.add_argument("dataset", nargs="?", choices=sorted(SOURCES), help="数据集名")
    parser.add_argument("--mirror", action="store_true", help="只打印候选 URL，不下载")
    args = parser.parse_args(argv)

    if not args.dataset:
        print("可用数据集：")
        for name, (filename, urls) in SOURCES.items():
            print(f"  {name:<14} → {filename}")
            for url in urls:
                print(f"      {url}")
        print("\n用法：python scripts/bench/fetch.py locomo")
        return 0

    filename, urls = SOURCES[args.dataset]
    if args.mirror:
        for url in urls:
            print(url)
        return 0

    print(f"获取 {args.dataset}：")
    target = DATA_DIR / filename
    if target.exists():
        print(f"  已存在，跳过下载：{target}")
    else:
        _download(urls, target)

    return 0 if _verify(args.dataset, target) else 1


if __name__ == "__main__":
    raise SystemExit(main())
