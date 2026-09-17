"""自建评测集的**机制回归**（G6：自证机制有效）。

与 `test_bench_metrics.py` 的分工：

- 那个验**判分器**（纯函数，离线）；
- 这个验**系统**——跑一遍自建集，断言"提取 → 入库 → 召回"这条链没有坏掉。

**零模型**（extract 模式 + 不配 LLM）：规则提取器与既定的降级路径也要能工作。
这条链红灯时，结论是**机制坏了**，而不是"模型不够强"——这两种结论的修法完全不同，
而只有一个分数的时候分不出来。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench.datasets import load_jsonl
from bench.metrics import Metrics
from bench.run import evaluate_group

CASES = ROOT / "scripts" / "bench" / "cases" / "selfcheck.jsonl"


# --------------------------------------------------------------------------- #
# R@k 的相关性口径（`_gold_ids` 的判据）
# --------------------------------------------------------------------------- #


def test_readable_ts_renders_month_name_for_keyword_matching():
    """`valid_from` 要能被**月份名**匹配到——时间类题的答案就是「7 May 2023」。

    渲染不出月份名，时间类题的相关记忆就永远找不到，而它们的日期
    **只存在于 `valid_from`** 里（LLM 改写 `content` 时把日期丢掉了）。
    """
    from bench.run import _readable_ts

    assert _readable_ts("2023-05-08T13:56:00+00:00") == "8 May 2023"
    assert _readable_ts("2023-06-09T19:55:00+00:00") == "9 June 2023"
    assert _readable_ts("") == ""
    assert _readable_ts(None) == ""
    assert _readable_ts("不是时间") == ""


def test_gold_matching_needs_two_keywords_when_available():
    """命中**一个**关键词不算相关——否则年份会命中全库，recall 虚高到接近 1。

    这不是"更严格一点"，它决定指标有没有意义：时间类题的关键词形如
    `['May', '2023']`，而 `2023` 几乎出现在每一条当年记忆里。
    用 `any` 的话，"相关记忆"就等于整个库。
    """
    from bench.run import _matches_gold

    assert not _matches_gold("2023 年 caroline 去了支持小组", ["may", "2023"]), (
        "只含年份不算相关——那会命中全库"
    )
    assert _matches_gold("8 may 2023 caroline 去了支持小组", ["may", "2023"])
    assert _matches_gold("这是 2022 年的事", ["2022"]), "只有一个关键词时，命中它就算"
    assert not _matches_gold("这是 2023 年的事", ["2022"])
    assert _matches_gold(
        "8 may 2023 那个周日", ["the", "sunday", "before", "25", "may", "2023"]
    ), "虚词不该坏事：`May` + `2023` 已经足够"
    assert not _matches_gold("随便什么", []), "没有关键词就不构成任何「相关」"


def test_selfcheck_dataset_loads_completely():
    """评测集本身要能被**完整**读出来。

    加载器默默少读几条，分数会偏低而无从察觉——然后有人会去"优化系统"，
    去修一个根本不存在的问题。所以题数与类别分布都要钉住。
    """
    questions = load_jsonl(CASES)

    assert len(questions) == 10, f"自建集应当有 10 题，实得 {len(questions)}"
    categories = {q.category for q in questions}
    assert categories >= {"preference", "temporal", "multihop", "entity"}, (
        f"类别覆盖不足：{sorted(categories)}"
    )
    for question in questions:
        assert question.turns, f"{question.id} 没有任何素材——它无从被验证"
        assert question.answer, f"{question.id} 没有标准答案"
        assert question.gold_keywords, f"{question.id} 没有相关关键词（无法判检索）"


def test_selfcheck_mechanisms_do_not_regress(tmp_path):
    """跑一遍自建集，断言**召回**没有退化。

    只断言两个指标：

    - `answer_rate`：召回走通的比例（0 分说明"根本没找到"，与生成无关）；
    - `recall@k`：相关内容找到了几成。

    **不断言 accuracy / F1**——extract 模式下它们由"答案文本比标准答案长多少"决定，
    与机制好坏无关，钉死它们只会让用例红在无意义的地方。

    阈值取 1.0 是刻意的：这 10 题都是"明确要求记住"的直白事实，
    召回全中是最低要求；把线降下来等于承认"漏一条也行"。
    """
    questions = load_jsonl(CASES)
    metrics = Metrics()

    for question in questions:
        # 自建集**每题一组**（素材是各自写的），所以这里逐个跑单题组。
        # `embedding="off"`：门槛用例必须**离线可跑**——它守的是机制没退化，
        # 不是"网关今天通不通"。要它依赖网络，红的时候第一件事是去查网关。
        row = evaluate_group(
            question.id,
            [question],
            root=tmp_path,
            top_k=10,
            mode="extract",
            ingest="rule",
            embedding="off",
        )[0]
        assert not row.get("error"), f"{question.id} 评测过程异常：{row.get('error')}"
        metrics.add(
            pred=row.get("prediction"),
            gold=question.answer,
            retrieved=row["retrieved_ids"],
            gold_ids=set(row["gold_ids"]),
            k=10,
        )

    assert metrics.answer_rate == 1.0, (
        f"有题没召回到任何东西（{metrics.answered}/{metrics.total}）——"
        "问题在检索或提取，与生成无关"
    )
    assert metrics.recall == 1.0, (
        f"R@10 = {metrics.recall:.3f}：库里明明有相关内容却一条都没召回"
    )
    assert metrics.mrr >= 0.9, f"MRR = {metrics.mrr:.3f}：相关内容排得太靠后"
