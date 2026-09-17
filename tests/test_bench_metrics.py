"""评测判分器的离线门禁。

判分器是整个量化门槛的**唯一执行点**——它错了，所有分数都错了，
而且是"看起来很合理"的错：数字平滑、趋势正常、结论似是而非。
所以它必须能**不联网、不调模型**地被验证。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench.metrics import (
    Metrics,
    contains_answer,
    exact_match,
    mrr,
    ndcg_at_k,
    normalize,
    recall_at_k,
    token_f1,
    tokens,
)

# --------------------------------------------------------------------------- #
# 归一化与分词
# --------------------------------------------------------------------------- #


def test_normalize_folds_width_case_and_punctuation():
    assert normalize("住在上海。") == normalize("住在上海")
    assert normalize("ＡＢＣ") == "abc"
    assert normalize("  A  B  ") == "ab"


def test_exact_match_is_the_strictest_metric():
    assert exact_match("住在上海", "住在上海") == 1.0
    assert exact_match("用户住在上海", "住在上海") == 0.0, "多说了字就是 0——这正是 EM 的定义"


def test_chinese_is_tokenized_as_bigrams():
    """中文必须按二元组切——否则整句一个 token，F1 只有 0 或 1。

    失去区分度的指标无法指导优化：它只会告诉你"这条错了"，
    而说不出"错到哪一步"。二元组是零依赖与有区分度之间的折中。
    """
    assert "住在" in tokens("住在上海")
    assert tokens("住所") == ["住所"]


def test_english_is_tokenized_by_word():
    assert "shanghai" in tokens("I live in Shanghai")


# --------------------------------------------------------------------------- #
# 答案质量指标
# --------------------------------------------------------------------------- #


def test_token_f1_endpoints_and_partial_credit():
    assert token_f1("用户住在上海", "用户住在上海") == 1.0
    assert token_f1("用户住在上海", "今天天气不错") == 0.0
    partial = token_f1("用户住在上海浦东", "用户住在上海")
    assert 0 < partial < 1, "多说了一部分应当有部分分"


def test_chinese_f1_has_resolution():
    """一字之差**不应**得到 0——那会让所有近似答案挤在同一个分数上。"""
    assert 0 < token_f1("用户住在上海", "用户住在北京") < 1


def test_english_f1_does_not_inflate():
    """英文按**词**切——按字符切会虚高。

    `I live in Shanghai` vs `I live in Beijing`：4 个词里 3 个相同 → **F1 = 0.75**。
    若按字符切，两串共享 13/17 个字母 → 0.8 以上，**区分度被稀释**，
    而"换个城市"与"换句话"在字符层面的差别本来就不该是同一量级。
    """
    assert token_f1("I live in Shanghai", "I live in Beijing") == pytest.approx(0.75)
    assert token_f1("I live in Shanghai", "Completely different") == 0.0


def test_contains_answer_is_lenient_about_wrapping_text():
    assert contains_answer("根据记忆，用户住在上海哦", "住在上海") == 1.0
    assert contains_answer("用户住在北京", "住在上海") == 0.0
    assert contains_answer("随便什么", "") == 0.0, "空标准答案不该判为命中"


# --------------------------------------------------------------------------- #
# 检索指标
# --------------------------------------------------------------------------- #


def test_recall_at_k_counts_all_gold_items():
    retrieved = ["a", "b", "c"]
    gold = {"b", "c"}
    assert recall_at_k(retrieved, gold, 1) == 0.0, "前 1 条没命中"
    assert recall_at_k(retrieved, gold, 3) == 1.0
    assert recall_at_k(retrieved, gold, 2) == 0.5
    assert recall_at_k(retrieved, set(), 3) == 0.0


def test_mrr_rewards_ranking_not_just_presence():
    gold = {"b", "c"}
    assert mrr(["a", "b", "c"], gold) == pytest.approx(0.5)
    assert mrr(["b", "a", "c"], gold) == pytest.approx(1.0)
    assert mrr(["a", "x", "y"], gold) == 0.0


def test_ndcg_sees_the_whole_ranking_unlike_mrr():
    """多条标准答案时，MRR 只看第一个命中——会**严重高估**。

    NDCG 看整段排序质量，所以两者必须都报：
    只报 MRR，等于假装"后面的答案掉到第 9 位"没发生。
    """
    gold = {"b", "c"}
    good = ["b", "c", "x"]
    bad = ["b", "x", "x", "x", "c"]
    assert mrr(good, gold) == mrr(bad, gold) == pytest.approx(1.0)
    assert ndcg_at_k(good, gold, 5) > ndcg_at_k(bad, gold, 5), "NDCG 应当看得见'后面的掉了'"


# --------------------------------------------------------------------------- #
# 汇总语义
# --------------------------------------------------------------------------- #


def test_unanswered_questions_count_as_wrong():
    """**未命中的题必须拉低准确率**（分母是 `total` 而不是 `answered`）。

    用 `answered` 当分母会得到一个漂亮的假分数：召回越差、分母越小、准确率越高——
    指标的退化方向与实际质量**相反**。这类指标比没有指标更危险。
    """
    m = Metrics()
    m.add(pred="住在上海", gold="住在上海", retrieved=["x"], gold_ids={"x"})
    m.add(pred=None, gold="住在北京", retrieved=[], gold_ids={"y"})

    assert m.total == 2
    assert m.answered == 1
    assert m.accuracy == pytest.approx(0.5), "没答上来的题不能不计分"
    assert m.answer_rate == pytest.approx(0.5)


def test_answer_rate_diagnoses_where_the_failure_is():
    """答题率与准确率**一起看**才有诊断力。

    - 答题率高、准确率低 → 检索找对了，是生成没说对；
    - 答题率低 → 问题在检索或提取，跟生成无关。

    这两个结论指向完全不同的修法，而在"只有一个准确率"的报告里看不出来。
    """
    m = Metrics()
    for _ in range(4):
        m.add(pred="完全无关的回答", gold="住在上海", retrieved=["x"], gold_ids={"x"})
    assert m.answer_rate == 1.0
    assert m.accuracy == 0.0, "召回到了但答错 → 该修生成，不是修召回"


def test_to_dict_is_json_serialisable_and_keeps_n():
    m = Metrics()
    m.add(pred="住在上海", gold="住在上海", retrieved=["x"], gold_ids={"x"})
    payload = m.to_dict()
    assert payload["total"] == 1
    assert payload["accuracy"] == 1.0
    assert "by_category" in payload
