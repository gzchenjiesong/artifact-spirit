"""数据集适配器的**格式门禁**。

LoCoMo / LongMemEval 的真实文件不在仓库里（几十上百 MB，各有许可证），
所以这里用**按官方结构手工构造的最小样本**验证适配器。

这不是退而求其次。适配器写错时，如果在真实数据集上才暴露，
意味着**你先花了几十次 LLM 调用的钱，才拿到一个错误的分数**——
而错误的分数比没有分数更糟：它会引导你去修一个不存在的问题。
用最小样本先把格式钉住，代价几乎为零。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench.datasets import (
    DatasetError,
    load,
    load_locomo,
    load_longmemeval,
    parse_selfcheck,
)


def _write(tmp_path: Path, name: str, payload) -> Path:
    target = tmp_path / name
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# 自建集
# --------------------------------------------------------------------------- #


def test_selfcheck_parser_keeps_all_material():
    question = parse_selfcheck(
        {
            "id": "x-1",
            "category": "temporal",
            "question": "住在哪？",
            "answer": "上海",
            "gold_keywords": ["上海"],
            "turns": [
                {"session_id": "s1", "ts": "2026-01-01T00:00:00+08:00", "text": "我住在上海"},
            ],
        },
        index=1,
    )
    assert question.turns[0].text == "我住在上海"
    assert question.gold_keywords == ("上海",)


def test_selfcheck_parser_reports_position_and_missing_fields():
    """报错要带**第几条、缺什么**——加载器的错误信息必须能直接定位。"""
    with pytest.raises(DatasetError) as excinfo:
        parse_selfcheck({"id": "x", "question": "q"}, index=7)
    message = str(excinfo.value)
    assert "第 7 条" in message
    assert "answer" in message and "turns" in message


def test_selfcheck_parser_refuses_empty_turn_text():
    with pytest.raises(DatasetError):
        parse_selfcheck(
            {"id": "x", "question": "q", "answer": "a", "turns": [{"session_id": "s", "text": ""}]},
            index=2,
        )


# --------------------------------------------------------------------------- #
# LoCoMo
# --------------------------------------------------------------------------- #


def _locomo_sample(*, with_adversarial: bool = False) -> list[dict]:
    qa = [
        {"question": "What does the user drink?", "answer": "coffee", "category": 4},
        {"question": "Where did they live in March?", "answer": "Shanghai", "category": 2},
    ]
    if with_adversarial:
        qa.append(
            {
                "question": "Is the user a doctor?",
                "answer": "Yes",
                "adversarial_answer": "No",
                "category": 5,
            }
        )
    return [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1": [
                    {"speaker": "A", "text": "I drink coffee", "dia_id": "D1:1"},
                    {"speaker": "B", "text": "Nice", "dia_id": "D1:2"},
                ],
                "session_1_date_time": "1:00 pm on 1 January, 2026",
                "session_2": [{"speaker": "A", "text": "I moved to Shanghai", "dia_id": "D2:1"}],
                "session_2_date_time": "1:00 pm on 1 March, 2026",
            },
            "qa": qa,
        }
    ]


def test_locomo_adapter_reads_every_session_as_one_turn(tmp_path):
    """一轮 = 一个 session：拼段会**低估**分段能力，但反过来会让时间线碎成几十段。

    取保守的那个，并在文档里写明。
    """
    questions = load_locomo(_write(tmp_path, "locomo10.json", _locomo_sample()))

    assert len(questions) == 2, "两题都该读出来"
    first = questions[0]
    assert len(first.turns) == 2, f"两个 session 应当变成两轮，实得 {len(first.turns)}"
    assert "coffee" in first.turns[0].text
    assert first.turns[1].ts == "2026-03-01T13:00:00+00:00", (
        "自由文本时间必须被解析成 ISO8601——原样传下去会**按字典序比较**时间"
    )
    assert first.category == "single_hop"
    assert questions[1].category == "temporal", "category 2 → temporal"


def test_locomo_adapter_skips_adversarial_questions(tmp_path):
    """对抗题的"答案"是**陷阱**：按普通题判分会让分数虚高。

    它不是"难题"，而是"答对说明你听信了错误信息"——判分口径与普通题相反。
    """
    questions = load_locomo(_write(tmp_path, "l.json", _locomo_sample(with_adversarial=True)))
    ids = [q.id for q in questions]
    assert len(questions) == 2, f"对抗题必须被排除，实得 {len(questions)} 题：{ids}"


def test_locomo_adapter_refuses_a_file_without_sessions(tmp_path):
    bad = [{"sample_id": "c", "conversation": {}, "qa": [{"question": "q", "answer": "a"}]}]
    with pytest.raises(DatasetError) as excinfo:
        load_locomo(_write(tmp_path, "bad.json", bad))
    assert "没解析出任何 session" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# LongMemEval
# --------------------------------------------------------------------------- #


def test_longmemeval_adapter_reads_haystack_and_dates(tmp_path):
    payload = [
        {
            "question_id": "q1",
            "question_type": "temporal-reasoning",
            "question": "When did they move?",
            "answer": "March",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I moved in March"},
                    {"role": "assistant", "content": "Noted"},
                ]
            ],
            "haystack_dates": ["2026-03-01"],
        }
    ]
    questions = load_longmemeval(_write(tmp_path, "lme.json", payload))

    assert len(questions) == 1
    assert questions[0].category == "temporal-reasoning"
    assert questions[0].turns[0].ts == "2026-03-01T00:00:00+00:00"
    assert "March" in questions[0].turns[0].text


def test_longmemeval_adapter_refuses_questions_without_haystack(tmp_path):
    payload = [{"question_id": "q1", "question": "q", "answer": "a", "haystack_sessions": []}]
    with pytest.raises(DatasetError) as excinfo:
        load_longmemeval(_write(tmp_path, "empty.json", payload))
    assert "没解析出任何会话" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 分派
# --------------------------------------------------------------------------- #


def test_unknown_dataset_name_lists_the_supported_ones(tmp_path):
    """未知名字**报错并列出支持的**——不回退到默认。

    静默回退到自建集，会让"我明明跑的是 LoCoMo"这个前提悄悄失效。
    """
    with pytest.raises(DatasetError) as excinfo:
        load("locomo_typo", tmp_path / "whatever.json")
    assert "selfcheck" in str(excinfo.value)
    assert "locomo" in str(excinfo.value)


def test_empty_jsonl_is_refused(tmp_path):
    """空评测集会**让所有分数失去意义**——它必须报错，而不是"0 题、0 分、通过"。"""
    target = tmp_path / "empty.jsonl"
    target.write_text("\n\n", encoding="utf-8")
    with pytest.raises(DatasetError) as excinfo:
        load("selfcheck", target)
    assert "一题都没读到" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 答案关键词（判"哪条记忆算相关"的判据）
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 时间戳解析（时态能不能工作的先决条件）
# --------------------------------------------------------------------------- #


def test_locomo_evidence_maps_to_the_session_id_we_ingest():
    """官方 `evidence`（`D1:3`）必须映射成**我们灌入时用的 session_id**。

    映射错了**不会报错**——只会让"相关记忆"恒为空、R@k 恒为 0，
    看起来像"检索全废"。所以这条映射规则必须被钉住。
    """
    from bench.datasets import _evidence_sessions

    assert _evidence_sessions(["D1:3"], sample_index=0) == ("locomo0-session_1",)
    assert _evidence_sessions(["D1:3;D4:7"], sample_index=5) == (
        "locomo5-session_1",
        "locomo5-session_4",
    ), "一段 evidence 里可以引用多个会话"
    assert _evidence_sessions([], sample_index=0) == ()
    assert _evidence_sessions(None, sample_index=0) == ()


def test_locomo_loader_reads_the_official_evidence(tmp_path):
    """LoCoMo 的题要带上官方 `evidence`——**它一直在数据里，只是没人读**。

    不读它的代价实测过：gold 集均值 29 条、最大等于整个库，
    `top_k=10` 时 R@k 的理论上限只有 0.24，分数低得像是检索坏了。
    """
    sample = _locomo_sample()
    sample[0]["qa"][0]["evidence"] = ["D1:2;D2:1"]
    questions = load_locomo(_write(tmp_path, "l.json", sample))
    assert questions[0].evidence_sessions == (
        "locomo0-session_1",
        "locomo0-session_2",
    ), f"实得 {questions[0].evidence_sessions}"


def test_parse_ts_handles_both_datasets_and_refuses_unknown():
    """两个数据集的格式都要认；**认不出来返回空串，不猜**。

    不猜的理由是不对称的：留空会让时态**退化**（有指标能看出来），
    而猜一个会让它**看起来正常但顺序是错的**——后者没有任何信号。
    """
    from bench.datasets import parse_ts

    assert parse_ts("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00+00:00"
    assert parse_ts("2023/04/10 (Mon) 17:50") == "2023-04-10T17:50:00+00:00"
    assert parse_ts("2026-03-01") == "2026-03-01T00:00:00+00:00"
    assert parse_ts("昨天下午") == ""
    assert parse_ts("") == ""
    assert parse_ts(None) == ""


def test_parsed_timestamps_sort_in_chronological_order():
    """解析之后**字典序必须等于时间序**——这是时态比较的前提。

    钉的是那个真实缺陷：`10:37 am on 27 June, 2023` 与 `1:56 pm on 8 May, 2023`
    按**原文字典序是反的**（第二个字符 `'0' < ':'`），而系统内部正是按字符串比较时态字段。
    后果不是"排序略差"，是 `valid_to <= valid_from` 的守卫在第 2 条记忆上就抛异常。
    """
    from bench.datasets import parse_ts

    earlier = parse_ts("1:56 pm on 8 May, 2023")
    later = parse_ts("10:37 am on 27 June, 2023")
    assert earlier < later, "解析之后字典序必须与时间顺序一致"


def test_answer_keywords_strip_quotes_and_brackets():
    """关键词要**去掉两端的引号与括号**。

    这是被真实数据逼出来的一条：LongMemEval 的答案里有
    `'Data Analysis using Python' webinar` 这种写法，按标点切出来的片段是 `'Data`
    —— 而记忆里写的是 `Data Analysis`。于是**明明召回了却判为没命中**，
    指标因为一段标点**假性偏低**。

    修正前后差的是实打实的分数，所以它值得一条用例。
    """
    from bench.datasets import _keywords_from_answer

    keywords = _keywords_from_answer("'Data Analysis using Python' webinar")
    assert "Data" in keywords, f"引号被带进关键词了：{keywords}"
    assert "webinar" in keywords
    assert not any(k.startswith("'") or k.endswith("'") for k in keywords)


def test_answer_keywords_drop_single_chars():
    """单字关键词在任何记忆里都能命中，必须丢掉——

    留着它们会让"相关记忆"的判定**虚高**到没有意义：
    任何一条记忆都会被算成相关的，于是 R@k 永远是 1。
    """
    from bench.datasets import _keywords_from_answer

    assert _keywords_from_answer("是，好") == (), "单字必须丢掉"
    assert _keywords_from_answer("是，好，的") == ()
    assert _keywords_from_answer("是的，好") == ("是的",), "两个字以上的保留"
