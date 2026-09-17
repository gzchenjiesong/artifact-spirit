"""评测集的统一加载（D-13 / G6）。

## 三种来源，一套 schema

| 来源 | 用途 |
|---|---|
| **自建集**（`cases/*.jsonl`） | G6 的核心：**证明机制有效**，可离线回归 |
| **LoCoMo** | 对外标尺（门槛 ≥ 70） |
| **LongMemEval** | 对外标尺（门槛 ≥ 75） |

统一 schema 的理由很实际：**分数可比的第一个前提是"喂进去的东西形状一样"**。
如果自建集走一条代码路径、公开集走另一条，那么"自建集 90 分、LoCoMo 62 分"
就解释不了——差在系统上，还是差在加载器上？

## 一条硬纪律：**读不懂就报错**

公开数据集的字段名会变（不同版本、不同 fork）。加载器必须做两件事：

1. **宽容地认字段**（把常见别名都试一遍）；
2. **认不出来就抛错**，绝不静默跳过。

第 2 条是这个文件里最重要的设计。加载器默默少读了 30% 的题，
分数会**偏低**而无从察觉——然后有人会去"优化系统"，去修一个根本不存在的问题。
`run.py --inspect` 就是为这个准备的：先看加载器读出了什么，再信分数。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "DATASET_LOADERS",
    "Question",
    "Turn",
    "load",
    "load_jsonl",
    "load_locomo",
    "load_longmemeval",
    "parse_selfcheck",
    "parse_ts",
]


@dataclass(frozen=True, slots=True)
class Turn:
    """一轮要灌进器灵的对话。"""

    session_id: str
    ts: str
    text: str


@dataclass(frozen=True, slots=True)
class Question:
    """一道题及其所需的全部素材。"""

    id: str
    category: str
    question: str
    answer: str
    turns: tuple[Turn, ...]
    gold_keywords: tuple[str, ...] = ()
    """标准答案对应的记忆**关键词**（不是 id）。

    为什么不用 id：id 是运行期生成的，评测脚本无法预知。用关键词按**内容**匹配，
    就不依赖任何 id 约定——评测集因此能在不同的库、不同的 id 方案上复用。
    """
    evidence_sessions: tuple[str, ...] = ()
    """官方标注的**相关会话**（由 `evidence` 映射而来），只在 LoCoMo 上有。

    有了它就不必用"从标准答案里抠关键词"那种近似——实测那种近似的 gold 集
    均值 29 条、最大等于**整个库**（122 条），于是 `top_k=10` 时 R@k 的
    **理论上限只有 0.24**：分数低不是检索差，是尺子量不到。
    """
    group: str = ""
    """**素材组**：共享同一批 `turns` 的题归为一组，评测时**只灌一次库**。

    这不是性能优化，是**正确性**问题。LoCoMo 的 1540 题只分布在 10 个 sample 上——
    每题重建一次库意味着同一批数据被灌 1540 遍：LLM 模式下是 1540 倍的钱与时间
    （实测 553 秒/题，全量要跑十天），而且每题都在一个**新建的库**上作答，
    引入了一个与题目无关的随机性来源。

    按组灌入之后，"同一场对话被问了 154 次"与"被问了 1 次"是同一份素材上的两种查询——
    这也正是原论文的评测方式。
    """
    source: str = "selfcheck"


class DatasetError(ValueError):
    """数据集格式不符。**带上是第几条、缺什么字段**——加载器报错要能直接定位。"""


# --------------------------------------------------------------------------- #
# 自建集（G6 的核心）
# --------------------------------------------------------------------------- #


def parse_selfcheck(raw: dict, *, index: int, source: str = "selfcheck") -> Question:
    """自建集一行 = 一道题。字段少了就报错，**不猜**。"""
    missing = [k for k in ("id", "question", "answer", "turns") if not raw.get(k)]
    if missing:
        raise DatasetError(f"{source} 第 {index} 条缺字段：{missing}；实得 {sorted(raw)}")

    turns: list[Turn] = []
    for turn_index, turn in enumerate(raw["turns"]):
        if not isinstance(turn, dict) or not turn.get("text"):
            raise DatasetError(f"{source} 第 {index} 条的 turns[{turn_index}] 非法：{turn!r}")
        turns.append(
            Turn(
                session_id=str(turn.get("session_id") or f"s{index}"),
                ts=str(turn.get("ts") or ""),
                text=str(turn["text"]),
            )
        )

    keywords = raw.get("gold_keywords") or ()
    if isinstance(keywords, str):
        keywords = [keywords]

    return Question(
        id=str(raw["id"]),
        category=str(raw.get("category") or "uncategorised"),
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        turns=tuple(turns),
        gold_keywords=tuple(str(k) for k in keywords),
        # 自建集默认**每题一组**：它们的素材是各自写的，共享反而会互相污染。
        group=str(raw.get("group") or raw["id"]),
        source=source,
    )


def load_jsonl(path: str | Path, *, source: str = "selfcheck") -> list[Question]:
    """读 `.jsonl`（一行一题）。空行忽略；**坏行报错并说明行号**。"""
    target = Path(path)
    out: list[Question] = []
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{target.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc
        out.append(parse_selfcheck(raw, index=lineno, source=source))
    if not out:
        raise DatasetError(f"{target} 里一题都没读到——空评测集会让所有分数失去意义")
    return out


# --------------------------------------------------------------------------- #
# LoCoMo（对外标尺）
# --------------------------------------------------------------------------- #

_LOCOMO_CATEGORY = {
    1: "multihop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

_TS_FORMATS = (
    # LoCoMo：`1:56 pm on 8 May, 2023`
    "%I:%M %p on %d %B, %Y",
    # LongMemEval：`2023/04/10 (Mon) 17:50`
    "%Y/%m/%d (%a) %H:%M",
    # 只有日期（有的样本给到天）
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
)


def parse_ts(raw: object) -> str:
    """把数据集里的时间戳解析成 ISO8601。**解析不出来就返回空串，不猜。**

    这一步不是"格式美化"，它决定**双时态能不能工作**。两个数据集给的都是自由文本：

    | 数据集 | 原样 |
    |---|---|
    | LoCoMo | `1:56 pm on 8 May, 2023` |
    | LongMemEval | `2023/04/10 (Mon) 17:50` |

    原样传下去会**按字典序比较**时间：`10:37 am on 27 June, 2023` 小于
    `1:56 pm on 8 May, 2023`（因为第二个字符 `'0' < ':'`）。
    于是 `valid_to <= valid_from` 的守卫在第 2 条记忆上就炸了——
    而它看起来像"时态逻辑抛异常"，很容易让人去修错地方。

    **时区**：两者都不给时区。统一按 UTC 存档，理由是**不假装知道**——
    猜一个 `+08:00` 会让"凌晨的对话"落到前一天，而按 UTC 只是绝对值偏移，
    **相对顺序完全不变**；时态比较要的正是相对顺序。
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    for fmt in _TS_FORMATS:
        try:
            stamp = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return stamp.replace(tzinfo=UTC).isoformat()
    return ""


def load_locomo(path: str | Path) -> list[Question]:
    """LoCoMo（`locomo10.json`）。

    ## 两处必须点明的取舍

    1. **一轮 = 一个 session**：LoCoMo 的一个 session 里有几十条发言。
       器灵的 `ingest_turn` 以"一轮对话"为单位，所以这里把一个 session 的发言
       拼成一段文本灌进去。这会**低估**系统的分段能力——
       但反过来（一条发言一轮）会让 `session_id` 失去"会话"的含义，
       时间线也会碎成几十段。**取保守的那个**。
    2. **`evidence` 只用于定位，不用于判分**：它给的是 `D1:3` 这样的发言编号，
       而拼段之后编号就对不上了。相关记忆的判定改用**答案关键词**（见下）。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):  # 有的 fork 包了一层
        payload = payload.get("data") or payload.get("samples") or [payload]
    if not isinstance(payload, list):
        raise DatasetError("LoCoMo 顶层应当是数组（或含 data/samples 的对象）")

    out: list[Question] = []
    for sample_index, sample in enumerate(payload):
        conversation = sample.get("conversation") or {}
        turns = _locomo_turns(conversation, session_index=sample_index)
        if not turns:
            raise DatasetError(f"LoCoMo 第 {sample_index} 个样本没解析出任何 session")

        for qa_index, qa in enumerate(sample.get("qa") or []):
            answer = qa.get("answer")
            if not answer or qa.get("adversarial_answer"):
                # 对抗题的"答案"是陷阱，不能按普通题判分——**跳过要留痕**
                continue
            out.append(
                Question(
                    id=f"{sample.get('sample_id') or sample_index}#{qa_index}",
                    category=_LOCOMO_CATEGORY.get(qa.get("category"), "unknown"),
                    question=str(qa.get("question") or ""),
                    answer=str(answer),
                    turns=turns,
                    gold_keywords=_keywords_from_answer(str(answer)),
                    evidence_sessions=_evidence_sessions(
                        qa.get("evidence"), sample_index=sample_index
                    ),
                    # 同一个 sample 的所有题共享这 19 个 session → 同组灌一次。
                    group=str(sample.get("sample_id") or sample_index),
                    source="locomo",
                )
            )
    if not out:
        raise DatasetError(f"{path} 里一题都没读到")
    return out


def _locomo_turns(conversation: dict, *, session_index: int) -> tuple[Turn, ...]:
    """把 `session_N` 的发言拼成"一轮 = 一个会话"。"""
    turns: list[Turn] = []
    index = 1
    while True:
        key = f"session_{index}"
        if key not in conversation:
            break
        lines = conversation.get(key) or []
        ts = parse_ts(conversation.get(f"{key}_date_time"))
        text = "\n".join(
            f"{line.get('speaker', '?')}: {line.get('text', '')}".strip()
            for line in lines
            if isinstance(line, dict)
        )
        if text:
            turns.append(
                Turn(session_id=f"locomo{session_index}-{key}", ts=ts, text=text)
            )
        index += 1
    return tuple(turns)


# --------------------------------------------------------------------------- #
# LongMemEval（对外标尺）
# --------------------------------------------------------------------------- #


def load_longmemeval(path: str | Path) -> list[Question]:
    """LongMemEval。

    它的 haystack 是"一堆历史会话"，而**每题可能带自己的 haystack**
    （`haystack_sessions`）。抽取式与小规模版本都按这个结构读。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("questions") or [payload]
    if not isinstance(payload, list):
        raise DatasetError("LongMemEval 顶层应当是数组（或含 data/questions 的对象）")

    out: list[Question] = []
    for index, item in enumerate(payload):
        answer = item.get("answer")
        if not answer:
            continue
        sessions = item.get("haystack_sessions") or item.get("sessions") or []
        dates = item.get("haystack_dates") or item.get("dates") or []
        turns = _longmem_turns(sessions, dates, index=index)
        if not turns:
            raise DatasetError(f"LongMemEval 第 {index} 题没解析出任何会话")
        out.append(
            Question(
                id=str(item.get("question_id") or item.get("id") or index),
                category=str(item.get("question_type") or "unknown"),
                question=str(item.get("question") or ""),
                answer=str(answer),
                turns=turns,
                gold_keywords=_keywords_from_answer(str(answer)),
                # LongMemEval **每题带自己的 haystack**，所以每题一组——
                # 分组的判据是"素材是否相同"，不是"名字像不像"。
                group=str(item.get("question_id") or item.get("id") or index),
                source="longmemeval",
            )
        )
    if not out:
        raise DatasetError(f"{path} 里一题都没读到")
    return out


def _longmem_turns(sessions: Any, dates: Any, *, index: int) -> tuple[Turn, ...]:
    turns: list[Turn] = []
    for session_index, raw_session in enumerate(sessions or []):
        session = (
            raw_session.get("messages") or raw_session.get("turns") or []
            if isinstance(raw_session, dict)
            else raw_session
        )
        text = "\n".join(
            f"{(m or {}).get('role', '?')}: {(m or {}).get('content', '')}".strip()
            for m in (session or [])
            if isinstance(m, dict)
        )
        if not text:
            continue
        ts = parse_ts(dates[session_index]) if session_index < len(dates) else ""
        turns.append(Turn(session_id=f"lme{index}-s{session_index}", ts=ts, text=text))
    return tuple(turns)


_EVIDENCE_RE = re.compile(r"D(\d+)")


def _evidence_sessions(raw: Any, *, sample_index: int) -> tuple[str, ...]:
    """把官方 `evidence`（形如 `D1:3`）映射成**灌入时用的 session_id**。

    `D1:3` = 第 1 段会话的第 3 条发言，而本适配器把**一整个 session 灌成一轮**，
    所以粒度对得上：`D1` → `locomo{sample_index}-session_1`。

    **这才是 qrel 该有的样子**：官方标了"答案出自哪几句"，
    就不必再从答案文本里猜——那种猜法不但宽，还会把语料性质
    （"LoCoMo 的答案里几乎总有年份"）混进指标里变成系统性偏差。
    """
    found: list[str] = []
    for item in raw or []:
        for number in _EVIDENCE_RE.findall(str(item)):
            session = f"locomo{sample_index}-session_{number}"
            if session not in found:
                found.append(session)
    return tuple(found)


def _keywords_from_answer(answer: str) -> tuple[str, ...]:
    """从标准答案里取关键词，用于判定"哪条记忆算相关"。

    两条纪律，各修一种**会让指标失真**的形态：

    1. **只取长度 ≥ 2 的片段**。单字关键词（"是"、"的"）在任何记忆里都能命中，
       算进"相关记忆"会让检索指标**虚高**到没有意义。
    2. **去掉两端的引号与括号**。LongMemEval 的答案里有
       `'Data Analysis using Python' webinar` 这种写法，按标点切出来的片段是 `'Data`
       ——而记忆里写的是 `Data Analysis`。于是**明明召回了却判为没命中**，
       指标因为一段标点**假性偏低**。

    第 2 条不是"顺手清理"，它是被真实数据逼出来的：修正前后差的是实打实的分数。
    """
    import re

    parts = re.split(r"[\s,，。.；;、:：!！?？]+", answer)
    cleaned = (x.strip("'\"\u201c\u201d\u2018\u2019()（）[]【】") for x in parts)
    return tuple(x for x in cleaned if len(x) >= 2)


# --------------------------------------------------------------------------- #
# 分派
# --------------------------------------------------------------------------- #

DATASET_LOADERS = {
    "selfcheck": load_jsonl,
    "locomo": load_locomo,
    "longmemeval": load_longmemeval,
}


def load(name: str, path: str | Path) -> list[Question]:
    """按 `name` 分派；未知名字**报错并列出支持的**（不猜、不回退到默认）。"""
    loader = DATASET_LOADERS.get(name)
    if loader is None:
        raise DatasetError(
            f"未知数据集 {name!r}；支持：{sorted(DATASET_LOADERS)}"
        )
    return loader(path)
