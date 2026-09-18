"""量化门槛的判分器（D-13：LoCoMo ≥ 70 / LongMemEval ≥ 75）。

## 为什么判分器必须单独成文件，且零依赖

标尺的价值全在**可比性**。如果"LoCoMo 得 70 分"这个数字依赖判分实现的细节，
它就没有意义——换个人跑出 62 或 78，谁也说不清差在哪。

所以这里刻意做到两件事：

1. **纯函数、零业务依赖**：不 import 器灵任何模块。判分与系统解耦，
   同一份预测在任何环境、任何版本上得分相同。
2. **每个指标写明它惩罚什么**——因为**选哪个指标，决定了系统朝哪个方向优化**：

   | 指标 | 惩罚 |
   |---|---|
   | `exact_match` | 多说了**一个字** |
   | `token_f1` | 多说了**一部分**（按重叠比例） |
   | `contains_answer` | 只要答案在句子里就行（对中文最宽容） |

   只报一个数字而不说它是什么，等于把"优化什么"藏起来。

## 未命中与答错**分开计**

"没召回相关记忆"与"召回了但答错"是**两种失败**，修法完全不同
（前者改召回，后者改生成或改提取）。合并成一个准确率，
会让"召回坏了"看起来像"模型变笨了"——而它们的排查方向相反。
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

__all__ = [
    "Metrics",
    "contains_answer",
    "exact_match",
    "hit_at_k",
    "mrr",
    "ndcg_at_k",
    "normalize",
    "recall_at_k",
    "time_marks",
    "time_match",
    "token_f1",
    "tokens",
]

_PUNCT = re.compile(
    "[\\s，。！？、；：\u201c\u201d\u2018\u2019（）《》【】,.!?;:\"'()\\[\\]{}<>—–\\-]+"
)
"""归一化用的"标点与空白"。**含空白**——EM / contains 关心的是内容，不是排版。"""

_WORD_SPLIT = re.compile(
    "[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+",
)
"""**分词用**的分隔符：把非字母数字非汉字一律当作词边界。

与 `_PUNCT` 分开是必须的，而且这个 bug 真的踩过：
`normalize` 会把空白一起去掉，于是 `I live in Shanghai` 变成 `iliveinshanghai`——
英文词全粘成一个。**归一化与分词对"空白算不算内容"的答案相反**，
共用一套正则就一定会错一个。
"""

_CJK = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def normalize(text: str) -> str:
    """归一化：全角转半角、大小写折叠、去标点与空白。

    **不做词干还原、不做同义词**——那些会引入判分器自己的判断，
    而"判分器不该有观点"是可比性的底线。
    """
    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    return _PUNCT.sub("", folded).strip()


def tokens(text: str) -> list[str]:
    """中英混排分词：**英文按词、中文按字符二元组**。

    为什么不按空格切中文：中文没有空格，整句会变成一个 token，F1 于是退化成
    "要么全对要么全错"——**指标失去区分度**，而失去区分度的指标无法指导优化。

    二元组是"不需要分词器"与"有区分度"之间的务实折中：它不引入 jieba 这类依赖，
    判分器因此保持零依赖（而这正是可比性的前提）。

    **注意这里用的是 `_WORD_SPLIT` 而不是 `_PUNCT`**：分词需要保留词边界，
    归一化不需要——两者对"空白算不算内容"的答案相反。
    """
    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    out: list[str] = []
    for segment in _WORD_SPLIT.split(folded):
        if not segment:
            continue
        if _CJK.match(segment[0]):
            out.extend(_cjk_units(segment))
        else:
            out.append(segment)
    return out


def _cjk_units(segment: str) -> list[str]:
    """把一段（以中文开头的）文本切成"连续中文按二元组 + 夹在中间的非中文单独成词"。"""
    pieces: list[str] = []
    buffer = ""
    for char in segment:
        if _CJK.match(char):
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(char)
        else:
            buffer += char
    if buffer:
        pieces.append(buffer)

    # 把被非中文隔开的**连续中文单字**各自合并成二元组
    merged: list[str] = []
    run: list[str] = []
    for piece in pieces:
        if len(piece) == 1 and _CJK.match(piece):
            run.append(piece)
        else:
            merged.extend(_bigrams(run))
            run = []
            merged.append(piece)
    merged.extend(_bigrams(run))
    return merged


def _bigrams(chars: list[str]) -> list[str]:
    """单字序列 → 二元组序列。单字时保持单字（不能凭空造词）。"""
    if not chars:
        return []
    if len(chars) == 1:
        return [chars[0]]
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def exact_match(prediction: str, gold: str) -> float:
    """归一化后完全相等。**多一个字就是 0**——最严的指标。"""
    return 1.0 if normalize(prediction) == normalize(gold) else 0.0


def contains_answer(prediction: str, gold: str) -> float:
    """归一化后 `gold` 是 `prediction` 的子串。

    对**中文长答**最稳：模型多说一句解释不该算错，少说核心事实才算错。
    代价是它对"答非所问但恰好包含"没有抵抗力——所以它只适合
    "标准答案是一小段关键事实"的题，不适合开放问答题。
    """
    gold_norm = normalize(gold)
    if not gold_norm:
        return 0.0
    return 1.0 if gold_norm in normalize(prediction) else 0.0


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip


def _marks(year: int, month: int, day: int | None = None) -> set[str]:
    """同一个时间点产出**三种粒度**：`2023-05`（年月）/ `2023-05-07`（日）/ `05-07`（省年）。

    **必须一次产出全部三种**，理由是三条边界各需要一种：

    - `7 May 2023` 与 `2023-05-07` 要**相等** → 两边都得产出同一组记号。
      只给一边多一种粒度就会不等（这正是第一版的 bug：`7 May 2023` 从
      `may 2023` 那支多拿了 `2023-05`，而 `2023-05-07` 没有，于是判成错）；
    - 标准答案只问"哪个月"（`June 2023`）而模型答了具体某天（`2023-06-15`）→
      **那是对的**，靠年月粒度相遇；
    - 标准答案只给"几月几号"（`13 August`）而模型答了带年的 → 靠省年粒度相遇。
    """
    out = {f"{year:04d}-{month:02d}"}
    if day is not None:
        out.add(f"{year:04d}-{month:02d}-{day:02d}")
        out.add(f"{month:02d}-{day:02d}")
    return out


def time_marks(text: str) -> set[str]:
    """把文本里的**时间表达**抽成可比较的记号。

    **只归写法，不做模糊匹配**：`7 May` 与 `8 May` 仍是两个记号。
    能识别的写法：`2023-05-07` / `2023/5/7` / `2023年5月7日` / `7 May 2023` /
    `May 7, 2023` / `June 2023` / `2023年6月` / `13 August`（无年）。
    """
    folded = str(text or "").casefold()
    out: set[str] = set()
    for m in re.finditer(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", folded):
        out |= _marks(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in re.finditer(r"\b(\d{4})[-/](\d{1,2})\b(?![-/]\d)", folded):
        out |= _marks(int(m.group(1)), int(m.group(2)))
    for m in re.finditer(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?", folded):
        out |= _marks(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in re.finditer(r"(\d{4})\s*年\s*(\d{1,2})\s*月", folded):
        out |= _marks(int(m.group(1)), int(m.group(2)))
    for m in re.finditer(r"\b(\d{1,2})\s+([a-z]{3,9})\.?\s*(\d{4})?\b", folded):
        month = _MONTHS.get(m.group(2))
        if not month or not 1 <= int(m.group(1)) <= 31:
            continue
        year_text = m.group(3)
        if year_text:
            out |= _marks(int(year_text), month, int(m.group(1)))
        else:
            out.add(f"{month:02d}-{int(m.group(1)):02d}")
    # `<Month> <day>, <year>`——美式写法（`May 7, 2023`），
    # 与上面的「日 月 年」是**同一批数据的两种排版**，都要认。
    month_day = r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?\b"
    for m in re.finditer(month_day, folded):
        month = _MONTHS.get(m.group(1))
        if not month or not 1 <= int(m.group(2)) <= 31:
            continue
        year_text = m.group(3)
        if year_text:
            out |= _marks(int(year_text), month, int(m.group(2)))
        else:
            out.add(f"{month:02d}-{int(m.group(2)):02d}")
    for m in re.finditer(r"\b([a-z]{3,9})\.?\s+(\d{4})\b", folded):
        month = _MONTHS.get(m.group(1))
        if month:
            out |= _marks(int(m.group(2)), month)
    return out


def time_match(prediction: str, gold: str) -> float:
    """答案里的**时间点**与标准答案是否一致——只归一写法，不做模糊匹配。

    ## 为什么需要它

    LoCoMo 的 `temporal` 题，标准答案是 `7 May 2023`，而模型答 `2023-05-07`：
    **同一个日期**，但按 token 重叠算**一个 token 都不共享**，于是被判成错。

    实测的规模：`temporal` 答错的题里 **195/321（60.7%）** 预测与标准答案
    含相同数字；加上归一后**全量准确率 20.5% → 27.0%**，其中
    `temporal` **13.1% → 43.3%**，而其余三类合计只动了 **0.5 个点**
    （`single_hop` +0.2 / `multihop` +0.3 / `open_domain` ±0）——
    **它只在该生效的地方生效**。

    ## 三条边界（**不许放松**）

    - `7 May` 与 `8 May` 是两个记号 → 仍判错；
    - **标准答案里没有时间记号时直接返回 0**：否则空集会与任意预测"相交"，
      把不含时间的题判成对；
    - `13 August`（无年）能对上 `2023-08-13`，靠的是降级形式（见 `_marks`）——
      那是刻意的，因为标准答案常只给到"几月几号"。

    ## 与 F1 的关系

    **它只进"准确率"，不进 F1。** F1 是 token 重叠、与 LoCoMo 官方口径一致，
    刻意保持原样：官方的 F1 同样会惩罚写法差异，所以那一列仍可与公开数字比较。
    """
    if not prediction or not gold:
        return 0.0
    wanted = time_marks(gold)
    if not wanted:
        return 0.0
    return 1.0 if wanted <= time_marks(prediction) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD 式 token F1（多集求交，取重叠计数）。

    `precision` 罚"多说了"、`recall` 罚"少说了"，F1 是两者的调和平均。
    这是 LoCoMo 一类的常用口径，也是"部分正确"能被体现的地方。
    """
    pred_tokens = tokens(prediction)
    gold_tokens = tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    from collections import Counter

    pred_count, gold_count = Counter(pred_tokens), Counter(gold_tokens)
    overlap = sum((pred_count & gold_count).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# 检索指标（衡量"召回"这一半，与生成解耦）
# --------------------------------------------------------------------------- #


def recall_at_k(retrieved: list[str], gold_ids: set[str], k: int) -> float:
    """前 k 条里命中了几成标准答案对应的记忆。

    **这是最该被单独盯住的指标**：它衡量"记忆有没有被找出来"，
    完全不掺生成模型的水平。它掉下去，说明是检索坏了——而不是"模型变笨了"。
    """
    if not gold_ids:
        return 0.0
    hit = len(set(retrieved[:k]) & gold_ids)
    return hit / len(gold_ids)


def recall_at_gold(retrieved: list[str], gold_ids: set[str]) -> float:
    """**上限恒为 1.0 的召回**：只看前 `|gold|` 条。

    为什么必须有它——`R@10` 的**上限是 `10 / |gold|`**：
    实测 LoCoMo 有 **46% 的题 gold 超过 10 条**，于是 R@10 再怎么完美
    也到不了 0.241。那个数字低**不代表检索差**，但从报告上读起来一模一样。

    这个口径把 `k` 与被测对象对齐，回答的是：
    **"把相关记忆按数量取回来，取对了几成"**——它不因 `top_k` 的设置而变。
    """
    if not gold_ids:
        return 0.0
    hit = len(set(retrieved[: len(gold_ids)]) & gold_ids)
    return hit / len(gold_ids)


def hit_at_k(retrieved: list[str], gold_ids: set[str], k: int) -> float:
    """前 k 条里**有没有**相关记忆——只问"找得到找不到"，不问"找到多少"。

    ## 为什么它与 `R@k` 是两个必须分开的问题

    `R@k` 的分母是 `|gold|`，而本仓库的 gold 是「**该会话产生的全部记忆**」。
    在**原子记忆**粒度下，同一个会话里往往只有一两条与某道题真正相关。
    实测（conv-26#82）：

    - 问：`What did the charity race raise awareness for?`
    - gold 6 条：`Melanie is a parent`、`Caroline is researching adoption agencies`、
      `Melanie ran a charity race for mental health…`、`…going camping`、
      `…carves out me-time`、`Caroline chose an adoption agency…`
    - **只有第 3 条相关**，而系统**把它排在第 1 位**

    于是 `R@|gold|` 把"准确地排在第 1 位"记成"只找到 1/6"。
    这个数字低**不代表检索差**——它代表**量尺与问题不匹配**。

    `hit@k` 不受 gold 集大小影响，所以它在两种粒度下都可读。
    """
    if not gold_ids:
        return 0.0
    return 1.0 if set(retrieved[:k]) & gold_ids else 0.0


def mrr(retrieved: list[str], gold_ids: set[str]) -> float:
    """首个命中的倒数排名（MRR）。**衡量"排得对不对"**，不只是"找没找到"。"""
    for rank, mem_id in enumerate(retrieved, start=1):
        if mem_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold_ids: set[str], k: int) -> float:
    """归一化折损累计增益。**二元相关性**版本——命中记 1、未命中记 0。

    与 MRR 的差别：MRR 只看第一个命中，NDCG 看**整段的排序质量**。
    多条标准答案时（"关于这个主题的三条记忆"），MRR 会严重高估。
    """
    if not gold_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, mem_id in enumerate(retrieved[:k], start=1)
        if mem_id in gold_ids
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_ids), k) + 1))
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #


@dataclass
class Metrics:
    """一次评测的汇总。**每个数字都带 n**——没有样本量的分数读不出可信度。"""

    total: int = 0
    answered: int = 0
    """生成了答案的题数。`total - answered` 就是**未命中**（没召回到相关记忆）。"""
    correct: int = 0
    em_sum: float = 0.0
    f1_sum: float = 0.0
    contains_sum: float = 0.0
    recall_sum: float = 0.0
    recall_gold_sum: float = 0.0
    """`R@|gold|` 的累加——**上限恒为 1.0**，不受 `top_k` 卡（见 `recall_at_gold`）。"""
    hit_sum: float = 0.0
    hit_at_1_sum: float = 0.0
    """`hit@k` / `hit@1` 的累加——问的是"**找得到找不到**"，
    不受 gold 集大小影响（见 `hit_at_k`），因此对 gold 口径不敏感。"""
    time_sum: float = 0.0
    """时间点一致的题数累加——**只归写法**（`2023-05-07` ≡ `7 May 2023`），见 `time_match`。"""
    mrr_sum: float = 0.0
    ndcg_sum: float = 0.0
    by_category: dict[str, Metrics] = field(default_factory=dict)
    """按类别拆分的分数。

    **只有总分是不够的**：总分 72 可能是"时间类 95、多跳类 40"，
    也可能是"样样 72"。前者有一个明确的下一步（攻多跳），后者没有——
    而这两种情况在总分上完全一样。
    """

    def add(
        self,
        *,
        pred: str | None,
        gold: str,
        retrieved: list[str],
        gold_ids: set[str],
        k: int = 10,
    ) -> None:
        self.total += 1

        # **检索指标先算，且与生成无关。**
        #
        # `pred is None` 的意思是"没走到生成"——那**恰恰是检索失败的证据**
        # （没召回到相关记忆）。早先这里先 `return` 再算检索指标，于是那些题
        # 既不进分子也不进分母：**召回越差，分母越小，指标反而越好看**。
        # 这与 `accuracy` 那条"分母不能用 answered"的教训是同一个错误，
        # 只是它藏在了另一个属性里。
        self.recall_sum += recall_at_k(retrieved, gold_ids, k)
        self.recall_gold_sum += recall_at_gold(retrieved, gold_ids)
        self.hit_sum += hit_at_k(retrieved, gold_ids, k)
        self.hit_at_1_sum += hit_at_k(retrieved, gold_ids, 1)
        # `mrr` / `ndcg` 衡量的是**前 k 名排得对不对**——传全长会把"多取了"算成好处，
        # 而它们要回答的是"在系统实际给出的 top_k 里，相关记忆排得如何"。
        self.mrr_sum += mrr(retrieved[:k], gold_ids)
        self.ndcg_sum += ndcg_at_k(retrieved, gold_ids, k)

        if pred is None:
            # **未命中**：不算答错，但 F1/EM 都是 0——它必须拉低分数，
            # 否则"召回坏了"会被"没答的题不计分"悄悄抹平。
            return
        self.answered += 1
        self.em_sum += exact_match(pred, gold)
        self.f1_sum += token_f1(pred, gold)
        self.contains_sum += contains_answer(pred, gold)
        self.time_sum += time_match(pred, gold)
        # **判对有三个来源**：字面全等 / 包含 / **时间点一致**。
        # 第三个是"写法归一"——`2023-05-07` 与 `7 May 2023` 是同一个答案，
        # 只因为写法不同就判错，那是**判分器在表达它对排版的观点**。
        if contains_answer(pred, gold) or exact_match(pred, gold) or time_match(pred, gold):
            self.correct += 1

    @property
    def accuracy(self) -> float:
        """**分母是 `total` 而不是 `answered`**——没答上来的题要算错。

        用 `answered` 当分母会得到一个漂亮的假分数：
        召回率越低，分母越小，准确率反而越高。
        """
        return self.correct / self.total if self.total else 0.0

    @property
    def f1(self) -> float:
        return self.f1_sum / self.total if self.total else 0.0

    @property
    def recall(self) -> float:
        """**分母是 `total`，不是 `answered`。**

        与 `accuracy` 同一条理由：没召回到的题**必须算 0**。
        用 `answered` 当分母时，召回越差、进分母的题越少、指标反而越好看——
        **在最坏的时候最好看**，而这正是最需要报警的时刻。
        """
        return self.recall_sum / self.total if self.total else 0.0

    @property
    def recall_gold(self) -> float:
        """`R@|gold|`——**不受 `top_k` 卡上限**的召回（见 `recall_at_gold`）。

        **读它之前先看 gold 的构成**：本仓库的 gold 是「该会话产生的全部记忆」，
        其中往往只有一两条与某道题真正相关。所以这个数字低**可能只是量尺错配**，
        而不是检索差——判断时请对照 `hit` 与 `mrr`。
        """
        return self.recall_gold_sum / self.total if self.total else 0.0

    @property
    def hit(self) -> float:
        """`hit@k`——**相关记忆进没进前 k**。

        它是这一层评测里**最该先看**的检索指标：不受 gold 集大小影响，
        也不会被"gold 口径过宽"打偏。它掉下去，才是真的没找到。
        """
        return self.hit_sum / self.total if self.total else 0.0

    @property
    def hit_at_1(self) -> float:
        """`hit@1`——**第一条就是相关记忆**的比例，衡量排序顶端的精度。"""
        return self.hit_at_1_sum / self.total if self.total else 0.0

    @property
    def time_hit(self) -> float:
        """时间点与标准答案一致（含写法归一）的题占比。

        **它是一条规模尺，不是能力指标**：它高，说明"答对了但写法不同"
        被原口径误判的多；它掉下去，才说明时间类的题答得差。
        """
        return self.time_sum / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        return self.mrr_sum / self.total if self.total else 0.0

    @property
    def ndcg(self) -> float:
        return self.ndcg_sum / self.total if self.total else 0.0

    @property
    def answer_rate(self) -> float:
        """**答题率**——即"召回成功、走到生成这一步"的比例。

        它与 `accuracy` 的关系诊断力最强：
        答题率高而准确率低 → 检索找对了、生成没说对；
        答题率低 → 问题在检索或提取，与生成无关。
        """
        return self.answered / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "answered": self.answered,
            "answer_rate": round(self.answer_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "em": round(self.em_sum / self.total, 4) if self.total else 0.0,
            "contains": round(self.contains_sum / self.total, 4) if self.total else 0.0,
            "time_match": round(self.time_hit, 4),
            "hit@1": round(self.hit_at_1, 4),
            "hit@k": round(self.hit, 4),
            "recall@k": round(self.recall, 4),
            "recall@gold": round(self.recall_gold, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@k": round(self.ndcg, 4),
            "by_category": {name: m.to_dict() for name, m in self.by_category.items()},
        }
