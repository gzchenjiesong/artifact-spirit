"""去重与更新决策（LLD-AL2 §2.5）。

| 阶段 | 支持的状态 |
|---|---|
| MVP（M2） | ``ADD`` / ``UPDATE`` / ``IGNORE`` |
| **M3（已补齐）** | 新增 ``MERGE``（真合并）与 ``INVALIDATE``（旧条失效、**不删**） |

> 分期曾是**显式的**（记录在 [DES-000 §1.2 非目标](../../../docs/design/00-方案设计.md) 与 DES-REV-001 §3.6），
> M3 把它补齐。**``FORGET`` 刻意不在本模块**——它只属于"合规清除"与"用户显式删除"，
> 不是去重决策能得出的结论（D-17）。

## M3 的关键区别：「取值变了」不等于「该改那条记录」

- ``UPDATE``：改那条记录本身（补字段、纠错别字）——**历史被覆盖**；
- ``INVALIDATE``：**旧条失效 + 新条新增**——历史保留，`asof` 仍答得出"上个月是什么"。

同一属性取值变化属于后者：**用户换地址不是"我们记错了"，而是"事实变了"**。
把它按 ``UPDATE`` 处理，等于把"搬家"和"改错别字"混为一谈——
而这两件事在"我上个月填的地址是什么"这个问题上，答案完全不同。

## 为什么先用规则而不是让 LLM 判

去重决策**错了会污染记忆**，因此它确实是"值得用 LLM"的地方。但规则先行的理由不变：

- 规则可测、可解释、零成本、零延迟
- 规则版已覆盖绝大多数情形（同事实重复出现、同属性值变化、同值不同表述）
- LLM 判重需要"候选集 — 模型 — 再校验"的闭环，放 M3 的交叉验证里更合理

``llm`` 参数保留为可选：接入后它只在规则无法判定时兜底。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..common import content_hash_of
from ..model.base import LLMError, LLMProvider, ProviderUnavailableError
from ..store.base import MemoryBackend, MemoryRecord

__all__ = ["Decision", "DedupDecision", "Deduplicator"]

Decision = Literal["ADD", "UPDATE", "IGNORE", "MERGE", "INVALIDATE"]
"""五态。**`FORGET` 不在其中**：它只属合规清除与用户显式删除（D-17），不是去重的结论。"""


@dataclass(slots=True)
class DedupDecision:
    """去重决策。**必须写 audit（含 before/after）**——决策本身也要可溯。"""

    decision: Decision
    target_id: str | None = None
    reason: str | None = None
    merged: dict | None = None
    source: str = "rule"
    """``rule`` 或 ``llm``——让"这条为什么被判为重复"可解释。"""

    @property
    def is_write(self) -> bool:
        return self.decision in ("ADD", "UPDATE", "MERGE")


@dataclass(slots=True)
class Deduplicator:
    """三态去重。"""

    backend: MemoryBackend
    llm: LLMProvider | None = None

    def decide(self, candidate: dict) -> DedupDecision:
        """对一条候选做去重决策。

        判定顺序（先精确、后宽松）：

        1. **内容指纹完全一致** → ``IGNORE``（同一事实已存在，且情境相同）
        2. **同一属性（subject+predicate）已有取值** → ``UPDATE``（值变了，取代旧的）
        3. 其余 → ``ADD``
        """
        digest = content_hash_of(
            content=candidate["content"],
            subject=candidate.get("subject"),
            predicate=candidate.get("predicate"),
            object_=candidate.get("object"),
            scope=candidate.get("scope"),
        )

        duplicates = self.backend.find_by_hash(digest)
        if duplicates:
            # 指纹按 **三元组** 算（D-21），**不含 `content`**。
            # 所以"指纹相同"只说"说的是同一件事"，**不说"说得一模一样"**——
            # 直接判 IGNORE 会丢掉新表述里的细节。
            return self._settle_same_fact(duplicates[0], candidate, why="内容指纹一致")

        subject = candidate.get("subject")
        predicate = candidate.get("predicate")
        if subject and predicate:
            same_attr = self._find_same_attribute(subject, predicate)
            if same_attr is not None:
                if _norm(same_attr.object) != _norm(candidate.get("object")):
                    # 取值变了 = **事实变了**（不是"我们记错了"）→ 旧条失效、新条新增。
                    # 走 INVALIDATE 而不是 UPDATE：后者会把历史覆盖掉，
                    # 于是"上个月填的地址"永远答不出来（D-17 / INV-7）。
                    return DedupDecision(
                        decision="INVALIDATE",
                        target_id=same_attr.id,
                        reason=(
                            f"同一属性（{subject}.{predicate}）取值变化："
                            f"{same_attr.object!r} → {candidate.get('object')!r}"
                        ),
                    )
                return self._settle_same_fact(same_attr, candidate, why="同一属性取值相同")

        return DedupDecision(decision="ADD", reason="未发现重复")

    @staticmethod
    def _settle_same_fact(
        same: MemoryRecord, candidate: dict, *, why: str
    ) -> DedupDecision:
        """同一件事已经存在：**表述一致就忽略，表述不同就合并**。

        这两者的区别不是文字游戏：``IGNORE`` 会丢掉新表述里的细节，
        而"合并"是让信息**只增不减**的那个选择。
        """
        if _norm(same.content) == _norm(candidate.get("content")):
            return DedupDecision(
                decision="IGNORE",
                target_id=same.id,
                reason=f"{why}、表述也一致——完全重复",
            )
        return DedupDecision(
            decision="MERGE",
            target_id=same.id,
            merged={"content": merge_texts(same.content, candidate["content"])},
            reason=f"{why}，但表述不同——合并为信息更全的一条",
        )

    # ------------------------------------------------------------------ #

    def _find_same_attribute(self, subject: str, predicate: str) -> MemoryRecord | None:
        """找同主体同谓词的最新一条（不区分状态——休眠的记忆也不能被重复写一遍）。"""
        for record in self.backend.query(status=None, limit=1000):
            if _norm(record.subject) == _norm(subject) and _norm(record.predicate) == _norm(
                predicate
            ):
                return record
        return None

    def llm_arbitrate(
        self, candidate: dict, existing: list[MemoryRecord], *, schema: dict
    ) -> DedupDecision:
        """规则判不了时请模型仲裁（M3 交叉验证的前身）。

        失败一律退回 ``ADD``——**宁可多存一条，也不要因为仲裁失败而丢掉信息**。
        """
        if self.llm is None or not existing:
            return DedupDecision(decision="ADD", reason="无可仲裁对象", source="rule")

        listing = "\n".join(f"- {r.id}: {r.content}" for r in existing[:20])
        try:
            payload = self.llm.complete_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你在做记忆去重决策。判断新候选与已有记忆的关系，"
                            '输出 JSON：{"decision": "ADD|UPDATE|IGNORE|MERGE", '
                            '"target_id": "…|null", "reason": "…"}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"新候选：{candidate['content']}\n已有记忆：\n{listing}",
                    },
                ],
                schema=schema,
            )
        except (LLMError, ProviderUnavailableError):
            return DedupDecision(decision="ADD", reason="仲裁失败，保守新增", source="llm")

        decision = str(payload.get("decision", "ADD")).upper()
        if decision not in ("ADD", "UPDATE", "IGNORE", "MERGE"):
            decision = "ADD"
        return DedupDecision(
            decision=decision,  # type: ignore[arg-type]
            target_id=payload.get("target_id"),
            reason=payload.get("reason"),
            source="llm",
        )


def merge_texts(old: str, new: str) -> str:
    """把两条表述合并成一条**信息更全**的文本。

    规则刻意简单、确定、可测：

    1. 一方包含另一方 → 取**更长的那条**（它是超集）；
    2. 否则 → ``旧；新``（全角分号连接，保持"一句话"的可读性）。

    **不做"智能摘要"**：合并的代价是不可逆的——合并之后就分不出哪句来自哪次陈述。
    所以宁可用一条可解释的规则，也不在这里引入 LLM 调用。
    """
    a, b = old.strip(), new.strip()
    if not a:
        return b
    if not b:
        return a
    if _norm(a) in _norm(b):
        return b
    if _norm(b) in _norm(a):
        return a
    return f"{a}；{b}"


def _norm(value: str | None) -> str:
    return (value or "").strip().casefold()
