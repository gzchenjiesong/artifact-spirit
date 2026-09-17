"""T4 —— **完整向量路径**（唯一能证明"语义召回"真的成立的一段）。

前三段都在没有 embedding 的环境里跑，召回走的是 BM25。BM25 能过，只说明"降级可用"，
**完全不能说明向量路可用**——而向量路才是这个系统"按意思找记忆"的能力本体。

所以这一段只做一件核心的事，且必须用**词面不重叠、语义相关**的查询：

    存：用户对花生过敏，非常严重
    查：有什么食材我必须避开        ← BM25 命中不了（没有任何词重叠）

如果这种查询能召回，说明向量路真的在工作；如果只测"用原话查原话"，
那测的其实是关键词匹配，向量路坏掉也照样通过。

另外验证：维度不符必须**拒绝写入**（INV-2）、重嵌入（reindex）真实可用、
六因子里的 semantic 分量真的非零。
"""

from __future__ import annotations

import time

from harness import EMBED_DIM, EMBED_MODEL, Report, fresh_db, make_home, real_env

SECTION = "T4 完整向量路径（含 embedding）"

# (存进去的原文, 语义相关但**词面完全不相交**的查询, 期望召回关键词)
#
# 查询必须与原文做到**中文 bigram 都不相交**，否则测的就不是向量能力：
# 本文件的 FTS 用的是"CJK 逐字归一化 + bigram 查询"，只要共享一个 bigram
# （例如"必须"），BM25 就会命中，于是向量路坏掉测试也照样通过。
# 第一版就踩了这个坑——三条查询的对照组全命中，等于什么也没证明。
SEMANTIC_PAIRS = [
    (
        "用户对花生过敏，非常严重，必须严格避开含花生的食物。",
        "哪些东西吃了会不舒服",
        "花生",
    ),
    (
        "用户偏好深色主题界面，浅色看久了眼睛疼。",
        "屏幕亮度调低一点更舒服",
        "深色",
    ),
    (
        "项目 RAGFlow 使用 Python 3.12，部署方式是 Docker。",
        "这套系统用什么语言写的",
        "Python",
    ),
]


def _cjk_bigrams(text: str) -> set[str]:
    """中文 bigram 集合——正是 FTS 索引与查询的匹配粒度。"""
    import re

    cjk = "".join(re.findall(r"[一-鿿]", text or ""))
    return {cjk[i : i + 2] for i in range(len(cjk) - 1)}


def run(report: Report) -> None:
    from artifact_spirit.core import RecallQuery, TurnEvent
    from artifact_spirit.model import EmbeddingError
    from artifact_spirit.observability import status
    from artifact_spirit.runtime import start

    report.section(SECTION)

    if not EMBED_MODEL:
        report.info(SECTION, "跳过：未配置 embedding", "设 REALTEST_EMBED_* 后自动启用")
        return

    home = make_home("t4", with_embedding=True)
    fresh_db(home)
    env = real_env()

    # ---------------------------------------------------------------- T4.1
    services = start(home, env=env, start_threads=False, reconcile=False)
    embed = services.core.embedding
    report.check(
        SECTION,
        f"T4.1 embedding provider 就绪（{EMBED_MODEL}）",
        embed is not None,
        f"model={getattr(embed, 'model', '?')} dim={getattr(embed, 'dim', '?')}",
    )
    if embed is None:
        services.stop()
        return

    report.check(
        SECTION,
        "T4.2 向量维度与配置一致（INV-2）",
        int(getattr(embed, "dim", 0)) == int(EMBED_DIM),
        f"provider={getattr(embed, 'dim', '?')} 配置={EMBED_DIM}",
    )

    # ---------------------------------------------------------------- T4.3 真实向量化
    t0 = time.time()
    try:
        vecs = embed.embed(["用户对花生过敏", "今天天气不错"])
        ms = (time.time() - t0) * 1000
        report.check(
            SECTION,
            "T4.3 真实 embedding 调用（批量保序）",
            len(vecs) == 2 and len(vecs[0]) == int(EMBED_DIM),
            f"{ms:.0f}ms · {len(vecs)} 条 × {len(vecs[0])} 维",
            ms,
        )
    except Exception as exc:
        report.check(SECTION, "T4.3 真实 embedding 调用", False, f"{type(exc).__name__}: {exc}")
        services.stop()
        return

    # 语义性自证：相近语义的向量距离应显著小于无关语义
    def cosine(a, b) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    v = embed.embed(
        ["用户对花生过敏，必须避开", "有什么食材我需要忌口", "周末打算去爬山顺便拍点风景"]
    )
    sim_related = cosine(v[0], v[1])
    sim_unrelated = cosine(v[0], v[2])
    report.check(
        SECTION,
        "T4.3b 语义可分：相关句相似度 > 无关句（否则向量路没有意义）",
        sim_related > sim_unrelated,
        f"相关 {sim_related:.4f} vs 无关 {sim_unrelated:.4f}（差 {sim_related - sim_unrelated:+.4f}）",
    )

    # ---------------------------------------------------------------- T4.4 写入带向量的记忆
    before = len(services.backend.query(status=None))
    for content, _, _ in SEMANTIC_PAIRS:
        services.write_now(services.core.ingest_turn(
            TurnEvent(session_id="s-vec", user=f"请记住：{content}", assistant="好的", ts="2026-09-15T10:00:00+08:00")
        ))
    services.write_now(services.core.drain_pending())
    after = services.backend.query(status=None)
    report.check(
        SECTION,
        "T4.4 真实提取 + 向量写入",
        len(after) > before,
        f"{before} → {len(after)} 条",
    )
    report.info(SECTION, "T4.4b 落库内容", " || ".join(r.content[:38] for r in after))

    # 向量真的落库了（不是只存了正文）
    try:
        row = services.backend.conn.execute("SELECT COUNT(*) FROM vec_memories").fetchone()
        n_vec = int(row[0])
    except Exception:
        n_vec = -1
    report.check(
        SECTION,
        "T4.5 向量确实写入 vec0（不只是存了正文）",
        n_vec > 0,
        f"vec_memories 行数 = {n_vec}",
    )

    # ---------------------------------------------------------------- T4.6 核心：语义召回
    print("      --- 语义召回（查询与原文**bigram 都不相交**）---")
    semantic_hits = 0
    disjoint_hits = 0
    kw_hits = 0
    for _, query, expect in SEMANTIC_PAIRS:
        qvec = embed.embed([query])[0]
        hits = services.core.recall(RecallQuery(text=query, vec=qvec, session_id="s-vec", top_k=5))
        top_record = hits[0].record if hits else None
        joined = " ".join(h.record.content for h in hits)
        ok = expect in joined
        semantic_hits += int(ok)

        # **自证**：命中的那条与查询在 bigram 上是否真的不相交？
        # 不相交却召回成功 → 只能来自向量路，这条断言才有证明力。
        overlap: set[str] = set()
        if top_record is not None:
            overlap = _cjk_bigrams(query) & _cjk_bigrams(
                " ".join(filter(None, [top_record.content, top_record.abstract, top_record.subject, top_record.object]))
            )
            if ok and not overlap:
                disjoint_hits += 1

        # 对照组：同一条查询走纯关键词
        kw = services.backend.keyword_search(query, top_k=5)
        kw_ok = any(
            expect in str(h.meta.get("record", {}).get("content", "") or h.content or "") for h in kw
        )
        kw_hits += int(kw_ok)

        print(
            f"        查 {query!r:<24} → {'向量:命中' if ok else '向量:未命中'} | "
            f"BM25:{'命中' if kw_ok else '未命中'} | bigram 交集={sorted(overlap) or '∅'} · top={(top_record.content[:26] if top_record else '（无）')}"
        )

    report.check(
        SECTION,
        "T4.6 **语义召回成立**（词面不重叠也能找回）",
        semantic_hits == len(SEMANTIC_PAIRS),
        f"{semantic_hits}/{len(SEMANTIC_PAIRS)} 个语义查询命中",
    )
    report.check(
        SECTION,
        "T4.6b **对照组必须低于向量路**——否则这段测试证明不了向量能力",
        kw_hits < semantic_hits,
        f"向量 {semantic_hits}/{len(SEMANTIC_PAIRS)} vs BM25 {kw_hits}/{len(SEMANTIC_PAIRS)}"
        + ("（BM25 打平 → 查询与原文仍有 bigram 重叠，需换查询）" if kw_hits >= semantic_hits else ""),
    )
    report.check(
        SECTION,
        "T4.6c 命中的记忆与查询 bigram 完全不相交（自证：这不是关键词匹配）",
        disjoint_hits >= 1,
        f"{disjoint_hits}/{len(SEMANTIC_PAIRS)} 条命中的 bigram 交集为空",
    )

    # ---------------------------------------------------------------- T4.7 六因子
    qvec = embed.embed([SEMANTIC_PAIRS[0][1]])[0]
    hits = services.core.recall(RecallQuery(text=SEMANTIC_PAIRS[0][1], vec=qvec, session_id="s-vec", top_k=5))
    factors = sorted(hits[0].raw) if hits else []
    sem = hits[0].raw.get("semantic", 0.0) if hits else 0.0
    report.check(
        SECTION,
        "T4.7 六因子融合：semantic 分量非零（向量路真的进入了排序）",
        "semantic" in factors and float(sem) > 0,
        f"因子={factors} · semantic={float(sem) if sem else 0:.4f}",
    )

    # ---------------------------------------------------------------- T4.8 门槛
    sal = services.core.sensory.filter(_ctx("请记住：我对花生过敏"))
    report.check(
        SECTION,
        "T4.8 有 embedding 时用完整门槛（novelty 参与打分）",
        sal.novelty_available,
        f"score={sal.score:.3f} · 门槛={services.config.salience.get('threshold', 0.35)} · novelty 可用={sal.novelty_available}",
    )

    # ---------------------------------------------------------------- T4.9 维度不符必须拒绝
    report.check(
        SECTION,
        "T4.9 维度不符时拒绝写入（不得静默截断，C6 / INV-2）",
        _dimension_mismatch_rejected(services),
        "已用错误维度调用 put → 抛 DimensionMismatchError",
    )

    # ---------------------------------------------------------------- T4.10 重嵌入
    from artifact_spirit.store import reindex

    res = reindex.reindex(services.backend, embed.embed, model=EMBED_MODEL, dim=int(EMBED_DIM), batch_size=8)
    report.check(
        SECTION,
        "T4.10 全库重嵌入（reindex）真实可用",
        int(res.get("done", 0)) >= int(res.get("total", 0)) > 0 or int(res.get("skipped", 0)) > 0,
        f"{res}",
    )
    hits_after = services.core.recall(
        RecallQuery(text=SEMANTIC_PAIRS[0][1], vec=embed.embed([SEMANTIC_PAIRS[0][1]])[0], session_id="s-vec", top_k=5)
    )
    report.check(
        SECTION,
        "T4.11 重嵌入后语义召回仍成立",
        bool(hits_after) and SEMANTIC_PAIRS[0][2] in " ".join(h.record.content for h in hits_after),
        f"{len(hits_after)} 条命中",
    )

    # ---------------------------------------------------------------- T4.12 降级告警应为空
    st = status(services)
    degs = st.get("degradations") or []
    report.check(
        SECTION,
        "T4.12 配置完整时不再报 embedding 降级",
        not any("embedding" in str(d).lower() or "BM25" in str(d) for d in degs),
        f"degradations={degs}",
    )

    services.stop()
    _ = EmbeddingError


def _ctx(text: str):
    from artifact_spirit.core.base import TurnContext

    return TurnContext(session_id="s-vec", ts="2026-09-15T10:00:00+08:00", user=text, assistant="")


def _dimension_mismatch_rejected(services) -> bool:
    from artifact_spirit.store.base import DimensionMismatchError, MemoryRecord

    try:
        services.backend.put(
            MemoryRecord(id="", layer="semantic", type="fact", content="维度探针"),
            [0.1] * 7,  # 明显错误的维度
        )
    except DimensionMismatchError:
        return True
    except Exception:
        return False
    return False
