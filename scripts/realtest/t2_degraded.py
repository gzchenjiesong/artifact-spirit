"""T2 —— 无 embedding 的真实降级路径。

本机（以及任何只配了 chat 网关的场景）没有 embedding 服务。这不是异常，
而是设计里明确规定的一条路径（INV-6 + F2）：

- 写入：仍然成功，只是**没有向量**（记忆不丢）
- 召回：**降级为 BM25**，中文 2 字词必须仍能命中
- 可观测：`status` / `doctor` 必须**明说**降级了，而不是安静地返回差结果

这一段的重点是"降级必须可见"。一个不告警的降级，等于数据质量的静默劣化。
"""

from __future__ import annotations

import time

from harness import Report, fresh_db, make_home, real_env

SECTION = "T2 无 embedding 降级路径"


def run(report: Report) -> None:
    from artifact_spirit.core import RecallQuery, TurnEvent
    from artifact_spirit.observability import doctor, doctor_text, status
    from artifact_spirit.runtime import start

    report.section(SECTION)
    home = make_home("t2", with_embedding=False)
    fresh_db(home)
    env = real_env()

    services = start(home, env=env, start_threads=False, reconcile=False)
    report.check(
        SECTION,
        "T2.1 装配成功（无 embedding 不阻断启动）",
        services.backend is not None,
        f"notes={services.notes[:1]} · warnings={services.warnings[:1]}",
    )
    report.check(
        SECTION,
        "T2.2 降级被显式记录（notes 里说明走了关键词模式）",
        any("embedding" in n.lower() or "关键词" in n or "BM25" in n for n in services.notes),
        f"notes={services.notes}",
    )
    report.check(
        SECTION,
        "T2.3 core.embedding 为空（未偷接 chat 模型）",
        services.core.embedding is None,
        f"core.embedding={services.core.embedding!r}",
    )

    # ---------------------------------------------------------------- 写入
    turns = [
        ("用户：请记住，我对花生过敏，非常严重。", "助手：已记住，会提醒你避开含花生的食物。"),
        ("用户：项目 RAGFlow 现在用 Python 3.12，部署方式是 Docker。", "助手：了解了。"),
        ("用户：我偏好深色主题，浅色看久了眼睛疼。", "助手：记下了。"),
    ]
    t0 = time.time()
    total_intents = 0
    for user, assistant in turns:
        intents = services.core.ingest_turn(
            TurnEvent(session_id="s1", user=user, assistant=assistant, ts="2026-09-14T10:00:00+08:00")
        )
        total_intents += len(intents)
        services.write_now(intents)
    ms = (time.time() - t0) * 1000
    stored = services.backend.query(status=None)
    report.check(
        SECTION,
        "T2.4 真实提取 + 写入（fidelity > LLM 失败）",
        len(stored) >= 1,
        f"{total_intents} 个意图 → 落库 {len(stored)} 条 · {ms:.0f}ms",
        ms,
    )
    report.info(SECTION, "T2.4b 落库内容", " || ".join(f"[{r.layer}/{r.type}] {r.content[:40]}" for r in stored[:5]))

    # ---------------------------------------------------------------- 召回
    queries = ["花生过敏", "Python", "深色主题", "偏好", "Docker"]
    hits_by_query = {}
    for q in queries:
        hits = services.core.recall(RecallQuery(text=q, session_id="s1", top_k=5))
        hits_by_query[q] = hits
        print(f"      召回 {q!r:<12} → {len(hits)} 条 " + str([h.record.content[:18] for h in hits[:2]]))
    hit_queries = [q for q, h in hits_by_query.items() if h]
    report.check(
        SECTION,
        "T2.5 BM25 降级召回可用（中文 2 字词不失效）",
        len(hit_queries) >= 4,
        f"命中 {len(hit_queries)}/{len(queries)} 个查询：{hit_queries}",
    )
    report.check(
        SECTION,
        "T2.5b 2 字中文词专项（'偏好'——这是 unicode61/trigram 方案会死掉的地方）",
        bool(hits_by_query["偏好"]),
        f"'偏好' → {len(hits_by_query['偏好'])} 条",
    )
    report.check(
        SECTION,
        "T2.6 召回轨迹标明语义分缺席（raw 里语义因子为 0 而非静默缺失）",
        all("semantic" in h.raw for h in hits_by_query["花生过敏"]) if hits_by_query["花生过敏"] else False,
        f"raw keys={sorted(hits_by_query['花生过敏'][0].raw) if hits_by_query['花生过敏'] else '无命中'}",
    )

    # ---------------------------------------------------------------- 可观测
    st = status(services)
    alerts = st.get("degradations", [])
    report.check(
        SECTION,
        "T2.7 status 明示降级告警",
        any("BM25" in a or "embedding" in a.lower() for a in alerts),
        f"{alerts}",
    )
    doc = doctor(services)
    text = doctor_text(doc)
    report.check(
        SECTION,
        "T2.8 doctor 报告包含 embedding 检查项并给出提示",
        "embedding" in text.lower(),
        f"检查项={[c['check'] for c in doc['checks']]}",
    )
    print("      doctor 摘要：")
    for line in text.splitlines()[:9]:
        print("        " + line)

    services.stop()
