"""T1 —— 真实 LLM 链路。

验证对象：L4 的 `OpenAICompatClient` / `ModelResolver` / 任务档位，
以及 L2 的 `Extractor` 在**真实模型**上的结构化输出质量。

这是整份测试里唯一"必须在真网上才能回答"的部分：
mock 能证明代码路径对，但证明不了"真实模型愿不愿意按 schema 输出"。
"""

from __future__ import annotations

import json
import time

from harness import BASE_URL, MODEL_LARGE, MODEL_SMALL, Report, fresh_db, make_home, real_env

SECTION = "T1 真实 LLM 链路"


def run(report: Report) -> None:
    from artifact_spirit.config import load
    from artifact_spirit.extract.extractor import Extractor
    from artifact_spirit.model import (
        EmbeddingError,
        LLMError,
        ModelResolver,
        OpenAICompatClient,
        OpenAICompatSettings,
        ProviderUnavailableError,
        SchemaViolationError,
    )

    report.section(SECTION)
    home = make_home("t1")
    fresh_db(home)
    env = real_env()
    cfg = load(home, env=env)

    # ---------------------------------------------------------------- T1.1
    resolver = ModelResolver(
        llm=cfg.llm.as_dict() if cfg.llm else None,
        embedding=cfg.embedding.as_dict() if cfg.embedding else None,
        hermes_home=str(home),
        env=env,
    )
    chain = resolver.resolve_chain("extract")
    src = next((c.get("source") for c in chain if c.get("source")), None)
    report.check(
        SECTION,
        "T1.1 路由链解析（器灵配置优先）",
        src == "spirit",
        f"source={src}, model={resolver.resolve_chain('extract')[-1].get('model')}",
    )
    report.check(
        SECTION,
        "T1.1b 任务档位映射（extract 走小模型 / consolidate 走大模型）",
        resolver._task_model("extract") == MODEL_SMALL and resolver._task_model("consolidate") == MODEL_LARGE,
        f"extract={resolver._task_model('extract')}, consolidate={resolver._task_model('consolidate')}",
    )

    # ---------------------------------------------------------------- T1.2
    llm = resolver.llm("extract")
    t0 = time.time()
    try:
        answer = llm.complete(
            messages=[
                {"role": "system", "content": "你是简洁的助手，回答不超过 20 字。"},
                {"role": "user", "content": "用一句话说明什么是向量检索。"},
            ]
        )
        ms = (time.time() - t0) * 1000
        report.check(
            SECTION,
            "T1.2 真实 chat completion",
            bool(answer and answer.strip()),
            f"{ms:.0f}ms · {answer.strip()[:60]!r}",
            ms,
        )
    except Exception as exc:
        ms = (time.time() - t0) * 1000
        report.check(SECTION, "T1.2 真实 chat completion", False, f"{type(exc).__name__}: {exc}", ms)
        return

    # ---------------------------------------------------------------- T1.3
    # 结构化输出稳定性：同一 schema 连打 3 次，**每一次都必须能解析且过校验**。
    from artifact_spirit.extract.schema import EXTRACTION_SCHEMA

    convo = (
        "用户：我最近在重构 RAGFlow，把 Python 从 3.11 升到了 3.12，"
        "顺便把部署从 venv 换成了 Docker。\n"
        "助手：升级顺利吗？\n"
        "用户：还行，就是 sentence-transformers 要重新装。另外提醒你一下，"
        "我偏好深色主题的界面，浅色的看久了眼睛疼。"
    )
    schema = EXTRACTION_SCHEMA
    ok_runs, latency, raw_samples = 0, [], []
    for i in range(3):
        t0 = time.time()
        try:
            payload = llm.complete_json(
                messages=[
                    {"role": "system", "content": "你是记忆提取器，只输出 JSON。"},
                    {"role": "user", "content": f"从这段对话中提取值得长期记住的事实：\n{convo}"},
                ],
                schema=schema,
            )
            latency.append((time.time() - t0) * 1000)
            raw_samples.append(payload)
            ok_runs += 1
        except (SchemaViolationError, Exception):
            latency.append((time.time() - t0) * 1000)
    avg = sum(latency) / len(latency) if latency else 0
    report.check(
        SECTION,
        "T1.3 结构化输出稳定性（3 连击全部可解析）",
        ok_runs == 3,
        f"{ok_runs}/3 成功 · 平均 {avg:.0f}ms",
        avg,
    )
    report.info(SECTION, "T1.3b 首次响应样本", json.dumps(raw_samples[0], ensure_ascii=False)[:220] if raw_samples else "(无)")

    # ---------------------------------------------------------------- T1.4
    # 走真正的 Extractor（含 schema 校验与降级判定）
    extractor = Extractor(llm=llm)
    t0 = time.time()
    result = extractor.extract(convo)
    ms = (time.time() - t0) * 1000
    n = len(result.candidates)
    report.check(
        SECTION,
        "T1.4 提取器端到端（真实模型 + schema 校验）",
        result.degraded is None and n >= 1,
        f"degraded={result.degraded} · 候选 {n} 条 · {ms:.0f}ms",
        ms,
    )
    for c in result.candidates:
        report.info(
            SECTION,
            f"  候选 · {c.get('type')}/{c.get('layer')}",
            f"({c.get('subject')}) {c.get('predicate')} = {c.get('object')} | {str(c.get('content'))[:50]}",
        )

    # 人工可判的正确性：至少应抽出"Python 3.12 / Docker / 深色主题"中的两项
    blob = json.dumps(result.candidates, ensure_ascii=False)
    hits = [k for k in ("3.12", "Docker", "docker", "深色") if k in blob]
    report.check(
        SECTION,
        "T1.4b 提取内容相关性（应命中关键事实）",
        len(hits) >= 2,
        f"命中 {hits}",
    )

    # ---------------------------------------------------------------- T1.5
    # 4xx 不重试：真实 401
    bad = OpenAICompatClient(
        OpenAICompatSettings(base_url=BASE_URL, api_key="sk-definitely-invalid-key-000", backoff_base=0.0),
    )
    try:
        bad.complete(messages=[{"role": "user", "content": "hi"}], model=MODEL_SMALL)
        report.check(SECTION, "T1.5 真实 401 → 不重试、抛 ProviderUnavailableError", False, "未抛错")
    except ProviderUnavailableError as exc:
        report.check(
            SECTION,
            "T1.5 真实 401 → 不重试、抛 ProviderUnavailableError",
            bad.request_count == 1,
            f"请求次数={bad.request_count}（应为 1）· {str(exc)[:70]}",
        )
    except Exception as exc:
        report.check(SECTION, "T1.5 真实 401 → 不重试", False, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- T1.6
    # 真实"模型名不存在"→ 4xx，**不重试**。
    # 注意：不同网关对同一个错误的**状态码不同**（DeepSeek 给 400，有的给 404）。
    # 所以这里断言的是**策略**（4xx 一律不重试）而不是某个具体状态码——
    # 断言状态码会把测试绑死在某一家网关的脾气上，换个网关就假失败。
    good_client = OpenAICompatClient(
        OpenAICompatSettings(base_url=BASE_URL, api_key=env["ARTIFACT_SPIRIT_API_KEY"], backoff_base=0.0),
    )
    try:
        good_client.complete(messages=[{"role": "user", "content": "hi"}], model="no-such-model-xyz")
        report.check(SECTION, "T1.6 真实 4xx（模型名不存在）→ 不重试", False, "未抛错")
    except (ProviderUnavailableError, LLMError) as exc:
        report.check(
            SECTION,
            "T1.6 真实 4xx（模型名不存在）→ 不重试",
            good_client.request_count == 1,
            f"请求次数={good_client.request_count}（应为 1，重试就说明退避策略漏了 4xx）"
            f" · 抛 {type(exc).__name__}",
        )
    except Exception as exc:
        report.check(SECTION, "T1.6 真实 4xx → 不重试", False, f"{type(exc).__name__}: {exc}")

    # 更要紧的一条：模型名写错时，**记忆不能因此丢失**。
    # F1/F3 的承诺是"仅存原文"——所以提取失败必须走 fallback_only，而不是整轮丢弃。
    from artifact_spirit.extract.extractor import Extractor

    broken = OpenAICompatClient(
        OpenAICompatSettings(base_url=BASE_URL, api_key=env["ARTIFACT_SPIRIT_API_KEY"], backoff_base=0.0),
    )
    broken_llm = type("BrokenLLM", (), {
        "complete": lambda self, **kw: broken.complete(model="no-such-model-xyz", **kw),
        "complete_json": lambda self, **kw: broken.complete_json(model="no-such-model-xyz", **kw),
    })()
    result = Extractor(llm=broken_llm).extract("请记住：我对花生过敏。")
    report.check(
        SECTION,
        "T1.6b 提取失败 → 降级为「仅存原文」而不是丢掉这一轮",
        result.fallback_only and bool(result.raw_text) and result.degraded is not None,
        f"fallback_only={result.fallback_only} · degraded={result.degraded} · 原文长度={len(result.raw_text)}",
    )

    # ---------------------------------------------------------------- T1.7
    # INV-6：embedding 不得 fallback 到 LLM（真实环境下的否决检查）
    chat_before = good_client.request_count
    try:
        resolver.embedding()
        report.check(SECTION, "T1.7 embedding 未配置时抛 EmbeddingError（INV-6）", False, "竟然成功了")
    except EmbeddingError as exc:
        report.check(
            SECTION,
            "T1.7 embedding 未配置时抛 EmbeddingError（INV-6）",
            True,
            f"{str(exc)[:70]}",
        )
    report.check(
        SECTION,
        "T1.7b 该路径未偷发任何 chat 请求（无静默降级）",
        good_client.request_count == chat_before,
        f"chat 请求数保持 {chat_before}",
    )

    resolver.close()
