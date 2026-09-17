"""远端实机装机验证（在**宿主所在的那台机器上**运行）。

它回答的问题不是"代码对不对"（那是单测的事），而是**"装在这台机器上，能不能用"**：

1. entry point 是否真被宿主的环境看见（装错 venv 是这一步最常见的失败）
2. `register(ctx)` 是否真能交出 provider
3. provider 是否满足宿主契约的**全部**方法（缺方法不报错，只静默失效）
4. 配置能否被装载、模型链能否解析
5. 可选：真跑一轮对话，确认提取与召回确实工作

用法（在远端）::

    python verify_host_install.py --hermes-home ~/.hermes
    python verify_host_install.py --hermes-home ~/.hermes --live     # 真跑一轮（需要 key）

**没有 --live 时绝不发起网络请求**——装机器不该因为网关暂时不通就报错。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

PASS, FAIL, INFO = "[PASS]", "[FAIL]", "[info]"
_results: list[tuple[bool | None, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def info(name: str, detail: str = "") -> None:
    _results.append((None, name, detail))
    print(f"  {INFO} {name}" + (f"  — {detail}" if detail else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="器灵装机验证")
    parser.add_argument("--hermes-home", required=True, help="宿主的 HERMES_HOME")
    parser.add_argument("--live", action="store_true", help="真跑一轮对话（需要模型凭据）")
    parser.add_argument("--key-env", default="ARTIFACT_SPIRIT_API_KEY", help="密钥所在的环境变量名")
    args = parser.parse_args(argv)

    home = Path(args.hermes_home).expanduser()
    print("=" * 72)
    print("器灵 · 远端装机验证")
    print("=" * 72)
    print(f"  HERMES_HOME : {home}")
    print(f"  Python      : {sys.executable}")
    print(f"  解释器版本  : {sys.version.split()[0]}")

    print("\n[1] 宿主环境")
    check("HERMES_HOME 存在", home.is_dir(), str(home))
    try:
        import agent.memory_provider as host_mp  # type: ignore[import-not-found]

        info("宿主可 import", host_mp.__file__ or "")
        host_available = True
    except Exception as exc:
        info("宿主不可 import（独立环境也能验证插件本身）", f"{type(exc).__name__}")
        host_available = False

    print("\n[2] 插件是否被这个环境看见")
    eps = {}
    for group in ("hermes_agent.memory_providers",):
        for ep in importlib.metadata.entry_points(group=group):
            eps[ep.name] = ep
    check(
        "entry point 已注册",
        "artifact-spirit" in eps,
        f"发现：{sorted(eps) or '（无）'}；若为空说明包装进了别的 venv",
    )
    if "artifact-spirit" not in eps:
        _summary()
        return 1

    try:
        register = eps["artifact-spirit"].load()
        info("entry point 可加载", f"{register.__module__}")
    except Exception as exc:
        check("entry point 可加载", False, f"{type(exc).__name__}: {exc}")
        _summary()
        return 1

    print("\n[3] register(ctx) 与契约完整性")

    class _Collector:
        """宿主 `_ProviderCollector` 的最小替身。"""

        def __init__(self) -> None:
            self.provider = None

        def register_memory_provider(self, provider) -> None:
            self.provider = provider

    ctx = _Collector()
    try:
        register(ctx)
    except Exception as exc:
        check("register(ctx) 执行成功", False, f"{type(exc).__name__}: {exc}")
        _summary()
        return 1
    provider = ctx.provider
    check("register(ctx) 交出了 provider", provider is not None)
    if provider is None:
        _summary()
        return 1

    if host_available:
        import agent.memory_provider as host_mp

        missing = [
            n
            for n in dir(host_mp.MemoryProvider)
            if not n.startswith("_")
            and callable(getattr(host_mp.MemoryProvider, n, None))
            and not callable(getattr(type(provider), n, None))
        ]
        check("宿主契约方法全部实现", not missing, f"缺失：{missing}" if missing else "全部齐备")
    else:
        # 无法取到宿主 ABC 时，退化为"按已知清单核对"
        # `name` 是**属性**（宿主 ABC 里是 property），其余是方法——分开核对，
        # 混在一起会把"属性不是可调用对象"误报成"缺失"。
        expected = [
            "is_available", "initialize", "get_tool_schemas", "handle_tool_call",
            "get_config_schema", "save_config", "prefetch", "queue_prefetch", "recall_status",
            "sync_turn", "system_prompt_block", "on_turn_start", "on_session_end",
            "on_session_switch", "on_pre_compress", "on_delegation", "on_memory_write",
            "shutdown", "unavailable_reason", "identity_signature", "backup_paths",
        ]
        missing = [n for n in expected if not callable(getattr(provider, n, None))]
        if not getattr(provider, "name", None):
            missing.append("name")
        check("契约方法（清单核对）全部实现", not missing, f"缺失：{missing}" if missing else f"{len(expected)} 项方法 + name 齐备")

    check("name 正确", getattr(provider, "name", "") == "artifact-spirit", f"{getattr(provider, 'name', '')!r}")
    schema = provider.get_config_schema()
    check("配置 schema 是扁平 list（宿主面板需要）", isinstance(schema, list) and bool(schema), f"{len(schema) if isinstance(schema, list) else type(schema).__name__} 项")
    raw = provider.handle_tool_call("spirit_recall", {"query": "自检"})
    check("工具返回 JSON 字符串（宿主契约）", isinstance(raw, str) and _is_json(raw), f"{type(raw).__name__}")

    print("\n[4] 可用性与配置")
    avail = provider.is_available(hermes_home=str(home))
    check("is_available 返回 True", avail, f"last_check_ms={provider.last_check_ms:.2f}")
    if not avail:
        info("不可用原因", provider.unavailable_reason() or "（未给出）")

    cfg_path = home / "artifact-spirit.toml"
    info("配置文件", f"{cfg_path}（{'存在' if cfg_path.exists() else '缺失 → 走宿主的模型配置 fallback'}）")

    print("\n[5] 初始化")
    try:
        provider.initialize(hermes_home=str(home), start_threads=False, reconcile=True)
        svc = provider.services
        check("initialize 成功", True, f"notes={svc.notes}")
        info("数据库", str(svc.backend.path))
        from artifact_spirit.observability import status as _status

        st = _status(svc)
        info("模型链", json.dumps(st.get("models", {}), ensure_ascii=False)[:160])
        info("降级告警", str(st.get("degradations") or "无"))
    except Exception as exc:
        check("initialize 成功", False, f"{type(exc).__name__}: {exc}")
        _summary()
        return 1

    if args.live:
        print("\n[6] 真跑一轮（需要模型凭据）")
        _live(provider, args.key_env)
    else:
        info("跳过真跑（加 --live 启用）", "验证安装不依赖网络")

    try:
        provider.shutdown()
        check("shutdown 干净收尾", True)
    except Exception as exc:
        check("shutdown 干净收尾", False, f"{type(exc).__name__}: {exc}")

    return _summary()


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def _live(provider, key_env: str) -> None:
    """真跑一轮。

    **必须区分"提取成功"与"降级成仅存原文"**——两者的落库条数看起来一样（都 +1），
    把后者报成成功，等于用一次假通过换掉一个真信号。所以这里先看 LLM 到底可不可用。
    """
    import os

    services = provider.services
    configured_env = ""
    if services.config.llm is not None:
        configured_env = services.config.llm.api_key_env or ""
    effective_env = configured_env or key_env

    if not os.environ.get(effective_env):
        check(
            "模型凭据已注入",
            False,
            f"环境变量 {effective_env} 未设置（配置里声明的是 {configured_env or '（未声明）'}）——真跑无法进行",
        )
        return
    check("模型凭据已注入", True, f"{effective_env}=（已设置，值不打印）")

    llm_ok = services.core.llm is not None or services.resolver.llm_available()
    info("LLM 链可用", "是" if llm_ok else "否 → 本轮会降级为「仅存原文」")

    from artifact_spirit.core import RecallQuery, TurnEvent

    before = len(services.backend.query(status=None))
    intents = services.core.ingest_turn(
        TurnEvent(
            session_id="verify",
            user="请记住：我在做 RAGFlow 重构，已经升级到 Python 3.12，用 Docker 部署。",
            assistant="好的，记下了。",
            ts="2026-09-14T20:00:00+08:00",
        )
    )
    services.write_now(intents)
    after = services.backend.query(status=None)
    new_rows = after[before:]

    if llm_ok:
        structured = [r for r in new_rows if r.layer != "episodic" and r.type != "event"]
        check(
            "真实提取产出结构化记忆",
            bool(structured),
            f"{before} → {len(after)} 条，其中结构化 {len(structured)} 条",
        )
    else:
        check(
            "LLM 不可用时仍保住原文（记忆不丢）",
            bool(new_rows) and all(r.layer == "episodic" for r in new_rows),
            f"{before} → {len(after)} 条，全部为原文兜底（这是设计内的降级，不是失败）",
        )
    for r in new_rows:
        info(f"  + [{r.layer}/{r.type}]", r.content[:60])

    hits = services.core.recall(RecallQuery(text="Python 版本", session_id="verify", top_k=5))
    check("真实召回有结果", bool(hits), f"{len(hits)} 条：{[h.record.content[:20] for h in hits[:2]]}")
    if services.core.embedding is None:
        info("召回走的是 BM25 降级路径", "embedding 未配置——装上后自动恢复向量路")


def _summary() -> int:
    failed = [r for r in _results if r[0] is False]
    total = len([r for r in _results if r[0] is not None])
    print("\n" + "=" * 72)
    print(f"结论：{total - len(failed)}/{total} 项通过" + (f"，失败 {len(failed)} 项" if failed else "，全部通过"))
    for _, name, detail in failed:
        print(f"  ✗ {name} — {detail}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
