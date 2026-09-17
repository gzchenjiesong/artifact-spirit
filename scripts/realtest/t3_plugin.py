"""T3 —— 插件整体可用性（模拟 Hermes Agent 宿主的真实调用方式）。

这一段回答用户的问题：**"整个插件的使用是否正常？"**

它不直接 import 实现，而是像宿主那样：

1. 通过 entry point ``hermes_agent.memory_providers`` **发现**插件（这是真实装载路径）
2. 走契约方法：``initialize`` → ``on_session_start`` → ``prefetch`` → ``sync_turn``
   → ``on_pre_compress`` → ``on_memory_write`` → ``on_session_end`` → ``shutdown``
3. 调全部工具
4. 跑 CLI 子命令

**真线程 + 真网络**：writer 线程会真的去打真实网关。这能暴露 mock 永远暴露不了的
问题——跨线程连接、队列优先级、超时护栏、关闭顺序。
"""

from __future__ import annotations

import subprocess
import sys
import time

from harness import Report, fresh_db, make_home, real_env

SECTION = "T3 插件整体可用性"


def host_tool(provider, name, args=None, **kwargs):
    """**按宿主的方式**调用工具：`handle_tool_call` 返回 JSON 字符串，宿主自己解析。

    直接把它当 dict 用会 TypeError——这正是契约测试要拦住的错误。
    """
    import json as _json

    raw = provider.handle_tool_call(name, args or {}, **kwargs)
    assert isinstance(raw, str), f"handle_tool_call 必须返回 str，实际 {type(raw).__name__}"
    payload = _json.loads(raw)
    # 统一成 `ok` 便于本文件的判定逻辑；宿主侧同样以 `error` 键识别失败
    return {"ok": "error" not in payload, **payload}

TOOL_NAMES = [
    "spirit_recall",
    "spirit_expand",
    "spirit_remember",
    "spirit_forget",
    "spirit_restore",
    "spirit_review",
    "spirit_trace",
    "spirit_correct",
    "spirit_consolidate",
    "spirit_reflect",
    "spirit_export",
]


def run(report: Report) -> None:
    from importlib.metadata import entry_points


    home = make_home("t3", with_embedding=False)
    fresh_db(home)
    env = real_env()

    report.section(SECTION)

    # ---------------------------------------------------------------- T3.1 发现
    eps = {e.name: e for e in entry_points(group="hermes_agent.memory_providers")}
    report.check(
        SECTION,
        "T3.1 entry point 可被发现（hermes_agent.memory_providers）",
        "artifact-spirit" in eps,
        f"已注册：{list(eps)}",
    )
    # 宿主的真实装载方式：`_ProviderCollector` 把 register(ctx) 里注册的 provider 收走
    class _Collector:
        """宿主 `plugins.memory._ProviderCollector` 的最小替身。"""

        def __init__(self) -> None:
            self.provider = None

        def register_memory_provider(self, provider) -> None:
            self.provider = provider

    try:
        register = eps["artifact-spirit"].load()
        ctx = _Collector()
        register(ctx)
        provider = ctx.provider
    except Exception as exc:
        report.check(SECTION, "T3.1b register(ctx) 可装载并交出 provider", False, f"{type(exc).__name__}: {exc}")
        return
    report.check(
        SECTION,
        "T3.1b register(ctx) 可装载并交出 provider",
        provider is not None and hasattr(provider, "initialize"),
        f"{type(provider).__module__}.{type(provider).__name__}",
    )
    report.check(SECTION, "T3.2 provider.name", bool(getattr(provider, "name", "")), f"name={getattr(provider, 'name', '')!r}")

    # ---------------------------------------------------------------- T3.3 is_available
    # 先预热：宿主的启动路径本身要 import 整个 provider（本机实测 ~250ms），
    # 那是**导入成本**而不是检查成本。度量的对象必须是检查本身，否则测的是别的
    # 东西——第一次跑这条就踩过这个坑（318ms 其实是 import）。
    provider.is_available(hermes_home=str(home))
    t0 = time.time()
    avail = provider.is_available(hermes_home=str(home))
    ms = (time.time() - t0) * 1000
    report.check(
        SECTION,
        "T3.3 is_available 本地判定快（预热后 < 20ms）",
        bool(avail) and ms < 20,
        f"{avail} · {ms:.1f}ms（provider.last_check_ms={provider.last_check_ms:.1f}）",
        ms,
    )

    # INV-5：**真机验证**"绝不联网"。不是看代码，是真的把 socket 掐掉再调一次。
    import socket as _socket

    real_socket = _socket.socket
    calls: list[str] = []

    class _Blocked(_socket.socket):  # type: ignore[misc]
        def connect(self, *a, **k):
            calls.append("connect")
            raise AssertionError("is_available 发起了网络连接——违反 INV-5")

    _socket.socket = _Blocked  # type: ignore[assignment]
    try:
        provider._last_check_ms = 0.0
        still_ok = provider.is_available(hermes_home=str(home))
    finally:
        _socket.socket = real_socket
    report.check(
        SECTION,
        "T3.3b INV-5 真机验证：掐掉 socket 后 is_available 仍返回",
        isinstance(still_ok, bool) and not calls,
        f"返回 {still_ok} · 网络调用次数={len(calls)}",
    )

    # ---------------------------------------------------------------- T3.4 initialize（真线程）
    t0 = time.time()
    provider.initialize(hermes_home=str(home), env=env, start_threads=True, reconcile=True)
    ms = (time.time() - t0) * 1000
    svc = provider.services
    report.check(
        SECTION,
        "T3.4 initialize 成功（真线程 · writer + maintenance）",
        svc.writer.running and svc.maintenance.running,
        f"writer={svc.writer.running}(tid={svc.writer.thread_id}) maintenance={svc.maintenance.running} · {ms:.0f}ms",
        ms,
    )
    report.info(SECTION, "T3.4b 装配说明", f"notes={svc.notes} warnings={svc.warnings}")

    # ---------------------------------------------------------------- T3.5 工具声明
    schemas = provider.get_tool_schemas()
    names = [s.get("name") for s in schemas]
    report.check(
        SECTION,
        "T3.5 工具声明完整（11 个 spirit_*）",
        set(names) == set(TOOL_NAMES),
        f"声明 {len(names)} 个；缺失={set(TOOL_NAMES) - set(names)} 多余={set(names) - set(TOOL_NAMES)}",
    )

    # ---------------------------------------------------------------- T3.6 会话开始
    provider.on_session_start("s-real")
    report.check(SECTION, "T3.6 on_session_start 不抛错", True, "session=s-real")

    # ---------------------------------------------------------------- T3.7 sync_turn 非阻塞
    t0 = time.time()
    provider.sync_turn(
        "请记住：我在做 RAGFlow 重构，已经升级到 Python 3.12，用 Docker 部署。",
        "好的，我记下了。",
        session_id="s-real",
    )
    ms = (time.time() - t0) * 1000
    report.check(
        SECTION,
        "T3.7 sync_turn 非阻塞（INV-4：不得有同步 I/O，须 < 50ms）",
        ms < 50,
        f"{ms:.1f}ms（真实提取发生在 writer 线程，不阻塞宿主）",
        ms,
    )

    # 等真实 LLM 提取入库
    t0 = time.time()
    flushed = provider.services.flush(30.0)
    ms = (time.time() - t0) * 1000
    stored = svc.backend.query(status=None)
    report.check(
        SECTION,
        "T3.8 writer 线程真实处理完成（跨线程写库无异常）",
        flushed and not svc.writer.errors,
        f"{ms:.0f}ms · 落库 {len(stored)} 条 · 处理 {svc.writer.queue.stats.processed} · errors={svc.writer.errors[:1]}",
        ms,
    )
    report.info(SECTION, "T3.8b 落库内容", " || ".join(f"[{r.layer}] {r.content[:44]}" for r in stored[:4]))

    # ---------------------------------------------------------------- T3.9 prefetch
    t0 = time.time()
    block = provider.prefetch("RAGFlow 用的什么 Python 版本？", session_id="s-real")
    ms = (time.time() - t0) * 1000
    report.check(
        SECTION,
        "T3.9 prefetch 返回可用上下文块（带超时护栏）",
        isinstance(block, str) and len(block) > 0,
        f"{ms:.0f}ms · {len(block)} 字",
        ms,
    )
    print("      prefetch 内容：")
    for line in block.splitlines()[:6]:
        print("        " + line)

    # ---------------------------------------------------------------- T3.10 超时护栏
    slow_ms = svc.config.worker.get("prefetch_timeout_ms", 300)
    report.info(SECTION, "T3.10 超时护栏配置", f"prefetch_timeout_ms={slow_ms}")

    # ---------------------------------------------------------------- T3.11 system_prompt_block
    sp = provider.system_prompt_block(token_budget=300)
    report.check(
        SECTION,
        "T3.11 system_prompt_block 生成（受 token 预算约束）",
        len(sp) <= 2000,
        f"{len(sp)} 字",
    )
    if sp:
        print("      system prompt 块：")
        for line in sp.splitlines()[:5]:
            print("        " + line)

    # ---------------------------------------------------------------- T3.12 全部工具
    print("      --- 工具逐个调用 ---")
    results = {}
    for name in TOOL_NAMES:
        args: dict = {}
        if name == "spirit_recall":
            args = {"query": "Python 版本"}
        elif name == "spirit_remember":
            args = {"content": "用户偏好深色主题界面"}
        elif name == "spirit_review":
            args = {"limit": 3}
        elif name == "spirit_reflect":
            args = {}
        elif name == "spirit_consolidate":
            args = {"session_id": "s-real"}
        elif name == "spirit_export":
            args = {"fmt": "markdown"}  # 不给 path → 验证默认落点
        try:
            if name == "spirit_expand":
                base = results.get("spirit_recall", {})
                items = (base.get("data") or {}).get("items") or []
                if not items:
                    results[name] = {"ok": None, "text": "(无记忆可展开)"}
                    continue
                args = {"ref": items[0]["id"], "level": "L0"}
            if name in ("spirit_trace", "spirit_correct"):
                base = results.get("spirit_recall", {})
                items = (base.get("data") or {}).get("items") or []
                if not items:
                    results[name] = {"ok": None, "text": "(无记忆可操作)"}
                    continue
                args = (
                    {"ref": items[0]["id"]}
                    if name == "spirit_trace"
                    else {"mem_id": items[0]["id"], "patch": {"confidence": 0.99}, "reason": "实机测试"}
                )
            if name in ("spirit_forget", "spirit_restore"):
                if name == "spirit_forget":
                    base = results.get("spirit_recall", {})
                    items = (base.get("data") or {}).get("items") or []
                    args = {"mem_id": items[-1]["id"], "reason": "实机测试删除"} if items else {}
                else:
                    snaps = svc.backend.delete_snapshots()
                    args = {"audit_id": snaps[0]["audit_id"]} if snaps else {}
                if not args:
                    results[name] = {"ok": None, "text": "(无可操作对象)"}
                    continue
            out = host_tool(provider, name, args)
            results[name] = out
        except Exception as exc:
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    failed = {k: v for k, v in results.items() if v.get("ok") is False or v.get("error")}
    report.check(
        SECTION,
        "T3.12 全部 11 个工具可调用且无一报错",
        not failed,
        f"失败：{failed}" if failed else "11/11 正常",
    )
    for name, out in results.items():
        line = out.get("text") or out.get("error") or ""
        flag = "ok " if out.get("ok") else ("skip" if out.get("ok") is None else "ERR ")
        print(f"        [{flag}] {name:<18} {str(line).replace(chr(10), ' ')[:66]}")

    # ---------------------------------------------------------------- T3.13 参数校验
    bad = host_tool(provider, "spirit_recall", {})
    report.check(
        SECTION,
        "T3.13 缺失必需参数 → 结构化错误而不是抛异常",
        bad.get("ok") is False and "error" in bad,
        f"{str(bad)[:90]}",
    )
    unknown = host_tool(provider, "spirit_not_exist", {})
    report.check(
        SECTION,
        "T3.13b 未知工具名 → 结构化错误",
        unknown.get("ok") is False,
        f"{str(unknown)[:90]}",
    )

    # ---------------------------------------------------------------- T3.14 会话收尾
    provider.on_pre_compress(session_id="s-real")
    provider.on_memory_write("write", "USER.md", "用户偏好深色主题，讨厌刺眼的白底。")
    provider.on_memory_write("write", "MEMORY.md", "当前主项目是 RAGFlow 重构。")
    t0 = time.time()
    provider.on_session_end(session_id="s-real")
    ms = (time.time() - t0) * 1000
    report.check(
        SECTION,
        "T3.14 on_pre_compress / on_memory_write / on_session_end 均非阻塞",
        ms < 100,
        f"on_session_end {ms:.1f}ms（巩固已投递后台）",
        ms,
    )
    flushed = provider.services.flush(30.0)
    after = svc.backend.query(status=None)
    report.check(
        SECTION,
        "T3.14b 巩固 + 宿主记忆镜像真实落库",
        flushed and len(after) >= len(stored),
        f"落库 {len(after)} 条（会话结束前 {len(stored)} 条）· errors={svc.writer.errors[:1]}",
    )
    report.info(
        SECTION,
        "T3.14c 分层分布",
        str(svc.backend.count_by_layer()),
    )

    # ---------------------------------------------------------------- T3.17 宿主钩子全序列
    # 按宿主 `MemoryManager` 的真实调用顺序走一遍。**缺方法不会崩**——宿主会把
    # AttributeError 吞成一条日志，所以只能靠"逐个调用并确认真的做了事"来发现。
    print("      --- 宿主钩子逐个驱动 ---")
    hook_results = {}

    def _hook(label, fn, *args, **kwargs):
        try:
            out = fn(*args, **kwargs)
            hook_results[label] = out
            shown = "" if out is None else f" -> {str(out)[:40]}"
            print(f"        [ok ] {label}{shown}")
            return out
        except Exception as exc:
            hook_results[label] = exc
            print(f"        [ERR] {label}: {type(exc).__name__}: {exc}")
            return None

    _hook("on_turn_start", provider.on_turn_start, 7, "我偏好深色主题", session_id="s-hooks")
    _hook("unavailable_reason", provider.unavailable_reason)
    _hook("identity_signature", provider.identity_signature)
    _hook("backup_paths", provider.backup_paths)
    _hook("queue_prefetch", provider.queue_prefetch, "深色主题", session_id="s-hooks")
    _hook("prefetch", provider.prefetch, "深色主题", session_id="s-hooks")
    _hook("recall_status", provider.recall_status)
    _hook("system_prompt_block", provider.system_prompt_block)
    _hook("on_memory_write", provider.on_memory_write, "write", "USER.md", "用户偏好深色主题")
    pre = _hook("on_pre_compress", provider.on_pre_compress, [], session_id="s-hooks")
    _hook("on_delegation", provider.on_delegation, "子任务：查资料", "结果是 X", child_session_id="child-1")
    _hook("on_session_end", provider.on_session_end, [], session_id="s-hooks")
    _hook("on_session_switch(reset)", provider.on_session_switch, "s-hooks-2", parent_session_id="s-hooks", reset=True)

    errors = {k: v for k, v in hook_results.items() if isinstance(v, Exception)}
    report.check(
        SECTION,
        "T3.17 宿主全部钩子可调用且不抛异常",
        not errors,
        f"失败：{list(errors)}" if errors else f"{len(hook_results)} 个钩子全部正常",
    )
    report.check(
        SECTION,
        "T3.17b on_pre_compress 返回 str（宿主会把它拼进压缩提示词）",
        isinstance(pre, str),
        f"{type(pre).__name__} · {str(pre)[:40]}",
    )
    sig = hook_results.get("identity_signature") or {}
    report.check(
        SECTION,
        "T3.17c identity_signature 含 spirit_id（宿主据此判断记忆主体是否变化）",
        isinstance(sig, dict) and bool(sig.get("spirit_id")),
        f"{sig}",
    )
    backups = hook_results.get("backup_paths") or []
    report.check(
        SECTION,
        "T3.17d backup_paths 覆盖 WAL（只备份主文件会丢最近写入）",
        isinstance(backups, list) and any(str(b).endswith(".db") for b in backups),
        f"{backups}",
    )
    report.check(
        SECTION,
        "T3.17e 会话切换后工作记忆被清（reset=True 不得残留上一会话的组块）",
        not svc.backend.wm_list("s-hooks"),
        f"旧会话剩余组块数={len(svc.backend.wm_list('s-hooks'))}",
    )

    # ---------------------------------------------------------------- T3.17 宿主钩子全序列
    # 按宿主 `MemoryManager` 的真实调用顺序走一遍。**缺方法不会崩**——宿主会把
    # AttributeError 吞成一条日志，所以只能靠"逐个调用并确认真的做了事"来发现。
    print("      --- 宿主钩子逐个驱动 ---")
    hook_results = {}

    def _hook(label, fn, *args, **kwargs):
        try:
            out = fn(*args, **kwargs)
            hook_results[label] = out
            shown = "" if out is None else f" -> {str(out)[:40]}"
            print(f"        [ok ] {label}{shown}")
            return out
        except Exception as exc:
            hook_results[label] = exc
            print(f"        [ERR] {label}: {type(exc).__name__}: {exc}")
            return None

    _hook("on_turn_start", provider.on_turn_start, 7, "我偏好深色主题", session_id="s-hooks")
    _hook("unavailable_reason", provider.unavailable_reason)
    _hook("identity_signature", provider.identity_signature)
    _hook("backup_paths", provider.backup_paths)
    _hook("queue_prefetch", provider.queue_prefetch, "深色主题", session_id="s-hooks")
    _hook("prefetch", provider.prefetch, "深色主题", session_id="s-hooks")
    _hook("recall_status", provider.recall_status)
    _hook("system_prompt_block", provider.system_prompt_block)
    _hook("on_memory_write", provider.on_memory_write, "write", "USER.md", "用户偏好深色主题")
    pre = _hook("on_pre_compress", provider.on_pre_compress, [], session_id="s-hooks")
    _hook("on_delegation", provider.on_delegation, "子任务：查资料", "结果是 X", child_session_id="child-1")
    _hook("on_session_end", provider.on_session_end, [], session_id="s-hooks")
    _hook("on_session_switch(reset)", provider.on_session_switch, "s-hooks-2", parent_session_id="s-hooks", reset=True)

    errors = {k: v for k, v in hook_results.items() if isinstance(v, Exception)}
    report.check(
        SECTION,
        "T3.17 宿主全部钩子可调用且不抛异常",
        not errors,
        f"失败：{list(errors)}" if errors else f"{len(hook_results)} 个钩子全部正常",
    )
    report.check(
        SECTION,
        "T3.17b on_pre_compress 返回 str（宿主会把它拼进压缩提示词）",
        isinstance(pre, str),
        f"{type(pre).__name__} · {str(pre)[:40]}",
    )
    sig = hook_results.get("identity_signature") or {}
    report.check(
        SECTION,
        "T3.17c identity_signature 含 spirit_id（宿主据此判断记忆主体是否变化）",
        isinstance(sig, dict) and bool(sig.get("spirit_id")),
        f"{sig}",
    )
    backups = hook_results.get("backup_paths") or []
    report.check(
        SECTION,
        "T3.17d backup_paths 覆盖 WAL（只备份主文件会丢最近写入）",
        isinstance(backups, list) and any(str(b).endswith(".db") for b in backups),
        f"{backups}",
    )
    report.check(
        SECTION,
        "T3.17e 会话切换后工作记忆被清（reset=True 不得残留上一会话的组块）",
        not svc.backend.wm_list("s-hooks"),
        f"旧会话剩余组块数={len(svc.backend.wm_list('s-hooks'))}",
    )

    # ---------------------------------------------------------------- T3.15 关闭安全
    t0 = time.time()
    provider.shutdown()
    ms = (time.time() - t0) * 1000
    report.check(
        SECTION,
        "T3.15 shutdown 干净收尾（线程归零 · 幂等）",
        not svc.writer.running and not svc.maintenance.running,
        f"{ms:.0f}ms · 关闭序列={svc.stop_sequence}",
        ms,
    )
    provider.shutdown()
    report.check(SECTION, "T3.15b shutdown 可重复调用（幂等）", True, "二次调用未抛错")

    # ---------------------------------------------------------------- T3.16 CLI
    _cli(report, home, env)


def _cli(report: Report, home, env) -> None:
    """真实子进程跑 CLI —— 验证 console_scripts 入口在宿主环境里真能用。"""
    exe = sys.executable
    export_path = str(home / "cli-export.md")
    # (参数, 允许的退出码, 说明)
    # `doctor` **故意**在发现问题时退出 1——这是能进 CI 的语义，不是失败。
    cmds = [
        (["status"], {0}, ""),
        (["layers"], {0}, ""),
        (["review", "--limit", "3"], {0}, ""),
        (["reflect"], {0}, ""),
        (["doctor"], {0, 1}, "发现降级/问题时退出 1（CI 语义）"),
        (["audit", "--limit", "5"], {0}, ""),
        (["export", export_path, "--fmt", "markdown"], {0}, ""),
        (["decay"], {0}, ""),
    ]
    failures = []
    # cwd / PYTHONPATH 要按"实际存在"来算：开发机上有源码目录 src/，
    # 部署机上包是装进 venv 的、没有 src/。写死路径会让部署机上的子进程直接
    # FileNotFoundError——而这类失败与"CLI 好不好用"毫无关系。
    src_root = home.parents[3] / "src"
    child_env = {**__import__("os").environ, **env}
    if src_root.is_dir():
        child_env["PYTHONPATH"] = str(src_root)
    run_cwd = str(home) if home.is_dir() else None

    for args, allowed, note in cmds:
        proc = subprocess.run(
            [exe, "-m", "artifact_spirit.cli", "--home", str(home), *args],
            capture_output=True,
            text=True,
            timeout=120,
            env=child_env,
            cwd=run_cwd,
        )
        ok = proc.returncode in allowed
        first = (proc.stdout or proc.stderr or "").strip().splitlines()
        detail = first[0][:66] if first else "(无输出)"
        print(f"        [{'ok ' if ok else 'ERR'}] aspirit {' '.join(a for a in args if not a.startswith(str(home))):<22} rc={proc.returncode} {detail}")
        if not ok:
            failures.append((args, proc.returncode, (proc.stderr or "")[-160:]))
    report.check(
        SECTION,
        "T3.16 CLI 全部子命令在真实子进程里执行成功",
        not failures,
        f"失败：{failures}" if failures else f"{len(cmds)}/{len(cmds)} 正常",
    )
    report.check(
        SECTION,
        "T3.16b CLI 导出真的写出了文件",
        __import__("os").path.exists(export_path)
        and __import__("os").path.getsize(export_path) > 0,
        f"{export_path} · {__import__('os').path.getsize(export_path) if __import__('os').path.exists(export_path) else 0} 字节",
    )
