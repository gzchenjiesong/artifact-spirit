"""远端集成验证：用**宿主自己的 MemoryManager** 驱动器灵。

为什么不跑 `hermes -z`：那需要给新 profile 配一个可用的模型，而那属于改用户的
模型配置。这里绕开 LLM，直接驱动宿主的**记忆管理层**——被验证的恰恰是我最需要
确认的那一段：宿主怎么调用 provider、传什么参数、返回值怎么被消费。

能证明的事：

- 宿主自己的 `plugins.memory.load_memory_provider()` 能找到并加载器灵
- `MemoryManager.add_provider()` 接受它（含"单外部 provider"约束）
- 工具的注册与路由经宿主完成（`get_all_tool_names` / `handle_tool_call`）
- 逐轮钩子（`on_turn_start` → 预取 → `sync_turn` → 会话结束）真被调用
- 结束后记忆真的落进库、且能被召回

用法::

    HERMES_HOME=/root/.hermes/profiles/spirit python host_integration.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HOME = os.environ.get("HERMES_HOME", "/root/.hermes/profiles/spirit")
SESSION = "host-integration-1"
TURN_USER = "请记住：我在做 RAGFlow 重构，已经升级到 Python 3.12，用 Docker 部署。"
TURN_ASSISTANT = "好的，记下了。"

_results: list[tuple[bool | None, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    print(f"  {'[PASS]' if ok else '[FAIL]'} {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def info(name: str, detail: str = "") -> None:
    _results.append((None, name, detail))
    print(f"  [info] {name}" + (f"  — {detail}" if detail else ""))


def _load_profile_env() -> int:
    """把 profile 的 `.env` 灌进 `os.environ`，**像宿主启动时那样**。

    为什么必须自己做这一步：宿主在 CLI 启动时会调 `hermes_cli.config.load_env()`
    读 `{hermes_home}/.env`；而本脚本直接构造 `MemoryManager`，**跳过了那条启动流程**。
    少了这一步，密钥不在环境里，器灵会安静地走进"仅存原文 + BM25 降级"路径 ——
    而验证结果**看起来仍然是全通过**。这是最危险的一类假通过：
    它测的是降级路径，却报告成"集成正常"。

    （第一次跑就踩了这个：`initialize` 的 notes 里写着 LLM 与 embedding 都不可用，
    而所有 check 都是 PASS。真实对话里那次是我手工 `source .env` 才没踩到。）

    用 ``setdefault``：调用者显式设的值优先。
    """
    try:
        from hermes_cli.config import load_env
    except Exception as exc:
        info("加载 .env", f"宿主的 load_env 不可用（{type(exc).__name__}）—— 依赖调用者的环境")
        return 0

    loaded = 0
    for key, value in (load_env() or {}).items():
        if os.environ.get(key) is None:
            os.environ[key] = str(value)
            loaded += 1
    return loaded


def main() -> int:
    print("=" * 70)
    print("用宿主真实 MemoryManager 驱动器灵（不依赖模型）")
    print("=" * 70)
    print(f"  HERMES_HOME : {HOME}")
    print(f"  Python      : {sys.executable}")

    env_loaded = _load_profile_env()
    info("已从 profile .env 注入环境变量", f"{env_loaded} 项（已有值不覆盖）")

    from agent.memory_manager import MemoryManager
    from plugins.memory import load_memory_provider

    print("\n[1] 宿主自己加载器灵")
    provider = load_memory_provider("artifact-spirit", register_skills=False)
    check("load_memory_provider('artifact-spirit') 成功", provider is not None)
    if provider is None:
        return _summary()
    info("provider", f"{type(provider).__module__}.{type(provider).__name__}")

    print("\n[2] 交给宿主的 MemoryManager")
    manager = MemoryManager()
    manager.add_provider(provider)
    names = sorted(manager.get_all_tool_names())
    check("add_provider 被接受（未被单外部约束拒绝）", any(n.startswith("spirit_") for n in names),
          f"路由到器灵的工具 {len([n for n in names if n.startswith('spirit_')])} 个")
    check(
        "宿主侧工具清单完整",
        len([n for n in names if n.startswith("spirit_")]) == 11,
        f"{[n for n in names if n.startswith('spirit_')]}",
    )

    print("\n[2b] 工具 schema 的形状（宿主会**原样**交给模型）")
    # 这一段用**宿主自己的规范化函数**，不是我们的副本。
    #
    # 为什么必须单独查：宿主把 schema 原样塞进
    # `{"type": "function", "function": schema}` 再交给模型，而 OpenAI 规范读的是
    # `function.parameters`。键名写成 `input_schema`（Anthropic 的叫法）**不会报错**——
    # 工具照样能被调用，只是模型收不到参数定义，于是"参数传不进去"。
    # 真机上就是这么栽的：单测全绿、集成验证也全绿，因为集成验证是自己构造 args
    # 直接调 `handle_tool_call`，**绕过了 schema**。
    try:
        from agent.memory_manager import normalize_tool_schema
    except Exception as exc:
        check("可导入宿主的 normalize_tool_schema", False, f"{type(exc).__name__}: {exc}")
    else:
        raw_schemas = provider.get_tool_schemas()
        problems: list[str] = []
        for raw in raw_schemas:
            norm = normalize_tool_schema(raw)
            if norm is None:
                problems.append(f"{raw!r} 无 name，宿主会跳过")
                continue
            name = norm.get("name")
            params = norm.get("parameters")
            if not isinstance(params, dict):
                problems.append(f"{name}: 缺 parameters（模型将看不到任何参数）")
                continue
            if params.get("type") != "object" or not isinstance(params.get("properties"), dict):
                problems.append(f"{name}: parameters 形状不对")
                continue
            for key in params.get("required") or []:
                if key not in params["properties"]:
                    problems.append(f"{name}: 必填参数 {key!r} 未在 properties 中定义")
        check(
            "每个工具的 parameters 都完整可解析",
            not problems,
            "；".join(problems) if problems else f"{len(raw_schemas)} 个工具齐备",
        )
        if not problems:
            sample = normalize_tool_schema(
                next(s for s in raw_schemas if s.get("name") == "spirit_remember")
            )
            info("  样例", f"spirit_remember.parameters.required = "
                           f"{sample.get('parameters', {}).get('required')}")

    print("\n[3] initialize（走宿主的 initialize_all，它会自动注入 hermes_home）")
    try:
        manager.initialize_all(SESSION)
        svc = provider.services
        check("initialize_all → provider.initialize 成功", True, f"notes={svc.notes}")
    except Exception as exc:
        check("initialize 成功", False, f"{type(exc).__name__}: {exc}")
        return _summary()

    db = Path(svc.backend.path)
    info("数据库", str(db))
    before = len(svc.backend.query(status=None))
    info("已有记忆条数", str(before))

    print("\n[4] 逐轮钩子（由宿主驱动）")
    manager.on_turn_start(1, TURN_USER)
    check("on_turn_start 经宿主调用未抛错", True)

    block = manager.build_system_prompt()
    check("build_system_prompt（含器灵的 system_prompt_block）返回 str", isinstance(block, str), f"{len(block)} 字")
    info("  提示词片段", block.replace(chr(10), " ")[:80])

    # 预取（宿主会在有超时护栏的线程里跑）
    pf = provider.prefetch("Python 版本", session_id=SESSION)
    check("prefetch 经调用返回 str", isinstance(pf, str), f"{len(pf)} 字")

    # 宿主的真实同步写入路径：它自己排到带 ctx_bound 的后台 worker 上
    manager.sync_all(TURN_USER, TURN_ASSISTANT, session_id=SESSION)
    check("sync_all 非阻塞返回", True, "宿主在自己的 worker 上执行")

    # 等后台兑现（显式 flush 比 sleep 更可靠）
    drained = manager.flush_pending(timeout=120) if hasattr(manager, "flush_pending") else True
    info("后台队列已冲刷", str(drained))
    deadline = time.time() + 60
    grew = False
    while time.time() < deadline:
        if len(svc.backend.query(status=None)) > before:
            grew = True
            break
        time.sleep(1.5)
    after = svc.backend.query(status=None)
    check("宿主投递的写入真的落库", grew, f"{before} → {len(after)} 条")
    for r in after[before:]:
        info("  + 新记忆", f"[{r.layer}/{r.type}] {r.content[:56]}")

    print("\n[5] 工具经宿主路由调用")
    raw = manager.handle_tool_call("spirit_recall", {"query": "Python 版本", "session_id": SESSION})
    check("宿主路由 handle_tool_call 返回 str", isinstance(raw, str), f"{type(raw).__name__}")
    try:
        payload = json.loads(raw)
    except ValueError:
        check("工具返回可解析的 JSON", False, raw[:120])
    else:
        blob = json.dumps(payload, ensure_ascii=False)
        hit = "Python" in blob
        check(
            "召回结果含刚写入的内容",
            hit,
            f"命中={hit} · 返回片段={str(payload.get('text') or payload)[:110]}",
        )

    unknown = manager.handle_tool_call("spirit_does_not_exist", {})
    check("未知工具经宿主得到错误而非崩溃", "error" in unknown, unknown[:80])

    print("\n[6] 会话收尾与切换")
    manager.on_session_end([{"role": "user", "content": TURN_USER}])
    manager.on_session_switch("host-integration-2", parent_session_id=SESSION, reset=True)
    check("on_session_end / on_session_switch 经宿主调用未抛错", True)
    # 清理是**经写队列异步投递**的（on_session_switch 不能阻塞宿主，INV-4/INV-6），
    # 所以要先冲刷再判定——否则测的是"投递到没到"而不是"清理有没有发生"。
    manager.flush_pending(timeout=60)
    wm_left = svc.backend.wm_list(SESSION)
    check(
        "切换后旧会话工作记忆已清（冲刷后）",
        not wm_left,
        f"剩余 {len(wm_left)} 组块" + (f"：{[(c.chunk_key, c.act_count) for c in wm_left]}" if wm_left else ""),
    )

    print("\n[7] 关闭")
    manager.shutdown_all()
    check("宿主 shutdown 干净收尾", not svc.writer.running and not svc.maintenance.running)

    return _summary()


def _summary() -> int:
    failed = [r for r in _results if r[0] is False]
    total = len([r for r in _results if r[0] is not None])
    print("\n" + "=" * 70)
    print(f"结论：{total - len(failed)}/{total} 项通过" + (f"，失败 {len(failed)} 项" if failed else "，全部通过"))
    for _, name, detail in failed:
        print(f"  ✗ {name} — {detail}")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
