"""契约一致性核对：把我的 provider 与**宿主真实 ABC** 逐条比对。

宿主 ABC 来自 `NousResearch/hermes-agent` 的 `agent/memory_provider.py`
（vender 在 `vendor/host_memory_provider.py`，来源可在文件中核对）。

这份检查回答的是"插件装到真实宿主上会不会出问题"——
**方法缺失会让宿主的钩子静默失效，返回值类型不符会让工具结果与配置面板直接坏掉。**

    python scripts/realtest/contract_diff.py            # 只用 vendor 的副本
    python scripts/realtest/contract_diff.py --fetch    # 重新从 GitHub 拉最新版再比
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = HERE / "vendor" / "host_memory_provider.py"
UPSTREAM = "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/agent/memory_provider.py"

sys.path.insert(0, str(ROOT / "src"))


def load_host_abc():
    if "--fetch" in sys.argv:
        print(f"重新拉取宿主 ABC：{UPSTREAM}")
        raw = urllib.request.urlopen(UPSTREAM, timeout=30).read().decode("utf-8")
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        VENDOR.write_text(raw, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("host_memory_provider", VENDOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # 必须先登记进 sys.modules：dataclass 装饰器会回头查 `sys.modules[cls.__module__]`，
    # 不登记就会在 @dataclass 处炸掉（与宿主代码无关，纯属加载方式问题）。
    sys.modules["host_memory_provider"] = module
    spec.loader.exec_module(module)
    return module.MemoryProvider


def main() -> int:
    Host = load_host_abc()
    from artifact_spirit.provider import ArtifactSpiritProvider as Mine

    host_names = {
        n for n in dir(Host) if not n.startswith("_") and callable(getattr(Host, n, None))
    }

    print(f"{'宿主方法':<24}{'宿主签名':<56}{'我的实现'}")
    print("-" * 110)

    missing: list[str] = []
    for name in sorted(host_names):
        host_sig = str(inspect.signature(getattr(Host, name)))
        mine = getattr(Mine, name, None)
        if mine is None or not callable(mine):
            mark = "❌ 未实现"
            missing.append(name)
            mine_sig = mark
        else:
            mine_sig = str(inspect.signature(mine))
        print(f"{name:<24}{host_sig[:54]:<56}{mine_sig[:44]}")

    print()
    if missing:
        print(f"❌ 缺失 {len(missing)} 个宿主方法：{missing}")
    else:
        print("✅ 宿主 ABC 的方法全部已实现")

    # 返回值类型的静态核查（宿主明确要求 str 的两处）
    print("\n返回值契约：")
    for name, expected, why in (
        ("handle_tool_call", "str", "宿主按 JSON 字符串解析工具结果"),
        ("on_pre_compress", "str", "v2 checkpoint 期望回传交接内容"),
    ):
        fn = getattr(Mine, name, None)
        if fn is None:
            print(f"  ❌ {name}：未实现（期望返回 {expected}）")
            continue
        ann = inspect.signature(fn).return_annotation
        ok = ann is not inspect.Signature.empty and "str" in str(ann)
        print(f"  {'✅' if ok else '❌'} {name} -> {ann if ann is not inspect.Signature.empty else '(无标注)'}（期望 {expected}：{why}）")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
