"""真实环境测试入口。

**全程零 mock**：所有 LLM 调用都打真实网关，所有"关闭校验"都是真的关闭。
凭据只经环境变量注入，不落任何文件。

    export REALTEST_API_KEY=<真实 key>
    python scripts/realtest/run.py                    # 全部
    python scripts/realtest/run.py t1 t2              # 指定段
    REALTEST_BASE_URL=... REALTEST_MODEL_SMALL=...    # 切到 TokenHub

切到 TokenHub（含 embedding，可覆盖完整向量路径）::

    export REALTEST_BASE_URL=https://tokenhub.tencentmaas.com/v1
    export REALTEST_API_KEY=<tokenhub key>
    export REALTEST_MODEL_SMALL=glm-5.3-flash
    export REALTEST_MODEL_LARGE=glm-5.3
    export REALTEST_EMBED_BASE_URL=https://tokenhub.tencentmaas.com/v1
    export REALTEST_EMBED_MODEL=kinfra-text-embedding-4b
    export REALTEST_EMBED_DIM=2560
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import API_KEY, BASE_URL, MODEL_LARGE, MODEL_SMALL, Report, require_key

SECTIONS = {
    "t1": ("t1_llm", "T1 真实 LLM 链路"),
    "t2": ("t2_degraded", "T2 无 embedding 降级路径"),
    "t3": ("t3_plugin", "T3 插件整体可用性"),
    "t4": ("t4_embedding", "T4 完整向量路径（含 embedding）"),
}


def main(argv: list[str]) -> int:
    wanted = [a.lower() for a in argv[1:]] or list(SECTIONS)

    out_dir = Path(__file__).resolve().parent
    report = Report()

    print("=" * 74)
    print("Artifact Spirit · 真实环境测试")
    print("=" * 74)
    print(f"  网关      : {BASE_URL}")
    print(f"  小模型档  : {MODEL_SMALL}")
    print(f"  大模型档  : {MODEL_LARGE}")
    print(f"  embedding : {os.environ.get('REALTEST_EMBED_MODEL') or '(未配置 → 走 BM25 降级路径)'}")
    print(f"  密钥      : {'已注入（来自环境变量，未落盘）' if API_KEY else '缺失'}")

    if not require_key(report):
        print("\n缺少凭据，无法进行真实环境测试。")
        return 2

    for key in wanted:
        if key not in SECTIONS:
            print(f"跳过未知段：{key}")
            continue
        module_name, title = SECTIONS[key]
        try:
            module = __import__(module_name)
        except Exception:
            print(f"\n!! 无法导入 {module_name}")
            traceback.print_exc()
            continue
        try:
            module.run(report)
        except Exception:
            print(f"\n!! {title} 执行中断：")
            traceback.print_exc()
            report.check(title, "段内执行未中断", False, "见上方堆栈")

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    print("  " + report.summary())
    failures = report.failed()
    if failures:
        print("\n  失败项：")
        for c in failures:
            print(f"    - [{c.section}] {c.name} — {c.detail}")
    report.dump(out_dir / "last-run.json")
    print(f"\n  明细已写入：{out_dir / 'last-run.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
