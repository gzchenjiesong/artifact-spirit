"""真实环境测试的公共支撑（**零 mock**：所有网络调用都打真实网关）。

凭据经环境变量注入，**永不写进任何文件**（C1 / C7）。默认目标为 DeepSeek
（本机可用的真实 OpenAI 兼容网关）；通过 ``REALTEST_*`` 环境变量可整体切到
TokenHub，无需改动脚本。

用法::

    export REALTEST_API_KEY=...
    python scripts/realtest/run.py            # 跑全部
    python scripts/realtest/run.py t1 t3      # 只跑指定段
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# --------------------------------------------------------------------------- #
# 真实网关参数（全部可被环境变量覆盖）
# --------------------------------------------------------------------------- #

BASE_URL = os.environ.get("REALTEST_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.environ.get("REALTEST_API_KEY", "")
API_KEY_ENV = "ARTIFACT_SPIRIT_API_KEY"  # 配置文件里只写这个**变量名**

# 任务档位 → 真实模型名。DeepSeek 只提供 flash / pro 两档，故全部映射过去。
MODEL_SMALL = os.environ.get("REALTEST_MODEL_SMALL", "deepseek-flash")
MODEL_LARGE = os.environ.get("REALTEST_MODEL_LARGE", "deepseek-v4-pro")

# embedding：DeepSeek 不提供。留空 → 触发 L4/L2 的既定降级路径（INV-6）。
EMBED_BASE_URL = os.environ.get("REALTEST_EMBED_BASE_URL", "")
EMBED_MODEL = os.environ.get("REALTEST_EMBED_MODEL", "")
EMBED_DIM = os.environ.get("REALTEST_EMBED_DIM", "")


# --------------------------------------------------------------------------- #
# 结果记录
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    """一条检查项。``ok`` 为 ``None`` 表示"仅记录信息，不作判定"。"""

    section: str
    name: str
    ok: bool | None
    detail: str = ""
    elapsed_ms: float = 0.0


@dataclass
class Report:
    cases: list[Case] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def check(self, section: str, name: str, ok: bool, detail: str = "", ms: float = 0.0) -> bool:
        self.cases.append(Case(section, name, bool(ok), detail, ms))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
        return bool(ok)

    def info(self, section: str, name: str, detail: str = "", ms: float = 0.0) -> None:
        self.cases.append(Case(section, name, None, detail, ms))
        print(f"  [info] {name}" + (f"  — {detail}" if detail else ""))

    def section(self, title: str) -> None:
        print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")

    def failed(self) -> list[Case]:
        return [c for c in self.cases if c.ok is False]

    def passed(self) -> list[Case]:
        return [c for c in self.cases if c.ok is True]

    def summary(self) -> str:
        p, f, i = len(self.passed()), len(self.failed()), len([c for c in self.cases if c.ok is None])
        total = time.time() - self.started
        return f"共 {p + f} 项判定：通过 {p}，失败 {f}；另有 {i} 项信息记录。总耗时 {total:.1f}s。"

    def dump(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "base_url": BASE_URL,
                    "models": {"small": MODEL_SMALL, "large": MODEL_LARGE},
                    "embedding": {"base_url": EMBED_BASE_URL, "model": EMBED_MODEL, "dim": EMBED_DIM},
                    "cases": [c.__dict__ for c in self.cases],
                    "summary": self.summary(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# 环境装配
# --------------------------------------------------------------------------- #


def require_key(report: Report) -> bool:
    if API_KEY:
        return True
    report.check("环境", "真实凭据已注入", False, "缺少环境变量 REALTEST_API_KEY")
    return False


def make_home(name: str, *, with_embedding: bool = False) -> Path:
    """造一个干净的 hermes_home。同一轮测试内复用（用 name 区分）。"""
    from artifact_spirit.config import save

    home = ROOT / "scripts" / "realtest" / ".home" / name
    home.mkdir(parents=True, exist_ok=True)
    values: dict[str, object] = {
        "spirit.name": "拾欢者·清欢",
        "backend.path": str(home / "spirit" / "spirit.db"),
        "models.llm.provider": "openai-compat",
        "models.llm.base_url": BASE_URL,
        "models.llm.api_key_env": API_KEY_ENV,
        "models.llm.extract": MODEL_SMALL,
        "models.llm.dedup": MODEL_SMALL,
        "models.llm.summarize": MODEL_SMALL,
        "models.llm.consolidate": MODEL_LARGE,
        "models.llm.soul": MODEL_LARGE,
    }
    if with_embedding:
        values.update(
            {
                "models.embedding.provider": "openai-compat",
                "models.embedding.base_url": EMBED_BASE_URL,
                "models.embedding.model": EMBED_MODEL,
                "models.embedding.api_key_env": API_KEY_ENV,
                "models.embedding.dim": int(EMBED_DIM) if EMBED_DIM else None,
            }
        )
    save(home, values)
    return home


def real_env() -> dict[str, str]:
    """交给器灵的环境变量视图。**只在这里出现密钥**。"""
    return {API_KEY_ENV: API_KEY}


def fresh_db(home: Path) -> None:
    db = home / "spirit" / "spirit.db"
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db) + suffix)
        if p.exists():
            p.unlink()
