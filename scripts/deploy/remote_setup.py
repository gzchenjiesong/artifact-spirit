"""在新 profile 里为器灵做配置（**幂等、可回滚、只碰指定 profile**）。

设计上的三条纪律：

1. **只碰传给它的那个 profile 家目录**——不读、不写、不猜其他 profile。
2. **改任何既有文件前先备份**，且只改自己要改的那一个键；YAML 用解析器改，
   不用文本替换（注释与缩进都保得住）。
3. **密钥永不写进本文件**：经环境变量传入，落到 ``{home}/.env``（0600）——
   这正是宿主自己的约定（`secret: true` 的字段进 `.env`）。

用法::

    ARTIFACT_SPIRIT_API_KEY=xxx python remote_setup.py --profile-home /root/.hermes/profiles/spirit
    python remote_setup.py --profile-home ... --no-secret     # 只写非密钥配置
    python remote_setup.py --profile-home ... --rollback      # 从备份还原
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
from datetime import datetime
from pathlib import Path

BASE_URL = "https://tokenhub.tencentmaas.com/v1"
PROVIDER_NAME = "artifact-spirit"
KEY_ENV = "ARTIFACT_SPIRIT_API_KEY"

TOML_TEMPLATE = """# 器灵（Artifact Spirit）配置
# 由 remote_setup.py 生成 —— 这是**真相源**，手改这里永远生效。
# 宿主配置面板写的是同目录下的 artifact-spirit.json / artifact-spirit/config.json（镜像），
# 三者同时存在时以本文件为准。

[spirit]
name = "拾欢者·清欢"

[backend]
kind = "sqlite"
path = ""            # 留空 = {{hermes_home}}/spirit/spirit.db

[models.embedding]
provider    = "tokenhub"
base_url    = "{base_url}"
model       = "kinfra-text-embedding-4b"
api_key_env = "{key_env}"     # 只写变量名，密钥在 .env 里
dim         = 2560

[models.llm]
provider    = "tokenhub"
base_url    = "{base_url}"
api_key_env = "{key_env}"
extract     = "glm-5.3-flash"
dedup       = "glm-5.3-flash"
summarize   = "glm-5.3-flash"
consolidate = "glm-5.3"
soul        = "kimi-k3"

[recall]
top_k        = 8
token_budget = 2000
candidate_k  = 24

[salience]
threshold = 0.35

[decay]
enabled = false      # 衰减**只影响排序**，不会删除任何记忆

[worker]
write_queue_max          = 1000
maintenance_interval_min = 30
prefetch_timeout_ms      = 300
optimize_autonomous      = false
"""


def log(msg: str) -> None:
    print(f"  {msg}")


def backup(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    target = path.with_suffix(path.suffix + f".bak-{stamp}")
    shutil.copy2(path, target)
    return target


def ensure_memory_provider(config_path: Path, stamp: str) -> bool:
    """在 `memory:` 段下写入 `provider: artifact-spirit`（保留其余内容与注释）。"""
    import yaml

    raw = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config.yaml 不是映射结构：{config_path}")

    memory = data.get("memory")
    if not isinstance(memory, dict):
        memory = {}
        data["memory"] = memory

    current = memory.get("provider")
    if current == PROVIDER_NAME:
        log(f"memory.provider 已是 {PROVIDER_NAME}，无需改动")
        return False
    if current:
        raise SystemExit(
            f"该 profile 已激活了别的 provider（{current!r}）——"
            "按纪律不覆盖，请先确认是否需要切换"
        )

    bak = backup(config_path, stamp)
    if bak:
        log(f"已备份 config.yaml → {bak.name}")
    memory["provider"] = PROVIDER_NAME
    config_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    log(f"已写入 memory.provider = {PROVIDER_NAME}（内置记忆保持开启，两者共存）")
    return True


def ensure_toml(home: Path) -> bool:
    path = home / "artifact-spirit.toml"
    if path.exists():
        log(f"artifact-spirit.toml 已存在，保持不动（{path}）")
        return False
    path.write_text(
        TOML_TEMPLATE.format(base_url=BASE_URL, key_env=KEY_ENV), encoding="utf-8"
    )
    log(f"已生成 {path.name}")
    return True


def ensure_secret(home: Path, key: str) -> bool:
    """把密钥写进 `{home}/.env`（0600）。已存在同名则**不覆盖**。"""
    env_path = home / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if f"{KEY_ENV}=" in existing:
        log(f".env 中已存在 {KEY_ENV}，保持不动")
        return False
    line = f'\n{KEY_ENV}="{key}"\n'
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    os.chmod(env_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    log(f"已把 {KEY_ENV} 追加到 .env（权限 600，值不打印）")
    return True


def rollback(home: Path) -> int:
    """从最新的备份还原 config.yaml。"""
    backups = sorted(home.glob("config.yaml.bak-*"))
    if not backups:
        log("没有找到备份，无需还原")
        return 1
    latest = backups[-1]
    shutil.copy2(latest, home / "config.yaml")
    log(f"已从 {latest.name} 还原 config.yaml")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="为指定 profile 配置器灵")
    parser.add_argument("--profile-home", required=True)
    parser.add_argument("--no-secret", action="store_true", help="不写 .env（密钥另行注入）")
    parser.add_argument("--rollback", action="store_true", help="从备份还原 config.yaml")
    args = parser.parse_args(argv)

    home = Path(args.profile_home).expanduser()
    if not home.is_dir():
        raise SystemExit(f"profile 目录不存在：{home}")

    print("=" * 68)
    print(f"器灵 · profile 配置：{home}")
    print("=" * 68)

    if args.rollback:
        return rollback(home)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print("\n[1] 激活 provider")
    ensure_memory_provider(home / "config.yaml", stamp)

    print("\n[2] 写入器灵配置（TOML 为真相源）")
    ensure_toml(home)

    print("\n[3] 密钥注入")
    if args.no_secret:
        log("按 --no-secret 跳过（请自行保证环境变量可用）")
    else:
        key = os.environ.get(KEY_ENV, "")
        if not key:
            log(f"环境变量 {KEY_ENV} 为空 → 跳过（插件仍可用，只会降级为仅存原文 + BM25）")
        else:
            ensure_secret(home, key)

    print("\n完成。回滚：--rollback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
