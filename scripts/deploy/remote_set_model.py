"""把指定 profile 的**主对话模型**指向 TokenHub（幂等、可回滚、只碰一个 profile）。

为什么需要这个脚本，而不是手写一段 YAML：

1. **形态要对**。宿主的自定义端点有两种历史形态——旧的 ``custom_providers:`` 列表
   和现行的 ``providers:`` 字典（配置版本 v12 迁移过来的）。写错了不会报错，
   只是模型选择器里看不到、解析时静默落到别的 provider。
2. **密钥不能落明文**。``providers.<key>.key_env`` 只写**变量名**，值留在
   ``{home}/.env``（0600）。宿主自己生成的配置里常见明文 ``api_key``
   （例如 auxiliary 段），但我们不该再添一处。
3. **只碰一个 profile**。宿主有 7 个 profile 在跑，其中 4 个挂着 gateway。
   脚本只认 ``--profile-home``，不读也不猜别的 profile。

用法::

    python remote_set_model.py --profile-home /root/.hermes/profiles/spirit --check
    python remote_set_model.py --profile-home /root/.hermes/profiles/spirit
    python remote_set_model.py --profile-home ... --model glm-5.3
    python remote_set_model.py --profile-home ... --rollback
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

BASE_URL = "https://tokenhub.tencentmaas.com/v1"
PROVIDER_KEY = "tokenhub"
PROVIDER_NAME = "TokenHub"
KEY_ENV = "ARTIFACT_SPIRIT_API_KEY"
DEFAULT_MODEL = "glm-5.3-flash"

# 备用模型：同一个 provider 下可直接 `-m <model>` 切换，不必改配置。
# 选这些的理由：都是 TokenHub 上已实测可用的对话模型，能力档次不同，
# 便于在"快但弱"与"慢但强"之间现场取舍。
EXTRA_MODELS = ("glm-5.3", "kimi-k3", "minimax-m3", "mimo-v2.5-pro")


def log(msg: str) -> None:
    print(f"  {msg}")


def load_yaml(path: Path) -> dict:
    import yaml

    if not path.exists():
        raise SystemExit(f"配置文件不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} 不是映射结构（顶层是 {type(data).__name__}）")
    return data


def describe(data: dict, provider_key: str) -> str:
    """当前模型配置的一句话摘要。"""
    model = data.get("model") or {}
    providers = data.get("providers") if isinstance(data.get("providers"), dict) else {}
    entry = providers.get(provider_key) or {}
    parts = [
        f"default={model.get('default')!r}",
        f"provider={model.get('provider')!r}",
    ]
    if entry:
        parts.append(f"providers.{provider_key}.api={entry.get('api')!r}")
        parts.append(f"key_env={entry.get('key_env')!r}")
    else:
        parts.append(f"providers.{provider_key}=（未定义）")
    return " · ".join(parts)


def already_configured(
    data: dict, *, provider_key: str, base_url: str, model: str, key_env: str
) -> bool:
    """幂等判据：主模型已指向该 provider **且就是目标模型**，provider 条目齐备。

    ``model`` 必须进判据。只比 provider 的写法会让 ``--model glm-5.3``
    在已是 ``glm-5.3-flash`` 时被判为"无需改动"——**用户换模型的意图被静默忽略**，
    而脚本还报告成功。幂等要判的是"是否已达到我这次要求的状态"，
    不是"是否已经指向过同一个 provider"。
    """
    current = data.get("model") or {}
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return False
    entry = providers.get(provider_key)
    if not isinstance(entry, dict):
        return False
    return (
        current.get("provider") == provider_key
        and current.get("default") == model
        and str(entry.get("api") or "").rstrip("/") == base_url.rstrip("/")
        and entry.get("key_env") == key_env
    )


def apply(
    data: dict,
    *,
    provider_key: str,
    provider_name: str,
    base_url: str,
    model: str,
    key_env: str,
    extra_models: tuple[str, ...],
) -> dict:
    """就地改写：主模型段 + providers 条目。返回新的 *副本*，不改原对象。"""
    import copy

    out = copy.deepcopy(data)

    models = data.get("model")
    if not isinstance(models, dict):
        models = {}
    models["default"] = model
    models["provider"] = provider_key
    models["base_url"] = base_url
    out["model"] = models

    providers = out.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    entry = providers.get(provider_key)
    if not isinstance(entry, dict):
        entry = {}

    # 有序去重：目标模型在 extra_models 里也有一份，直接相加会写出重复项。
    # 用户手工加进列表的模型追加在后面，不覆盖他的选择。
    known: list[str] = []
    for candidate in (model, *extra_models, *(entry.get("models") or [])):
        if isinstance(candidate, str) and candidate and candidate not in known:
            known.append(candidate)

    entry.update(
        {
            "name": provider_name,
            "api": base_url,
            "key_env": key_env,
            "default_model": model,
            "models": known,
        }
    )
    providers[provider_key] = entry
    out["providers"] = providers
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="为指定 profile 配置 TokenHub 主模型")
    parser.add_argument("--profile-home", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"主模型（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--provider-key", default=PROVIDER_KEY)
    parser.add_argument("--provider-name", default=PROVIDER_NAME)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--key-env", default=KEY_ENV)
    parser.add_argument("--check", action="store_true", help="只报告现状，不做任何改动")
    parser.add_argument("--rollback", action="store_true", help="从最新备份还原 config.yaml")
    args = parser.parse_args(argv)

    import yaml  # 延后导入：--help 不该因为缺 pyyaml 就失败

    home = Path(args.profile_home).expanduser()
    if not home.is_dir():
        raise SystemExit(f"profile 目录不存在：{home}")
    config_path = home / "config.yaml"

    print("=" * 70)
    print(f"宿主模型配置 · {home}")
    print("=" * 70)

    if args.rollback:
        backups = sorted(home.glob("config.yaml.bak-*"))
        if not backups:
            log("没有找到备份，无需还原")
            return 1
        shutil.copy2(backups[-1], config_path)
        log(f"已从 {backups[-1].name} 还原 config.yaml")
        return 0

    data = load_yaml(config_path)
    print("\n[现状]")
    log(describe(data, args.provider_key))

    if already_configured(
        data,
        provider_key=args.provider_key,
        base_url=args.base_url,
        model=args.model,
        key_env=args.key_env,
    ):
        log(f"已是目标状态（provider={args.provider_key} · model={args.model}），不做改动")
        return 0

    if args.check:
        print("\n[将要做的事]")
        log(f"model.default  : {data.get('model', {}).get('default')!r} → {args.model!r}")
        log(f"model.provider : {data.get('model', {}).get('provider')!r} → {args.provider_key!r}")
        log(f"model.base_url : → {args.base_url}")
        log(f"providers.{args.provider_key}: 新增/更新（api / key_env={args.key_env} / models）")
        log("（--check 模式：未做任何改动）")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = config_path.with_suffix(config_path.suffix + f".bak-{stamp}")
    shutil.copy2(config_path, bak)

    updated = apply(
        data,
        provider_key=args.provider_key,
        provider_name=args.provider_name,
        base_url=args.base_url,
        model=args.model,
        key_env=args.key_env,
        extra_models=EXTRA_MODELS,
    )
    config_path.write_text(
        yaml.safe_dump(updated, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    # 写回后立刻**回读校验**：YAML 往返出问题必须在这里暴露，而不是等宿主启动失败。
    verify = load_yaml(config_path)
    entry = (verify.get("providers") or {}).get(args.provider_key) or {}
    ok = (
        verify.get("model", {}).get("provider") == args.provider_key
        and verify.get("model", {}).get("default") == args.model
        and entry.get("key_env") == args.key_env
        and entry.get("api") == args.base_url
    )

    print("\n[已写入]")
    log(f"备份：{bak.name}")
    log(describe(verify, args.provider_key))
    log(f"回读校验：{'通过' if ok else '❌ 不一致，请检查'}")

    if not ok:
        shutil.copy2(bak, config_path)
        log("已自动还原（回读校验失败）")
        return 2

    print("\n[提醒]")
    log(f"密钥由 {args.key_env} 提供，应在 {home}/.env 中（不落 config.yaml）")
    if not (home / ".env").exists():
        log(f"⚠️  {home}/.env 不存在 —— provider 会因取不到密钥而报错")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
