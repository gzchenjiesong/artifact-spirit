# 器灵 · 远端部署指南

面向"装到一台已经在跑的 Hermes Agent 实例上"，不是开发文档。

---

## 0. 先搞清三件事，否则一定踩坑

**① 装进哪个 Python 环境。** 宿主跑在哪个解释器/venv 里，插件就必须装进同一个。
装错环境时 `entry point` 发现不了它——而失败表现是"器灵在面板里根本不出现"，
不是报错。远端先用宿主自己的解释器确认一遍：

```bash
python -c "import sys; print(sys.executable)"
python -c "import agent.memory_provider; print(agent.memory_provider.__file__)"
```

**② `HERMES_HOME` 在哪。** 器灵所有路径都基于它（数据库、配置、镜像都在里面）。
默认 `~/.hermes`，可被环境变量覆盖。同一台机器上可能有多个 profile，别装错。

**③ 两种安装形态，行为不同。**

| | 目录插件 | pip 安装（entry point） |
|---|---|---|
| 位置 | `$HERMES_HOME/plugins/artifact-spirit/` | `pip install` 进宿主 venv |
| entry point | 不需要 | 需要（`hermes_agent.memory_providers`） |
| `plugin.yaml` 是否被读 | **是**（可自动装依赖、提示所需环境变量） | **否**（宿主不读 manifest） |
| 升级方式 | 覆盖目录 | `pip install -U` |
| 适合 | 想控制版本、不想动宿主环境 | 干净、可复现 |

**推荐 pip 安装**：器灵零外部服务依赖，pip 装完就能用；目录插件更适合"要和宿主源码一起改"的场景。

---

## 1. 安装

### 方式 A · pip

```bash
# 在宿主所用的 venv 里（或先用 pipx 装到宿主环境）
pip install /path/to/artifact-spirit          # 或 pip install artifact-spirit
```

### 方式 B · 目录插件

```bash
TARGET="$HERMES_HOME/plugins/artifact-spirit"
mkdir -p "$TARGET"
cp -r /path/to/artifact-spirit/src/artifact_spirit/* "$TARGET"/
# 目录必须是 <name>/__init__.py 且 __init__.py 里含 register_memory_provider
ls "$TARGET/__init__.py" "$TARGET/plugin.yaml"
```

---

## 2. 配置

配置有**两个写入位置**，都是对的，取决于你怎么改：

- **手改 / CLI**：`$HERMES_HOME/artifact-spirit.toml` ← **真相源**
- **宿主面板**：`$HERMES_HOME/artifact-spirit.json` 与 `$HERMES_HOME/artifact-spirit/config.json`
  （宿主两套面板约定各写一处；器灵会把它们当镜像读）

三处同时存在时 **TOML 优先**——所以你手改 TOML 永远生效，不会被面板覆盖掉。

### 最小可用配置

```toml
# $HERMES_HOME/artifact-spirit.toml
[spirit]
name = "拾欢者·清欢"

[backend]
path = ""            # 留空 = $HERMES_HOME/spirit/spirit.db

[models.embedding]
provider    = "tokenhub"
base_url    = "https://tokenhub.tencentmaas.com/v1"
model       = "kinfra-text-embedding-4b"
api_key_env = "ARTIFACT_SPIRIT_API_KEY"   # 只写变量名，密钥永不入文件
dim         = 2560

[models.llm]
provider    = "tokenhub"
base_url    = "https://tokenhub.tencentmaas.com/v1"
api_key_env = "ARTIFACT_SPIRIT_API_KEY"
extract     = "glm-5.3-flash"
dedup       = "glm-5.3-flash"
summarize   = "glm-5.3-flash"
consolidate = "glm-5.3"
soul        = "kimi-k3"
```

### 密钥

**只经环境变量注入，永不落盘。**

```bash
export ARTIFACT_SPIRIT_API_KEY="<你的 TokenHub key>"
```

宿主的 `config.yaml` 里若已有模型配置，器灵会自动借用（第二段 fallback），
这时上面整段 `[models]` 都可以不写。缺 embedding 时召回降级为关键词（BM25）——
**不会静默返回差结果**，`aspirit status` / `doctor` 会明说。

### 激活

在宿主的 `config.yaml` 里：

```yaml
memory:
  provider: artifact-spirit      # 同一时刻只能有一个外部 provider
```

---

## 3. 验证（**必做**）

```bash
# 纯本地检查，不联网
python scripts/deploy/verify_host_install.py --hermes-home "$HERMES_HOME"

# 真跑一轮（需要凭据已 export）
python scripts/deploy/verify_host_install.py --hermes-home "$HERMES_HOME" --live
```

它逐项回答：entry point 是否被宿主环境看见 → `register(ctx)` 是否交出 provider →
**契约方法是否全部实现**（缺方法不报错、只静默失效，所以必须显式核对）→
配置能否装载 → 初始化后模型链与降级告警 → （可选）真跑一轮提取与召回。

`--live` 会**区分"结构化提取成功"与"降级成仅存原文"**——两者落库条数都是 +1，
不区分就会用一次假通过换掉一个真信号。

---

## 4. 装完之后的日常

```bash
aspirit status      # 层计数 / 健康度 / 生效模型链 / 降级告警
aspirit layers      # 五类记忆分布
aspirit review      # 器灵记住了什么（人类可读）
aspirit doctor      # 自检；有问题时退出码 1（可进 CI）
aspirit reflect     # 记忆健康度
aspirit audit       # 审计日志：谁、何时、以何因做了什么
aspirit export ~/memories.md    # 人类可读档案，不依赖器灵即可阅读
```

---

## 5. 常见故障对照

| 现象 | 大概率原因 | 怎么办 |
|---|---|---|
| 面板里看不到 artifact-spirit | 装进了别的 Python 环境 | 用宿主的解释器重装；`verify_host_install.py` 第 2 节会直接指出 |
| 面板字段是空的 | 两套 schema 约定只满足了一套 | 本版本两套都提供；若仍空，跑 `verify_host_install.py` 看第 3 节 |
| 面板改了值、刷新后回弹 | 面板写的位置与器灵读的位置不一致 | 本版本已对齐三处路径；确认没有旧版本的残留文件 |
| 记不住东西（`review` 一直空） | 显著性门槛把内容挡住了 | `aspirit status` 看降级告警；无 embedding 时门槛会自动按比例下调 |
| 召回结果偏弱 | embedding 未配置 → 走了 BM25 | `aspirit doctor` 看 `Embedding 可达性`；配上即自动恢复向量路 |
| 每轮日志里有 `on_xxx failed` | 契约方法缺失 | `verify_host_install.py` 第 3 节会列出来 |
| 换过 embedding 模型后召回全乱 | 维度/模型变了，旧向量失效 | `aspirit reindex` 全库重嵌入 |
| 备份后丢数据 | 只备份了 `.db`，漏了 WAL | 用 `provider.backup_paths()` 给出的清单（含 `-wal` / `-shm`） |

---

## 6. 卸载

```bash
# 记忆是你自己的，先带走
aspirit export ~/memories.md

# pip 安装
pip uninstall artifact-spirit

# 目录插件
rm -rf "$HERMES_HOME/plugins/artifact-spirit"

# 数据（可选——不删就是留着）
rm -rf "$HERMES_HOME/spirit" "$HERMES_HOME/artifact-spirit.toml" \
       "$HERMES_HOME/artifact-spirit.json" "$HERMES_HOME/artifact-spirit"
```

---

## 主对话模型：指向 TokenHub

`remote_set_model.py` 把指定 profile 的**主对话模型**指向 TokenHub（或任意 OpenAI 兼容端点）。
幂等、可回滚、只碰一个 profile。

```bash
# 先看现状与将要做的改动（不改任何文件）
python remote_set_model.py --profile-home /root/.hermes/profiles/spirit --check
# 执行
python remote_set_model.py --profile-home /root/.hermes/profiles/spirit
# 换模型（同 provider 下的其它模型）
python remote_set_model.py --profile-home ... --model glm-5.3
# 回滚
python remote_set_model.py --profile-home ... --rollback
```

写入的形态：

```yaml
model:
  default: glm-5.3-flash
  provider: tokenhub            # 指向 providers 字典里的 key
  base_url: https://tokenhub.tencentmaas.com/v1
providers:
  tokenhub:
    name: TokenHub
    api: https://tokenhub.tencentmaas.com/v1
    key_env: ARTIFACT_SPIRIT_API_KEY   # 只写变量名，值留在 {home}/.env（0600）
    default_model: glm-5.3-flash
    models: [glm-5.3-flash, glm-5.3, kimi-k3, minimax-m3, mimo-v2.5-pro]
```

**为什么用 `providers:` 字典**：宿主旧的 `custom_providers:` 列表已由配置迁移 v12 转到
`providers:` 字典（桌面端 "Custom Endpoints" 也写这里）。字典形态支持 **`key_env`**，
**密钥不必落明文**——宿主自己生成的配置里常见明文 `api_key`，我们不该再添一处。

**密钥放哪**：`{profile_home}/.env` 里的 `ARTIFACT_SPIRIT_API_KEY`。
已实证宿主的 `get_env_path()` 在该 profile 上下文下指向 `{profile}/.env`，
`get_env_value_prefer_dotenv()` 取得到值 —— 判据是 `investing` profile 的
`DEEPSEEK_API_KEY` 就在它自己的 profile `.env` 里且工作正常。

**换模型只需 `-m`**：`models` 列表里的模型可以直接 `hermes -p <profile> -m glm-5.3` 使用，
不必改配置。

### 跑集成验证前必须知道的一件事

`host_integration.py` 现在会**自己加载 profile 的 `.env`**（像宿主 CLI 启动时那样）。
早期版本没做这一步，导致密钥不在环境里 → 器灵安静地走"仅存原文 + BM25 降级"路径，
而**所有 check 仍然 PASS** —— 测的是降级路径却报告成"集成正常"。
最危险的一类假通过，已修。
