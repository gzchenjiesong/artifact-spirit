# LLD-AL1 适配层详细设计

> 上级文档：[DES-000 概要设计 §5 / §3.2](../00-方案设计.md)
> 状态：初版，待评审

---

## 1. 职责与边界

### 1.1 负责

| 职责 | 说明 |
|---|---|
| **宿主契约适配** | 实现 `agent.memory_provider.MemoryProvider` 全部必需方法 |
| **生命周期** | `initialize` / `shutdown` 的对外入口，转交 AL5 组合根 |
| **工具注册** | `spirit_*` 工具的 schema 声明与调用分发 |
| **CLI** | `aspirit` 子命令入口 |
| **配置面板** | `config_schema.py` 声明式描述 |
| **参数校验与错误翻译** | 把内部异常翻译为宿主/工具可理解的错误 |

### 1.2 不负责（明确排除）

| 不负责 | 归属 |
|---|---|
| 记忆算法（显著性、巩固、召回融合） | AL2 核心层 |
| SQL / 存储格式 | AL3 存储层 |
| HTTP 调用与重试 | AL4 模型层 |
| 线程与队列的实现 | AL5 辅助机制 |
| 具体依赖装配逻辑 | AL5 `lifecycle.py`（AL1 只是调用它） |

> **边界判据**：AL1 **只做翻译，不做决策**。任何"if 这条记忆重要"的判断都不属于 AL1。

---

## 2. 对外接口

### 2.1 宿主入口

```python
# src/artifact_spirit/__init__.py
def register(ctx) -> None:
    from .provider import ArtifactSpiritProvider
    ctx.register_memory_provider(ArtifactSpiritProvider())
```

```python
# src/artifact_spirit/provider.py
class ArtifactSpiritProvider:
    name = "artifact-spirit"
    # 其余方法见 §5 M1 契约映射表
```

### 2.2 分发方式（硬约束）

| 项 | 要求 |
|---|---|
| entry point | `hermes_agent.memory_providers` |
| 包布局 | `config_schema.py`、`cli.py`、`plugin.yaml` 须与包 `__init__.py` **同级** |
| `__init__.py` | 只做 `register()`，**零业务逻辑**（R6） |
| 密钥 | 不落配置文件，只走环境变量 |

### 2.3 对外工具（`spirit_*`）

| 工具 | 作用 | 对应价值 | 里程碑 |
|---|---|---|---|
| `spirit_recall` | 跨层召回，可指定层与时间范围；**须能解释召回原因** | V1 | M1 |
| `spirit_expand` | **逐级展开 L0 / L1 / L2**（分级加载，P1） | V1 | M2 |
| `spirit_remember` | 显式记住（带层提示与强度） | — | M1 |
| `spirit_forget` | 遗忘（两级：休眠 / 删除）；**删除仅两个入口**，默认 dry-run 且须显式确认 | — | M1 |
| `spirit_restore` | **从审计快照恢复被删记忆**（D-22 安全网） | **V2 / V3** | M1 |
| `spirit_review` | **逐条审查"Agent 记住了什么"**，含来源 / 时间 / 置信度 | **V2** | M1 |
| `spirit_trace` | **溯源**：某条记忆的完整变更史 | **V2** | M1 |
| `spirit_correct` | **主动干预**：修正内容 / 调层 / 改状态 / 调置信度 | **V2** | M4 |
| `spirit_consolidate` | 手动触发巩固（含层间提升） | — | M1 |
| `spirit_reflect` | 记忆健康度报告（层分布、遗忘曲线、热点关联） | — | M1 |
| `spirit_export` | **导出人类可读档案 / 完整包**（P3） | **V3** | **M1** |
| `spirit_soul` | 查看 / 修订器灵本体 | — | M4 |
| `spirit_import` | **从档案 / 完整包重建（器灵传承）**，与 `spirit_export` 对称 | **V3** | M3 |
| `spirit_ingest` | **批量素材导入**（外部文档 / 手册 / SOP，以 `type` 区分）；幂等 + 只存不编 + 记来源 | — | M3（D-27 提案） |

> **工具与核心价值的对应是可验证的**：P 档目标必须在 MVP 内产出可用工具，不能只躺在设计里。
>
> **两个"导入"不可混**：`spirit_import` = 拿 `spirit_export` 的产物**重建器灵**（往返保真）；`spirit_ingest` = 把**外部资料**灌进库（不进热路径）。命名裁决与理由见 [TERM-000 §11.4 ⑦](../06-术语与命名规范.md)。

### 2.4 对外 CLI

```bash
# —— 已实现（M0–M2）——
aspirit init            # 初始化：生成配置、命名器灵
aspirit status          # 状态：层计数、健康度、生效模型链、降级告警
aspirit layers          # 五类记忆分布可视化
aspirit review          # 逐条审查：器灵记住了什么（V2 主入口）
aspirit reflect         # 记忆健康度报告（层分布 / 遗忘曲线 / 热点关联）
aspirit trace <id>      # 溯源：某条记忆的完整变更史
aspirit export <path> [--fmt markdown|json]   # 导出人类可读档案 / 完整包（V3 主入口）
aspirit import <path>   # 从档案 / 完整包重建（器灵传承，与 export 对称）
aspirit consolidate <session_id>  # 手动触发巩固
aspirit decay [--apply]           # 衰减排序与降级（**不删除**；默认只预演）
aspirit forget <id> --reason …    # 物理删除（**两个入口之一**；默认预演 + reason 必填）
aspirit restore <audit_id>        # 从审计快照恢复被删记忆（D-22 安全网）
aspirit correct <id> --set k=v    # 主动干预：修正某条记忆
aspirit audit [--limit N] [--since T]  # 审计日志：谁、何时、以何因做了什么（V2 复盘入口）
aspirit optimize [--apply] [--autonomous]  # 记忆优化：治理性删除与降级（默认只报告）
aspirit reindex         # 换 embedding 模型后全库重嵌入（经 AL3 入口，不直接操作表）
aspirit replay          # 从 audit 重放/重建派生字段
aspirit doctor          # 自检：配置 / DB / 模型**配置**可用性 / schema 版本（**纯本地，不联网**）

# —— 规划中（尚未实现）——
aspirit soul            # 查看 / 编辑器灵本体（M4）
aspirit awaken <pack>   # 觉醒器灵（M3）
aspirit ingest <path…>  # 批量素材导入：外部资料入库（幂等 + 只存不编 + 记来源；M3，D-27 提案）
```

主命令 `aspirit`，别名 `artifact-spirit`。

> **清单以 `cli.py` 为准**：本文与实现不一致时**改本文**。此处的"已实现/规划中"分界
> 对应 `cli.py` 的 subparser 与 §10 未决项（DES-REV-003 发现原清单缺 `reflect`/`optimize`、
> 多了未实现的 `soul`/`awaken`/`ingest`）。
>
> **`doctor` 不做网络探活**：`is_available()` 与自检都必须是纯本地操作（INV-5）。
> 它检查的是"模型**配置**是否可用"（`api_key_env` 有没有配、生效模型链是什么），
> 不是"网关是否连得通"——后者属于集成验收（[LLD-AL4 §9](./AL4-模型层详细设计.md)）。

---

## 3. 内部结构

```
src/artifact_spirit/
├── __init__.py        # register(ctx)，零业务逻辑
├── provider.py        # MemoryProvider 实现 + 宿主探测（R4 唯一豁免点）
├── plugin.yaml        # 插件元信息
├── config_schema.py   # 声明式配置面板（零依赖，共享内核；由 T-AL1-04 产出）
├── cli.py             # aspirit 子命令
└── tools/             # 工具实现（schema + handler 分离）
    ├── schemas.py     # JSON schema 声明
    └── handlers.py    # 调用分发
```

**组合根不在这里**：`runtime/lifecycle.py` 是 AL5 的文件（依赖装配的唯一位置）。
AL1 的 `provider.py` / `cli.py` 只是它的调用方——把 `lifecycle.py` 归到 AL1 是本文档
曾经的口径错误（DES-REV-003）。

---

## 4. 依赖

| 方向 | 内容 |
|---|---|
| **依赖** | Hermes 契约类型（**仅 `provider.py`**，R4）；AL2 层服务与元认知；AL5 `lifecycle`（组合根）/ `config` / `observability`；AL3 的 `probe` / `reindex` / `replay` 入口 |
| **禁止依赖** | `store.sqlite_backend` 等**层内实现模块**（R9）；在 `provider.py` 之外 import 宿主（R4）；在 `provider.py` 内写业务逻辑 |
| **被依赖** | Hermes 宿主（通过 entry point） |

> **AL1 是唯一"可与任意层交互"的层**：适配层的职责就是翻译与分发，所以它不需要白名单
> （R1/R2/R3/R10 都不约束 AL1）。AL1 的边界由另外三条守：
> **R4**（宿主耦合只在 `provider.py`）、**R9**（不得摸层内实现模块）、**R8**（不得创建线程）。
>
> **`.base` 优先**：`provider.py` 做库可用性预检时必须走 `store/probe.py`，
> 不能 import `store/sqlite_backend.py`——后者会把 sqlite-vec 一起拖进最外层，
> 而这正是 `store/base.py` 里那句"DES-REV-003 的 R9 要拦的事"。

---

## 5. 关键设计

### M1 契约映射表

| Hermes 方法 | 必需 | 器灵实现 | 阻塞约束 |
|---|---|---|---|
| `name` | ✅ | `"artifact-spirit"` | — |
| `is_available()` | ✅ | 仅本地检查（M2），**不联网** | 必须快 |
| `unavailable_reason()` | 可选 | 与 `is_available()` **共用同一条 home 解析链**——否则会出现"可用性说可用、原因说没 home"的自相矛盾 | 必须快 |
| `initialize(session_id, **kwargs)` | ✅ | 转交 AL5 `lifecycle.start()` 装配 | 允许慢（启动期） |
| `get_tool_schemas()` | ✅ | 返回 `tools/schemas.py` 声明 | — |
| `handle_tool_call(name, args, **kwargs)` | ✅ | 分发到 `tools/handlers.py` | 主线程 |
| `get_config_schema()` | ✅ | 声明式最小集 | — |
| `save_config(values, hermes_home)` | ✅ | 写 TOML，**过滤密钥** | — |
| `system_prompt_block()` | 可选 | 核心记忆摘要 + 器灵状态 | **须快**（有预算） |
| `prefetch(query, *, session_id)` | 可选 | AL2 召回（M4） | **同步，有超时护栏** |
| `queue_prefetch(query, *, session_id)` | 可选 | 预热（可选，M2+） | 非阻塞 |
| `recall_status()` | 可选 | "上一轮注入几条"（无数据返回 `None`，宿主走通用渲染） | 必须快 |
| `on_turn_start(turn_number, message, **kwargs)` | 可选 | 会话登记 + 轮次计数；**只改内存状态、不落库不决策** | 必须快（热路径） |
| `sync_turn(user, assistant, *, session_id, messages)` | 可选 | 投递事件（M5） | **必须非阻塞** |
| `identity_signature()` | 可选 | spirit_id / schema_version / db_path——宿主据此判断"是否还是同一个记忆主体" | 必须快 |
| `on_session_end(messages)` | 可选 | 投递 CommitTask | **必须非阻塞** |
| `on_session_switch(new_session_id, *, parent_session_id, reset, rewound, **kwargs)` | 可选 | 旧会话投递巩固 + 清工作记忆（旧 session_id 不换掉，新会话开头会混进上一会话的组块） | **必须非阻塞** |
| `on_pre_compress(messages)` | 可选 | 抢救归档（**不声明 v2 checkpoint API**，v1 best-effort） | **必须非阻塞** |
| `on_memory_write(action, target, content)` | 可选 | 镜像 MEMORY.md → 语义记忆；USER.md → 核心记忆 | 主线程→writer |
| `on_delegation(task, result, *, child_session_id, **kwargs)` | 可选 | 委派结果投递为情景记忆（**"做过什么"必须留下**，否则跨会话复盘看不到子代理贡献） | **必须非阻塞** |
| `backup_paths()` | 可选 | DB + `-wal`/`-shm`（存在才给）+ `artifact-spirit.toml` | 必须快 |
| `shutdown()` | 可选 | 转交 AL5 `lifecycle.stop()` | 允许慢（收尾） |

> **本表必须与宿主 ABC 一一对应（22/22）**。此前的版本只有 15 行，漏掉了
> `unavailable_reason` / `recall_status` / `on_turn_start` / `identity_signature` /
> `on_session_switch` / `on_delegation` / `backup_paths` 共 **7 个已实现的钩子**——
> 它们全是**没有契约的孤儿**：宿主加钩子时没人会发现（P1-1）。
> 对照源：`tests/vendor/host_memory_provider.py`（宿主 `MemoryProvider` ABC）。

**其他宿主硬约束**：

- **单激活**：同时只能有一个 external provider 生效（内置 MEMORY.md/USER.md 始终并存）
- **存储隔离**：所有路径基于 `hermes_home`，禁止硬编码 `~/.hermes`（INV-9）

### M2 `is_available()` —— 本地判定（INV-5）

```
检查（全部为本地操作，无网络）：
  1. 配置文件存在且语法合法
  2. DB 路径所在目录可写
  3. DB 中 schema_version 与代码期望一致
返回 bool
```

**明确不做**：HTTP 探活、DNS 解析、端口探测。理由：Hermes 契约硬性要求，且探活会拖慢激活流程。

### M3 `initialize()` —— 装配

```
initialize(session_id, **kwargs):
  hermes_home = kwargs["hermes_home"]        # 恒存在
  cfg = config.load(hermes_home)             # AL5
  problems = config.validate(cfg)            # AL5
  if problems: raise ConfigError(...)        # 给人话
  services = lifecycle.start(cfg)            # AL5 组合根
  self._services = services
```

**关键**：`initialize` 只负责编排调用，**装配逻辑本身在 AL5**（AL1 是调用方，不是实现方）。

### M4 `prefetch()` —— 同步只读 + 超时护栏

```
prefetch(query, session_id):
  1. 启动超时计时（默认 300ms）
  2. AL4 向量化 query —— 失败则跳过向量路（不阻断）
  3. AL2 多路召回 + 融合 + 预算裁剪
  4. 超时或异常 → 返回核心记忆摘要保底（保证 prompt 非空）
```

**保底策略是硬要求**：无论内部发生什么，`prefetch` 都必须返回**非空且合法**的内容，绝不向上抛异常。

### M5 `sync_turn()` —— 非阻塞投递（INV-4）

```
sync_turn(user, assistant, *, session_id, messages):
  event = TurnEvent(session_id, user, assistant, messages, ts=now())
  write_queue.submit(event)     # 非阻塞，立即返回
  return
```

**断言要求**：p99 < 5ms 且调用期间无 I/O。

**同一原则适用于** `on_session_end`（投递 CommitTask）与 `on_memory_write`。

### M6 工具设计要点

- **schema 与 handler 分离**：`schemas.py` 纯声明，`handlers.py` 纯分发 —— 便于测试与审查
- **参数校验前置**：非法参数返回**结构化错误**，不抛异常到宿主（F10）
- **危险操作默认安全**：`spirit_forget` / `aspirit decay` 默认 dry-run，需显式确认才执行
- **删除是收敛操作**（D-17 / D-22）：只有**两个入口**——① 记忆优化任务（发现错误 / 冲突 / 不可达）② 用户显式要求。`reason` 必填，**请求本身写审计**；默认保留快照因而**可恢复**（`spirit_restore`）。**合规删除**是唯一例外（`purge_snapshot=True`，不可恢复）
- **可解释性**：`spirit_recall` 必须能输出召回原因（来自 AL2 的 `Scored.raw`）
- **人类可读优先（V2）**：`spirit_review` / `spirit_trace` 的输出面向人，不是面向程序——**这是核心价值，不是格式化细节**
- **逐级展开（V1）**：`spirit_expand` 只返回所请求的级别；不得"顺手"把 L2 全文一并返回，否则分级加载的注意力收益失效
- **自包含导出（V3）**：`spirit_export` 的产物必须能在**没有器灵的环境**里被读懂；不得引入需专有工具才能解析的格式

### M7 CLI 设计要点

- CLI 是**薄壳**：只做参数解析 + 调用 + 格式化输出
- CLI **允许**直接调用 AL3 的工具型能力（`reindex` / `replay`），因为它们是运维操作而非业务链路
- 破坏性命令（`decay` / `forget`）**默认 dry-run**
- `--json` 输出选项，便于脚本化与集成测试
- `doctor` 是排障主入口：配置 → DB → schema → **模型配置可用性**逐项体检（纯本地，不联网）

### M8 错误翻译

| 内部异常 | 对外表现 |
|---|---|
| `ConfigError` | 启动报错，含具体字段与期望 |
| `ProviderUnavailableError` | 工具返回"模型不可用，记忆将以未结构化形式保存" |
| `StorageFatalError` | 工具返回"存储不可用"；**绝不崩溃宿主** |
| `SchemaViolationError` | 工具返回"提取未通过校验，已保留原文" |
| 参数非法 | 结构化错误，指明字段 |

**总原则**：**记忆系统绝不阻断宿主**。任何异常都在 AL1 边界被翻译为可理解的结果。

---

## 6. 编码注意事项

| # | 注意点 | 说明 |
|---|---|---|
| C1 | **`__init__.py` 保持极简** | 只 export `register`；重逻辑会拖慢宿主加载（R6） |
| C2 | **`is_available` 内禁止任何网络** | 包括"只是 ping 一下"（INV-5） |
| C3 | **`sync_turn` 里禁止同步 I/O** | 连日志写文件都要考虑；用内存缓冲（INV-4） |
| C4 | **`prefetch` 必须有 try/except 全包** | 任何异常都要回到"保底内容" |
| C5 | **`save_config` 用白名单** | 只写已知字段，其余一律不落盘 |
| C6 | **不要缓存跨 session 的可变状态** | provider 实例可能被复用于不同 session |
| C7 | **路径一律 `Path(hermes_home) / …`** | 禁止字符串拼接，禁止硬编码 `~/.hermes`（INV-9） |
| C8 | **工具名与 schema 名严格一致** | `spirit_recall` 在 schema / handler / 文档三处不得拼错 |
| C9 | **`shutdown` 必须幂等** | 可能被调用多次；重复调用不得报错 |
| C10 | **CLI 与 provider 共用同一套服务装配** | 不重复实现装配逻辑（统一走 AL5 lifecycle） |
| C11 | **`system_prompt_block` 有 token 预算** | 核心记忆摘要要截断，不得无限增长 |

---

## 7. 错误处理与降级

| 故障 | AL1 行为 |
|---|---|
| 配置缺失/非法 | `initialize` 抛 `ConfigError`（带可操作提示）；`is_available` 返回 `False` |
| **F10** 工具参数非法 | 返回结构化错误，不抛异常到宿主 |
| **F6** `prefetch` 超时 | 返回核心记忆摘要保底 |
| 任何内部异常冒泡到 AL1 | 捕获 → 日志 → 返回安全值（空记忆/默认状态），**绝不崩溃宿主** |
| LLM/embedding 不可用 | 透传 AL5 `status` 的降级告警，工具层如实说明 |
| `shutdown` 时 flush 超时 | 记录未完成任务数，仍正常返回 |

---

## 8. 独立验收标准

**测试环境**：**fake 核心层**（返回固定召回结果）+ fake 队列。**不需要**真实 Hermes 与真实 DB。

### 8.1 契约

- [ ] `name` 返回 `"artifact-spirit"`
- [ ] `get_tool_schemas()` 返回的每个工具名都能被 `handle_tool_call()` 处理（**一一对应断言**）
- [ ] **钩子全集与宿主 ABC 一一对应**（**22/22**：无遗漏、无悬空）——宿主日后新增钩子时，这条会红
- [ ] `get_config_schema()` 输出可被渲染为合法表单结构
- [ ] `shutdown()` 连续调用两次不报错（幂等，C9）

### 8.2 非阻塞与非联网（关键不变量）

- [ ] **`is_available()` 全程无网络调用**（socket 打桩断言，INV-5）
- [ ] **`sync_turn()` p99 < 5ms 且无 I/O**（INV-4）
- [ ] **`on_session_end()` 非阻塞**（立即返回，不等待巩固完成）
- [ ] **`on_pre_compress()` 非阻塞**（投递后台归档后立即返回）
- [ ] **`on_memory_write()` 非阻塞**（镜像转交 writer，主线程不等待）
- [ ] `prefetch()` 在 fake 核心层抛异常时仍返回非空内容（保底生效）

### 8.3 安全与隔离

- [ ] `save_config()` 落盘结果**不含**任何密钥字段（白名单断言）
- [ ] 所有产出路径都位于传入的 `hermes_home` 之下（INV-9 断言）
- [ ] `spirit_forget` 不带 `confirm` 时**不产生任何写操作**（dry-run 断言）

### 8.4 工具与 CLI

- [ ] 非法参数 → 返回结构化错误而非抛异常（F10）
- [ ] **各内部异常均有对应翻译**（M8）：`StorageFatalError` / `SchemaViolationError` / `ProviderUnavailableError` / `ConfigError` → 可操作文案，而**不是** `内部错误：<类型名>`
- [ ] `handle_tool_call` 对未知工具名返回明确错误
- [ ] CLI 每个子命令 `--help` 正常
- [ ] `aspirit status --json` 输出合法 JSON
- [ ] `aspirit decay` 默认 dry-run（不写库）
- [ ] `doctor` 在配置损坏时能指出具体问题

### 8.5 核心价值验证（V1 / V2 / V3）

- [ ] **V1**：`spirit_expand(ref,'L1')` **不含** L2 全文（逐级展开断言）
- [ ] **V1**：返回体量随级别单调增长 L0 < L1 < L2
- [ ] **V2**：`spirit_review` 输出可直接阅读，含来源与时间（非 JSON 亦可读）
- [ ] **V2**：`spirit_trace` 展示完整变更史
- [ ] **V3**：`spirit_export` 产物用纯文本读取即可理解，无需器灵代码
- [ ] **V3**：导出的档案能通过 `spirit_import` 往返而不丢核心信息

---

## 9. 集成验收关注点

| 关注点 | 验证方式 |
|---|---|
| 真实激活 | 在 Hermes 中成功激活为 external provider |
| 单激活约束 | 验证与宿主内置其他 memory provider 的互斥关系可被用户理解 |
| 全链路 | `prefetch` / `sync_turn` / `on_session_end` / `shutdown` 端到端跑通 |
| 压缩救援 | 触发 `on_pre_compress`，验证工作记忆已归档 |
| 内置记忆镜像 | `on_memory_write` 后 MEMORY.md 内容出现在语义记忆 |
| 崩溃安全 | 会话中途 kill → 重启后 `is_available` 与 `status` 正常 |
| 工具可用性 | 在真实对话中调用 `spirit_recall` 并展示召回原因 |

---

## 10. 待决策 / 未决项

| # | 事项 | 现状 | 影响 |
|---|---|---|---|
| 1 | `prefetch` 超时阈值 | 建议 300ms，待实测调整 | 召回完整度 vs 响应速度 |
| 2 | `system_prompt_block` 的 token 预算 | 待定 | 核心记忆注入量 |
| 3 | ~~`on_pre_compress` 的 v1/v2 API 选择~~ | **已关闭**（见下方裁决记录） | — |
| 4 | `queue_prefetch` 是否实现 | 可选，M2+ 再评估 | 预热门槛 |
| 5 | 与内置 provider 的切换指引文档 | 待写（README） | 用户上手 |

**裁决记录（2026-09-16）**——原第 3 项「`on_pre_compress` 的 v1/v2 API 选择」**已关闭**：**不声明 v2 checkpoint API**，按宿主默认的 v1 best-effort 语义（编号 3 保留不重排）。

- 宿主 `pre_compress_checkpoint_api_version` 的**声明前提**是「每一次成功调用都 durable checkpoint」（`tests/vendor/host_memory_provider.py:78-80`），而器灵只能**投递**后台巩固任务（投递 ≠ 完成，且成败取决于 LLM）——**满足不了该前提，声明即为虚假声明**；
- v2 的 fail-closed 语义（`strict-mode failure propagation`）要求"失败必须被感知并向上传播"，即**必须等结果**，与「不阻塞主线程」**互斥**；
- 详见 [评审日志 P1-3](../07-设计评审.md)。
