# LLD-AL5 辅助机制详细设计

> 上级文档：[DES-000 概要设计 §5 / §3.2](../00-方案设计.md)
> 状态：**已复核**（DES-REV-008 · 2026-09-16）——§2 契约副本按实现重写、§3 文件树补全、
> §5 M3 补两张档位表、§8 条目修正（详见 [07-设计评审.md §16](../07-设计评审.md)）

---

## 1. 职责与边界

### 1.1 负责

| 机制 | 说明 |
|---|---|
| **组合根** | `runtime/lifecycle.py`：装配全部依赖（DI 容器）。**这是全工程唯一 import 各层具体实现的位置**（R10 的显式例外），AL1 的 `initialize` 只是调用方 |
| **运行时** | 线程与队列：单写者串行化、离线维护调度、生命周期启停 |
| **配置** | TOML 装载、环境变量注入、校验、fallback 链的配置部分 |
| **可观测性** | `status` / `layers` / `reflect` 的数据来源；审计视图；**人类可读渲染**（`render.py`：INV-1 说文本是投影，投影属于这里而不属于 AL2） |
| **合规** | CI 依赖扫描、架构规则测试、许可校验 |
| **恢复** | 崩溃后对账、`replay` 重建、`reindex` 触发 |

> **组合根为什么在 AL5 而不是 AL1**：装配要 import 各层具体实现，而 AL1 是"只做翻译"的
> 适配层。把 DI 容器放在 AL5 之后，**"AL5 不依赖业务层"这条约束才只需要一个例外**
> （R10）；反过来会让 AL1 同时成为"宿主适配"与"依赖装配"两个东西的所在地，
> 边界反而更模糊（DES-REV-003 P0-1）。

### 1.2 不负责（明确排除）

| 不负责 | 归属 |
|---|---|
| 记忆算法与决策 | AL2 核心层 |
| 具体持久化与 SQL | AL3 存储层 |
| HTTP 协议与模型调用 | AL4 模型层 |
| 宿主契约方法本身 | AL1 适配层 |

> **边界判据**：AL5 提供"**怎么安全地跑起来**"，不提供"跑什么"。

---

## 2. 对外接口

> **唯一权威副本是代码。** AL2 / AL3 / AL4 的复核都抓出过"契约副本比实现少若干成员"
> （AL2 P1-8、AL3 P1-21、AL4 P1-31），AL5 也不例外——本节原先虚构了
> `TurnEvent` / `ExtractTask` / `CommitTask` / `MaintenanceScheduler` 四个**不存在**的符号，
> 且三节都缺成员（**P1-53**）。**本节是实现的镜像；两者不一致时改本节。**

### 2.1 运行时（`runtime/`）

**任务与档位（`events.py`）**——队列里放的是"要做什么"，不是"怎么做"（C8）：

```python
TaskKind = str        # "intent.<op>" / "turn" / "commit" / "maintenance"

PRIORITY_MEMORY = 0    # 真相源写入——丢了不可重建
PRIORITY_RELATION = 1  # 关联与实体——可由记忆重建，但成本高
PRIORITY_EXTRACT = 2   # 提取与审计——可重跑
PRIORITY_SUMMARY = 3   # 概览投影——纯派生物，随时可重算

PRIORITY: dict[str, int] = {          # **任务类型**档位（穷举，见 §5 M3）
    "intent.memory": PRIORITY_MEMORY, "intent.relation": PRIORITY_RELATION,
    "turn": PRIORITY_EXTRACT, "commit": PRIORITY_SUMMARY,
    "maintenance": PRIORITY_SUMMARY,
}

@dataclass(frozen=True, slots=True)
class Task:
    kind: TaskKind
    payload: object
    priority: int = PRIORITY_MEMORY
    seq: int = 0                # 由队列分配，同优先级下 FIFO
    submitted_at: str = ""
    @property
    def sort_key(self) -> tuple[int, int]: ...

@dataclass(slots=True)
class QueueStats:
    depth / max_depth / accepted / dropped / dropped_by_kind / processed / failed
```

**写队列与单写者（`writer.py`）**：

```python
INTENT_PRIORITY: dict[str, int]         # 写意图 op 档位（穷举，见 §5 M3）
def intent_priority(op: str) -> int      # 未知 op → ValueError
def kind_priority(kind: str) -> int      # 未知 kind → ValueError

class WriteQueue:
    def __init__(self, maxsize: int = 1000) -> None      # maxsize <= 0 → ValueError
    def submit(self, payload: object, *, kind: str = "intent.memory",
               priority: int | None = None) -> bool      # **非阻塞**；队满丢最低优先级
    def get(self, timeout: float | None = None) -> Task | None
    def flush(self, timeout: float = 5.0) -> bool         # 判据 = 堆 **或** 在途
    def depth(self) -> int                                # 堆深
    def pending(self) -> int                              # 堆 + 在途（收尾报数用它）
    def drain(self) -> list[Task]
    def notify_consumed(self) -> None                     # 销账 + 唤醒
    def notify_waiters(self) -> None                      # **只唤醒**，不动在途计数
    stats: QueueStats

@dataclass(slots=True)
class Writer:
    backend: MemoryBackend
    apply: Callable[[object], None]   # 由组合根注入（writer 不 import AL2/AL3）
    queue: WriteQueue
    clock / on_error / thread_factory
    def start(self) -> None
    def stop(self, *, drain: bool = True, timeout: float = 5.0) -> int   # 返回未完成数
    def submit(self, payload, *, kind=..., priority=None) -> bool
    thread_id: int | None    # **缓存**，stop() 之后仍可取
    running: bool
```

**维护调度（`maintenance.py`）**：

```python
@dataclass(frozen=True, slots=True)
class TaggedTask:
    payload: object
    key: str = ""            # 去重键：相同 key 的任务只保留一个

@dataclass(slots=True)
class Maintenance:
    writer: Writer
    handler: Callable[[object], None]
    on_cycle: Callable[[], None] | None
    interval_seconds: float = 1800.0
    queue: WriteQueue                    # 维护任务自己的队列
    cycles / errors / thread_factory
    def start(self) -> None
    def stop(self, *, drain: bool = True, timeout: float = 5.0) -> None
    def schedule(self, payload, *, kind: str = "maintenance",
                 dedupe_key: str | None = None) -> bool
    def trigger_now(self) -> None
    def run_cycle_now(self) -> None
    thread_id / running
```

> 维护任务的载荷是 `{"action": "consolidate" | "decay" | "optimize" | "refresh_overviews", ...}`
> 形式的**字典**，由 `on_session_end` / 定时器 / CLI 各自组装。**没有** `MaintenanceScheduler`
> Protocol，也没有 `schedule_commit` / `schedule_decay`。

**线程铸造与超时护栏（`threading_.py` / `timeout.py`）**：

```python
ThreadFactory = Callable[..., Any]      # (target=, name=, daemon=) -> Thread
def default_thread_factory(**kwargs) -> threading.Thread    # 宿主缺席时的实现

def run_with_timeout(action: Callable[[], T], timeout: float, fallback: T, *,
                     on_timeout: Callable[[], None] | None = None,
                     thread_factory: ThreadFactory | None = None) -> tuple[T, bool]
class TimeoutRunner:
    def __call__(self, action, timeout, fallback, **kwargs) -> tuple[T, bool]
    timeouts: int        # enabled=False 时同步直调（确定性测试）
```

> **硬约束**：`action` **必须无副作用**。超时后调用方放弃等待，但那条被遗弃的线程
> **仍会跑完**——"结果没人要"不等于"没有影响"。

**组合根（`lifecycle.py`）**——唯一可 import 各层具体实现的位置（R10 的单文件例外）：

```python
class IntentApplier:          # 写意图 → AL3 调用；**只翻译，不做决策**
    def apply(self, intent: WriteIntent) -> None
    failures: list[str]       # 向量化失败等降级信号

@dataclass
class Services:
    config / backend / resolver / core / writer / maintenance / applier
    timeout_runner: TimeoutRunner
    warnings: list[str]       # validate() 的"警告"级问题（status 会展示）
    notes: list[str]          # 装配期说明（降级原因、对账结果）
    started: bool
    def submit(payload, *, kind=..., priority=None) -> bool    # 非阻塞
    def enqueue(intents: list[WriteIntent]) -> None   # **所有写路径都走这里**
    def write(intents) -> None                         # = enqueue（热路径别名）
    def write_now(intents) -> None                     # 同步落库（CLI / 测试）
    def drain_pending() -> list[WriteIntent]
    def flush(timeout: float = 5.0) -> bool
    def run_with_timeout(action, timeout, fallback)
    def stop(self, *, drain: bool = True) -> list[str]  # **幂等**，返回调用序列
    stop_sequence: list[str]

def start(hermes_home: str, *, config=None, env=None, transport=None,
          embedding=None, llm=None, start_threads=True, reconcile=True,
          thread_factory=None) -> Services
def stop(services: Services, *, drain: bool = True) -> list[str]
```

> `write_now` 是**同步**入口，只在"此刻没有 writer / maintenance 线程"（CLI、
> `start_threads=False` 的测试）时安全。任何**后台产出**都必须走 `enqueue`——
> 否则 AL3 会多出一条写连接，"单写者"静默作废（P0-12）。

### 2.2 配置（`config/`）

```python
CONFIG_FILENAME = "artifact-spirit.toml"
DEFAULT_DB_RELPATH = "spirit/spirit.db"
ENV_PREFIX = "ARTIFACT_SPIRIT_"

@dataclass(frozen=True, slots=True)
class BackendConfig:   kind: str = "sqlite"; path: str = ""

@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    provider="tokenhub" / base_url="" / model="" / api_key_env="" / dim: int | None = None
    def as_dict() -> dict

@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider="tokenhub" / base_url="" / api_key_env=""
    task_models: dict[str, str]     # extract / dedup / summarize / consolidate / soul
    def as_dict() -> dict

@dataclass(frozen=True, slots=True)
class SpiritConfig:
    name: str = "" / id: str = "" / hermes_home: str = ""
    backend: BackendConfig / embedding: EmbeddingConfig | None / llm: LLMConfig | None
    recall / salience / decay / worker: dict
    source: str                     # "file" | "defaults"（doctor 说清"读到了哪份"）
    @property db_path -> str        # 默认 {hermes_home}/spirit/spirit.db（INV-9）
    @property embedding_dim -> int  # 显式配置 → KNOWN_EMBEDDING_DIMS → 2560
    def toml_payload() -> dict      # 可写盘结构（**不含密钥**）

def load(hermes_home, *, env: Mapping[str, str] | None = None) -> SpiritConfig
def load_file(path) -> dict                      # 语法错误 → ConfigError（含可操作提示）
def validate(cfg: SpiritConfig) -> list[str]     # 空 = 通过；**警告项以"警告："开头**
def toml_example() -> str                        # `aspirit init` 的注释模板
def save(hermes_home, values: Mapping[str, object]) -> Path   # 白名单写入
SAVE_WHITELIST: frozenset[str]                   # 只有列出的点号路径会落盘
def config_path(hermes_home) -> Path
def mirror_paths(hermes_home) -> tuple[Path, Path]   # 宿主面板的两套镜像路径
class ConfigError(Exception)
```

**密钥纪律（三条都是机械保证，不是口头约定）**：

1. 配置文件里**只写 `api_key_env`**（变量名）；发现疑似密钥字段 → `ConfigError`（**不静默忽略**）；
2. `save()` 走**白名单**：不在 `SAVE_WHITELIST` 里的键根本不可能落盘；
3. 判据看**最后一个词**：`api_key` 命中，`api_key_env` / `token_budget` 放行。

> **`validate()` 的严重性表达是"文案前缀"**：`start()` 用 `p.startswith("警告")` 把返回值
> 分成 fatal / warning（P2-29）。改这条契约**必须同时改 `start()` 的分流**。

### 2.3 可观测性（`observability/`）

**全部是 AL3 之上的只读投影**（INV-1），实现为**模块级函数**（不是 Protocol）：

```python
def status(services) -> dict          # 五区块，见下表
def layers(services) -> dict          # **四层**分布（sensory / working 不落表）
def reflect(services) -> dict         # 为什么遗忘 / 为什么召回 / 哪些在成为候选信念
def audit_view(services, *, since=None, limit=50) -> list[dict]   # 委托 AuditView
def doctor(services) -> dict          # 逐项自检，**全程本地不联网**（INV-5）
def status_text(report) -> str        # 人类可读渲染（默认视图）
def doctor_text(report) -> str

class AuditView:                      # audit_view.py
    def entries(self, *, since=None, limit=50, actor=None) -> list[dict]
    def forgetting(self, *, limit=50) -> list[dict]
    def restorable(self) -> list[dict]
def format_audit_text(entries, *, now=None) -> str
def format_restorable_text(snapshots, *, limit=50) -> str

# render.py：AL2 数据结构的文本投影（INV-1 的落点）
def format_review_text(rows, *, now=None) -> str
def format_trace_text(events) -> str
```

**`aspirit status` 必须展示**（这是排障的主要入口）：

| 区块 | 内容 |
|---|---|
| 身份 | `spirit_name` / `spirit_id` / schema 版本 / hermes_home / 配置来源 |
| 层计数 | 四层各自的 active / dormant / forgotten + 合计 |
| 健康度 | 平均强度、待重算概览、写队列深度与统计、线程存活 |
| **生效模型链** | `resolve_chain("extract")` 与 `embedding_chain()` 的**实际结果**（看清是否走了 fallback） |
| **降级告警** | embedding 不可用（BM25 模式）、LLM 不可用（仅存原文）、队列丢弃/失败、**配置校验告警**、装配期说明 |

> **审计条目的编号一律是真实 `audit_id`**，不是"过滤结果里的第几条"——`aspirit restore <编号>`
> 只认真实 id，而位置编号在 `--since` 下会错位（P1-47）。

---

## 3. 内部结构

```
runtime/
├── lifecycle.py     # 组合根：装配依赖、启停顺序（**唯一可 import 各层实现**，R10 单文件例外）
├── writer.py        # 单写者线程 + 写队列 + 优先级穷举表
├── maintenance.py   # 维护线程（巩固 / 衰减 / 概览重算 / 摘要）
├── events.py        # 任务、档位表、队列统计
├── threading_.py    # **唯一的线程创建出口**（可注入的线程铸造策略）
├── timeout.py       # 超时护栏（AL1 的 prefetch 300ms 靠它——AL1 自己不能建线程，R8）
└── __init__.py      # 层内公开面

config/
├── model.py         # 配置数据类 + 默认值表
├── loader.py        # TOML + env 装载 + 镜像合并 + 校验 + 白名单落盘
└── __init__.py

observability/
├── status.py        # status / layers / reflect / audit_view / doctor 数据组装
├── audit_view.py    # 审计视图（只读投影；删除账本的可恢复性）
├── render.py        # **人类可读渲染**（INV-1：文本是投影，投影属于 AL5）
└── __init__.py

compliance/          # CI 与本地自检
├── arch_rules.py    # R1–R10 架构规则（自研检查器，见 §5 M7）
├── dep_scan.py      # R7 依赖扫描
└── __init__.py
```

**每个模块的关键约束**：

| 模块 | 关键约束 |
|---|---|
| `runtime/*` | 层内**唯一可建线程**（R8）；不 import 业务层具体实现（R10，除 `lifecycle.py`） |
| `runtime/lifecycle.py` | 组合根；R10 / R9 的**单文件例外**——例外不得扩散到第二个文件 |
| `runtime/writer.py` | 优先级表**必须穷举**，未知值抛错（C8） |
| `config/*` | 只依赖 `model.base` 与标准库（`tomllib`）——不碰 `model.resolver`，否则 httpx 顺着签名漏进 AL5 |
| `observability/*` | **只读投影**（INV-1）；不 import AL2 的具体模块（"文本是投影，投影属于 AL5"） |
| `compliance/*` | 规则与代码同源；**每条规则配一个注入违规的反例用例** |

---

## 3. 内部结构

```
runtime/
├── lifecycle.py     # 组合根：装配依赖、启停顺序
├── writer.py        # 单写者线程 + 写队列
├── maintenance.py   # 维护线程（巩固 / 衰减 / 摘要）
└── events.py        # 内部事件与审计投递

config/
├── model.py         # 配置数据类
└── loader.py        # TOML + env 装载 + fallback 解析 + 校验

observability/
├── status.py        # status / layers / reflect 数据组装
└── audit_view.py    # 审计视图（只读投影）

compliance/          # CI 与本地自检
├── arch_rules.py    # R1–R10 架构规则（自研检查器，见 §5 M7）
└── dep_scan.py      # R7 依赖扫描
```

---

## 4. 依赖

| 方向 | 内容 |
|---|---|
| **依赖** | 标准库；`tomllib`；AL3/AL4/AL2 的**协议**（`store/base.py`、`model/base.py`）与共享内核 `common`；**唯一例外**是组合根 `runtime/lifecycle.py`（装配具体实现） |
| **禁止依赖** | 业务层**具体实现**（`store.sqlite_backend` / `model.resolver` / `core.*` …，R10）；任何**业务语义**（不得判断"这条记忆重不重要"） |
| **被依赖** | AL1（调起组合根、投递队列）、AL2（通过队列驱动后台任务）、AL3（线程安全前提） |

> **组合根例外必须写进规则，不能靠"整层放行"**：`runtime/lifecycle.py` 是 DI 容器，
> 必然要 import 各层具体实现——这是它存在的**目的**。所以例外精确到一个文件：
> **R10** 约束 AL5 的其余部分（`writer` / `maintenance` / `config` / `observability` /
> `compliance`），**R9** 再单独约束"实现模块不得被层外当接口用"，同样只放行组合根。
>
> 修订前这里写的是"R1 的约束针对 `core/*`，不针对组合根"——把 AL5 的边界说成了
> 别人的边界，等于 AL5 没有边界（DES-REV-003 P0-2）。

---

## 5. 关键设计

### M1 线程模型（R8：唯一可建线程的层）

| 线程 | 职责 | 读写 | 生命周期 |
|---|---|---|---|
| Hermes 主线程 | `prefetch` 只读召回、工具调用 | 只读 | 宿主控制 |
| **writer 线程** | 消费写队列，串行执行**所有**写操作 | 写 | 常驻 |
| **maintenance 线程** | 巩固 / 衰减 / 摘要（低频长跑） | 写（经队列） | 常驻 |

### M2 单写者串行化（D-03）

**所有写操作**（写记忆、更新活性、写审计、写关联）都经 writer 线程的**单一队列串行执行**。

- **收益**：从根本上消除 `SQLITE_BUSY` 与写竞争，不需要复杂重试
- **代价**：写吞吐上限受限（本场景远未触及）
- **实现要点**：maintenance 线程的写**也投递到同一队列**，不持有独立写连接

```
主线程 ──submit(TurnEvent)──▶ [write_queue] ──▶ writer 线程（唯一写者）──▶ AL3
maintenance ──submit(...)────────────────┘
```

### M3 队列与优先级

| 队列 | 生产者 | 消费者 | 溢出策略 |
|---|---|---|---|
| `write_queue` | 主线程 / maintenance | writer | **不丢弃**；极端情况下降级为丢弃 + 告警 |
| `extract_queue` | writer | maintenance | 丢弃最旧（提取可延后） |
| `maintenance_queue` | 定时器 / `on_session_end` | maintenance | 合并同类任务 |

**优先级约定**：记忆写入 > 关联更新 > 提取/审计 > 摘要。溢出时从低优先级开始丢（数字越大越先丢）。

**优先级必须是穷举表，不是"默认档兜底"**（`runtime/writer.py::INTENT_PRIORITY`）：

```python
PRIORITY_MEMORY = 0    # 真相源写入——丢了不可重建
PRIORITY_RELATION = 1  # 关联与实体——可由记忆重建，但成本高
PRIORITY_EXTRACT = 2   # 提取与审计——可重跑
PRIORITY_SUMMARY = 3   # 概览投影——纯派生物，随时可重算

PRIORITY: dict[str, int] = {           # **任务类型**档位（穷举，问"这类任务丢了会怎样"）
    "intent.memory": PRIORITY_MEMORY, "intent.relation": PRIORITY_RELATION,
    "turn": PRIORITY_EXTRACT, "commit": PRIORITY_SUMMARY,
    "maintenance": PRIORITY_SUMMARY,
}

INTENT_PRIORITY: dict[str, int] = {    # **写意图 op** 档位（穷举，须与 AL2 的 IntentOp 一致）
    "put": PRIORITY_MEMORY, "update": PRIORITY_MEMORY, "set_status": PRIORITY_MEMORY,
    "forget": PRIORITY_MEMORY, "restore": PRIORITY_MEMORY,
    "wm_put": PRIORITY_MEMORY, "wm_delete": PRIORITY_MEMORY,
    # 会话登记 / 收尾——工作记忆的外键前提，必须先于 wm_put 应用
    "session_create": PRIORITY_MEMORY, "session_end": PRIORITY_MEMORY,
    "link": PRIORITY_RELATION, "reinforce": PRIORITY_RELATION,
    "touch": PRIORITY_RELATION, "entity_upsert": PRIORITY_RELATION,
    "audit": PRIORITY_EXTRACT,
    "overview_put": PRIORITY_SUMMARY, "overview_invalidate": PRIORITY_SUMMARY,
}

def kind_priority(kind: str) -> int: ...    # 未知 kind → ValueError
def intent_priority(op: str) -> int: ...    # 未知 op → ValueError
```

> **两张表都必须穷举。** op 表守着"写意图"（DES-REV-003 P1-6 初修，`session_create` /
> `session_end` 随后补齐）；`kind` 表守着"任务类型"——它在同一轮复核里被漏掉了一半，
> 直到 **DES-REV-008 P1-43** 才补上（原先 `PRIORITY.get(kind, PRIORITY_MEMORY)` 会把
> 未登记的任务类型**静默排进"记忆档"**）。

**为什么必须穷举**：分档依据是"**这条写丢了能不能重建**"，这是个需要按 op 逐条判断的
问题，所以映射表就是判断结果的载体。用 `PRIORITY.get(op, PRIORITY_SUMMARY)` 会让
未登记的 op **静默落到最慢/最先被丢的一档**——`audit` 与 `wm_delete` 都曾因此被误降级，
而队列只会"看起来在工作"（DES-REV-003 P1-6）。现在的行为是：未知 op 直接 `ValueError`，
并提示去登记。

这张表**必须与 AL2 的 `IntentOp` 穷举一致**——`tests/test_runtime_adapter.py::test_intent_priority_covers_every_intent_op` 守着（AL2 每加一个写意图，这里就有义务登记；缺登记即红灯）。

### M4 配置装载与校验

**装载顺序**（fallback 链的配置侧）：

```
1. {hermes_home}/artifact-spirit.toml       ← 主配置
2. 环境变量覆盖（ARTIFACT_SPIRIT_*）
3. LLM/Embedding 缺失时 → 读宿主 Hermes 配置（hermes_home/config.yaml）
4. 仍未 → 通用环境变量（OPENAI_API_KEY / OPENAI_BASE_URL）
```

**配置文件形态**：

```toml
[spirit]
name = ""                # 器灵名（aspirit init 写入）
id   = ""                # ULID（自动生成）

[backend]
kind = "sqlite"          # 目前仅 sqlite（后端可插拔，见 ADR-002）
path = ""                # 默认 {hermes_home}/spirit/spirit.db

[models.embedding]
provider    = "tokenhub"
base_url    = "https://tokenhub.tencentmaas.com/v1"
model       = "kinfra-text-embedding-4b"
api_key_env = "ARTIFACT_SPIRIT_API_KEY"
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

[recall]
top_k        = 8
token_budget = 2000
weights      = { semantic = 0.40, importance = 0.20, recency = 0.15, entity = 0.10, diffusion = 0.10, core = 0.05 }

[salience]
threshold = 0.35

[decay]
enabled     = false      # MVP 不自动触发（M3 启用）
# 注意：D-16/D-17 之后**不再有 θ_forget 删除阈值**——衰减只影响排序，删除仅白名单
window_days = 180        # 排序用的时间窗

[worker]
write_queue_max          = 1000
maintenance_interval_min = 30
```

**密钥约定（硬约束）**：

- 所有密钥经环境变量注入，前缀 `ARTIFACT_SPIRIT_`
- 配置里只写 `api_key_env`（**变量名**），**永不写 key 本身**
- `save_config()` 落盘前过滤任何疑似密钥字段

**校验时机**：

| 时机 | 校验内容 | 是否联网 |
|---|---|---|
| `is_available()` | 配置文件存在、DB 路径可写、schema 版本匹配 | **否**（INV-5） |
| `initialize()` | 打开 DB、`migrate()`、**校验 `embedding_model` 与库中记录一致** | 否 |
| 首次调用 | LLM / embedding **配置**可用性（懒校验，失败走降级） | 是 |

### M5 可观测性

- `status` / `layers` / `reflect` / `audit_view` 全部是 **AL3 之上的只读投影**，无副作用（INV-1）
- **可读 ≠ 可溯**：查当前状态看投影；查历史看 `audit` 事件表
- `reflect` 的重点是**可解释**：为什么遗忘、为什么召回、哪些在成为候选信念
- **复盘视图（V2 核心价值）**：`review` / `trace` 的数据来自 AL2（`CoreFacade.review` / `.trace`），AL5 只负责**组织与人类可读渲染**——`aspirit review` 是用户了解"Agent 到底记住了什么"的主入口
- **分级视图**：`status` 与 `layers` 默认给 L0 级信息；需要细节时由 `aspirit review` 逐级下钻——**默认输出过载等于没给信息**（与分级加载同一个道理，只是作用在"人读界面"而不是"模型上下文"上）

### M6 崩溃恢复与对账

| 场景 | 动作 |
|---|---|
| 进程崩溃 → 重启 | WAL 自动恢复已提交事务；未提交的回滚 |
| 队列中未消费的任务 | **不持久化**（内存队列），崩溃即丢——可接受，因为都是可重建的派生工作 |
| 派生字段可疑 | `aspirit replay --rebuild-derived` 从 `audit` 重放重建（INV-12） |
| `vec_memories` 与 `memories` 不一致 | 启动对账：缺失向量补算、孤儿向量删除 |
| `embedding_model` 变更 | 启动校验不一致 → **拒绝启动** + 提示 `aspirit reindex`（INV-2） |

> **重要取舍**：写队列**不做持久化**。理由：它的成员都是"可重建的加工任务"（提取、摘要、巩固），丢失只影响及时性，不影响真相源。持久化队列会引入复杂度与新的失败面。

### M7 合规 CI

| 检查 | 规则 | 手段 |
|---|---|---|
| 架构规则 | **R1–R6、R8–R10** | 自研检查器 `compliance/arch_rules.py`（`pytest tests/test_arch.py`、`aspirit layers --check`） |
| 依赖扫描 | R7（禁止传染性 / 许可不明的依赖） | 扫描 `pyproject.toml` 与源码 |
| 密钥扫描 | 代码与配置中无真实 key | 预提交钩子 |

> **为什么改用自研检查器而不是 `import-linter` / `pytest-archon`**：通用工具只能表达
> "包 A 不许 import 包 B"这类**包级**规则，而 DES-REV-003 暴露的四个洞恰恰都在包级之外——
> 宿主模块要按**点分路径**匹配（`agent.memory_provider`）、实现模块要按**模块名**点名
> （`store.sqlite_backend`）、例外要精确到**单个文件**（`runtime/lifecycle.py`），
> 还要顺带守住"新增包必须登记层表"。自研检查器 200 行、零依赖，规则即代码，也就
> 不存在"规则文档改了、门禁没改"的漂移。

### M8 启停顺序

```
start:
  1. 装载配置 + 校验（失败即拒绝启动）
  2. 打开 AL3 backend + migrate()
  3. 校验 embedding 一致性（INV-2）
  4. 启动 writer 线程
  5. 启动 maintenance 线程
  6. 触发一次启动对账

stop:
  1. 停止接收新任务
  2. flush write_queue（超时保护）
  3. 停止 maintenance
  4. 停止 writer
  5. 关闭 AL3（WAL checkpoint）
```

**顺序不可颠倒**：writer 必须先于 maintenance 停止，否则 maintenance 的写会落空。

---

## 6. 编码注意事项

| # | 注意点 | 说明 |
|---|---|---|
| C1 | **`submit()` 必须非阻塞** | 队列满时的行为要明确（抛错 or 丢弃），不得静默阻塞主线程 |
| C2 | **守护线程 + 显式停止** | 线程必须 `daemon=True` 且提供 `stop()`，避免进程无法退出 |
| C3 | **停止必须 drain** | `shutdown` 时先 flush 再停线程，否则丢写 |
| C4 | **异常不得杀死线程** | writer / maintenance 的主循环必须 try/except 包住单任务，记录后继续 |
| C5 | **配置校验失败要给人话** | 报错需指出具体字段与期望，不要只抛 `KeyError` |
| C6 | **`spirit_id` 只生成一次** | 已存在时不得覆盖（否则器灵"换身份"了） |
| C7 | **`save_config` 必须过滤密钥** | 白名单式写入，非白名单字段一律不落盘 |
| C8 | **禁止在 AL5 里判断业务语义** | AL5 不知道也不该知道"哪条记忆重要"。它能做的只有**按任务类型（`op`）分档**——而这张档位表必须是**穷举白名单**（M3），未知 `op` 抛错。用"默认档兜底"等于让 AL5 替业务层猜优先级：`audit` / `wm_delete` 都曾静默落到最易被丢的一档 |
| C9 | **对账要可 dry-run** | 清理孤儿向量前先报告数量 |
| C10 | **线程间通信只用队列** | 不用共享可变状态；唯一共享是 AL3 backend（只由 writer 写） |
| C11 | **`worker` 配置要有安全上限** | `write_queue_max` 必须 > 0，配置校验拦截 |

---

## 7. 错误处理与降级

| 故障 | AL5 行为 |
|---|---|
| **F4** SQLite 锁超时 | writer 重试（最多 3 次）→ 仍失败则丢弃该任务 + 告警 |
| **F5** 写队列积压 | 按优先级丢弃（先丢摘要，保记忆写入）→ `status` 显示队列深度告警 |
| 队列 `submit` 在满时 | 按策略丢弃并计数，**绝不阻塞主线程**（INV-4） |
| 线程内任务异常 | 捕获 → 记 `audit` / 日志 → 继续下一任务（C4） |
| 配置缺失关键项 | 校验期拦截并给出可操作提示（C5） |
| 启动对账发现不一致 | 默认修复并报告；`--dry-run` 只报告 |
| `embedding_model` 变更 | 拒绝启动 + 提示 `aspirit reindex` |
| 关闭时 flush 超时 | 强制停止但记录未完成的任务数（可观测） |

---

## 8. 独立验收标准

**测试环境**：fake backend + fake 层服务 + 计数用队列。**不启动真实 Hermes。**

### 8.1 运行时

- [ ] `submit()` 在队列满时**不阻塞**（有超时断言）
- [ ] 所有写操作实际由**同一线程**执行（记录 thread id 断言）
- [ ] 并发 `submit` 大量写入 → 无 `SQLITE_BUSY`（用真实 SQLite 验证串行化收益）
- [ ] writer 主循环内单任务抛异常 → 线程存活且后续任务继续执行（C4）
- [ ] `stop(drain=True)` 后队列中残留任务数为 0
- [ ] 启停顺序符合 M8（用调用序列断言）
- [ ] 队列溢出时按优先级丢弃（断言摘要先于记忆被丢）

### 8.2 配置

- [ ] 无配置文件时**用默认值启动**（缺失不是错误——这是"零门槛接入"的前提），并在 `doctor` 里给出可操作提示（C5；**P1-57** 改：原条款写的是"给出可操作错误"，与实现和用例相反）
- [ ] 环境变量能覆盖 TOML 值
- [ ] `api_key_env` 指向的环境变量不存在时，报错信息**不含** key 内容
- [ ] `save_config()` 落盘结果中**不含**任何密钥字段（白名单断言）
- [ ] `weights` 之和不为 1 时给出警告（不阻断，但要可见）
- [ ] `write_queue_max <= 0` 被校验拦截
- [ ] `spirit_id` 已存在时 `init` 不覆盖（C6）

### 8.3 可观测性

- [ ] `status()` 包含"生效模型链"与"降级告警"两个区块
- [ ] 造一个 embedding 不可用的场景 → `status` 显示 BM25 降级告警
- [ ] `layers()` 输出与 AL3 实际计数一致
- [ ] 连续两次调用 `status()` 结果一致（纯投影，INV-1）

### 8.4 恢复与合规

- [ ] 启动对账能发现并修复"有记忆无向量"
- [ ] `embedding_model` 与库中不一致 → 拒绝启动
- [ ] ⏳ **缺口（P1-58）**：`replay --rebuild-derived` 后派生字段与原始一致（INV-12）——**全仓无用例**；按 E7，`gap` 的处置是**补用例**，不是改这一行
- [ ] 架构规则测试（R1–R6、R8–R10）全绿
- [ ] 依赖扫描：注入一行违禁 import → CI 失败（R7 有效性自证）

---

## 9. 集成验收关注点

| 关注点 | 验证方式 |
|---|---|
| 与 AL1 的装配 | `initialize()` 后所有层可用，`status` 正常 |
| 与 AL2 的后台驱动 | `on_session_end` → 巩固在 maintenance 线程执行 |
| 与 AL3 的线程安全 | 真实 SQLite 下压测无锁冲突 |
| 与 AL4 的降级联动 | 配错 key → `status` 显示 fallback 生效链 |
| 崩溃恢复 | 写入过程中 kill 进程 → 重启后一致、无半写、对账通过 |
| 关闭安全 | Ctrl+C 后无残留线程、无丢写 |

---

## 10. 待决策 / 未决项

> **口径**（与 AL2 §10 一致）：**"决策未定"与"已定默认值待调优"是两件事**。
> 1–5 均已由实现固化——原先"待确认 / 倾向"的字样属**状态过期**（AL2 P2-14 同型）。

| # | 事项 | 现状 | 影响 |
|---|---|---|---|
| 1 | 写队列是否持久化 | ✅ **已定：不持久化**（§5 M6）。队列成员都是可重建的加工任务，持久化只会引入新的失败面 | 崩溃时丢加工任务（可接受） |
| 2 | maintenance 调度间隔 | ✅ **已定：默认 30 min**（`worker.maintenance_interval_min`，可配） | 巩固/衰减及时性 |
| 3 | 队列满时的精确策略 | ✅ **已定：丢最低优先级 + 计数 + `status` 告警**（`QueueStats.dropped_by_kind`） | 极端负载行为 |
| 4 | `pytest-archon` vs `import-linter` | ✅ **已定：都不用**，改自研检查器（§5 M7——通用工具表达不了"点分路径 / 模块名点名 / 单文件例外 / 新增包必须登记"这四类需要） | CI 依赖 |
| 5 | 是否提供 `aspirit doctor` | ✅ **已提供**（T-AL5-12），且**全程本地不联网**（INV-5） | 可运维性 |
| 6 | `validate()` 的严重性表达 | ⏳ **新登记**：目前靠"警告："中文前缀分流（P2-29）——文案即契约 | 改文案会变成"拒绝启动" |
| 7 | `Services.write_now` 是否加机械约束 | ⏳ **新登记**：是否在 `started` 为真时禁止调用（P0-12 的根因收口；当前只靠 docstring 自觉） | "靠自觉" vs "不可能做错" |
