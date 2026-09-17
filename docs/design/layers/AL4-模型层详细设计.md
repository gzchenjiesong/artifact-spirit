# LLD-AL4 模型层详细设计

> 上级文档：[DES-000 概要设计 §5 / §3.2](../00-方案设计.md)
> 方案依据：[ADR-003 模型方案](../05-模型方案.md)
> 状态：初版，待评审

---

## 1. 职责与边界

### 1.1 负责

| 职责 | 说明 |
|---|---|
| **统一协议接入** | 只实现 OpenAI 兼容协议（`/chat/completions`、`/embeddings`） |
| **三档模型解析** | 按任务名（extract / dedup / summarize / consolidate / soul）解析到具体模型 |
| **Embedding 服务** | 向量化，含 batch；对外暴露 `model` 与 `dim` |
| **可用性判定（配置侧）** | `embedding_available()` / `llm_available()`：**只读配置**判断"能不能用"，不发起任何请求 |
| **生效链自述** | `resolve_chain()` / `embedding_chain()`：返回链的逐段结果与**被跳过段的跳过原因**（`status` 的"生效模型链"区块直接渲染它） |
| **结构化输出** | `complete_json()` 强制 JSON schema 输出与校验 |
| **轻量 schema 校验器** | `validate_schema()`：**住在 `model/base.py`**，AL4（`complete_json`）与 AL2（`extract/schema.py`）共用同一份实现 |
| **fallback 链** | 器灵配置 → 宿主配置 → 通用环境变量 |
| **重试策略** | 网络瞬断重试；非 JSON 输出的"加提示重试" |
| **连接生命周期** | `close()`：释放共享的 `httpx.Client`（由 AL5 组合根在停止时调用） |

> **AL4 刻意不提供"探活 / ping"接口**。契约里只有两类东西：
> **真调用**（`complete` / `complete_json` / `embed`）和**配置侧判定**
> （`*_available()`、`resolve_chain()`）。
>
> 原因在 INV-5：`is_available()` 与 `aspirit doctor` 都必须是纯本地操作。
> 一旦 AL4 暴露 `probe()`，这两个调用点迟早会用它——于是"本地判定"变成
> "依赖网络与额度"，宿主启动时多一次外部请求，离线环境下直接报错。
> 所以这里不是"还没实现探活"，而是**设计上不该有**。
>
> **要验证网关真的连得通**：走集成验收（§9），用真实凭证实调一次 embedding +
> 一次 `complete_json`。那是唯一能证明"通"的方式，也只是验收动作，不进热路径。

### 1.2 不负责（明确排除）

| 不负责 | 归属 |
|---|---|
| 提示词内容（提取什么、怎么问） | AL2 核心层 / `extract` |
| 何时调用（频率、时机） | AL2 元认知 / AL5 运行时 |
| 成本决策（用哪个档） | AL2 决定"任务"，AL4 只做"映射" |
| 向量存储与检索 | AL3 存储层 |
| 交叉验证的裁决逻辑 | AL2 核心层（AL4 只提供"多模型可用"） |

> **边界判据**：AL4 是"哑管道"——**知道怎么连、知道用哪个模型名，但不知道要问什么**。所有提示词与任务语义属于 AL2。

---

## 2. 对外接口

### 2.1 协议（`model/base.py`）

> **本副本是唯一权威**：与 `src/artifact_spirit/model/base.py` 逐字对齐。
> 上层（AL1/AL2/AL5）只按本节对契约——**本节少一个方法，上层就"合法地"不知道它存在**。
> 故 §8 配一条 `inspect` 白名单断言，两个方向都比（详见 DES-REV-007 P1-31）。

```python
from typing import Any, Literal, Protocol

TASKS = ("extract", "dedup", "summarize", "consolidate", "soul")
HttpTransport = Any          # 不透明注入点：写成 httpx.BaseTransport 会逼 AL5 的公开签名 import httpx
KNOWN_EMBEDDING_DIMS = {"kinfra-text-embedding-4b": 2560, "kinfra-text-embedding-0.6b": 1024}

@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str
    def as_dict(self) -> dict: ...


class LLMProvider(Protocol):
    def complete(self, *, messages: list[ChatMessage] | list[dict],
                 model: str | None = None,
                 timeout: float | None = None,
                 temperature: float | None = None) -> str: ...

    def complete_json(self, *, messages: list[ChatMessage] | list[dict],
                      schema: dict,
                      model: str | None = None,
                      timeout: float | None = None) -> dict: ...


class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str: ...
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ModelRegistry(Protocol):
    def llm(self, task: str = "extract") -> LLMProvider: ...
    def embedding(self) -> EmbeddingProvider: ...
    def resolve_chain(self, task: str = "extract") -> list[dict]: ...
    def embedding_chain(self) -> list[dict]: ...
    def llm_available(self) -> bool: ...
    def embedding_available(self) -> bool: ...
    def close(self) -> None: ...
```

**三处必须写对、也是 C9/INV-6 的载体**：

| 要点 | 说明 |
|---|---|
| `model` 可省略 | 实现携带由**任务档位**决定的默认模型——核心层不该知道 `glm-5.3-flash` 这种名字（C9） |
| **`temperature` 默认 `None`** | `None` = **请求体里根本不发这个字段**（网关默认值生效），**不是** `0.0`。要确定性输出必须显式传值 |
| `resolve_chain` / `embedding_chain` 返回 `list[dict]` | 每段含 `source` / `usable` / `reason`，供 `status` 渲染；**不是** `list[str]` |

### 2.2 异常类型（`model/base.py`）

```python
class ModelError(Exception): ...
class LLMError(ModelError): ...
class SchemaViolationError(LLMError): ...          # 非 JSON 或不符合 schema
class ProviderUnavailableError(LLMError): ...      # 未配置 / 连接失败 / 鉴权失败 / 429
class EmbeddingError(ModelError): ...              # 绝不降级为 LLM 调用（INV-6）
class SchemaValidationError(ValueError): ...        # **刻意不是 ModelError**（见下）
```

> **关键约定**：AL2 依据异常类型决策降级，因此异常分类是接口的一部分，不可随意合并。

三条不可动摇的语义：

1. **`ProviderUnavailableError` 挂在 `LLMError` 下**（不是直接挂 `ModelError`）。
   因为"没配好模型"与"模型返回了坏东西"在上层是**同一个降级分支**（仅存原文 + 告警），
   而与 embedding 的降级分支（降级 BM25）**必须分得开**——后者由 `EmbeddingError` 承担。
   > 若把 `ProviderUnavailableError` 直接挂 `ModelError`，`except LLMError` 就会漏掉"未配置"这条最常见的情况。
2. **`EmbeddingError` 不得是 `LLMError` 的子类**：这是 INV-6（embedding 绝不 fallback 到 LLM）的**类型载体**。
   一旦成为子类，`except LLMError` 会顺手把 embedding 失败当成"LLM 不行"处理，静默走错降级路径。
3. **`SchemaValidationError` 继承 `ValueError` 而**不是** `ModelError`**：它表示"**数据**不符合 schema"，
   是调用方（`validate_schema` 的使用者）的问题，不是模型层故障；对模型层而言校验结果只是 `list[str]` 错误列表。

---

## 3. 内部结构

```
model/
├── base.py           # 协议 + 数据类 + 异常族 + schema 校验器（零 I/O）
├── openai_compat.py  # 唯一实现：OpenAI 兼容 httpx 客户端
├── resolver.py       # 三档解析 + fallback 链 + 宿主 YAML 子集解析
└── __init__.py       # 汇总导出（零逻辑，R6）
```

| 文件 | 职责 | 关键约束 |
|---|---|---|
| `base.py` | 定义契约 + `validate_schema()` | 不得 import httpx（它是协议层）；校验器**刻意与 AL2 共用**，见下 |
| `openai_compat.py` | 实际 HTTP 调用、超时、重试、schema 校验 | 唯一构造 `httpx.Client` 的地方；`api_key` 的 `repr` 必须屏蔽 |
| `resolver.py` | 依据配置 + 环境变量决定"用哪个 base_url / model / key" | 链信息**一次解析后缓存**；`re-export` `KNOWN_EMBEDDING_DIMS` 以兼容旧取值路径 |
| `__init__.py` | 汇总导出 | 零逻辑（R6） |

> **校验器为什么住在 AL4 的协议模块里**：`validate_schema()` 被 AL4（`complete_json`）
> 与 AL2（`extract/schema.py`）**同时需要**，而 R1 的白名单只允许 AL2 依赖
> `store/base.py`、`model/base.py`、`common` 与 `store/ids.py`。
> 把它放进 `base.py` 是**唯一**既不重复实现、又不越界的位置——
> 复制一份"两份校验器逐渐不一致"是经典缺陷。
> （注：AL2 LLD §6 的 **C12**「提示词与 schema 同源」要求提示词与校验规则一致，
> 共用同一份实现正是这条的落地手段；但**归属依据是 R1 白名单**，不是 C12 本身。）

---

## 4. 依赖

| 方向 | 内容 |
|---|---|
| **依赖** | `httpx`；`json`；标准库（含**自写的 YAML 子集解析器**，见 M3）。**不依赖** `openai` SDK（避免额外依赖与版本耦合），**不依赖** PyYAML |
| **禁止依赖** | `core/*`、`store/*`（R3）；`model.openai_compat` / `model.resolver` 只准 AL4 内部与 AL5 组合根使用（R9） |
| **被依赖** | AL2 核心层（仅 `model/base.py` 协议与 `validate_schema`）；AL5（`config/model.py` 取 `KNOWN_EMBEDDING_DIMS`、组合根取 `ModelResolver`） |
| **可选依赖** | 宿主 `hermes_home` 下的 `config.yaml`（fallback 链第 2 级）——**只支持 YAML 的映射/标量子集** |

---

## 5. 关键设计

### M1 一套 httpx 客户端接所有 provider

```python
Provider ≈ base_url + api_key + model_name
```

- TokenHub 本质就是 `base_url` 指向网关的 provider，`model` 字段即路由键
- 切换网关 / 加模型 / 换厂商**只改配置，代码零改动**
- **不引入 `openai` SDK**：它底层也是 httpx，引入只是增加依赖面与版本耦合

### M2 三档模型映射（ADR-003）

| 档 | 任务 | 模型 | 频率 |
|---|---|---|---|
| embedding | 写入向量化 / 检索向量化 | `kinfra-text-embedding-4b`（2560 维） | 极高 |
| 小模型 | extract / dedup / summarize | `glm-5.3-flash` | 高 |
| 大模型 | consolidate / soul | `glm-5.3` / `kimi-k3` | 低 |

**交叉验证池**（M3+ 启用）：`glm-5.3` / `kimi-k3` / `minimax-m3` / `mimo-v2.5-pro`

> AL4 只提供"能按名字调任意模型"的能力与 `resolve_chain()`；是否交叉验证、如何裁决由 AL2 决定。

### M3 fallback 链（ADR-003 §3.6）

```
resolve_llm(task):
  1. 器灵配置 [models.llm] 有值 且 provider 可用 → 用之
  2. 未配置 → 读宿主 Hermes 的模型配置（hermes_home/config.yaml）
  3. 仍未 → 读通用环境变量（OPENAI_API_KEY / OPENAI_BASE_URL）
  4. 全无 → 抛 ProviderUnavailableError（由 AL2 降级为"仅存原文"）

resolve_embedding():
  1. 器灵配置 [models.embedding]（需 base_url + model 齐备）→ 用之
  2. 未配置 → 宿主配置的 embedding 节点
  3. 仍未 → 环境变量：OPENAI_API_KEY + **OPENAI_EMBEDDING_MODEL**（两个都要）
  4. 全无 → 抛 EmbeddingError，由 AL2 降级 BM25
```

**两条铁律**：

- **LLM 可 fallback 到宿主**（兜底，非推荐——大模型做高频提取成本高）
- **embedding 不可 fallback 到 LLM**（INV-6）；embedding 未配置时抛 `EmbeddingError`，由 AL2 降级 BM25

**第 3 段 embedding 的一个额外开关**：仅设置 `OPENAI_API_KEY` **不足以**启用 env 段 embedding，
还必须给出 `OPENAI_EMBEDDING_MODEL`（否则无法判断向量维度，猜错维度会让全库向量失效）。
两个变量都不在配置里存**值**，只存变量名（C1）。

**宿主配置的读取路径（第 2 段）**——按序探测，命中即停：

| 段 | LLM | embedding |
|---|---|---|
| 嵌套 | `memory.llm` → `llm` → `openai` → `providers.openai` | `memory.embedding` → `embedding` |
| 扁平 | `llm_base_url` / `llm_model` / `llm_api_key_env` | — |

**宿主 YAML 只支持极小子集**（映射 + 标量 + 简单列表）：宿主 `config.yaml` 是**外部文件**，
器灵刻意不引 PyYAML（依赖面越小越好）。解析失败**一律返回空 dict**——
"读不到宿主配置"不是错误，只是少一段 fallback。

> ⚠️ 这意味着：宿主用了锚点 / 多行标量（`|`）/ 复杂嵌套时，**器灵会静默读不到**，
> 表现为"宿主配置中未找到可用模型服务"。
> 处置：`resolve_chain()` 必须把"宿主段为什么没用上"原样展示（§8 有断言），
> 让用户能从 `status` 输出直接看出是"没有"还是"格式没读进来"。

### M4 结构化输出与重试

```
complete_json(model, messages, schema):
  1. 请求：**总是**带 response_format={"type": "json_object"}
  2. 解析 JSON → 校验 schema
     ✅ 通过 → 返回 dict
     ❌ 失败 → 追加"请严格按以下 schema 输出"提示，重试 1 次
        仍失败 → 抛 SchemaViolationError
```

- **只重试 1 次**：避免成本失控与延迟抖动
- 校验用轻量手写校验器（不引 `jsonschema` 重依赖），**实现只有一份**（`model/base.py`），
  AL2 的 `extract/schema.py` 直接复用同一函数——校验规则与提示词的"同源"由这一点保证
- **`response_format` 是硬依赖**：部分网关不认这个字段会返回 4xx，
  而不重试的 4xx 会直接抛 `LLMError` → 上层降级为"仅存原文"。
  **这是可接受但要记载的耦合**（见 §10 未决项 2）
- 失败必须留痕：由 AL2 写 `audit(reason='extract_parse_failed')`

> ⚠️ **最坏请求数 = 2 × 3 = 6**。M4 的"重试 1 次"与 M6 的"连接类重试 2 次"是**相乘**的：
> 一次 `complete_json` 最坏会发出 `(1+2) × 2 = 6` 个 HTTP 请求。
> 两节各自看都没问题，合起来才是真实上界——**成本与延迟预算必须按 6 次算**（§8 有断言）。

### M5 batch embedding

- `embed(texts: list[str])` 内部按批合并请求
- 依据：TokenHub 限流 TPM 100 万 / RPM 300（极富余），但 batch 仍能**省请求数 + 省成本**
- 批大小可配置（默认 `DEFAULT_EMBED_BATCH = 32`），超长文本单独成批
  （阈值 `SINGLE_TEXT_MAX_CHARS = 2000`——避免一条长文拖累同批其它请求）
- **必须保持输入顺序**：返回的向量顺序与入参严格一致（否则向量与记忆错配，是灾难性 bug）
  - 落地手段：先按入参长度建好 `[None] * n` 的槽位，再**按服务端返回的 `index` 归位**
    （缺 `index` 时退化为顺序归位）；末尾断言长度并检查无 `None` 空槽
- 返回数量与入参不符 → 抛 `EmbeddingError`（**不猜、不补**）

### M6 限流与超时

| 项 | 值 | 策略 |
|---|---|---|
| TPM | 1,000,000 | 极富余，**不实现限流器** |
| RPM | 300 | 富余，仅靠 batch 降低请求数 |
| 单次超时 | LLM 60s / Embedding 30s | 可配置（`DEFAULT_TIMEOUT_LLM` / `DEFAULT_TIMEOUT_EMBED`）；**httpx 客户端构造时也要给 timeout**，不能只靠每次调用传 |
| 网络重试 | 连接类错误重试 2 次（`DEFAULT_MAX_RETRIES`），指数退避基数 0.5s | 仅对连接类错误，不对 4xx 重试 |
| 5xx | 重试（同上） | 耗尽后抛 `ProviderUnavailableError`——服务端错误属于"暂时不可用"，与 4xx 的"配置/配额问题"处置不同 |
| 429 | 不重试 | 抛 `ProviderUnavailableError`（配额/频率受限） |

> 明确**不做**：令牌桶、排队、退避队列。理由：限流参数远高于实际用量，加限流器是过度设计（ADR-003 §3.7）。
>
> ⚠️ **重试次数要连乘看**：M4 的 schema 重试与本节的重试叠加，最坏 6 次请求（见 M4）。

### M7 配置形态

```toml
[models.embedding]
provider    = "tokenhub"
base_url    = "https://tokenhub.tencentmaas.com/v1"
model       = "kinfra-text-embedding-4b"
api_key_env = "ARTIFACT_SPIRIT_API_KEY"   # 只存变量名，不存 key
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

**`dim` 的判定顺序**（`resolve_embedding` 内部）：

1. 配置里显式给了 `dim` → 必须是**正整数**，否则抛 `EmbeddingError`；
2. 未显式给 → 查 `KNOWN_EMBEDDING_DIMS`（`kinfra-text-embedding-4b` = 2560 / `-0.6b` = 1024）；
3. 都不匹配 → 抛 `EmbeddingError`（提示已知模型清单）——**绝不给默认维度**。

> **为什么宁可不启动也不猜**：维度不对 → 向量写入后与检索的向量空间不一致（INV-2），
> 是"能跑但结果全错"的静默故障。`dim` 在链路里是**契约数据**，不是实现细节。
> 这张表住在 `model/base.py` 而不是 `resolver.py`：AL5 的配置层（`Config.vec_dim`）
> 要按模型名推断维度，若表留在 `resolver`，AL5 就必须 import AL4 的实现模块（R9 / R10）。
> 注意 `dim` 只在这里**声明与自洽**，**与库内 `meta.embedding_dim` 的比对不在 AL4**（见 §7 F7）。

---

## 6. 编码注意事项

| # | 注意点 | 说明 |
|---|---|---|
| C1 | **key 只从环境变量读** | 配置里只有 `api_key_env`；日志与异常中**禁止**打印 key |
| C2 | **`K` 名不可出现在代码里** | 任何示例、测试夹具都不得硬编码真实 key |
| C3 | **`embed()` 必须保序** | 见 M5，向量错配是灾难性 bug；**必须**加长度断言与 `None` 空槽检查 |
| C4 | **`dim` 必须与 AL3 建表一致** | AL4 **只负责**"从配置/已知模型表得到 dim 并自洽"，**不做**库内比对（INV-2 / F7）；真正的"拒绝启动"归 AL5 装配期 + AL3（见 §7 F7） |
| C5 | **`complete_json` 不要用 `eval`** | 用 `json.loads`；解析失败走重试而非容错解析 |
| C6 | **异常要分类抛出** | 未配置/连接失败/401/429 → `ProviderUnavailableError`；非 JSON 或不合 schema → `SchemaViolationError`；embedding 一切失败 → `EmbeddingError`；AL2 靠类型决策 |
| C7 | **超时必须显式传入** | 不依赖 httpx 默认（默认无超时，会挂死）；客户端构造时给 `httpx.Timeout`，每次调用传业务超时 |
| C8 | **不重试 4xx** | 4xx 是配置/配额问题，重试无意义且浪费配额；**5xx 属于"暂时不可用"，要重试** |
| C9 | **模型名允许运行时改写** | 便于测试（注入 fake model 名）与灰度；`model=None` 时用任务档位默认值 |
| C10 | **base_url 结尾斜杠归一化** | 避免 `//chat/completions` 类拼接错误 |
| C11 | **链信息只由对应的解析器写** | `_llm_chain` 只准 LLM 解析路径写、`_embedding_chain` 只准 embedding 解析路径写。**写错链的后果不是显示错，而是判错可用性**——"链非空"被当作"已解析完成"的判据，污染会让另一条链**跳过解析**并误报不可用（DES-REV-007 P0-10） |
| C12 | **`*_available()` 是一次性判定** | 首次解析后结果被缓存（含失败结果）；同一 resolver 实例内**不会**因为"环境变量后来被设上"而恢复。AL5 若需要重判，必须重建 `ModelResolver`（见 §10 未决项 6） |

---

## 7. 错误处理与降级

| 故障 | 检测 | AL4 行为 | 上层承接 |
|---|---|---|---|
| **F1** LLM 未配置 | resolver 全链失败 | 抛 `ProviderUnavailableError` | AL2 → 跳过提取，仅存原文 + 告警 |
| **F2** embedding 未配置 / 调用失败 | 异常 | 抛 `EmbeddingError`（**不 fallback 到 LLM**，INV-6） | AL2 → 降级 BM25 检索 |
| **F3** 非 JSON 输出 | schema 校验失败 | 加提示重试 1 次 → 抛 `SchemaViolationError` | AL2 → 仅存原文 + `audit` 留痕 |
| **F7** 维度不匹配 | 库内 `meta.embedding_dim` ≠ 配置 `dim` | **AL4 不做这个比对**（它看不到库——R3 禁止依赖 `store`）。AL4 只保证 `dim` 来自配置或已知模型表且为正整数 | AL5 装配期：`cfg.embedding_dim` 传入 AL3（`assert_embedding_model` / `DimensionMismatchError`）+ `doctor` 的"向量维度一致性"检查 → 拒绝启动并提示 `reindex` |
| **F9** 成本超预算 | 由 AL2/AL5 计数（非 AL4 职责） | — | AL2 → 降级为纯启发式 |
| 网络瞬断 | 连接异常 | 指数退避重试 2 次 | 仍失败则抛 `ProviderUnavailableError` |
| 5xx | HTTP ≥ 500 | 重试 2 次 | 同上 |
| 429 / 配额超限 | HTTP 429 | 不重试，抛 `ProviderUnavailableError` | AL5 记录告警 |
| 网关不认 `response_format` | HTTP 4xx | 不重试，抛 `LLMError`（非"不可用"） | AL2 → 仅存原文；**需人工判断是否换网关或去掉该字段**（§10 未决项 2） |

---

## 8. 独立验收标准

**测试环境**：本地 mock HTTP 服务（不连真实网关）。

**契约类**（可机械判定，见 DES-REV-007 §15.6）

- [ ] **协议对齐**：`inspect` 比对 `model/base.py` 的三个 Protocol 与 §2.1 **两个方向**——
      文档有的方法实现必须有、实现有的方法文档必须有；数据类字段与异常继承关系同样比对
      （**不要**写成"签名与文档一致"这种无法判定的句子）
- [ ] 代码扫描：AL4 无 `core` / `store` 导入；未引入 `openai` SDK 与 PyYAML

**调用类**

- [ ] `complete()` 能对 mock 返回正确文本
- [ ] `complete()` **不传 `temperature` 时请求体里没有该字段**；显式传 `0.0` 时字段存在且为 0.0
- [ ] `complete_json()` 对合法 JSON 直接返回 dict，且请求体**总是**含 `response_format={"type":"json_object"}`
- [ ] `complete_json()` 对非法 JSON 触发**恰好 1 次**重试，第二次成功则返回（断言 `request_count == 2`）
- [ ] 两次都失败 → 抛 `SchemaViolationError`
- [ ] **最坏请求数上界**：mock 持续返回 5xx 或持续抛连接错误 + schema 重试 → `request_count == 6`
- [ ] `embed(["a","b","c"])` 返回 3 个向量，**顺序与入参一致**（C3 断言）
- [ ] `embed()` 返回数量与入参不符 → 抛 `EmbeddingError`（mock 少返一条）
- [ ] `embed()` 超长输入（> 2000 字符）**单独成批**，且结果顺序不变
- [ ] `dim` 属性返回 2560（显式配置 `dim` 与按已知模型表推断两条路径各一次）

**链与 fallback 类**

- [ ] `resolve_chain()` 输出"器灵配置 → 宿主 → env"的实际生效链，**每段带 `usable` 与 `reason`**
- [ ] fallback：删除器灵 LLM 配置 → 能读到宿主配置（用 fixture 模拟 `hermes_home`）
- [ ] 宿主段跳过时，`reason` 必须**指名**是哪一步没过（环境变量名 / 未找到节点），
      而不是一律"未找到可用模型服务"——用户要能区分"没有"与"格式没读进来"（M3）
- [ ] fallback 全断 → 抛 `ProviderUnavailableError`
- [ ] **P0-10 回归**：给定"器灵 LLM 未配置 + 宿主 LLM 节点的 `api_key_env` 指向未设置变量 +
      `[models.embedding]` 完全合法"，**先调用 `llm_available()`（或 `resolve_chain()`）
      再调用 `embedding_available()`，结果必须为 `True`**，且 `embedding_chain()` 里
      **不得**出现 LLM 解析产生的段（链污染回归，见 §6 C11）
- [ ] env 段 embedding：只有 `OPENAI_API_KEY` 时为不可用；加上 `OPENAI_EMBEDDING_MODEL` 后可用
- [ ] 宿主 YAML 子集：遇到不支持的结构（多行标量 / 嵌套列表）时**读不到就是读不到**，
      不得猜出错误值，且不抛异常

**降级与安全类**

- [ ] embedding 未配置时**不会**调用 chat 接口（INV-6 断言：mock 记录无 `/chat/completions` 请求）
- [ ] HTTP 429 不重试（断言请求计数为 1）且抛 `ProviderUnavailableError`
- [ ] HTTP 5xx 重试（断言请求计数为 3 = 1 + 2 次重试）
- [ ] `EmbeddingError` **不是** `LLMError` 的子类；`ProviderUnavailableError` **是**（异常契约断言）
- [ ] 超时参数被真实传入（mock 校验）
- [ ] 日志中不含 key（扫描日志输出）；`repr(OpenAICompatSettings(api_key=...))` 不含 key

> **断言纪律**：以上每条都要有明确的失败条件。**"代码扫描无 `store` 导入"这类断言必须同时配一个
> 注入违规的反例**，否则规则报绿不能证明规则在工作（DES-REV-003 的结论）。

---

## 9. 集成验收关注点

| 关注点 | 验证方式 |
|---|---|
| 与 AL2 的异常契约 | AL2 依据异常类型走对应降级分支（三个降级路径各跑一次） |
| 与 AL3 的维度契约 | 真实 TokenHub 返回 2560 维可直接落库 |
| 真实网关连通 | 对 TokenHub 实调一次 chat 与一次 embedding |
| fallback 实战 | 故意配错 key → 观察是否按链降级而非崩溃 |
| 成本可观测 | 统计一轮对话的实际 token 消耗，与预估对比 |

---

## 10. 待决策 / 未决项

| # | 事项 | 现状 | 影响 |
|---|---|---|---|
| 1 | `glm-5.3-flash` 的结构化输出稳定性 | **需实测**——不稳则提取退到 `glm-5.3` | 提取质量与成本 |
| 2 | `response_format` 的使用方式 | **已定默认（待调优）**：**总是**发送 `{"type": "json_object"}`（代码已如此）；**未**用 `json_schema` 严格模式。若目标网关不认该字段，会退化为 4xx → 降级"仅存原文" | 提取准确率 vs 网关兼容性 |
| 3 | 是否支持 OpenAI 原生 SDK 作为可选后端 | **已定：不做** | 依赖面 |
| 4 | 交叉验证的模型池与仲裁策略 | M3+ 决策，AL4 只保证"可多模型调用" | — |
| 5 | `complete()` 不传 `temperature` 时**不发该字段** | **已定默认（待调优）**：不传即不发送（网关默认值生效），因此提取结果**默认不保证确定性**。若实测发现网关默认偏高影响提取稳定，应改为显式发送 `0.0` | 提取可复现性 |
| 6 | `*_available()` 的一次性缓存是否够用 | **已定：够用（一次性）**，代价是同一实例内不恢复。若将来要做"配置热重载"，需重建 `ModelResolver` 而非改缓存（C12） | 配置热重载可行性 |
| 7 | 宿主 YAML 子集解析失败的可观测性 | **已定**：不引 PyYAML，解析失败返回空 dict；**代价是宿主用了锚点/多行标量时静默读不到**。已用"链段必须自述跳过原因"补偿（M3 / §8） | 宿主 fallback 命中率 |
