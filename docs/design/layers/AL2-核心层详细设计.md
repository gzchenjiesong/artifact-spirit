# LLD-AL2 核心层详细设计

> 上级文档：[DES-000 概要设计 §5 / §2](../00-方案设计.md)
> 质量依据：[RES-001 记忆质量调研](../03-记忆质量调研.md)
> 状态：初版，待评审

---

## 1. 职责与边界

### 1.1 负责

| 职责 | 说明 |
|---|---|
| **五类记忆语义** | 感觉 / 工作 / 情景 / 语义 / 程序性 各自的组织规则 |
| **元认知调度** | 巩固提升、衰减遗忘、扩散激活、健康度评估 |
| **显著性判断** | 决定"值不值得记"（零 LLM 启发式） |
| **召回融合** | 多路召回结果的多因子加权融合与预算裁剪 |
| **提取与去重** | 提示词 + 结构化 schema + 六态更新决策 |
| **核心记忆维护** | identity / soul 的装载与高门槛更新 |

### 1.2 不负责（明确排除）

| 不负责 | 归属 |
|---|---|
| 任何 I/O（SQL、HTTP、文件） | AL3 / AL4（R5 硬约束） |
| 线程与队列的创建和管理 | AL5 辅助机制 |
| 宿主的生命周期回调 | AL1 适配层 |
| 向量存储格式、SQL 语句 | AL3 存储层 |
| HTTP 重试与协议细节 | AL4 模型层 |

> **边界判据**：AL2 是**纯函数式领域逻辑**——给它输入（记忆、查询、配置）与协议句柄，它产出决策与数据。**任何 `import` 了 I/O 库的代码都不属于 AL2。**

---

## 2. 对外接口

### 2.1 核心门面（`core/facade.py`）—— AL1 与 AL5 的唯一入口

> **评审补充（P0-1）**：AL1 需要"用 fake 核心层测试"，就要求 AL2 暴露一个**单一可替换门面**。此前 AL2 只有散落的 `LayerService` / `Metacognition` / 函数，无入口可替。

```python
from typing import Protocol

class CoreFacade(Protocol):
    """AL2 对外的唯一门面。AL1（适配层）与 AL5（运行时）只依赖它。"""

    # ---- 写入路径（同步产出决策，落库由 AL5 writer 执行）----
    def ingest_turn(self, event: "TurnEvent") -> list["WriteIntent"]: ...

    # ---- 召回路径（AL1 的 prefetch 调用）----
    def recall(self, q: "RecallQuery") -> list["Scored"]: ...

    # ---- 分级加载（P1 首要机制）----
    def expand(self, ref: str, level: str = "L0", *, hot_path: bool = True) -> str: ...
    """level ∈ L0 | L1 | L2。`hot_path=True`（默认）时**只读现有 L1，缺失则降级返回 L0 而不生成**
    （N-P0-2：L1 生成是模型调用，绝不能进热路径）；`hot_path=False` 供离线维护使用。
    评审 P1-8 补充：此参数是 N-P0-2 的唯一落地点，此前未进契约。"""

    # ---- 审查与主动干预（P2 首要机制）----
    def review(self, *, layer: str | None = None, since: str | None = None,
               limit: int = 50) -> list[dict]: ...            # 逐条审查视图
    def trace(self, mem_id: str) -> list[dict]: ...           # 该条记忆的完整变更史
    def correct(self, mem_id: str, patch: dict, *, reason: str) -> list["WriteIntent"]: ...
    def forget(self, mem_id: str, *, reason: str, source: str,
               purge_snapshot: bool = False) -> list["WriteIntent"]: ...
    """物理删除的**唯一人工收口**（D-17 入口 2）。

    - `reason` 与 `source` 必填——删除请求本身要可审计
    - `purge_snapshot=True` 仅用于**合规删除**：连 `audit` 快照一并清除（不可恢复）
    - 默认 `False` → 保留 `audit.before` 快照，**可经 `restore()` 恢复**（D-22）
    - 入口 1（记忆优化删除）由 `consolidation` 内部调用，不经此入口
    """
    def restore(self, audit_id: int) -> list["WriteIntent"]: ...
    """从 `audit.before` 快照恢复被删记忆（D-22 安全网）。恢复本身也写审计。"""

    # ---- 会话生命周期 ----
    def on_session_end(self, session_id: str) -> "ConsolidationReport": ...
    """会话收尾巩固（M2）。返回**巩固报告**，不是写意图列表——收尾要跨会话统计与决策，
    粒度与 `ingest_turn` 的"一轮一意图"不同（评审 P1-8：此前误写作 `list[WriteIntent]`，
    且与 §5 M2 的 `ConsolidationReport` 定义矛盾）。"""

    # ---- 元认知（AL5 maintenance 调用）----
    def consolidate(self, *, session_id: str) -> "ConsolidationReport": ...
    def optimize(self) -> "OptimizationReport": ...
    """记忆优化（D-23）：**降级与物理删除的唯一发起者**（入口 1）。
    - 不可达 → 产出 `op="forget"` 意图（不经 `forget()`）
    - 无入边且低重要度 → 产出 `set_status(dormant)` 意图
    ⚠️ 当前仅实现这两类；"错误记忆 / 冲突记忆"**未实现**（`superseded_by` 无生产者）——
    见评审 P0-7。"""
    def decay(self, *, now: str | None = None, dry_run: bool = True) -> "DecayReport": ...
    def drain_pending(self) -> list["WriteIntent"]: ...
    """取走**非热路径**产生的写意图（`expand` 的概览失效标记等），由 AL5 maintenance 在后台排空
    （N-P0-1：读路径不得直接落库）。"""
    def refresh_overviews(self, *, limit: int = 20) -> list["WriteIntent"]: ...
    """**唯一**允许生成 / 刷新 L1 概览的入口（离线，只应由 maintenance 线程调用）。
    热路径绝不生成 L1（N-P0-2），只做 `overview_invalidate` 标记 + `stale` 返回旧缓存。"""
    def health(self) -> "HealthReport": ...

    # ---- 核心记忆 ----
    def system_prompt_block(self, *, token_budget: int = 400) -> str: ...
```

> **`WriteIntent` 的意义**：AL2 是纯逻辑，**不能直接落库**（R5）。因此 `ingest_turn` 返回"写意图"列表，由 AL5 writer 翻译为 AL3 调用并落库。这保证了 AL2 的可测性（断言意图即可，不需要真库）。

```python
@dataclass(frozen=True, slots=True)
class WriteIntent:
    op: str                        # 见下方 op 全集（评审 P1-9：此前只列了 6 个，实际 14 个）
    record: "MemoryRecord | None" = None
    patch: dict | None = None
    audit: "AuditEvent | None" = None
    ...
```

**`op` 全集（14 个，权威副本＝`runtime/writer.py::INTENT_PRIORITY`，评审 P1-9）**：

| 分组 | op | 语义 |
|---|---|---|
| 写入 | `put` / `update` / `set_status` | 新增 / 补丁更新 / 状态迁移（含 `dormant` 降级） |
| 关系 | `link` / `entity_upsert` / `reinforce` | 建关系边 / 实体表 upsert / 加权重 |
| 访问 | `touch` | 记一次访问（`access_count` / `last_access`，热路径之外排空） |
| 删除 | `forget` / `restore` | 物理删除（含 `purge_snapshot` 合规语义）/ 从审计快照恢复 |
| 概览 | `overview_put` / `overview_invalidate` | 写 L1 概览 / 标记概览失效（**热路径只允许后者**） |
| 工作记忆 | `wm_put` / `wm_delete` | 工作记忆槽写入 / 淘汰 |
| 审计 | `audit` | 纯审计事件（不落数据），如去重忽略 |

> 新增 op **必须**同时登记本表与 `INTENT_PRIORITY`——AL5 writer 按 op 穷举分派，未登记的 op 会抛 `ValueError`（不用"默认档兜底"）。

### 2.2 召回相关（`core/base.py` 定义数据类 · `core/recall.py` 实现算分）

```python
from dataclasses import dataclass
from store.base import MemoryRecord      # R1 允许：AL2 可依赖 store 的**协议与数据类**

@dataclass(slots=True)
class Scored:
    record: MemoryRecord               # 显式类型（评审 P0-2：此前写作 object，丢失契约）
    raw: dict[str, float]              # {"semantic":…, "importance":…, "recency":…, "entity":…, "diffusion":…, "core":…}
    score: float                       # 融合后总分

@dataclass(slots=True)
class RecallQuery:
    text: str
    vec: list[float] | None
    session_id: str
    layers: list[str] | None = None
    token_budget: int = 2000
    top_k: int = 8                       # 进入预算裁剪前的候选上限（评审 P1-9 补）

@dataclass(slots=True)
class RecallWeights:
    """六因子权重（合计 1.0）。

    D-20：原 `vitality`（strength 与 access_count 合成）拆为两个**独立**因子——
          `recency`（时间邻近）与 `importance`（重要度）。
          原因：strength 内部已含时间衰减，与 recency 语义重叠；
                而"低频但关键"的记忆需要 importance 独立支撑（见 DES-RES-002 §4.1）。
    """
    semantic:   float = 0.40
    importance: float = 0.20    # 置信度 / 显著性 / 用户标注 / 被反复引用
    recency:    float = 0.15    # 时间邻近度（衰减形状，仅影响排序）
    entity:     float = 0.10    # 实体匹配（D-10）
    diffusion:  float = 0.10    # 扩散激活
    core:       float = 0.05    # 与核心记忆的一致性

def strength_at(record: MemoryRecord, *, now: str, params: DecayParams,
                last_access: str | None = None) -> float: ...      # 遗忘曲线求值（M3 §一）
def fuse(items: Sequence[Scored], weights: RecallWeights) -> list[Scored]: ...
    # 缺因子的权重重归一化；六因子齐全时不归一化
def clip_to_budget(items: Sequence[Scored], token_budget: int,
                   cost_of: "Callable[[Scored], int] | None" = None) -> list[Scored]: ...
    # L0 摘要优先；**至少保留 1 条**（首条超预算也返回），否则空预算等于静默不召回
def choose_level(items: Sequence[Scored], token_budget: int) -> dict[str, str]: ...
    # 逐条决定该以 L0 / L1 / L2 交付（§5 M9）
def normalize_minmax(values: Sequence[float]) -> list[float]: ...
    # 全部相等时返回全 1（而非全 0）——避免"都重要"退化成"都不重要"
def score_semantic(similarity: float | None) -> float: ...
    # 入参是**已转成 [0,1] 的相似度**（`vector_search` 已做余弦距离转换），不是 (hit, q)
def score_importance(record: MemoryRecord, *, now: str, params: DecayParams) -> float: ...  # D-20：与频率解耦
def score_recency(record: MemoryRecord, *, now: str, params: DecayParams,
                  last_access: str | None = None) -> float: ...   # 内含 Wixted 衰减形状
def score_entity(record: MemoryRecord, query: RecallQuery,
                 entity_index: Mapping[str, float]) -> float: ...  # 实体命中度（D-10）
def score_diffusion(record: MemoryRecord, seeds: set[str],
                    neighbors: Mapping[str, float]) -> float: ...
def score_core_alignment(record: MemoryRecord, core_index: Mapping[str, float]) -> float: ...

@dataclass(slots=True)
class Recaller:
    """把六路信号拼起来的薄协调器：多路召回 → 归一化 → 融合 → 裁剪。"""
    def recall(self, q: RecallQuery) -> list[Scored]: ...
```

> **`RecallQuery.top_k` 与 `clip_to_budget` 的关系**：**先**按 `top_k` 截候选，**再**按 `token_budget` 裁
> （`limit` 是"最多看几条"，`budget` 是"最多给多少 token"）。两者都设，缺一不可——只设预算时，
> 十条各 200 token 的记忆会和两条各 1000 token 的记忆抢同一个预算。
> 评审 P2-9 补充：此前契约里这三处边界（至少留 1 条 / `top_k` 先截 / 全 0 分时返回全部）均未记载。
```

> **R1 澄清（评审补充）**：R1 禁止的是"依赖**具体实现**"（`store/sqlite_backend.py`）。依赖 `store/base.py` 的**协议与数据类**是允许且必要的——否则 AL2 无法表达"我处理的是记忆记录"。

### 2.3 层服务（`core/layers/`）

```python
@dataclass(slots=True)
class BaseLayerService:            # 评审 P1-11：实现是**共用基类**，不是 Protocol
    layer: str                     # 层标签（episodic / semantic / ...）；子类以类属性声明
    def recall(self, q: RecallQuery) -> list[Scored]: ...         # 默认空实现（子类覆写）
    def candidates(self, ctx: "TurnContext") -> list[dict]: ...   # 从一轮输入产出候选（未落库）
    def ingest(self, ctx: "TurnContext", intents: list[WriteIntent]) -> None: ...  # 产出写意图
```

> **为什么是基类而不是 Protocol**：六个层服务共享 `layer` 的默认处理与共用装配（门面按 `layer` 建索引），
> 用 `Protocol` 会让每个实现重复一遍这类样板；而"可替换"的需求已由 `CoreFacade` 单例门面满足（P0-1）。
> 评审 P1-11：原契约写作 `Protocol`，与 `core/layers/base.py` 的实际形态不符。

| 服务 | 特有语义 |
|---|---|
| `sensory.py` | 只做显著性过滤，**不产出落库候选**（噪声不落盘） |
| `working.py` | 组块聚类 + 容量控制（4±1）+ 意图槽提取 |
| `episodic.py` | 会话 → 事件序列，带时空上下文 |
| `semantic.py` | 去情境化事实 / 偏好 / 实体 |
| `procedural.py` | 技能与可复用轨迹（MVP 仅占位） |
| `core_memory.py` | identity / soul 装载与更新 |

### 2.4 元认知（**无独立模块**——门面即调度中枢）

> **评审 P1-10 修正**：本节此前声明 `core/metacognition.py` 与 `Metacognition` Protocol，但**该模块不存在**，
> §3 文件树也列了它（幽灵模块）。实际形态是：`CoreFacade` 直接持有 `Consolidator` / `Decayer` /
> `Optimizer` / `Activator` 并对外暴露 `consolidate()` / `decay()` / `optimize()` / `health()`（§2.1）。
>
> **裁决理由**：元认知的**消费者只有 AL5 maintenance 一处**，而它只认 `CoreFacade`（P0-1 的单门面约定）。
> 再抽一层 Protocol 既无第二个实现、也无第二个消费者 → **不为无消费者造抽象**。四件能力各自是独立模块：

```python
# core/consolidation.py
@dataclass(slots=True)
class Consolidator:                 # M2：巩固提升（幂等、单向、离线）
    def consolidate(self, *, session_id: str) -> ConsolidationReport: ...

@dataclass(slots=True)
class Optimizer:                    # M3 §三：不可达删除 + 低重要度降级（入口 1）
    def run(self) -> OptimizationReport: ...

# core/decay.py
@dataclass(slots=True)
class Decayer:                      # M3 §一：只重排与降级，**无删除调用**（红线）
    def run(self, *, now: str | None = None, dry_run: bool = True) -> DecayReport: ...

# core/activation.py
@dataclass(slots=True)
class Activator:                    # M5：扩散激活（沿 relations 边传播）
    def activate(self, seeds: list[str], *, hops: int = 1) -> dict[str, float]: ...
    # 返回 {mem_id: 激活量}——此前写作 list[tuple[str, float]]，与实现不符
```

### 2.5 提取契约（`extract/schema.py`）

**提取输出（严格 JSON）**：

```json
{
  "memories": [
    {
      "type": "fact|preference|event|entity|skill|identity|soul|intent",
      "layer": "episodic|semantic|procedural|core",
      "subject": "RAGFlow",
      "predicate": "uses_python_version",
      "object": "3.12",
      "content": "项目 RAGFlow 使用 Python 3.12",
      "abstract": "RAGFlow 用 Python 3.12",
      "scope": {"type": "project", "id": "ragflow"},
      "confidence": 0.98,
      "salience": 0.7,
      "valid_from": null,
      "valid_to": null
    }
  ]
}
```

**去重决策输出**：

```json
{"decision": "ADD|UPDATE|MERGE|IGNORE", "target_id": "sem_01J…", "merged": { }}
```

| 阶段 | 支持的状态 |
|---|---|
| **MVP（M2）** | `ADD` / `UPDATE` / `IGNORE`（`MERGE` 可选） |
| **M3+** | 补齐 `MERGE` / `INVALIDATE` / `FORGET`（依赖时态与置信度模型） |

---

## 3. 内部结构

```
core/
├── facade.py              # CoreFacade —— AL1/AL5 的唯一入口（M9/M11 也由此暴露）
├── layers/
│   ├── sensory.py         # 显著性过滤
│   ├── working.py         # 工作记忆容量管理 + 意图槽
│   ├── episodic.py        # 情景记忆
│   ├── semantic.py        # 语义记忆
│   ├── procedural.py      # 程序性记忆（MVP 占位）
│   └── core_memory.py     # 核心记忆
├── base.py                # 协议数据类（Scored / RecallQuery / RecallWeights / WriteIntent / …）
├── consolidation.py       # 巩固流水线
├── decay.py               # 衰减排序 + 治理性删除（不做自动删除）
├── activation.py          # 扩散激活 + Hebbian
├── progressive.py         # 分级加载 L0/L1/L2（M9 · P1）
├── evolve.py              # 记忆进化 A-MEM 式（M10 · M4）
├── review.py              # 审查 / 溯源 / 干预（M11 · P2）
├── salience.py            # 显著性打分
└── recall.py              # 多因子融合召回

extract/
├── schema.py              # 记忆对象 schema + 校验器
├── extractor.py           # 结构化提取（Mem0 范式）
└── dedup.py               # 六态更新决策
```

> **评审 P1-10**：树中**没有 `metacognition.py`**（此前列了它，属幽灵模块）——"调度中枢"由 `facade.py` 承担，
> 理由见 §2.4。`evolve.py`（M10 · M4）同样尚未落地，是**前瞻占位**，实现前不得被任何模块 import。

**内部数据流**：

```
一轮对话输入
   └─▶ sensory.filter()    显著性打分（零 LLM）
         ├─ 低于阈值 → working.ingest()（仅会话内）
         └─ 达标    → extractor.extract()（LLM）→ schema 校验
                        └─▶ dedup.decide()（LLM 或规则）
                              └─▶ 产出写操作意图（不落库，交 AL5 writer）
```

---

## 4. 依赖

| 方向 | 内容 |
|---|---|
| **依赖（仅协议）** | `store/base.py` 的 `MemoryBackend`、`model/base.py` 的 `LLMProvider` / `EmbeddingProvider` |
| **禁止依赖** | `store/sqlite_backend.py`、`model/openai_compat.py`（R1）；任何 Hermes 代码（R4）；`httpx` / `sqlite3` / `openai`（R5） |
| **被依赖** | AL1 适配层（调用层服务与工具）；AL5 运行时（驱动巩固/衰减任务） |

**依赖注入方式**：所有外部能力（backend、llm、embedding、clock、id 生成器）**通过构造函数注入**，禁止模块级单例或全局变量。

> `clock` 与 `id_gen` 也必须注入——否则衰减与时序逻辑无法稳定测试。

---

## 5. 关键设计

### M1 显著性过滤（感觉记忆门口）

> **依据**：重要性是**编码期**就打上的标记（情绪唤醒 / 自我相关会加强编码），与访问频率是两个独立维度——见 [DES-RES-003 §4.1](../10-神经科学依据与机制映射.md)。

零 LLM 成本的启发式打分，避免每轮烧 token：

```
salience = w1 · 信息新颖度        （与现有记忆最大相似度的补：1 − max_sim）
         + w2 · 显式指令信号      （"记住" / "以后" / "别忘" / "我不喜欢"）
         + w3 · 实体密度          （人名 / 项目名 / 专有名词计数）
         + w4 · 情绪强度          （叹号、强调词、重复标点）
         + w5 · 核心记忆一致性偏移 （与核心记忆偏离的反而更值得记）

salience < θ_salience  → 只进工作记忆，不提交长期记忆
```

- 权重与阈值**可配置**，集中在 `core/salience.py::SalienceWeights` / `SalienceConfig`（C9）。**默认值**（评审 P2-8 补：此前只有符号 `w1..w5`，数值只活在代码里）：

| 因子 | 符号 | 默认权重 | 备注 |
|---|---|---|---|
| 信息新颖度 | `w1` | **0.30** | 需一次 embedding；取 `novelty_top_k = 5` 条候选的最大相似度 |
| 显式指令信号 | `w2` | **0.25** | 零 LLM |
| 实体密度 | `w3` | **0.20** | 零 LLM |
| 情绪强度 | `w4` | **0.15** | 零 LLM |
| 核心记忆一致性偏移 | `w5` | **0.10** | 与核心记忆偏离越大越高 |

  阈值 `θ_salience = 0.35`。
- `w1` 需要相似度，故需要一次 embedding 调用——**这是唯一的模型依赖**，失败时退化为纯规则打分（置 `w1 = 0`，其余权重重归一化）。

#### M1-a 降级时「门槛必须同量纲缩放」（评审 P1-19 补）

`w1` 置 0 后**如果门槛不动**，分数会整体从 [0,1] 塌到 [0, 0.70]，而 0.35 的门槛留在原处 —— 等于要求
**规则因子独自凑满原本五因子的全部证据量**。实测 `请记住：我对花生过敏` 只拿 0.298，**永远过不了线**：
器灵一句都记不住。这不是"更严格的筛选"，而是把「降级打分」偷偷变成了「**关闭写入**」（M0–M2 的 W4 缺陷）。

**契约要求（两条缺一不可）**：

```
① 分数：w1 = 0，其余权重按 Σw' 归一化          → 分母从 1.00 变成 0.70
② 门槛：θ' = θ_salience × (1 − w1) = 0.35 × 0.70 = 0.245   → 门槛与分数同量纲
```

- 只做 ① 不做 ②（W4 原始形态）→ **门槛事实上永久关闭**；
- 只做 ② 不做 ① → 等价于悄悄放宽门槛。
- 落地点：`SalienceConfig.effective_threshold()`（`degraded_threshold` 可显式覆盖，`None` 时按上式推算）。

### M2 巩固提升（元认知驱动，单向递进）

| 迁移 | 触发条件 | 动作 |
|---|---|---|
| 工作 → 情景 | 会话结束 / 上下文压缩前 | 整段会话固化为 episode |
| 情景 → 语义 | 同一主题/实体在 ≥ N 个**不同会话**出现，或事件被反复引用 | 提炼去情境化知识 → `profile` / `preferences` / `entities` |
| 语义 → 程序 | 某模式成功复用 ≥ K 次（`access_count` 统计） | 固化为 skill / trajectory |
| 任意 → 核心 | soul / identity 相关表述出现 | 更新 identity / soul，**低频、高门槛** |

**硬约束**：

- 迁移**单向**，不出现逆向搬移
- 提升产生**新记录**，用 `relations(rel_type='derived_from')` 保留来源链
- 巩固**在离线后台执行**（AL5 maintenance 线程），不在在线热路径
- 核心记忆更新须过阈值，**M3+ 必须经交叉验证**（INV-11）

### M3 衰减排序、降级与记忆优化删除（D-16 / D-17 / D-20 / D-22 / D-23）

> **定位修正**（[DES-RES-002](../09-机制决策依据研究.md) §4）：Agent 没有生物大脑的资源约束，**不需要模拟"必然遗忘"**；它真正稀缺的是**注意力**。
>
> **形状依据**：衰减**只作排序**，且取"近期下降快、远期长尾"的混合形状（[DES-RES-003 §4.3](../10-神经科学依据与机制映射.md)）。
> 因此本节不做"遗忘机制"，只做**排序**与**治理性删除**两件事。

#### 一、衰减只作排序信号（D-16）

`strength` **不再是删除触发器**，只是召回排序的一个因子。

```
strength(t) = base_retention × ( w · exp(−Δt / τ_fast) + (1−w) · (1 + Δt)^(−β) )
                Δt = 距上次访问时长（天）；访问时回升（间隔重复效应）

用途限定：只喂给 score_recency()，不参与任何删除判定。
```

Wixted 混合模型**保留，但只用于描述时间衰减的形状**（近期快、远期长尾）——这是统计规律，不需要生物学论证。

**为什么移除"衰减触发删除"**（DES-RES-002 §4.1）：

- 按访问频率衰减会**系统性优先淘汰"低频但关键"的记忆**——身份证号 / 血型 / 过敏史 / 紧急联系人，全部调用频率极低，但一旦需要就必须有
- **它与记忆研究的结论相悖**：记忆中"重要性"与"频率"是两个独立维度，高重要性的记忆衰减极慢（此处引用的是研究的**可检验结论**，不是"像不像脑"）

→ 对应 D-20：`importance` 升为**独立排序因子**，与频率解耦。

#### 二、降级语义：`dormant` = 只保留 L0 参与召回（D-23）

`dormant` **不是"待删队列"**，而是**注意力层面的降级**——用户澄清：*"dormant 更多是变成抽象 L0 被遗忘"*。

| 状态 | 召回语义 | 存储 |
|---|---|---|
| `active` | L0 初筛 → 命中后 L1 判断 → 确认需要展开 L2 | 完整 |
| `dormant` | **只有 L0 参与召回候选**；L2 不自动展开（需显式 `expand`） | **完整——详情仍在库里** |
| `forgotten` | 不参与任何召回 | 物理删除（见 §五 安全网） |

**关键点**：`dormant` **不丢数据**，只丢"被自动展开的资格"。这正是用户说的"分层抽象本身就是遗忘"——**淡出的是注意力，不是信息**。

**实现口径**（评审 P1-13：本条曾被读成"只保留 L0 参与召回候选"，与实现不同）：

- "不参与召回候选"与"降权参与召回"**只能二选一**。本设计的裁决是**降权**：
  `dormant` 的记忆**仍参与六因子召回**，但总分乘以 `dormancy_penalty`（默认 **0.6**，可配置——评审 P2-10 要求离开硬编码）。
  理由：`dormant` 是"注意力降级"而非"索引移除"；若直接从候选里剔除，用户明确提过的旧事会**彻底召不回**，
  而 `dormant` 的定位恰恰是"变成抽象 L0 **被想起**"。
- "能看到多少细节"由 `choose_level` + `expand` 决定（§5 M9），**与状态无关**；
  `dormant` 的唯一差别是：**不会被自动展开到 L2**（需显式 `expand(ref, "L2")`）。

| 关心的问题 | 由谁负责 | 是否不可逆 |
|---|---|---|
| 在不在**注意力**里 | `dormant` | ✅ 可逆 |
| 看到多少**细节** | L0 / L1 / L2 | ✅ 可逆 |
| 数据**还在不在** | `forgotten` | ⚠️ 见 §五（可恢复） |

> 原 `archive`（移出活跃索引 + 丢弃 L2 详情）被取消——它把"细节多少"塞进了"删除机制"，而那本质是**摘要层**的职责。

**降级由谁触发**：**记忆优化任务**（巩固 / 整理，LLM 驱动，离线）在整理时判定。
**明确不用时间或频率**——理由见 §一。

#### 三、L2 详情的删除：只在"不可达"时由优化任务发起（D-23）

用户给出的删除判据非常明确：**某条记忆不应该被时间或者频率衡量而删除**；但"错误记忆 / 多条记忆冲突"如果不处理，问题会一直存在。

因此把删除收敛到**记忆优化任务**里，触发条件是**语义性**的，而非统计性的：

| 触发情形 | 说明 | 实现状态 |
|---|---|---|
| **错误记忆** | 被优化任务判定为错误（如与更高置信度记忆直接矛盾） | ❌ **未实现** |
| **冲突记忆** | 多条记忆互相冲突，优化后需淘汰被取代者 | ❌ **未实现** |
| **不可达记忆** | **图结构上的孤立**：无入边关联 + 无实体关联 + 不在任何 L1 概览中 | ✅ 已实现 |

> **"不可达"是本节最重要的判据**：它不是"很久没用"，而是"**从任何路径都到不了它**"——这才是真正没有认知价值的记忆。

**⚠️ 未实现两类的根因（评审 P0-7）**："错误记忆"的判据载体是 `superseded_by`，而**该字段全库没有生产者**——
"新记录取代旧记录"的写入语义（UPDATE 语义、跨 AL2/AL3/AL5）从未落地，于是这条触发**永远不可能命中**；
"冲突记忆"则依赖尚未实现的 `MERGE` 与置信度比较。**裁决**：本轮如实登记为未实现（`OptimizationReport`
的 docstring 已同步），并在**拆解阶段**为"`superseded_by` 的生产者 + 错误/冲突判定"新增一个任务
（编号续接 `T-AL2-16`，不重排既有编号），标注依赖 UPDATE 语义。

#### 三-a "不可达"与"降级"的落地判据（评审 P1-14 补：此前只有定性描述，无法复核）

**不可达（→ `forget`，物理删除）当且仅当同时满足全部五条**（`consolidation.py::Optimizer.run`）：

| # | 判据 | 实现 |
|---|---|---|
| 1 | **层可治理** | `layer ∉ {core, procedural}`（人格与技能永不触碰）且 `status == 'active'` |
| 2 | **已定形** | 无来源会话（全局 / 核心类），或来源会话 `session.status == 'committed'` |
| 3 | **无实体关联** | 实体表里没有任何实体的名字出现在该记忆的 `subject` / `object` / `content` 中 |
| 4 | **不被概览覆盖** | 其 `abstract`（缺则 `content`）**前 40 字符**未出现在任何 L1 概览文本中 |
| 5 | **图上完全孤立** | `relations` 中**既无入边也无出边** |

**降级（→ `dormant`，可逆）**：第 1–4 条相同，第 5 条换成 **无入边且 `importance < 0.5`**
（`LOW_IMPORTANCE_THRESHOLD`）。**有入边的记忆一律不动**——被引用的东西不该降级。

两条纪律（必须能断言）：

- 判据 2 是**生命周期**判据（会话是否已收尾），**不是时间判据**——会话未结束时宣称"孤立"会把刚写下、
  下轮才被关联的记忆误删；这正是 §一"不用时间/频率"的直接落点。
- 判据 4 是**启发式**（前 40 字符子串匹配）。它必须与 L1 概览的生成策略**同源**，否则会出现
  "概览其实收了它、判定却说不覆盖"的漏删；概览生成策略改动时**必须同步**这里的取样方式。
- 判据 3 在实体表为空时整体退化：此时不可达判定**偏保守（可能漏报）**，报告 `notes` 里必须留话
  （实现已如此）。

> **⚠️ 扫描上限（评审 P1-18）**：`Optimizer.run` 以 `scan_limit = 5000` 取候选，且
> `_overview_covered()` / `_committed_sessions()` 各自**再全表扫一遍**。**记忆总数超过上限后，
> 治理会静默失效**（只治理前 5000 条）。裁决方向：跨会话计数与覆盖判定应走聚合查询
> （`COUNT(*)` / 带索引的子查询），而不是拉全行到 Python 里数；在改造前，**上限必须写进文档并在
> 超限时告警**。

**发起形式**：可**系统自主**，也可**用户确认**（可配置）。用户的态度是：*"不管何种形式，保留记忆删除的操作记录用于审查或者恢复就可以兜底防止误删。"*

#### 四、删除的两个入口（D-17）

| # | 入口 | 触发者 | 说明 |
|---|---|---|---|
| **1** | **记忆优化删除** | 记忆优化任务（离线，LLM 驱动） | 发现"错误 / 冲突 / 不可达"时发起；系统自主或用户确认 |
| **2** | **用户显式要求** | 用户 | `aspirit forget <id>` / `spirit_forget`，须显式确认（默认 dry-run） |

**不在两个入口内的一律不删除。** `aspirit decay` **只做降级与重排，不做删除**。

> **合规删除**不单列第三项——它走入口 2，但需**额外清除审计快照**（见 §五 例外）。

#### 五、安全网：删除可恢复（D-22）

用户的关键判断：**"保留记忆删除的操作记录用于审查或者恢复，就可以兜底防止误删。"**

于是删除的安全性**不靠"禁止删除"，而靠"可恢复"**：

| 机制 | 实现 |
|---|---|
| **快照留底** | `audit(op='forget').before` 保留被删记忆的**完整快照**（含 `content` / 层级 / 时间 / 置信度 / 关联） |
| **同事务** | 审计与删除**同一事务**提交；审计不得因删除而消失 |
| **可恢复** | `aspirit restore <audit_id>`（/ `spirit_restore`）从快照重建记忆 |
| **恢复也留痕** | 恢复本身写 `audit(op='restore')`，形成完整链条 |
| **请求可审** | `actor`（`optimizer` / `user`）、`reason`（必填）、`source`（入口）、`ts` 全部落库 |

**唯一例外 · 合规删除**：

> 合规要求的是"**数据真的不存在**"，因此合规删除**必须同时清除 `audit` 中的快照**——这是**唯一不可恢复**的删除。
> 该操作本身仍需留痕（记录"何时因合规要求执行了不可恢复删除"，但**不留内容快照**）。

#### 六、硬约束

- 所有状态变更与删除**必须写 `audit`**（INV-7 / INV-8）
- **先算候选集，再处置**；候选集与处置结果都要可见
- CLI `aspirit decay` **默认 dry-run**；`aspirit forget` **默认 dry-run 且须显式确认**
- `decay.py`（降级/排序）中**不存在删除调用**——删除只发生在 `consolidation.py` / `review.py` 的优化路径（对应可断言的代码级红线）
- 衰减参数（`w` / `τ_fast` / `β`）可配置；**不再有 `θ_forget` 阈值**
- **降级与删除都不允许由时间或频率直接触发**——这是 D-16 的核心纪律

### M4 扩散激活与 Hebbian 学习

```
on_co_access(a, b):
    w_ab ← w_ab + η · (1 − w_ab)          # 共激活增强，趋近 1

召回时：
    activated = seeds ∪ { x | w(seed, x) > θ_diffusion }   # 一跳扩散
```

- 关联边落 AL3 的 `relations`（`rel_type='co_activation'`）
- **边权与 `co_count` 双写**：权用于打分，计数用于巩固判定
- MVP 只做**一跳扩散**；多跳（`hops > 1`）留到 M4 之后（避免组合爆炸）

### M5 多因子融合召回

```
score = α · semantic    （AL3.vector_search）
      + β · importance  （重要度：置信度 / 显著性 / 用户标注 / 被反复引用，D-20）
      + γ · recency     （时间邻近度，内含 Wixted 衰减形状，D-16）
      + δ · entity      （实体命中度，D-10）
      + ε · diffusion   （扩散激活值）
      + ζ · core        （与核心记忆的一致性）
```

> **D-20 的关键改动**：原 `vitality`（`strength` 与 `access_count` 合成）被**拆开**——
> - `recency` 承担**时间邻近**（这是统计规律）
> - `importance` 承担**重要度**（与访问频率**解耦**）
>
> 拆分理由：`strength` 内部本就含时间衰减，与 `recency` 语义重叠；而"低频但关键"的记忆（身份证号 / 过敏史）**只能靠 importance 救回来**，频率给不了它任何分（见 [DES-RES-002 §4.1](../09-机制决策依据研究.md)）。

- 默认权重见 `RecallWeights`（可配置，合计 1.0）
- 该式是斯坦福 generative agents `relevance + recency + importance` 的器灵化扩展（RES-001 §3）
- **各路原生分必须先归一化到 [0,1]**，否则不同量纲的分数相加无意义

**`entity` 因子的取法**（D-10）：

1. 从 query 中抽取候选实体名（与 AL3 `entities.name` / `aliases` 做匹配，**无需 LLM**）
2. 命中实体的记忆：按命中强度打分（完全匹配 > 别名匹配 > 部分匹配）
3. 再叠加 `relations(rel_type='mentions')` 的连通度作为加权

> **成本口径（评审 P0-4 修正）**：实体抽取在**提取期**已完成，因此召回期"只是多一路 SQL 匹配"。
> ⚠️ **当前实现违反本条承诺**：`recall.py::Recaller._entity_index` 在**每个命中实体的循环内**做一次
> 全表查询（`query(status=None, limit=500)`）＋ Python 侧 `casefold()` 子串匹配 —— 真实成本是
> `实体数 × 500 行`，并且因 `limit=500` **无排序保证**而在高基数下**静默漏匹配**（"排序上浮"的承诺失效）。
> **裁决待定，见评审 §13.8-1**（推荐：删兜底，只走 `entity_find` + `relations(mentions)` 索引）。

**与业界成熟检索范式的对照**（评审 DES-REV-001 §3.1——把"放弃什么 / 用什么补偿"写明）：

| 业界做法 | 器灵裁决 | 补偿方案 |
|---|---|---|
| 意图分析（LLM 生成 0–5 个 typed query） | **放弃** | 成本不可接受（`prefetch` 是每轮热路径）；用多因子融合替代 |
| 目录递归下钻 | **放弃** | 器灵无目录树（结构化真相源，INV-1）；用**层过滤 + `scope` 过滤**替代。**注意区分**：放弃的是"以目录树为存储 / 检索主结构"，**不是**知识型记忆的"**目录卡**"（每条记录上的短索引字段，见 [DES-RES-004 §3.2](../11-知识型记忆研究与机制增补.md)） |
| rerank（额外模型/LLM 打分） | **放弃** | 命中热路径；用**六路加权融合**替代 |
| 语义 + BM25 混合 | **采纳** | 语义（`vector_search`）+ BM25（`keyword_search`）并行 |
| 实体匹配 | **采纳（D-10）** | 纳入第六因子，见上 |

> 这几项**不是遗漏，是主动放弃 + 有补偿**。写在正文里，避免未来读者误判为设计缺口。

**召回流程**：

```
1. 向量化 query（AL4）——失败则跳过向量路（降级 BM25）；semantic 因子缺席 → 权重重归一化
2. 多路召回：semantic（vector_search） / BM25（keyword_search） / 结构索引（core_terms、entity、diffusion）
3. 各路 → Scored 明细（raw 六键，便于解释"为什么召回这条"）
4. normalize_minmax()：各因子归一化到 [0,1]（全相等时取全 1，不退化为 0）
5. fuse() 加权融合（缺因子的权重重归一化）
6. [:top_k] 截候选 → clip_to_budget()：L0 摘要优先，**至少保留 1 条**
7. 输出带层标签与来源文本
```

> **⚠️ 第 2 条的实现现状（评审 P1-16 / P1-14）**——写实，避免下一个人按声明去"修 bug"：
>
> - **工作记忆路名存实亡**：`layers/working.py::WorkingLayer.recall()` 从未被调用；`Recaller.recall` 里
>   只做 `for chunk in backend.wm_list(...)` 后**把结果丢弃**——热路径白付一次 I/O，工作记忆候选
>   既不返回也不参与打分。**待裁决（§13.8-7）**：接通该路 / **删掉该路声明与空转 I/O**（推荐后者：
>   工作记忆的"当场价值"已由宿主上下文承担，塞进六因子会与 `wm_put` 的注意力语义重复）。
> - **episodic 时间窗路未实现**：M2 实际只有 semantic + BM25 + 结构索引。**待裁决（§13.8-7）**：
>   倾向改文档——时间窗检索应随 M3+ 的时间语义（`valid_from`/`valid_to`）一起做，不属于 M2。

**可解释性要求**：`Scored.raw` 必须保留，`spirit_recall` 需能展示"这条为什么被召回"。

### M6 工作记忆容量管理（4±1）

> **依据**：容量限制作用于**组块数**而非信息量（约 4±1），所以上限取 5 并允许同话题合并累积——见 [DES-RES-003 §4.6](../10-神经科学依据与机制映射.md)。

```
容量上限 = 5 个组块（Baddeley 4±1 的上界）
超出时：淘汰 act_count 最低且 last_touched 最早的组块
组块聚类：按 chunk_key（话题标识）归并，同话题的累积激活
```

- 被淘汰的组块**不是丢弃**——它会随会话结束一并固化进情景记忆
- **"淘汰" = 移出注意力窗口，不删数据**（评审 P1-12：此前只写"淘汰"，容易被读成删除）：
  被淘汰的组块退出**注意力容量**（不再参与 `system_prompt_block` / 上下文注入），但**仍保留在工作记忆存储里**
  （AL3 的 `wm_*` 表——工作记忆本来就不进 `memories` 长期记忆表），会话收尾时作为 episode 固化。
  因此"容量 5"约束的是**注意力**，不是**存储**。"淘汰一次"与"存了 30 个组块"并不矛盾。
- 意图槽（前瞻记忆）独立于组块容量，不受 4±1 限制

### M7 前瞻记忆（意图槽）

```
从对话识别"待办 / 计划 / 目标" → intents 表
召回时不仅看"过去相关"，也看"未来相关"
```

- MVP：**表已建**；当前落地的是**关键词启发式最小实现**（评审 P1-15）
- 识别同样走提取器（`type='intent'`），不新增 LLM 通道

> **⚠️ 实现现状（评审 P1-15）**：`core/layers/working.py` 用 `INTENT_MARKERS` 关键词表在**工作记忆摄入时**
> 就地识别意图（写入 `chunk_key = '__intent__'` 的槽位，且**不受 4±1 容量限制**），并已被用例钉住
> （`test_intent_slot_not_limited_by_capacity`）。这与"MVP 逻辑 M4 补齐"的声明冲突。
> **待裁决（§13.8-6）**：① 承认启发式为 M2 的最小实现、M4 再换回提取器（**推荐**——意图槽**不落长期记忆**，
> 误判的代价只是多一条待办，漏判的代价是忘事）；② 立即回退，M2 不做意图槽。

### M8 提取器分工

| 组件 | 用什么 | 为什么 |
|---|---|---|
| `sensory` 显著性 | **零 LLM**（启发式） | 每轮都跑，不能烧 token |
| `extractor` 提取 | 小模型（`glm-5.3-flash`） | 高频，需结构化输出 |
| `dedup` 去重决策 | 小模型；M3+ 交叉验证 | 错了会污染记忆 |
| `summarize` L0 摘要 | 小模型 | 中频 |
| `consolidate` 巩固 | 大模型（`glm-5.3`） | 低频，质量优先 |
| `soul` 核心记忆 | 最强（`kimi-k3`） | 极低频，错了污染人格 |

### M9 分级加载 L0 / L1 / L2（P1 · 首要机制）

> 这是器灵**存在的原始动因之一**（DES-000 §1.1 P1）：贴合人类"模糊印象 → 大致判断 → 回想细节"的回忆过程，同时优化 LLM 注意力——**不把几十年的记忆一次性灌进上下文**。
>
> **机理与判据**：分级的动因是**注意力预算**（稀缺的是注意力而非存储），可用约束为"同时在线的信息量有上限"，判据是**注意力效率**（被引用条数 / 注入条数）——见 [DES-RES-003 §2](../10-神经科学依据与机制映射.md)。

**三级定义**：

| 级 | 是什么 | 体量 | 生成时机 | 存储 |
|---|---|---|---|---|
| **L0** | 单条记忆的一句话摘要 | ~50 token | 写入后异步（与提取同批） | `memories.abstract` |
| **L1** | **主题 / 实体级概览**——"关于 X 我大致记得什么" | ~2k token | 首次需要时生成 + 缓存；底层变更置 stale | `overviews` 表 |
| **L2** | 原始记忆 + 来源会话 + 关联链 | 按需 | 实时组装 | `memories.content` + `relations` + `audit` |

**认知对齐**：

```
L0  "好像有这回事"          → 模糊印象，用于快速筛掉不相关的
L1  "关于 X 我大致记得…"     → 判断有没有用，决定要不要深入
L2  "当时具体是这么说的…"    → 努力回想，拿到完整细节
```

**召回时的三级展开**：

```
1. 粗筛   在 L0（abstract）上做 向量 + BM25 + 实体匹配 → 候选集
2. 判断   命中主题/实体 → 拉 L1 概览 → 交给 LLM 判断相关性
3. 展开   确认需要的条目 → 展开 L2（原文 + 来源 + 关联）
4. 裁剪   token 预算不足时，按 L2 → L1 → L0 的顺序**优先舍弃细级**
```

**LLM 注意力收益**：默认只注入 L0（便宜、覆盖广）；需要时升到 L1（成本可控）；极少情况才注入 L2（贵但精确）。

**硬约束**：

- **L1 是缓存，不是真相源**（INV-1）：底层记忆变更 → `overviews.stale = 1` → 下次读取时重生成
- ⚠️ **L1 不得在热路径生成**（评审 N-P0-2）：`prefetch` 有 300ms 护栏，**同步调 LLM 生成 L1 会直接击穿它**。因此：
  - **预生成**：写入后异步生成（与 L0 同批）
  - **重算异步**：发现 `stale` 时，**热路径只读旧缓存或降级到 L0**，重算投递到 AL5 maintenance
  - 冷路径（`spirit_expand` 显式调用、`aspirit review`）允许等待生成
- **L0 缺失时可退化**：摘要生成失败 → 截断 `content` 首句，不得让召回失败
- 三级展开**必须在 token 预算内完成**，不得因展开而超预算
- L2 展开**只读**，不得触发写入
- **知识型记忆默认不常驻注入**（`type="knowledge"`）：平时以 L0 **目录卡**参与候选，命中后只注入**片段 + 出处**；注入预算与单片段上限见 [DES-RES-004 §3.5](../11-知识型记忆研究与机制增补.md)（**提案，未生效**）

### M10 记忆进化（A-MEM 式 · M4 · D-12）

> 借鉴 A-MEM 的 Zettelkasten 自组织：**新记忆入库时，触发对相邻记忆的更新**。器灵此前只做了"建边"（Hebbian），没做"被激活记忆的自我更新"。

**流程**：

```
新记忆落库
  └─▶ 取 relations 上的一跳邻居（上限 N 条）
        └─▶ 对每个邻居：更新 abstract（L0）与关联边
              └─▶ 写 audit(op='evolve')
```

**硬约束（红线）**：

| # | 约束 | 理由 |
|---|---|---|
| 1 | **只改 `abstract` 与 `relations`，绝不改 `content`** | `content` 是事实来源；改写它等于篡改历史，破坏 INV-1 与可审计性 |
| 2 | 每次入库的进化范围**有上限 N** | 防连锁反应与成本爆炸 |
| 3 | 每次进化**写 `audit`** | 必须可追溯"这条摘要为什么变了" |
| 4 | **可整体关闭**（配置开关） | 属于 A 档增强，不能影响 P 档可用性 |
| 5 | 只更新 `stale=1` 的 L1 概览，不主动重算全部 | 避免 O(n²) 开销 |

> **与 P 档的关系**：M10 是纯增强。若它与 M9（分级加载）冲突——例如进化导致概览频繁失效——**优先保 M9**（D-14：P 档优先）。

### M11 审查、溯源与主动干预（P2 · 首要机制）

> 另一个原始动因：**运行一段时间后能逐条看到 Agent 记住了什么**，从而具备复盘与主动优化能力。人类可读不是"附带好处"，是一等公民。

**三件事**：

| 能力 | 接口 | 说明 |
|---|---|---|
| **审查** | `CoreFacade.review()` | 逐条列出：层 / 内容 / L0 摘要 / 时间 / 置信度 / 来源会话 / 状态 |
| **溯源** | `CoreFacade.trace(mem_id)` | 该条记忆的完整变更史（从 `audit` 重放） |
| **干预** | `CoreFacade.correct(mem_id, patch, reason)` | 修正内容 / 调层 / 改状态 / 调置信度 / 删除 |

**主动干预的红线**：

- **必须走正常写入路径**（产出 `WriteIntent` → AL5 writer → AL3），**不得绕过 `audit`**
- 干预的 `audit.actor` 记为 `user`，与自动流程（`extractor` / `consolidator` / `decay`）可区分
- 用户修正后，相关 `overviews` 置 `stale`
- 破坏性干预（删除 / 归档）**默认 dry-run**

> **可解释性要求**：`review` 与 `trace` 的输出必须**人类可读**——这是 V2 核心价值的直接兑现。

---

## 6. 编码注意事项

| # | 注意点 | 说明 |
|---|---|---|
| C1 | **AL2 内禁止任何 `import` 的 I/O 库** | 违反 R5；CI 架构测试会拦 |
| C2 | **时钟与 ID 生成器必须注入** | 否则衰减与 ULID 无法稳定测试 |
| C3 | **分数归一化** | 融合前各路必须归一到 [0,1]（M5） |
| C4 | **保留 `raw` 明细** | 可解释性是一等公民，不允许只留总分 |
| C5 | **提取输出必须先校验再落库** | 校验失败的记忆不得进入 AL3 |
| C6 | **巩固幂等** | 同一 session 重复触发巩固不应产生重复记忆 |
| C7 | **衰减必须可 dry-run** | 计算与处置分两步，处置前可审查候选集 |
| C8 | **禁止在 AL2 里写 SQL 或拼 HTTP** | 一律走协议 |
| C9 | **权重与阈值集中配置** | 不散落在各处魔法数字 |
| C10 | **`scope` 语义要一致** | `global` / `project` / `session` 三档，召回时据此过滤 |
| C11 | **程序性记忆 MVP 只建表** | 不要提前实现固化逻辑（范围蔓延） |
| C12 | **提示词与 schema 同源** | 提示词中的字段说明与 `extract/schema.py` 的校验规则必须一致，避免"提示说 A、校验要 B" |

---

## 7. 错误处理与降级

| 故障 | AL2 行为 |
|---|---|
| **F1** LLM 不可用 | 跳过提取，**仅存原文** + 告警。记忆不丢，只是未结构化 |
| **F2** embedding 不可用 | 显著性打分置 `w1=0` 并归一化；召回跳过向量路，只用 BM25 |
| **F3** 提取返回非 JSON | 由 AL4 重试 1 次；仍失败 → 仅存原文 + `audit(reason='extract_parse_failed')` |
| **F9** 成本超预算 | 降级为纯启发式：不做 LLM 提取，仅按显著性存原文 |
| **F6** 召回超时 | 返回核心记忆摘要保底（保证 `system_prompt_block` 非空） |
| 巩固时后端不可用 | 记 `audit` 并放弃本次巩固（下次重试），不产生半成品 |
| 衰减阈值取值异常 | 参数校验在配置装载期拦截（AL5），AL2 假定配置合法 |

---

## 8. 独立验收标准

**测试环境**：内存 fake backend + fake LLM + fake embedding + 可注入的固定 clock。**不连任何网络与数据库。**

### 8.1 显著性（M1）

- [ ] 明确指令（"记住 X"）的打分显著高于中性句
- [ ] 与已有记忆高度相似的输入打分显著低（新颖度惩罚生效）
- [ ] 低于阈值不产出落库候选
- [ ] embedding 不可用时退化为纯规则打分且不抛错
- [ ] **降级态门槛同量纲缩放**（P1-19 补）：embedding 不可用时 `θ' = θ_salience × (1 − w1) = 0.245`，注入 `请记住：我对花生过敏`（规则分仅 0.298）**仍能落库**——直接钉住 W4 缺陷（只归一化分数、不缩放门槛 → 写入被永久关闭）

### 8.2 巩固（M2）

- [ ] 会话结束产出 1 条 episode
- [ ] 同一主题跨 ≥N 会话后被提升为语义记忆
- [ ] 提升产生 `derived_from` 关联边
- [ ] 同一 session 重复巩固不产生重复记忆（幂等，C6）

### 8.3 衰减排序与删除治理（M3 · D-16 / D-17 / D-20）

- [ ] `strength(t)` 随 Δt 单调递减
- [ ] 近期衰减快于远期（指数项主导 → 幂律项主导的转折可见）
- [ ] 访问后 `strength` 回升
- [ ] `decay()` **不产生任何 `hard_delete` 调用**（红线断言）
- [ ] **降级与删除均不由时间或频率触发**（D-16 纪律断言：注入"很久未访问"的记忆 → 不自动降级/删除）
- [ ] `dormant` 记忆**详情仍在库中**，`expand(ref,'L2')` 仍可取到（D-23）
- [ ] `dormant` 记忆**仍参与召回**（降权：总分 × `dormancy_penalty`，默认 0.6），且**不会被自动展开到 L2**（需显式 `expand(ref,'L2')`）（P1-13 修正：此前写作"L0 仍参与召回候选"，与实现不同）
- [ ] **低频但高重要度**的记忆（如身份证号）在排序上**优于**高频低重要度的记忆（D-20 核心断言）
- [ ] `importance` 与 `access_count` 独立：修改 `access_count` 不影响 `importance`
- [ ] 降级只产生 `active → dormant`，**不存在 `archive` 目标状态**
- [ ] **"不可达"五条判据逐条可翻红**（P1-14 补）：会话未收尾 / 有实体关联 / 被概览覆盖 / 有入边 / 有出边——**任一成立即不得删除**（只可能降级）
- [ ] 记忆总数超过 `scan_limit`（5000）时治理**必须告警**——不得静默只治理前 N 条（P1-18 补）
- [ ] `dormant` 记忆可被显式搜索命中，且可恢复为 `active`
- [ ] `forget` 仅由白名单触发；非白名单调用抛错
- [ ] `dry_run=True` 时不产生任何写操作
- [ ] 每次状态变更都产生 `audit` 事件
- [ ] **删除时 `audit(op='forget')` 与 `hard_delete` 同事务**，且审计在删除后仍可读（不得级联删除）
- [ ] 删除审计含 `actor='user'`、`reason`（非空）、`source`（入口来源）

### 8.4 激活（M4）

- [ ] 共激活后边权按 `η·(1−w)` 增长
- [ ] 反复共激活时权重趋近 1 但不越界
- [ ] 一跳扩散返回的节点满足 `w > θ`

### 8.5 召回（M5）

- [ ] 融合前各路分已归一化到 [0,1]
- [ ] 调整权重可改变排序（权重生效）
- [ ] `raw` 明细包含全部**六个**分量：semantic / importance / recency / entity / diffusion / core
- [ ] **`importance` 与 `recency` 相互独立**：单独改变其一能独立影响排序（D-20）
- [ ] **`entity` 因子生效**：query 命中实体名时，提及该实体的记忆排序上浮（D-10）
- [ ] **`entity` 因子成本有上界**（P0-4 补）：注入 N 条与实体无关的记忆后，单次 `recall` 在该路的查询行数**不随库规模线性增长**
- [ ] **边界语义**（P2-9 补）：`clip_to_budget` **至少保留 1 条**（首条即超预算也返回）；全 0 分时返回 `below or fused`（不空手而归）
- [ ] `clip_to_budget` 优先保留 L0 摘要
- [ ] 向量路不可用时仅用 BM25 且不抛错

### 8.6 工作记忆（M6）

- [ ] 组块数超过 5 时自动淘汰最冷组块（`act_count` 最低且 `last_touched` 最早）
- [ ] **被淘汰的组块仍在存储中**——只退出注意力容量，**不删数据**（P1-12 补）
- [ ] 同话题输入累积到同一 `chunk_key`
- [ ] 意图槽不受组块容量限制

### 8.7 提取（M8）

- [ ] 合法 JSON 通过校验并产出候选
- [ ] 非法 JSON 不产出任何候选（不得部分接受）
- [ ] 去重决策 `ADD/UPDATE/IGNORE` 三个分支各有用例
- [ ] 决策为 `MERGE` 时**审计如实记账**：`op='add'` + `after.downgraded_from='merge'`（**不得**写 `op='merge'` 而实际新增）（P0-6 补）
- [ ] **代码扫描：AL2 内无 `httpx` / `sqlite3` / `openai` / Hermes 导入**

### 8.8 分级加载（M9 · P1）

- [ ] `expand(ref, 'L0')` 返回一句话摘要
- [ ] `expand(ref, 'L1')` 返回主题级概览
- [ ] `expand(ref, 'L2')` 返回原文 + 来源 + 关联链
- [ ] **L0 缺失时退化**为 `content` 首句截断，不报错
- [ ] 底层记忆变更后，对应 `overviews.stale = 1`
- [ ] **热路径不生成 L1**（N-P0-2）：`stale` 时 `prefetch` 只读旧缓存或降级 L0，**不产生 LLM 调用**（断言无 model 调用）
- [ ] 三级展开在 token 预算内完成，超预算时**先舍 L2**
- [ ] `expand(..., 'L2')` **不产生任何写操作**（只读断言）
- [ ] 召回默认只用到 L0/L1，L2 仅在明确需要时展开（注意力收益可观察）

### 8.9 记忆进化（M10 · M4）

- [ ] 新记忆入库后，一跳邻居的 `abstract` 被更新
- [ ] **`content` 未被修改**（红线断言）
- [ ] 每次进化产生 `audit(op='evolve')`
- [ ] 单次入库的进化邻居数不超过上限 N
- [ ] 关闭开关后，进化完全不发生
- [ ] 与 M9 冲突时（概览频繁失效）不影响召回正确性

### 8.10 审查与干预（M11 · P2）

- [ ] `review()` 返回字段包含层 / 内容 / 摘要 / 时间 / 置信度 / 来源会话 / 状态
- [ ] `trace(mem_id)` 按时间序返回全部变更事件
- [ ] `correct()` **产出 `WriteIntent` 而非直接写库**（不绕过 AL5 writer）
- [ ] `forget()` 缺 `reason` / `source` 时**拒绝执行**（N-P0-1）
- [ ] `restore(audit_id)` **能完整恢复**被删记忆（含 ID 与关联），且恢复本身写 `audit(op='restore')`（D-22）
- [ ] `forget(purge_snapshot=True)` 后**不可恢复**（合规路径，D-22 例外）
- [ ] **全代码库中通向 `hard_delete` 的路径只有 `forget()` 与优化删除两处**（路径唯一性断言）
- [ ] 干预产生的 `audit.actor` 可区分（`user` / `optimizer`）
- [ ] 干预后相关 `overviews` 置 stale
- [ ] 破坏性干预默认 dry-run
- [ ] `review` / `trace` 返回**结构化数据**（`list[dict]`）；**人类可读渲染属 AL5**（`observability/render.py`，INV-1）
  ——评审 P1-17 修正：此前写作"输出为人类可读文本（V2 验收）"，与 INV-1 冲突且与实现相反

---

## 9. 集成验收关注点

| 关注点 | 验证方式 |
|---|---|
| 与 AL3 的落库一致 | 提取候选经 AL5 writer 落库后，`get()` 字段完整 |
| 与 AL4 的异常契约 | 三个降级路径（F1/F2/F9）各跑一次真实链路 |
| 与 AL5 的后台驱动 | `on_session_end` 后巩固确实在 maintenance 线程执行 |
| 与 AL1 的可解释性 | `spirit_recall` 能展示召回原因（`raw` 分量） |
| 质量对标 | 以 LoCoMo / LongMemEval 为标尺做端到端评测（RES-001 §5） |

---

## 10. 待决策 / 未决项

> **口径（评审 P2-14 补）**：本表分两类——**"决策未定"**（还在二选一）与**"已定默认值、数值待调优"**（实现已固化，缺的是真机数据拟合）。
> 此前两类混在一张表里，导致"M2 已落地的实现"与"仍标待定的未决项"长期并存：**读者无法判断某条到底有没有权威默认值**。

| # | 事项 | 现状 | 影响 |
|---|---|---|---|
| 1 | 显著性各权重与阈值默认值 | **✅ 已定默认（实现固化）**：`SalienceWeights` = novelty 0.30 / instruction 0.25 / entity 0.20 / emotion 0.15 / core_deviation 0.10，`θ_salience = 0.35`，`novelty_top_k = 5`（已写入 §5 M1）；**数值仍需真机调优** | 记忆质量 |
| 2 | 巩固的跨会话命中阈值 N | **✅ 已定默认**：`promotion_threshold = 2`（`CoreSettings` 可覆盖）。**仍需验证 N=2 的产出量是否合适** | 语义记忆产出量 |
| 3 | 复用计数阈值 K | **决策未定**（程序性记忆 MVP 仅占位） | 程序性记忆固化时机 |
| 4 | 衰减参数 `w / τ_fast / β` | **✅ 已定默认**：`DecayParams` = `w 0.6 / τ_fast 7.0 / β 0.5`（`base_retention 1.0`）；**仍需真实数据拟合** | 遗忘曲线形状 |
| 5 | 扩散阈值 `θ_diffusion` 与是否多跳 | **✅ 已定默认**：`threshold = 0.2`，MVP **一跳**（`hops > 1` 抛 `NotImplementedError`，非静默降级）；**多跳仍待 M4 后评估** | 召回召回率 |
| 6 | 融合权重默认值（D-06） | **✅ 已定默认**：`RecallWeights` 六因子合计 1.0（semantic 0.40 / importance 0.20 / recency 0.15 / entity 0.10 / diffusion 0.10 / core 0.05，见 §2.2）；**数值仍需调优** | 排序质量 |
| 7 | `glm-5.3-flash` 提取稳定性 | 需实测（见 LLD-AL4 §10） | 是否退到 `glm-5.3` |
| 8 | **六态更新 → MVP 三态的偏差** | **已决策**：MVP 做 `ADD/UPDATE/IGNORE`，M3 补 `MERGE/INVALIDATE/FORGET`（因后三者依赖时态与衰减模型） | 记录于 [DES-REV-001 §3.6](../07-设计评审.md) |
| 9 | **实体匹配信号**（评审 Q1） | ✅ **已决策：引入**为第六因子（D-10） | — |
| 10 | **双时态语义启用时机**（评审 Q2） | ✅ **已决策**：MVP 只留字段，M3 随 `INVALIDATE` 启用（D-11） | — |
| 11 | **A-MEM 记忆进化**（评审 Q3） | ✅ **已决策**：M4 部分引入，只改 `abstract` 与关联边（D-12） | — |
| 12 | L1 概览的 **scope 粒度**（实体 / 主题 / 层） | **✅ 已实现**：`entity`（优先，命中 `entities` 表）→ `topic`（退化到主题名）→ `scope` → `memory`（`_scope_of`），即"以 entities 为主、scope 为辅" | `overviews` 表设计 |
| 13 | L1 概览的**生成策略**（按需 vs 预生成） | **✅ 已实现：按需 + `stale` 重算**——热路径只读缓存/降级 L0，重算投递 `overview_invalidate` 给 maintenance（N-P0-2）；冷路径现场生成并 `overview_put` | 成本与延迟 |
| 14 | M10 单次进化的**邻居上限 N** | 待定（建议 5） | 成本 |
| 15 | **衰减只作排序信号** | ✅ **已决策**（D-16）——移除强度触发删除 | — |
| 16 | **三级遗忘降为两级**，取消 `archive`；删除白名单仅两项 | ✅ **已决策**（D-17） | — |
| 17 | **`vitality` 拆为 `recency` + `importance`** | ✅ **已决策**（D-20） | — |
| 18 | **实验特性清单**（RIF / 模式分离 / 交错重放 / 元记忆）默认关闭 | ✅ **已决策**（D-19） | 见 [DES-RES-002 §6.1](../09-机制决策依据研究.md) |
| 19 | 合规删除的**独立入口** | ✅ **已决**（D-22）：并入"用户显式要求"，但以 `purge_snapshot=True` 走**不可恢复**路径 | — |
| 20 | `importance` 的**计算方式** | ✅ **已决**：`用户标注 > 置信度 > 显著性 > 被引用次数`，归一化加权；**必须能从 `spirit_review` 看到构成**；参数后续迭代调优 | — |
| 21 | **"不可达"的判定阈值**（入边数 / 实体关联数 / 是否在 L1 概览中） | 待定——建议 M3 实现时用真实数据校准 | 删除精度 |
| 22 | **降级（active → dormant）的判据** | 待定——建议"长期未进入任何活跃主题簇 / L1 概览"，仍**不得用纯时间或频率** | 降级精度 |
| 23 | **`superseded_by` 的生产者**（错误 / 冲突记忆的判定源） | **决策未定**——字段有 schema 与读取方，但 AL2/AL3/AL5 三处**均无写入方**；故 §5 M3 §三 的"错误 / 冲突"两类删除触发**当前不可能命中**（评审 P0-7）。需随 `UPDATE` 语义落地，拆解阶段登记为 `T-AL2-16` | D-23 删除路径完整性 |
