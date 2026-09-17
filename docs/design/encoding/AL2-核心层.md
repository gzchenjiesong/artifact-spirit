# AL2 核心层 · 编码任务

> 上游：[ENC-000 总纲](./README.md) / [LLD-AL2 核心层详细设计](../layers/AL2-核心层详细设计.md) / [DES-RES-002 机制决策依据研究](../09-机制决策依据研究.md)
> 里程碑覆盖：**M1 – M5**（M0 无 AL2 任务）
> 任务数：**23**（M1–M2: 15；M3: 3；M4: 4；M5: 1——本轮随 DES-REV-005 的复核结论补齐，编号顺延不重排）
> **本层是纯逻辑层**——任何任务中出现 I/O 库导入即为缺陷（R5）

---

## M0 期的并行工作（不计入 M1/M2 任务）

AL2 可在 AL3/AL4 之前用 **fake backend + fake model** 开工。开工前先做 T-AL2-01（接口与数据类），其余任务即可并行。

---

## M1 · 召回与首要目标（T-AL2-01 ~ 08）

### T-AL2-01 · `CoreFacade` 与共享数据类

- **里程碑**：M1
- **依赖**：无
- **产出**：`core/facade.py`、`core/recall.py` 中的数据类
- **要求**：
  1. `CoreFacade` 方法**全集**（按实现，**不是** LLD 的旧副本）：`ingest_turn` / `recall` / `expand` /
     `review` / `trace` / `correct` / `forget` / `restore` / `on_session_end` / `consolidate` /
     `decay` / `health` / `system_prompt_block` / **`optimize`** / **`drain_pending`** / **`refresh_overviews`**
  2. 数据类：`Scored`（含 `raw` 六分量）、`RecallQuery`、`RecallWeights`、`WriteIntent`、`TurnContext`
  3. `RecallWeights` 默认值：`semantic .40 / importance .20 / recency .15 / entity .10 / diffusion .10 / core .05`
  4. `MemoryRecord` 从 `store/base.py` 导入（R1 允许协议与数据类）
  5. **`WriteIntent.op` 必须穷举登记**（经 DES-REV-005 P1-9 补齐后共 **17** 个）：
     `put` / `update` / `set_status` / `forget` / `restore` / `wm_put` / `wm_delete` /
     `session_create` / `session_end` / `link` / `reinforce` / `touch` / `entity_upsert` /
     `audit` / `overview_put` / `overview_invalidate`——**新增一个就同步 `INTENT_PRIORITY`**
  6. **签名细节**（按实现，DES-REV-005 P1-8）：
     - `expand(ref, level, *, hot_path=True)` —— `hot_path=False` 表示"显式下钻"，允许生成 L1
     - `on_session_end(...)` 返回 **`ConsolidationReport`**（含 `intents` / `skipped` / 统计）
     - `decay(*, now=None, dry_run=...)` —— `now` 可注入（测试与重放）
     - `system_prompt_block(*, token_budget=...)` —— **有默认值**，调用方可以不传
     - `system_prompt_block` **不是** LLD 写的那样只返回 str：它必须**自己守预算**（见 T-AL1-12）
  7. **`Metacognition` Protocol 删除**（P1-10）：门面即调度中枢，**不为无消费者造抽象**
- **约束**：R1（只依赖协议）；R5（禁止 I/O）；DES-REV-005 P1-8 / P1-9 / P1-10
- **验收**：
  - [ ] `CoreFacade` 的方法集合与**实现**逐字一致（`inspect` 白名单，**两个方向都比**：不多也不少）
  - [ ] `WriteIntent.op` 的取值集合与 `runtime.writer.INTENT_PRIORITY` **互为子集**（任一新增未登记即红）
  - [ ] `RecallWeights` 默认值合计为 1.0
  - [ ] `Scored.raw` 为六键字典
  - [ ] **`core/` 内无 `httpx` / `sqlite3` / `openai` / Hermes 导入**（R5 断言）

### T-AL2-02 · 显著性过滤（M1）

- **里程碑**：M1
- **依赖**：T-AL2-01
- **产出**：`core/salience.py`
- **要求**：
  1. 五因子启发式打分：信息新颖度 / 显式指令信号 / 实体密度 / 情绪强度 / 核心记忆一致性偏移
  2. **零 LLM**——新颖度需一次 embedding；**embedding 失败时置 `w1=0` 并重新归一化**，不得抛错
  3. 权重与阈值（默认 0.35）可配置，**不得散落魔法数字**（C9）
  4. **权重数值必须写进文档**（P2-8）：`novelty 0.30 / instruction 0.25 / entity 0.20 / emotion 0.15 / core_deviation 0.10`
  5. **降级时门槛必须同量纲缩放**（P1-19，**这是 M0–M2 修过的 W4 缺陷**）：
     `w1` 置 0 之后，其余权重需重新归一化，**门槛要同步缩放**（实现取 `0.35 × 0.70 = 0.245`）——
     否则降级态下**写入被永久关闭**：分数永远够不到那个没跟着缩的门槛
- **约束**：C9 / DES-REV-005 P1-19 / P2-8
- **验收**：
  - [ ] 明确指令（"记住 X"）得分显著高于中性句
  - [ ] 与已有记忆高度相似的输入得分显著低
  - [ ] 低于阈值不产出落库候选
  - [ ] embedding 不可用时退化为纯规则打分且不抛错
  - [ ] **降级态下门槛同步缩放**：同一句话在"有 embedding"与"无 embedding"两种模式下**都能过线**
    （反过来断言：**故意不缩放门槛，这条必须红**——P1-19 的注入式验证）

### T-AL2-03 · 工作记忆容量管理与意图槽（M6 / M7）

- **里程碑**：M1
- **依赖**：T-AL2-01
- **产出**：`core/layers/working.py`
- **要求**：
  1. 组块聚类（按 `chunk_key`），容量上限 **5**（Baddeley 4±1 上界）
  2. 超容量淘汰 `act_count` 最低且 `last_touched` 最早的组块
  3. 意图槽**不受组块容量限制**
  4. **"淘汰" = 移出注意力窗口，不是删数据**（P1-12）——被淘汰组块随会话结束一并固化进情景记忆
  5. 意图槽的判定实现是 **`INTENT_MARKERS` 关键词启发式**（P1-15）：承认它是最小实现并写明
     **为什么够用**（不落库、只在工作记忆内），**不等 M4 的提取器**
- **约束**：D-20 / DES-REV-005 P1-12 / P1-15
- **验收**：
  - [ ] 组块数超 5 时自动淘汰最冷组块
  - [ ] **淘汰不删数据**：被淘汰组块仍能随会话固化进情景记忆（P1-12）
  - [ ] 同话题输入累积到同一 `chunk_key`
  - [ ] 意图槽不受容量限制

### T-AL2-04 · 六层 `LayerService` 骨架

- **里程碑**：M1
- **依赖**：T-AL2-01
- **产出**：`core/layers/{sensory,working,episodic,semantic,procedural,core_memory}.py`
- **要求**：
  1. 统一 `recall(q)` / `candidates(ctx)` 接口
  2. `sensory` 不产出落库候选
  3. `procedural` MVP 仅占位（**不实现固化逻辑**，C11；`D-28~D-32` 未生效，见**为什么没做**：`docs/design/00` §10 台账标为"提案，未生效"）
  4. **共享行为用基类而非 Protocol**（P1-11）：实现是 `BaseLayerService` dataclass 基类——
     因为它要共享 `hydrate` / `layer` 的默认实现；**LLD 原先写成 Protocol 是错的**（无共享实现的 Protocol 只是噪音）
- **约束**：C11 / DES-REV-005 P1-11；R5
- **验收**：
  - [ ] 六个服务均可实例化并返回约定类型
  - [ ] `procedural` 无业务实现（占位断言）
  - [ ] **`BaseLayerService` 的共享方法只实现一次**（子类不得各自复制 `hydrate`）

### T-AL2-05 · 召回六因子打分

- **里程碑**：M1
- **依赖**：T-AL2-01 / T-AL2-04
- **产出**：`core/recall.py` 的六个 `score_*` 函数
- **要求**：
  1. `semantic` / `importance` / `recency` / `entity` / `diffusion` / `core`
  2. **`importance` 与 `access_count` 完全解耦**（D-20）：`用户标注 > 置信度 > 显著性 > 被引用次数`，归一化加权
  3. `recency` 内含 Wixted 衰减形状；`strength` **只喂给 `score_recency`**（D-16）
  4. `entity` 无需 LLM：query 实体名与 `entities.name` / `aliases` 匹配 + `relations(mentions)` 连通度
  5. **各路输出必须归一化到 [0,1]**
  6. **`entity` 必须是"多一路 SQL"，不是"循环里扫表"**（P0-4）：LLD §5 M1 写的是
     「成本为零：召回时只是多一路 SQL 匹配」，而实现曾写成 **实体循环内全表扫描 + 静默截断**——
     两者相反。要求：实体匹配走**一次关联查询**，**不得在循环里逐实体扫表**；
     若确有上限，必须**显式配置、显式计数**，不得静默截断
  7. **权重组装方式必须记载**（P2-9）：`importance` / `relevance` / `entity` 三路的分子分母怎么来的，
     必须能从文档复现——"跑出来是对的"不算记载
- **约束**：D-16 / D-20 / DES-REV-005 P0-4 / P2-9
- **验收**：
  - [ ] 六路输出均落在 [0,1]
  - [ ] 修改 `access_count` **不影响** `importance`（D-20 断言）
  - [ ] 低频高 `importance` 的记忆在 `importance` 维度**高于**高频低重要度者
  - [ ] `entity` 命中时该路得分上浮
  - [ ] **`entity` 打分的 SQL 执行次数有上界**（次数计数断言，**与实体数无关**）——P0-4 的注入式验证

### T-AL2-06 · 融合与预算裁剪

- **里程碑**：M1
- **依赖**：T-AL2-05
- **产出**：`core/recall.py::fuse` / `clip_to_budget`
- **要求**：
  1. 加权求和，权重来自 `RecallWeights`
  2. `clip_to_budget` 按 **L0 优先**，超预算时**先舍 L2**
  3. **保留 `raw` 明细**（可解释性一等公民，C4）
  4. 向量路不可用时跳过该路并重归一化权重，**不抛错**
  5. **三条边界语义必须写进文档并被用例钉住**（P2-9）：
     - `clip_to_budget` **至少保留 1 条**（首条超预算也要返回，否则预算小的时候召回直接空）
     - `recall` **全 0 分时返回全部**候选（`below or fused` 的写法）——不是返回空
     - `top_k` 是**先截后裁**（顺序不可交换）
- **约束**：C4 / DES-REV-005 P2-9
- **验收**：
  - [ ] 调整权重可改变排序
  - [ ] `raw` 含全部六分量
  - [ ] 超预算时先舍 L2
  - [ ] 向量路缺失时不抛错
  - [ ] **首条超预算时仍返回 1 条**（至少保留）
  - [ ] **全 0 分时返回全部**（不是空列表）
  - [ ] 交换 `top_k` 与裁剪顺序 → 结果不同（证明"先截后裁"确实是当前语义）

### T-AL2-07 · 分级加载 `expand`（**P 档前置**）

- **里程碑**：M1
- **依赖**：T-AL2-01
- **产出**：`core/progressive.py`、`CoreFacade.expand(ref, level, *, hot_path=True)`
- **要求**：
  1. L0 = `memories.abstract`；L1 = `overviews`（缓存）；L2 = 实时组装（`content` + 来源 + 关联）
  2. **L1 不得在热路径生成**（N-P0-2）：`stale` 时**只读旧缓存或降级到 L0**，重算投递 AL5 maintenance
  3. **L0 缺失时退化**为 `content` 首句截断，不报错
  4. L2 展开**只读**，不触发任何写入
  5. **逐级展开，不得越级**（V1 验收）
  6. **`hot_path` 是契约的一部分**（P1-8）：`hot_path=True`（默认，宿主热路径）不得生成 L1；
     `hot_path=False`（用户显式下钻，如 `spirit_expand` 工具）允许生成
- **约束**：INV-1（L1 是缓存）；P6（不阻断宿主）；DES-REV-005 P1-8
- **验收**：
  - [ ] 三级返回体量单调 L0 < L1 < L2
  - [ ] `expand(ref,'L1')` **不含** L2 全文
  - [ ] `expand(ref,'L2')` **无写操作**（只读断言）
  - [ ] **热路径不产生 LLM 调用**（断言无 model 调用）
  - [ ] L0 缺失时退化不报错
  - [ ] **`hot_path=True` 且 L1 stale 时：不生成、走旧缓存或降级**（P1-8 的落点）

### T-AL2-08 · 审查、溯源与干预（**P 档前置**）

- **里程碑**：M1
- **依赖**：T-AL2-01
- **产出**：`core/review.py`、`CoreFacade.review/trace/correct/forget/restore`
- **要求**：
  1. `review()` 字段：层 / 内容 / L0 摘要 / 时间 / 置信度 / 来源会话 / 状态 / **`importance` 及其构成**
  2. `trace(mem_id)` 从 `audit` 重放变更史
  3. `correct()` **产出 `WriteIntent`**，不直接写库
  4. `forget(mem_id, *, reason, source, purge_snapshot=False)`——**人工唯一收口**，`reason` 必填
  5. `restore(audit_id)` 从快照恢复
  6. **干预的 `audit.actor` 记为 `user`**，与自动流程可区分
  7. **AL2 只返回数据结构（`list[dict]`），不渲染文本**（P1-17）：
     LLD §8.10 与旧任务书写的"输出为人类可读文本"**与 INV-1 冲突**——
     "文本是投影"，投影属于 AL5 的 `observability/render.py`。AL2 若自己渲染，AL5 就只能反向依赖 AL2
  8. **`SalienceScorer.score(session_id=...)` 不得留死参数**（P2-12）：未使用的参数就是误导
- **约束**：INV-1 / INV-7 / INV-8 / D-17 / D-22 / DES-REV-005 P1-17 / P2-12
- **验收**：
  - [ ] `review()` 含 `importance` 构成
  - [ ] `correct()` / `forget()` 产出 `WriteIntent` 而非直接写库
  - [ ] `forget()` 缺 `reason` / `source` 时拒绝执行
  - [ ] `restore()` 能恢复且恢复本身留痕
  - [ ] **`review()` / `trace()` 返回 `list[dict]`，且 `core/` 内不存在 `format_*_text` 渲染函数**（P1-17）
  - [ ] **渲染由 AL5 的 `format_review_text` / `format_trace_text` 完成**（跨层用例）

---

## M2 · 提取、巩固与治理（T-AL2-09 ~ 15）

### T-AL2-09 · 提取 schema 与校验器

- **里程碑**：M2
- **依赖**：T-AL2-01
- **产出**：`extract/schema.py`
- **要求**：定义记忆对象 schema 与**轻量手写校验器**（不引 `jsonschema`）；**校验规则必须与提示词中的字段说明同源一致**（C12）；
  **校验器的实现只有一份**（住在 `model/base.py`，AL4 的协议层，`extract/schema.py` import 它——
  这是 **R1 白名单**允许的方向，不是"就地实现一份"）
- **约束**：C12 / R1 / DES-REV-007 P1-35
- **验收**：
  - [ ] 合法结构通过校验
  - [ ] 字段缺失 / 类型错误被识别
  - [ ] 校验器无第三方依赖
  - [ ] **`extract/schema.py` 不重复实现校验逻辑**（import 自 `model.base`，`inspect` 断言）

### T-AL2-10 · 结构化提取（含 F1 / F3 降级）

- **里程碑**：M2
- **依赖**：T-AL2-09 / T-AL4-04
- **产出**：`extract/extractor.py`
- **要求**：
  1. 用 `llm('extract')`（`glm-5.3-flash`）产出结构化候选
  2. **F1**：LLM 不可用 → 跳过提取，**仅存原文** + 告警
  3. **F3**：非 JSON → 由 AL4 重试 1 次；仍失败 → 仅存原文 + `audit(reason='extract_parse_failed')`
  4. **校验失败的记忆不得进入 AL3**（C5）
  5. 提取为 ADD-only 单遍（Mem0 范式）
- **约束**：C5 / F1 / F3
- **验收**：
  - [ ] 合法返回产出候选
  - [ ] 非法 JSON **不产出任何部分候选**
  - [ ] LLM 不可用时降级为"仅存原文"且不抛错
  - [ ] 失败路径产生 `audit` 留痕

### T-AL2-11 · 去重决策（三态）

- **里程碑**：M2
- **依赖**：T-AL2-10
- **产出**：`extract/dedup.py`
- **要求**：
  1. 支持 `ADD` / `UPDATE` / `IGNORE`；`MERGE` 在 M3 落地（见 T-AL2-17）
  2. 决策**必须写 `audit`（含 before/after）**
  3. **决策与审计必须一致**（P0-6）：实现曾把 `MERGE` **静默降级为 `ADD`**，
     而审计却写 `op="merge"`——**账本在撒谎**。红线：**审计里的 `op` 必须等于真正执行的那个分支**；
     做不到就把该分支从决策集里去掉，不许"写着一套、做着一套"
  4. `limit=500` 的相似样本上限**必须记载**（P2-9）：它决定"看起来不重复"何时退化成"真的不重复"
- **约束**：六态分期的偏差已记录于 00 §1.2 非目标；DES-REV-005 P0-6 / P2-9
- **验收**：
  - [ ] 三个分支各有用例
  - [ ] 每次决策产生 `audit` 事件（含 `before` / `after`）
  - [ ] `UPDATE` 分支同步更新 `updated_at`
  - [ ] **`audit.op` 与真正执行的分支逐个相等**（参数化：对每个决策分支断言 `audit.op == 分支名`）——P0-6 的注入式验证

### T-AL2-12 · 巩固：工作 → 情景

- **里程碑**：M2
- **依赖**：T-AL2-03
- **产出**：`core/consolidation.py`
- **要求**：会话结束把整段会话固化为 episode（带时空上下文）；**离线后台执行**，不在在线热路径
- **约束**：P6 / C6（幂等）
- **验收**：
  - [ ] 会话结束产出 1 条 episode
  - [ ] **同一 session 重复触发不产生重复记忆**（幂等，C6）
  - [ ] 巩固过程全部在后台（无主线程调用）

### T-AL2-13 · 巩固：情景 → 语义（基础版）

- **里程碑**：M2
- **依赖**：T-AL2-12
- **产出**：`core/consolidation.py` 提升路径
- **要求**：
  1. 同一主题/实体在 **≥ N 个不同会话**出现后提升为语义记忆
  2. 提升产生**新记录** + `relations(rel_type='derived_from')` 保留来源链
  3. 迁移**单向**，不出现逆向搬移
  4. **"跨会话计数"不得靠拉全行**（P1-18）：实现曾用 `query(layer='episodic', limit=500)` 计数——
     **超过上限即计数偏小 → 提升漏判**（而且静默）。要求改为
     **`COUNT(*)` 聚合**（一次聚合，不该拉全行）；同理 `Optimizer.run` 的 `scan_limit=5000` 全表扫 4 遍
     也必须要么改聚合、要么**超限时告警**
- **约束**：DES-REV-005 P1-18
- **验收**：
  - [ ] 跨会话复现后产生语义记忆
  - [ ] 存在 `derived_from` 关联边
  - [ ] 提升幂等
  - [ ] **跨会话计数走 `COUNT(*)` 聚合**（SQL 计数断言：不出现 `limit=500` 式拉行）
  - [ ] **注入"超过旧上限的会话数" → 提升仍命中**（P1-18 的注入式验证）

### T-AL2-14 · 衰减排序与降级

- **里程碑**：M2
- **依赖**：T-AL2-05
- **产出**：`core/decay.py`
- **要求**：
  1. `strength` **只作排序信号**，**`decay.py` 中不存在任何删除调用**（红线）
  2. 降级 `active → dormant`（**只保留 L0 参与召回，详情仍在库**，D-23）
  3. **降级不得由时间或频率直接触发**——由记忆优化任务判定
  4. 先算候选集再处置；候选集与处置结果都可见
  5. **`dormant` 的召回语义要写准**（P1-13）：实现是"**六因子总分 × 降权系数**"，
     而 LLD 写的是"只保留 L0 参与召回"——两者不等价。裁决：**按实现改措辞**为
     "降权参与召回（系数配置化）"，且 **`0.6` 必须离开硬编码**
- **约束**：D-16 / D-17 / D-23；INV-7；DES-REV-005 P1-13
- **验收**：
  - [ ] **注入"很久未访问"的记忆 → 不自动降级/删除**
  - [ ] `decay()` 不产生 `hard_delete` 调用
  - [ ] 降级只产生 `active → dormant`，**不存在 `archive` 目标状态**
  - [ ] `dry_run=True` 时不产生写操作
  - [ ] `dormant` 详情仍可用 `expand(ref,'L2')` 取到
  - [ ] **降权系数可在配置里改**（硬编码断言：源码中不出现裸 `0.6`）

### T-AL2-15 · 记忆优化删除（"不可达"判定）

- **里程碑**：M2
- **依赖**：T-AL2-14 / T-AL2-08
- **产出**：`core/consolidation.py` 的优化删除路径
- **要求**：
  1. 三类触发：**错误记忆 / 冲突记忆 / 不可达记忆**——其中
     **错误记忆（`superseded_by` 非空）与冲突记忆**在 M2 曾**完全未实现**（P0-7），
     M3 必须补齐（见 T-AL2-16 / T-AL2-17）
  2. **"不可达"= 图结构孤立**：无入边关联 + 无实体关联 + 不在任何 L1 概览中；
     **判据必须写进 LLD**（P1-14）：`_is_settled`（会话 committed）+ 无入边 + `importance < 0.5`、
     以及 `_overview_covered`（"摘要前 40 字符 ⊆ 概览文本"）**都是启发式**，不写进文档就无法复核
  3. 可**系统自主**或**用户确认**（可配置）
  4. 删除经 `forget()` 语义（保留快照，可恢复）
  5. `audit.actor = 'optimizer'`
- **约束**：D-17 / D-22；DES-REV-005 P0-7 / P1-14
- **验收**：
  - [ ] 造"孤立记忆"→ 命中不可达判据
  - [ ] 删除产生 `audit(actor='optimizer')`
  - [ ] 删除后可通过 `restore()` 恢复
  - [ ] **时间与频率不参与判定**（断言）
  - [ ] **`superseded_by` 非空的记忆被识别为"错误记忆"**（M2 缺口的回归用例）
  - [ ] **判据里每个魔数（`0.5` / `[:40]`）都能在 LLD 里找到出处**

---

## M3 · 时态与遗忘治理（T-AL2-16 ~ 18）

### T-AL2-16 · 双时态语义

- **里程碑**：M3
- **依赖**：T-AL3-24（存储侧字段启用）/ T-AL2-11
- **产出**：`core/temporal.py`、`CoreFacade` 的时态查询入口
- **要求**：
  1. 三件事分清：**`valid_from` / `valid_to` = 事实在真实世界有效的区间**；
     **`created_at` / `updated_at` = 器灵记录变化的时刻**；**`superseded_by` = 被哪条取代**
  2. `asof(ref, ts)` 返回 `ts` 时刻**有效**的那一版；无则返回结构化空结果
  3. **"失效" ≠ "删除"**：`INVALIDATE` 写 `valid_to` + `superseded_by`，**记录仍在库里、仍可查历史**
  4. 时态判定**不得用当前时间做隐式默认**：`now` 必须可注入（重放与测试）
- **约束**：D-24 之后的时态口径 / INV-7（删除纪律）/ INV-12（可溯源）
- **验收**：
  - [ ] 同一条 `ref` 在两个 `ts` 上返回**不同内容**（证明时态真的生效，而非返回当前值）
  - [ ] `ts` 早于首次写入 → 返回"无有效版本"，**不是抛错、也不是返回当前值**
  - [ ] 失效后**记录仍在**（`query(status=None)` 仍能查到），且 `trace` 能看到失效事件
  - [ ] `now` 可注入（不读系统时钟）

### T-AL2-17 · 三态去重（MERGE / INVALIDATE / FORGET）

- **里程碑**：M3
- **依赖**：T-AL2-11 / T-AL2-16
- **产出**：`extract/dedup.py` 的三态决策
- **要求**：
  1. **MERGE**：真的合并（内容融合 + 关联并集 + 强度取语义上的合理值），
     **不是"记一笔 audit 了事"**——这是 P0-6 的**根治**
  2. **INVALIDATE**：新事实取代旧事实 → 旧条写 `valid_to` + `superseded_by`，**不删**
  3. **FORGET**：仅用于合规清除与用户显式删除，**不用于"被取代"**
  4. 三态**与审计一一对应**（`audit.op` == 真正执行的分支）
  5. MERGE 与 INVALIDATE 都**必须幂等**（重复导入同一批不产生重复合并）
- **约束**：INV-7 / INV-14 / DES-REV-005 P0-6
- **验收**：
  - [ ] 三态各有用例，且**每个用例先断言"旧数据变成了什么"**（不是只看 audit）
  - [ ] `MERGE` 后**只剩一条**记忆，且内容包含双方信息
  - [ ] `INVALIDATE` 后旧条**仍可 `asof` 查到**（在某时刻之前）
  - [ ] 重复执行同一批导入 → **状态不变**（幂等）
  - [ ] **`audit.op` 与执行分支逐个相等**（P0-6 的守门断言）

### T-AL2-18 · 交叉验证（LLM 仲裁）

- **里程碑**：M3
- **依赖**：T-AL2-17 / T-AL4-13
- **产出**：`core/crosscheck.py`
- **要求**：
  1. 对**冲突候选**（同一主题的新旧事实）走 LLM 仲裁池判定，产出 `MERGE` / `INVALIDATE` / `KEEP_BOTH`
  2. **异步 + 限频 + 高门槛低频**（INV-11）：不得进热路径，不得每轮调用
  3. **LLM 不可用时降级**为确定性规则（时间优先），并留 `audit` 说明"未交叉验证"
  4. 仲裁结果**必须可溯源**（保留了哪些证据、为什么这样判）
- **约束**：INV-11 / P6 / F1
- **验收**：
  - [ ] 冲突候选触发仲裁；无冲突时**一次 LLM 调用都不发生**（调用计数断言）
  - [ ] LLM 不可用 → 走确定性规则且 `audit` 标注
  - [ ] 仲裁**不在热路径**（`prefetch` / `sync_turn` 调用计数为 0）

---

## M4 · 关联与进化（T-AL2-19 ~ 22）

### T-AL2-19 · 扩散激活

- **里程碑**：M4
- **依赖**：T-AL2-05 / T-AL3-09
- **产出**：`core/recall.py::score_diffusion` 的实装
- **要求**：
  1. 沿 `relations` 做**一跳**扩散，衰减因子可配
  2. **`hops > 1` 明确抛 `NotImplementedError`**（MVP 已确立的做法）——不得静默按一跳处理
  3. 扩散**只读图**，不做写入；单次召回的图遍历有**节点数上界**
- **约束**：C9（不散落魔数）/ P6（热路径有界）
- **验收**：
  - [ ] 一跳邻居的 `diffusion` 得分上浮；**隔离节点得 0**
  - [ ] `hops=2` → `NotImplementedError`（不是静默降级）
  - [ ] 图遍历节点数有上界（计数断言）

### T-AL2-20 · 记忆进化

- **里程碑**：M4
- **依赖**：T-AL2-17
- **产出**：`core/evolution.py`
- **要求**：借鉴 A-MEM——**新记忆到来时允许改写旧记忆的抽象/关联**（而非只做 ADD），
  每次改写**留审计 + 保留 before**；改写不得丢原文（原文在 `content`，抽象在 `abstract`）
- **约束**：INV-7 / INV-8 / INV-12
- **验收**：
  - [ ] 进化后 `abstract` 变化、`content` **不变**
  - [ ] `audit` 里有 `before` / `after`
  - [ ] 关掉进化开关 → 行为退回纯 ADD（配置生效断言）

### T-AL2-21 · 核心记忆

- **里程碑**：M4
- **依赖**：T-AL2-08 / T-AL2-13
- **产出**：`core/layers/core_memory.py` 的实装
- **要求**：高门槛升格（INV-11）；核心记忆**参与召回但占比小且可配**；
  升格/降格都有审计；`system_prompt_block` 读的就是它
- **约束**：INV-11
- **验收**：
  - [ ] 未达门槛**不升格**（注入式：差一点点的候选不上）
  - [ ] 升格后 `system_prompt_block` **能看到**它
  - [ ] 升格/降格各有 `audit` 事件

### T-AL2-22 · 审查干预（增强）

- **里程碑**：M4
- **依赖**：T-AL2-08 / T-AL2-16
- **产出**：`core/review.py` 的干预能力扩展
- **要求**：审查 → 溯源 → 干预闭环中新增**时态干预**（"这条在某时刻起就错了"）
  与**核心记忆干预**（升格/降格）；一切干预 `actor='user'`、必带 `reason`
- **约束**：V2 / INV-7 / D-22
- **验收**：
  - [ ] 时态干预后 `asof` 结果随之改变，且历史仍可查
  - [ ] 核心记忆干预有审计
  - [ ] 缺 `reason` 一律拒绝

---

## M5 · 传承的核心侧（T-AL2-23）

### T-AL2-23 · 传承重建（导入侧）

- **里程碑**：M5
- **依赖**：T-AL2-17 / T-AL3-26
- **产出**：`core/transfer.py`、`CoreFacade` 的导入入口
- **要求**：
  1. 导入走**与在线写入同一条路径**（写意图 → 单写者），不另开一条"批量直插"
  2. 按 `content_hash` 判重（INV-14）；**时态字段一并恢复**（`valid_from` / `valid_to` / `superseded_by`）
  3. **不重新提取**（`spirit_import` 是重建，不是 ingest——两个入口语义不得重叠）
  4. 导入**不覆盖**已有且内容相同的条目；冲突时按三态规则处置并留审计
- **约束**：V3 / INV-13 / INV-14
- **验收**：
  - [ ] 重复导入同一档案 → **零新增、零改写**
  - [ ] 导入后时态字段**逐字相等**
  - [ ] 导入过程**不产生 LLM 调用**（`spirit_import` ≠ `spirit_ingest` 的机械证据）
  - [ ] 导入的写**全部经写队列**（线程断言）

---

## 完成检查表（AL2 层）

- [ ] M1 八个任务全通过
- [ ] M2 七个任务全通过
- [ ] **M3 三个任务通过**（T-AL2-16 ~ 18）
- [ ] **M4 四个任务通过**（T-AL2-19 ~ 22）
- [ ] **M5 一个任务通过**（T-AL2-23）
- [ ] **`core/` 内零 I/O 导入**（R5）
- [ ] **无循环依赖**；只依赖 `store/base` 与 `model/base`
- [ ] 时钟与 ID 生成器**均为注入**（C2）
- [ ] 关键不变量逐条有测试：INV-1（L1 是缓存）/ INV-7（删除纪律）/ INV-11（核心记忆门槛）/ INV-12（可溯源）
- [ ] P 档前置三项（T-AL2-07 / 08 与 T-AL3-13 / 15）在 M1 内完成
- [ ] **`audit.op` 与真正执行的分支处处相等**（P0-6 的根治——账本不许撒谎）
- [ ] **所有"高门槛低频"的 LLM 能力（交叉验证 / 进化）都不在热路径**（调用计数断言）
