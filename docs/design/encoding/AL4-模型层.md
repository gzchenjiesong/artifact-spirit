# AL4 模型层 · 编码任务

> 上游：[ENC-000 总纲](./README.md) / [LLD-AL4 模型层详细设计](../layers/AL4-模型层详细设计.md) / [ADR-003 模型方案](../05-模型方案.md)
> 里程碑覆盖：**M0 – M5**
> 任务数：**13**（M0–M2: 12；M3: 1——本轮随 DES-REV-007 的复核结论补齐，编号顺延不重排）

---

## M0 · 协议与客户端骨架（T-AL4-01 ~ 03）

### T-AL4-01 · 协议与异常族

- **里程碑**：M0
- **依赖**：无
- **产出**：`src/artifact_spirit/model/base.py`
- **要求**：
  1. 定义 `LLMProvider` / `EmbeddingProvider` / `ModelRegistry` 三个 Protocol，**并且协议副本必须与实现逐字一致**
     （P1-31）：`base.py` 是**唯一权威副本**，LLD §2.1 是它的镜像。曾出现的偏差共
     **少 15 个成员**（`ChatMessage` / `TASKS` / `KNOWN_EMBEDDING_DIMS` / `HttpTransport` /
     `validate_schema` / `SchemaValidationError` / `TaskLLM` / `TaskEmbedding` / `ResolvedRoute` /
     `ModelResolver` / `load_host_config_yaml` / `embedding_chain` / `llm_available` /
     `embedding_available` / `close`）+ **3 处签名不符**
  2. 定义异常族（**父类归属不可动**，它决定 AL2 的降级分支）：
     ```
     ModelError
      ├─ LLMError
      │   ├─ ProviderUnavailableError  # 连接失败 / 鉴权失败 —— 必须是 LLMError 的子类（P1-33）
      │   └─ SchemaViolationError      # 非 JSON 或不符合 schema
      └─ EmbeddingError                # 绝不降级为 LLM 调用（INV-6）
     ```
  3. **`SchemaValidationError` 刻意不是 `ModelError`**（P1-34）：它继承 `ValueError`——
     表示"**数据**不合 schema"，不是模型层故障。写错归属，下一个人会把它并进 `ModelError`
  4. **本文件禁止 import `httpx`**（协议层零 I/O）；`HttpTransport = Any` **不透明别名**住在这里
     （跨层需要不透明类型的先例——否则 httpx 会顺着 `lifecycle.start` 的签名漏进 AL5）
  5. **`temperature` 默认值语义要写准**（P1-32）：实现是 `None` = **请求体里不带该字段**（网关默认生效），
     不是 `0.0`。两者语义不同——照 `0.0` 实现会把温度**钉死**，而实际是"提取结果默认不保证确定性"
  6. 本层注意事项共 **12 条**（C1–C12）：C11（链信息只由对应的解析器写）/ C12（`*_available()` 是
     **一次性判定**，同实例内不恢复）**必须写进文档**（P1-36）
- **约束**：R3（不得依赖 core/store）/ INV-6 / DES-REV-007 P1-31 ~ P1-36
- **验收**：
  - [ ] 三个 Protocol 定义完整（`inspect` 白名单，**两个方向都比**：实现有而协议没有的也要报）
  - [ ] **异常继承关系逐条断言**：`ProviderUnavailableError` 是 `LLMError` 子类；
        `EmbeddingError` **不是** `LLMError` 子类；`SchemaValidationError` **不是** `ModelError`
  - [ ] **不传 `temperature` → 请求体里没有该字段**；传 `0.0` → 字段存在（P1-32）
  - [ ] `base.py` 无 `httpx` 导入

### T-AL4-02 · OpenAI 兼容客户端骨架

- **里程碑**：M0
- **依赖**：T-AL4-01
- **产出**：`model/openai_compat.py::OpenAICompatClient.complete()`
- **要求**：
  1. `POST {base_url}/chat/completions`，`Authorization: Bearer {key}`
  2. **base_url 结尾斜杠归一化**，避免 `//chat/completions`（C10）
  3. **超时必须显式传入**（默认 LLM 60s）——不得依赖 httpx 默认（C7）
  4. **不引入 `openai` SDK**——直接用 httpx
- **验收**：
  - [ ] 对 mock HTTP 返回正确文本
  - [ ] `base_url` 带/不带尾斜杠均能正确拼接
  - [ ] 超时值真实传入（mock 校验）

### T-AL4-03 · 重试策略与异常分类

- **里程碑**：M0
- **依赖**：T-AL4-02
- **产出**：重试装饰器 / 逻辑
- **要求**：
  1. **连接类错误**：指数退避重试 **2 次**
  2. **4xx（含 429）不重试**（C8）——配置 / 配额问题，重试无意义
  3. 重试耗尽后抛**分类异常**（连接失败 → `ProviderUnavailableError`）
- **验收**：
  - [ ] 连接失败触发恰好 2 次重试
  - [ ] HTTP 429 **不重试**（断言请求计数为 1）
  - [ ] HTTP 401 不重试，抛 `ProviderUnavailableError`

---

## M1 · 三档解析与结构化输出（T-AL4-04 ~ 08）

### T-AL4-04 · 结构化输出与校验重试

- **里程碑**：M1
- **依赖**：T-AL4-03
- **产出**：`complete_json(*, model, messages, schema, timeout)`
- **要求**：
  1. 解析用 `json.loads`——**禁止 `eval`**（C5）
  2. 校验用**轻量手写校验器**（不引 `jsonschema`）
  3. 失败时追加"请严格按以下 schema 输出"提示，**重试恰好 1 次**
  4. 仍失败 → 抛 `SchemaViolationError`
- **验收**：
  - [ ] 合法 JSON 直接返回 dict
  - [ ] 非法 JSON 触发**恰好 1 次**重试，第二次成功则返回
  - [ ] 两次都失败 → 抛 `SchemaViolationError`
  - [ ] 校验器能识别字段缺失 / 类型错误

### T-AL4-05 · Embedding 批量与保序

- **里程碑**：M1
- **依赖**：T-AL4-03
- **产出**：`model/openai_compat.py::embed(texts)`
- **要求**：
  1. `POST /embeddings`，批大小可配置（**默认 32**）
  2. 超长文本单独成批
  3. **返回向量顺序必须与入参严格一致**——加断言（C3，错配是灾难性 bug）
- **验收**：
  - [ ] `embed(["a","b","c"])` 返回 3 个向量且**顺序一致**
  - [ ] 超长输入自动拆批，结果顺序不变
  - [ ] 批大小配置生效

### T-AL4-06 · 三档模型解析与 fallback 链

- **里程碑**：M1
- **依赖**：T-AL4-04 / T-AL4-05
- **产出**：`model/resolver.py`、`ModelRegistry` 实现
- **要求**：按 LLD-AL4 §5 M3 实现四段链：
  ```
  1. 器灵配置 [models.llm] 有值且 provider 可用 → 用之
  2. 未配置 → 读宿主 Hermes 配置（hermes_home/config.yaml）
  3. 仍未 → 通用环境变量（OPENAI_API_KEY / OPENAI_BASE_URL）
  4. 全无 → 抛 ProviderUnavailableError
  ```
  任务 → 模型映射：`extract` / `dedup` / `summarize` → `glm-5.3-flash`；`consolidate` → `glm-5.3`；`soul` → `kimi-k3`
  2. **env 段要两个变量**（P1-39）：embedding 段要求 `OPENAI_API_KEY` **且** `OPENAI_EMBEDDING_MODEL`
     ——缺后者直接 `EmbeddingError`。**只写 KEY、不写 MODEL，用户永远启用不了这一段**（文档漏写即等于没这功能）
  3. **宿主 YAML 读取的能力边界必须写明**（P1-40）：实现读 4 个嵌套路径
     （`memory.llm` / `llm` / `openai` / `providers.openai`）+ 3 个扁平键，用**自写的 YAML 子集解析器**
     （不引 PyYAML，**解析失败返回 `{}`**）。宿主用了锚点 / 多行标量时**静默读不到**，
     而用户看到的是"宿主配置中未找到可用模型服务"——**分不清"没有"与"没读进来"**
- **约束**：**embedding 不可 fallback 到 LLM**（INV-6）/ DES-REV-007 P1-39 / P1-40
- **验收**：
  - [ ] `resolve_chain(task)` 输出实际生效链（**链段带 `usable` 与 `source`**，不是只有 `model`）
  - [ ] 删除器灵 LLM 配置 → 能读到宿主配置（fixture 模拟 `hermes_home`）
  - [ ] 全断 → 抛 `ProviderUnavailableError`
  - [ ] **embedding 未配置时不调用 chat 接口**（mock 断言无 `/chat/completions`）
  - [ ] **只给 `OPENAI_API_KEY` → 不可用；加上 `OPENAI_EMBEDDING_MODEL` → 可用**（P1-39）
  - [ ] **宿主 YAML 读不到就是读不到**：不得"猜"出一个错误值（P1-40 的注入式验证）

### T-AL4-07 · 维度与模型属性

- **里程碑**：M1
- **依赖**：T-AL4-05
- **产出**：`EmbeddingProvider.model` / `.dim`
- **要求**：
  1. `dim` 判定**三步**：显式配置 → `KNOWN_EMBEDDING_DIMS` 查表 → **抛错，绝不给默认维度**
  2. **与库中 `meta.embedding_dim` 的比对不在 AL4**（P0-11）：`meta` 属于 AL3，
     而 R3 禁止 AL4 依赖 `store/*`。**比对由 AL5 装配期调用 AL3 完成**——
     旧任务书写"AL4 启动时比对"是**要求 AL4 做架构上禁止它做的事**
- **约束**：INV-2 / R3 / DES-REV-007 P0-11
- **验收**：
  - [ ] `model` 返回 `kinfra-text-embedding-4b`；`dim` 返回 2560
  - [ ] **未知模型且未显式配置 `dim` → 抛错**（不给默认值）
  - [ ] **AL4 代码中不出现 `meta` / `store` 访问**（R3 扫描断言）
  - [ ] 维度不一致的**拒绝启动发生在 AL5 装配期**（跨层用例，见 T-AL5-08）

### T-AL4-08 · 配置对接

- **里程碑**：M1
- **依赖**：T-AL4-06
- **产出**：从 AL5 的 `SpiritConfig` 构造 provider
- **要求**：`api_key_env` 是**变量名**，运行时 `os.environ` 取值（C1）；**key 不落任何文件**
- **验收**：
  - [ ] 从配置的 `api_key_env` 正确读到 key
  - [ ] 环境变量缺失时给出**不含 key** 的可操作错误
  - [ ] 日志中不含 key（扫描日志输出）

---

## M2 · 降级与契约测试（T-AL4-09 ~ 12）

### T-AL4-09 · 降级路径实现

- **里程碑**：M2
- **依赖**：T-AL4-08
- **产出**：三条降级路径
- **要求**（对应 LLD-AL4 §7）：

  | 故障 | AL4 行为 | 上层承接 |
  |---|---|---|
  | **F1** LLM 未配置 | 抛 `ProviderUnavailableError` | AL2 → 跳过提取，仅存原文 |
  | **F2** embedding 未配置/失败 | 抛 `EmbeddingError`（**不 fallback 到 LLM**） | AL2 → 降级 BM25 检索 |
  | **F3** 非 JSON 输出 | 重试 1 次 → 抛 `SchemaViolationError` | AL2 → 仅存原文 + audit 留痕 |
  | **F9** 成本超预算 | **不属 AL4 职责**（由 AL2/AL5 计数） | AL2 → 降级纯启发式 |

  2. **最坏请求数必须连乘看**（P1-37）：M4 的"重试 1 次"与 M6 的"重试 2 次"**叠加**，
     最坏是 **`(1+2)×2 = 6` 次请求**。两节各自看都没毛病，**合起来才是真实上界**——
     成本与延迟预算按"2 次"估会**低估 3 倍**
  3. **链末段不得撒谎**（P1-41）：`resolve_chain()` 的末段曾恒定
     `{"task":…, "model":"", "source":"none", "active": True}`——**全链失败也标 `active: True`**。
     按裁决：**末段不追加**；若追加则 `active=False` 并把"全链失败"编码进 `source`
- **约束**：INV-6 / DES-REV-007 P1-37 / P1-41
- **验收**：
  - [ ] 三条降级路径各有用例
  - [ ] 降级时异常类型**正确可区分**（AL2 靠类型决策）
  - [ ] **持续 5xx + schema 重试 → `request_count == 6`**（P1-37 的真实上界断言）
  - [ ] **全链失败时链末段不出现 `active: True`**（P1-41）

### T-AL4-10 · 超时与限流纪律

- **里程碑**：M2
- **依赖**：T-AL4-03
- **产出**：超时配置项
- **要求**：LLM 60s / Embedding 30s（可配置）；**不实现限流器**（TPM 100万 / RPM 300 极富余，见 ADR-003 §3.7）——仅靠 batch 降低请求数
- **验收**：
  - [ ] 超时可配置且生效
  - [ ] 代码中**不存在**令牌桶 / 排队 / 退避队列（防过度设计回归）

### T-AL4-11 · 密钥与日志纪律

- **里程碑**：M2
- **依赖**：T-AL4-08
- **产出**：日志脱敏处理
- **要求**：异常信息与日志**禁止**包含 key；`repr` 不得泄露
- **约束**：ADR-001 `save_config` 约定
- **验收**：
  - [ ] 构造一次带 key 的失败调用 → 日志与异常文本中**无 key 子串**
  - [ ] 密钥扫描门禁通过

### T-AL4-12 · 契约测试（mock HTTP）

- **里程碑**：M2
- **依赖**：全部
- **产出**：`tests/test_model.py`（**注意文件名**：旧任务书写的是 `test_model_contract.py`，
  实际文件是 `tests/test_model.py`——按名字**找不到文件**的验收等于没验收，P2-25）
- **要求**：全部测试基于本地 mock HTTP，**不连真实网关**；
  `transport` 必须在**装配时**注入（`start()` 之后再改 `resolver._transport` 是**死代码**——
  客户端已缓存，请求照样打真网络，症状是"单测偶发变慢"而不是报错，P1-49）
- **验收**：
  - [ ] 覆盖 LLD-AL4 §8 全部验收项
  - [ ] 测试运行不需要网络（**真网络兜底：注入后请求计数为 0**）
  - [ ] 代码扫描：AL4 无 `core` / `store` 导入，且未引入 `openai` SDK

---

## M3 · 交叉验证（T-AL4-13）

### T-AL4-13 · 交叉验证仲裁池

- **里程碑**：M3
- **依赖**：T-AL4-04 / T-AL4-09
- **产出**：`model/resolver.py` 新增 `crosscheck` 任务档位 + 仲裁用提示模板
- **要求**：
  1. 为 AL2 的交叉验证（T-AL2-18）提供**一个专用档位**（模型名可配，默认走轻量档）
  2. 输出是**结构化裁决**（`MERGE` / `INVALIDATE` / `KEEP_BOTH` + 理由 + 证据引用），
     走与 T-AL4-04 相同的 schema 校验与重试路径
  3. **失败必须可降级**（F1 / F3 同型）：LLM 不可用或两次都解析不出 → 抛分类异常，
     由 AL2 退回确定性规则——**仲裁失败绝不能阻断写入**
  4. **不新增请求类型**：仍走同一条 `complete_json`，避免第二条 HTTP 路径（重复实现 = 双真相源）
- **约束**：INV-11（高门槛低频）/ INV-6 / F1 / F3
- **验收**：
  - [ ] `resolve_chain("crosscheck")` 能解析到模型（配置缺失时可降级）
  - [ ] 仲裁输出**经 schema 校验**（非法输出不产生部分裁决）
  - [ ] **LLM 不可用 → 抛分类异常**，AL2 侧用例断言"退回确定性规则"
  - [ ] **不存在第二条 HTTP 调用路径**（`httpx` 调用点计数断言）

---

## 完成检查表（AL4 层）

- [ ] M0 三个任务全通过
- [ ] M1 五个任务全通过
- [ ] M2 四个任务全通过
- [ ] **M3 一个任务通过**（T-AL4-13）
- [ ] 关键不变量逐条有测试：INV-6（embedding 不降级到 LLM）/ INV-2（维度一致）
- [ ] 依赖规则：**R3 通过**（AL4 不 import `core` / `store`，也不用 `openai` SDK）
- [ ] **协议副本与 `base.py` 双向一致**（P1-31：每条成员都在，签名逐字对）
- [ ] **重试连乘上界有断言**（`request_count == 6`，P1-37）
- [ ] **实测确认 `glm-5.3-flash` 的结构化输出稳定性**（若不稳，提取任务退到 `glm-5.3`——见 LLD-AL4 §10）
