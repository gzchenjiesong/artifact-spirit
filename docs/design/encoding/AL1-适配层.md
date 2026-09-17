# AL1 适配层 · 编码任务

> 上游：[ENC-000 总纲](./README.md) / [LLD-AL1 适配层详细设计](../layers/AL1-适配层详细设计.md)
> 里程碑覆盖：**M0 – M5**
> 任务数：**18**（M0–M2: 13；M3–M5: 5——本轮随 DES-REV-004 的复核结论补齐，编号顺延不重排）
> **本层只做翻译，不做决策**——任何"这条记忆重不重要"的判断都不属于 AL1

---

## M0 · 契约打通（T-AL1-01 ~ 03）

### T-AL1-01 · 包结构与 `register` 入口

- **里程碑**：M0
- **依赖**：无
- **产出**：`src/artifact_spirit/__init__.py`、`plugin.yaml`、`pyproject.toml`
- **要求**：
  1. `pyproject.toml` 声明 entry point `hermes_agent.memory_providers`
  2. `__init__.py` **只做** `register(ctx) → ctx.register_memory_provider(...)`，**零业务逻辑**（R6 / C1）
  3. `config_schema.py` / `cli.py` / `plugin.yaml` 与包 `__init__.py` **同级**
  4. `provider.py::ArtifactSpiritProvider.name = "artifact-spirit"`
- **约束**：R6
- **验收**：
  - [ ] `__init__.py` 内无任何 import 业务模块（R6 断言）
  - [ ] 包布局符合要求
  - [ ] `name` 返回 `artifact-spirit`

### T-AL1-02 · `is_available` 本地判定

- **里程碑**：M0
- **依赖**：T-AL1-01
- **产出**：`provider.py::is_available`
- **要求**：仅做本地检查——配置文件存在且语法合法 / DB 路径可写 / `schema_version` 匹配。**明确不做** HTTP 探活、DNS、端口探测
- **约束**：**INV-5（禁止网络调用）**
- **验收**：
  - [ ] **全程无网络调用**（socket 打桩断言）
  - [ ] 配置缺失时返回 `False` 而非抛错
  - [ ] 调用耗时在毫秒级

### T-AL1-03 · `initialize` 调用装配

- **里程碑**：M0
- **依赖**：T-AL1-02 / T-AL5-03
- **产出**：`provider.py::initialize`
- **要求**：`hermes_home` 从 `kwargs` 取（恒存在）；装载配置 → 校验 → 调用 `lifecycle.start(cfg)`；**装配逻辑本身在 AL5**，AL1 只调用
- **验收**：
  - [ ] `initialize` 后各层服务可用
  - [ ] 配置非法时抛 `ConfigError`（含可操作提示）
  - [ ] `initialize` 内不含装配实现细节

---

## M1 · 工具与召回（T-AL1-04 ~ 09）

### T-AL1-04 · `MemoryProvider` 其余契约方法

- **里程碑**：M1
- **依赖**：T-AL1-03
- **产出**：`get_tool_schemas` / `handle_tool_call` / `get_config_schema` / `save_config` / `shutdown`
- **要求**：`shutdown` **幂等**（C9）；`save_config` 用**白名单**（C5）
- **验收**：
  - [ ] `get_tool_schemas()` 每个工具名都能被 `handle_tool_call()` 处理（**一一对应断言**）
  - [ ] `get_config_schema()` 输出可渲染为合法表单
  - [ ] `save_config()` 落盘结果**不含任何密钥字段**
  - [ ] `shutdown()` 连调两次不报错

### T-AL1-05 · `prefetch` 超时护栏与保底

- **里程碑**：M1
- **依赖**：T-AL1-03 / T-AL2-06
- **产出**：`provider.py::prefetch`
- **要求**：
  1. 启动超时计时（默认 **300ms**）
  2. 向量化失败 → 跳过向量路（不阻断）
  3. **超时或异常 → 返回核心记忆摘要保底**，保证 prompt 非空
  4. `try/except` **全包**（C4）
  5. **护栏内的闭包"只算不写"**：超时后调用方放弃等待，但那条被遗弃的线程**仍会跑完**——
     因此闭包只返回 `(文本, ids)`，**状态一律由主线程在"未超时"分支里写**；
     超时与异常两个分支都要**显式清空** `_last_recall` / `_last_injected`
- **约束**：P6（绝不阻断宿主）/ **DES-REV-004 P1-4**（"结果没人要"≠"没有影响"）
- **验收**：
  - [ ] **核心层抛异常时仍返回非空内容**
  - [ ] 超时返回保底内容
  - [ ] 向量不可用时仍能返回（仅 BM25）
  - [ ] 返回体的 token 量在预算内
  - [ ] **超时之后 `recall_status()` 报告的是"本轮 0 条"**，不是上一轮的陈旧条数（P1-4）
  - [ ] **迟到的召回 id 不会计入下一轮**（P1-4；注入式用例：动作里 sleep 超过护栏）

### T-AL1-06 · `sync_turn` / `on_session_end` 非阻塞投递

- **里程碑**：M1
- **依赖**：T-AL1-03 / T-AL5-04
- **产出**：`provider.py::sync_turn` / `on_session_end`
- **要求**：构造事件对象 → `write_queue.submit()` → **立即返回**；**禁止同步 I/O**（连写日志都要用内存缓冲，C3）
- **约束**：**INV-4**
- **验收**：
  - [ ] `sync_turn` **p99 < 5ms 且无 I/O**（断言）
  - [ ] `on_session_end` 立即返回，不等待巩固完成
  - [ ] 队列满时不阻塞（配合 T-AL5-04）

### T-AL1-07 · 工具 schema 与 handlers

- **里程碑**：M1（含 **P 档前置**）
- **依赖**：T-AL1-04 / T-AL2-07 / T-AL2-08
- **产出**：`tools/schemas.py`、`tools/handlers.py`
- **要求**：
  1. **schema 与 handler 分离**（便于测试与审查）
  2. 工具集：`spirit_recall` / `spirit_expand` / `spirit_remember` / `spirit_forget` / `spirit_restore` / `spirit_review` / `spirit_trace` / `spirit_correct` / `spirit_consolidate` / `spirit_reflect` / `spirit_export`
  3. **参数校验前置**，非法参数返回**结构化错误**而非抛异常（F10）
  4. 危险操作默认安全：`spirit_forget` 默认 dry-run + 须显式确认
  5. `spirit_recall` **必须能输出召回原因**（来自 `Scored.raw`）
  6. `spirit_expand` **只返回所请求级别**，不得顺手返回 L2 全文（V1）
  7. `spirit_review` / `spirit_trace` 输出**人类可读**（V2）
  8. `spirit_export` 产物**不依赖器灵运行时**（V3）
  9. **schema 的形状是宿主的接口，不是我们的命名偏好**（真机踩过，见实现记录 §9.2）：
     - 每条必须是 **bare function schema**：`{"name", "description", "parameters"}`
       —— 宿主把它**原样**塞进 `{"type": "function", "function": schema}` 交给模型
       （`agent_init.py` / `memory_manager.py` 两处），而 OpenAI 规范读的是
       `function.parameters`。写成 `input_schema`（Anthropic 的叫法）**不会报错**，
       只会让模型收到一个**没有参数定义的函数**：工具能调、参数传不进去。
     - 因此键名必须用 `parameters`；每个**必填参数必须在 `properties` 里可见且有
       `description`**（模型据此知道传什么）
     - **验收必须用宿主自己的规范化函数**，vendor 一份做基准
       （`tests/vendor/host_tool_schema.py`），并**模拟宿主包装后的形状**再断言
- **约束**：V1 / V2 / V3
- **验收**：
  - [ ] 工具名与 schema 三处（schema / handler / 文档）严格一致（C8）
  - [ ] **每个 schema 经宿主 `normalize_tool_schema` 后都有 `parameters`**，
        且包装成 `{"type":"function","function":…}` 后**必填参数逐个可在 `properties` 中找到**
  - [ ] 非法参数返回结构化错误
  - [ ] `spirit_expand(ref,'L1')` 不含 L2 全文
  - [ ] `spirit_forget` 不带确认时不产生写操作
  - [ ] `spirit_recall` 输出含六分量原因

### T-AL1-08 · CLI 核心命令

- **里程碑**：M1（含 **P 档前置**）
- **依赖**：T-AL1-07
- **产出**：`cli.py`
- **要求**：CLI 是**薄壳**（解析 → 调用 → 格式化）；破坏性命令默认 dry-run；提供 `--json`
  命令：`init` / `status` / `layers` / `review` / `trace` / `export` / `consolidate` / `decay` / `forget` / `restore` / `audit` / `doctor`
  （`import` / `awaken` / `correct` / `soul` 属 M2–M4，见 T-AL1-10）
- **验收**：
  - [ ] 每个子命令 `--help` 正常
  - [ ] `aspirit status --json` 输出合法 JSON
  - [ ] `aspirit decay` 默认 dry-run（不写库）
  - [ ] `aspirit review` 输出人类可读
  - [ ] **CLI 与 provider 共用同一套装配**（不重复实现，C10）

### T-AL1-09 · 错误翻译

- **里程碑**：M1
- **依赖**：T-AL1-04
- **产出**：`tools/handlers.py::translate_fault`（工具出口的统一翻译点）
- **要求**：按 LLD-AL1 §5 M8 翻译**四族**异常，每条都要给出**"下一步动作"**，并**保留原始细节**便于排查：
  1. `StorageFatalError` → **绝不崩溃宿主**（提示检查磁盘 / 权限，必要时 `reindex`）
  2. `SchemaViolationError` → 说明"已保留原文、未结构化"
  3. `ProviderUnavailableError` → 说明"模型不可用"，指向配置或宿主 fallback
  4. `ConfigError` → 指出**具体字段**与期望值
  5. **未登记的异常 → 退回兜底**：仍返回结构化错误，**绝不抛给宿主**
- **约束**：P6 / **DES-REV-004 P1-2**——翻译写在函数里不算数，**必须从"工具出口"验证**（异常要先被 `handle_tool_call` 接住才有意义）
- **验收**：
  - [ ] 四族异常**逐个**断言其文案必须出现的语义（参数化用例，一条一个族）
  - [ ] 注入 `StorageFatalError` → 工具返回可操作错误，**宿主进程存活**
  - [ ] **未登记异常**走兜底且不抛给宿主
  - [ ] 任何内部异常冒泡到 AL1 都返回**安全值**而非抛出

---

## M2 · 契约收尾与价值验收（T-AL1-10 ~ 13）

### T-AL1-10 · `on_pre_compress` 与 `on_memory_write`

- **里程碑**：M2
- **依赖**：T-AL1-06
- **产出**：`provider.py::on_pre_compress` / `on_memory_write`
- **要求**：
  1. `on_pre_compress`：抢救性归档当前工作记忆（**不声明 v2 checkpoint API**，按宿主默认的 v1 best-effort 语义）
  2. `on_memory_write`：镜像 `MEMORY.md` → 语义记忆；`USER.md` → 核心记忆
- **约束**：宿主在**主线程**调用这两条钩子，故**都必须非阻塞**（投递后台任务后立即返回，见 [LLD-AL1 §5 M1](../layers/AL1-适配层详细设计.md)）
- **验收**：
  - [ ] 压缩前工作记忆已归档
  - [ ] 镜像写入后可在语义记忆/核心记忆中查到
  - [ ] 两者均不对主线程造成阻塞

### T-AL1-11 · `save_config` 与密钥过滤

- **里程碑**：M2
- **依赖**：T-AL1-04
- **产出**：`provider.py::save_config`
- **要求**：**白名单式**写入——只写已知非密钥字段；任何疑似密钥字段一律不落盘
- **约束**：ADR-001 `save_config` 约定
- **验收**：
  - [ ] 传入含 key 的 values → 落盘结果无该字段
  - [ ] 白名单外字段一律忽略
  - [ ] 二次写入不产生重复段

### T-AL1-12 · `system_prompt_block`

- **里程碑**：M2
- **依赖**：T-AL1-05
- **产出**：`provider.py::system_prompt_block`
- **要求**：
  1. 核心记忆摘要（identity / soul）+ 器灵状态（层计数、健康度）
  2. **必须有 token 预算并截断**（C11）
  3. **给核心记忆预留预算时，预留量不得超过总预算**：`reserve = min(80, token_budget)`。
     写成 `max(1, token_budget - 80)` 会在 `token_budget < 80` 时**派生出永远无法满足的预算 1**，
     而 `truncate_to_tokens` 在 `budget == 1` 下**曾原地死循环 → 挂死宿主**（DES-REV-009 P0-1）
- **约束**：C11 / **DES-REV-009 P0-1**
- **验收**：
  - [ ] 输出含核心记忆摘要与状态
  - [ ] **先灌入远超预算的核心记忆，再断言"小预算块真的更短"**——空库下断言"长度受限"是**恒真**的（DES-REV-004 P2-6）
  - [ ] **极小预算逐个调用都能返回**（`token_budget` 取 1 / 10 / 60 / 79 / 80），不挂起（DES-REV-009 P0-1）
  - [ ] 调用耗时在预算内（有界）

### T-AL1-13 · 核心价值验收测试

- **里程碑**：M2
- **依赖**：全部
- **产出**：`tests/test_value.py`（**值测试单独成文件**，不混进 `test_al1_provider.py`——DES-REV-004 P2-7）
- **要求**：对三条核心价值 + 两条不变量写**端到端**断言（走真实链路：工具 → 队列 → 落库 → 召回），
  不是直接调内部函数。**每条验收项与一条用例一一对应**——不许多条条款共用一条断言
- **约束**：V1 / V2 / V3 / INV-4 / INV-5
- **验收**：
  - [ ] **V1**：`spirit_expand(ref,'L1')` 不含 L2 全文；返回体量随级别**单调增长**
  - [ ] **V2**：`spirit_review` 输出可直接阅读（含来源与时间）；`spirit_trace` 展示完整变更史
  - [ ] **V3**：`spirit_export` 产物用**纯文本**读取即可理解（可 UTF-8 解码、原文逐字可寻、无 NUL 字节）；**导出 → 导入一个全新的库 → 内容不丢**
  - [ ] **INV-4**：`sync_turn` p99 < 5ms（**有超时上界**的断言，不是"跑完没报错"）
  - [ ] **INV-5**：`is_available` 全程无网络调用（socket 打桩）

---

## M3 · 时态工具面（T-AL1-14）

### T-AL1-14 · `spirit_asof`（按时间点查询）

- **里程碑**：M3
- **依赖**：T-AL3-24（双时态字段启用） / T-AL2-16
- **产出**：`tools/schemas.py`、`tools/handlers.py` 新增 `spirit_asof`
- **要求**：
  1. 入参 `ref` + `ts`（ISO8601），返回**该时刻有效**的那一版内容
  2. 无该时刻的有效版本时返回**结构化空结果**，不是空字符串（用户要能区分"没有"与"内容为空"）
  3. 工具**只做翻译**：时态判定在 AL2 / AL3，AL1 不得自己比对时间字段（R4 的精神）
- **约束**：V1（不越级）/ DES-REV-008 C8（AL1 不判业务）
- **验收**：
  - [ ] `spirit_asof(ref, ts)` 在 `ts` 早于首次写入时返回"无有效版本"
  - [ ] 同一 `ref` 在两个 `ts` 上返回**不同内容**（证明时态真的生效，而非返回当前值）
  - [ ] schema 三处（schema / handler / 文档）一致

---

## M4 · 干预工具面（T-AL1-15）

### T-AL1-15 · 审查干预闭环

- **里程碑**：M4
- **依赖**：T-AL2-22（审查干预）
- **产出**：`tools/handlers.py` 的 `spirit_review` / `spirit_correct` / 新增核心记忆干预入口
- **要求**：V2「记忆可见、可审、可干预」在**工具面**闭环：
  审查（`review`）→ 溯源（`trace`）→ 修正（`correct`）/ 提升（核心记忆）→ **每一步都留审计**
- **约束**：V2 / INV-12（可溯源）
- **验收**：
  - [ ] `review → trace → correct` 三步串起来能改变一条记忆，且 `trace` 里**看得到那次变更的 actor 与 reason**
  - [ ] 干预**不绕过写队列**（写意图投递，不直接落库）
  - [ ] 危险操作（删除/降级）默认 dry-run

---

## M5 · CLI 全套与传承入口（T-AL1-16 ~ 18）

### T-AL1-16 · CLI 命令全集

- **里程碑**：M5
- **依赖**：T-AL1-08 / T-AL5-13
- **产出**：`cli.py` 补齐命令并分组
- **要求**：仍是**薄壳**（解析 → 调用 → 格式化）；破坏性命令默认 dry-run；`--json` 全支持。
  命令全集（含 M0–M2 已有的 12 条）：
  `init` / `status` / `layers` / `review` / `trace` / `export` / `import` / `ingest` /
  `consolidate` / `decay` / `forget` / `restore` / `audit` / `doctor` / `replay` / `reindex` /
  `optimize` / `soul` / `awaken` / `reflect`
  > `soul` / `awaken` 若在 v1.0 不实现，**必须从命令表移除并写明原因**——不得留下"能列出但会报未实现"的空壳命令（DES-REV-003 P1-10 的教训）
- **约束**：C10（与 provider 共用同一套装配）
- **验收**：
  - [ ] `cli.py` 声明的子命令集与**文档表**一致（**逐个**执行 `--help` 均退出码 0，不是抽查）
  - [ ] `status --json` 输出合法 JSON；人类视图走 `cli._print_out` 兜底（**Windows GBK 控制台不得因 emoji 崩溃**）
  - [ ] `decay` / `optimize` / `forget` 默认 dry-run（不写库）

### T-AL1-17 · 传承入口（`spirit_export` / `spirit_import` / `spirit_ingest`）

- **里程碑**：M5
- **依赖**：T-AL3-26（传承导出） / T-AL2-23（传承重建）
- **产出**：`tools/schemas.py`、`tools/handlers.py`、`cli.py`
- **要求**（**命名已裁决，不得混用**）：
  - `spirit_export` = 导出档案（纯文本，+ CLI `export`）
  - `spirit_import` = **传承重建**（对称于 export：把一个档案导进本器灵）
  - `spirit_ingest` = **批量素材导入**（把外部素材喂进来做提取，工具 + CLI 都给）
- **约束**：V3 / INV-13 / INV-14（以 `content_hash` 为去重锚点）
- **验收**：
  - [ ] 三个入口**语义不重叠**：`import` 不做提取、`ingest` 不做重建（各有用例钉住）
  - [ ] 重复 `import` 同一档案 → **零新增**
  - [ ] 导出档案在**无器灵环境**下可读（用例里不 import 器灵的任何模块就能解析）

### T-AL1-18 · 传承往返验收（V3 的最终判决）

- **里程碑**：M5
- **依赖**：T-AL1-17
- **产出**：`tests/test_value.py` 的 V3 段
- **要求**：**端到端**——真实链路导出 → 导进一个**全新的库** → 逐字段比对
- **约束**：V3 / INV-13 / INV-14
- **验收**：
  - [ ] 往返后核心信息**逐字相等**（不是"条数相同"）
  - [ ] 二次 `import` **零新增**（幂等）
  - [ ] 档案里**没有**任何只有器灵能解析的字段（用正则扫"私有标记"）

---

## 完成检查表（AL1 层）

- [ ] M0 三个任务全通过
- [ ] M1 六个任务全通过
- [ ] M2 四个任务全通过
- [ ] **M3 一个任务通过**（T-AL1-14）
- [ ] **M4 一个任务通过**（T-AL1-15）
- [ ] **M5 三个任务通过**（T-AL1-16 ~ 18）
- [ ] **`__init__.py` 零业务逻辑**（R6）/ **`core/` 无 Hermes 导入**（R4，由 AL5 架构测试守）
- [ ] **只有 `provider.py` 依赖宿主契约**（R4）
- [ ] 工具名三处一致，无悬空工具
- [ ] P 档前置：`spirit_expand` / `spirit_review` / `spirit_export` 在 M1 内可用
- [ ] 三条核心价值验收测试（T-AL1-13）全绿 —— **这是 MVP 的最终判决项**
- [ ] **`DES-REV-004` 的 P1-1（M1 表补 7 个钩子）已收口**：LLD-AL1 的钩子表与本任务书**两处**都不再漏钩子
  （当前状态：**仍待补文档**——钩子全集为 `initialize` / `is_available` / `prefetch` / `sync_turn` /
  `on_session_end` / `on_pre_compress` / `on_memory_write` / `system_prompt_block` /
  `get_tool_schemas` / `handle_tool_call` / `get_config_schema` / `save_config` / `shutdown` /
  `on_turn_start` / `on_session_switch` / `on_delegation` / `queue_prefetch`）
