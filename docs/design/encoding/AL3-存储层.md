# AL3 存储层 · 编码任务

> 上游：[ENC-000 总纲](./README.md) / [LLD-AL3 存储层详细设计](../layers/AL3-存储层详细设计.md)
> 里程碑覆盖：**M0 – M5**
> 任务数：**26**（M0–M2: 23；M3: 2；M5: 1——本轮随 DES-REV-006 的复核结论补齐，编号顺延不重排）

---

## 进度

| 里程碑 | 任务 | 状态 |
|---|---|---|
| **M0** | T-AL3-01 ~ 04 | ✅ **已完成**（31 项测试通过，2026-09-14） |
| **M1** | T-AL3-05 ~ 16 | ✅ **已完成**（2026-09-14） |
| **M2** | T-AL3-17 ~ 23 | ✅ **已完成**（2026-09-14） |
| **M3** | T-AL3-24 ~ 25 | ✅ **已完成**（2026-09-17；双时态**启用**与完整包往返——见 T-AL3-24 的核实前置结论） |
| **M5** | T-AL3-26 | ⏳ 待开工（传承导出与重建） |

**实现记录**：
- 代码位于 `src/artifact_spirit/store/`（`base.py` / `schema.sql` / `sqlite_backend.py` / `ids.py` / `text.py` / `archive.py` / `reindex.py`）
- `base.py`（数据结构 + 异常族 + `MemoryBackend` 协议）**提前到 M0 落地**——它是 schema 与协议共享的数据基础，且为纯声明无 I/O
- 测试：`tests/test_store_m0.py`（31）+ `test_store_m1.py`（60）+ `test_store_m2.py`（28）+ `test_arch.py`（13）
- 合规模块 `compliance/`（架构规则 R1–R10 / 依赖扫描 / 密钥扫描）随 T-AL3-23 一并落地

### 实现期的三处**设计偏离**（守 E4：显式记录，不留隐性遗漏）

| # | 偏离 | 理由 |
|---|---|---|
| 1 | **T-AL3-03 的 `rowid` 退化路径裁掉** | sqlite-vec 0.1.9 实测支持 TEXT 主键，退化分支无法被测试；而它位于所有向量读写关键路径——引入不可测分支比明确失败更危险 |
| 2 | **`mem_fts` 由"外部内容表"改为"普通 FTS5 表 + CJK 逐字归一化"** | 外部内容表要求索引与正文逐字一致，无法承载归一化；而 FTS5 内置的 `unicode61` 让整句中文变成单一 token，`trigram` 又要求查询 ≥3 字符（"偏好""花生"这类 2 字词全部失效）。器灵面向中文使用者，2 字词是主流，故采用 CJK 逐字归一化（见 `store/text.py` 文件头）。表仍是**派生索引**，可由 `rebuild_fts()` 重建（INV-1） |
| 3 | **删除的内容快照移入独立表 `delete_snapshots`**（不再塞进 `audit.before`） | INV-8 要求 `audit` append-only，而 D-22 的合规删除要求"数据真的不存在"——两者直接矛盾。把**内容**（可清除）与**账本**（永不删）分开后两个约束同时成立：`audit` 只记"谁/何时/因何/对哪条"，内容快照另存且可被合规清除 |
| 4 | **记忆包不含向量**（收窄 LLD-AL3 的"完整包含向量"） | T-AL3-22 明文要求"向量不必入档案，导入后由 reindex 补"；且 2560 维浮点既大又不可读，与 INV-13 的"人类可读"冲突 |

**其他实现说明**：
- `working_memory.last_touched` 采用**微秒精度**——秒级精度会让同一秒内的多次触碰退化为插入顺序，无法满足"按 `last_touched DESC` 排序"的验收
- `audit` 的 `op` 枚举新增 `touch` / `set_status` / `reconcile`——派生字段（`strength`/`access_count`/`last_access_at`）要能精确重放重建（INV-12），`touch` 必须留痕且记录绝对值
- `aspirit`/`spirit_*` 之外的额外方法（`entity_list` / `all_relations` / `assert_embedding_model` / `rebuild_fts` / `delete_snapshots` / `last_audit_id`）属实现补充，不改动协议既有语义

---

## M0 · 地基（任务 T-AL3-01 ~ 04）

### T-AL3-01 · 建库、PRAGMA 与迁移框架

- **里程碑**：M0
- **依赖**：无
- **产出**：`src/artifact_spirit/store/schema.sql`、`sqlite_backend.py::migrate()`
- **要求**：
  1. `schema.sql` 写入 LLD-AL3 §5 M4 的**全部 DDL**（含索引、触发器）——逐字对齐，不得增删字段
  2. 连接初始化按序执行：`journal_mode=WAL` → `synchronous=NORMAL` → `foreign_keys=ON` → `busy_timeout=5000`
  3. `migrate()` **幂等**：可重复执行不报错；全部 DDL 用 `IF NOT EXISTS`
  4. `schema_version` 写入 `meta` 表；启动时比对，不匹配则走迁移函数表
  5. DB 路径默认 `{hermes_home}/spirit/spirit.db`，目录不存在则创建
- **约束**：INV-9（路径基于 `hermes_home`）
- **验收**：
  - [ ] 空库上 `migrate()` 建全部表成功
  - [ ] 连续调用 `migrate()` 两次不报错（幂等）
  - [ ] `PRAGMA journal_mode` 返回 `wal`
  - [ ] `meta.schema_version` 已写入

### T-AL3-02 · meta 键值读写

- **里程碑**：M0
- **依赖**：T-AL3-01
- **产出**：`sqlite_backend.py::meta_get` / `meta_set`
- **要求**：键值直接映射；`meta_set` 用 upsert；约定键（`spirit_id` / `spirit_name` / `embedding_model` / `embedding_dim` / `schema_version`）已由 §T-AL3-01 初始化默认值
- **验收**：
  - [ ] `meta_set` → `meta_get` 往返一致
  - [ ] 同键二次 `meta_set` 为覆盖而非报错
  - [ ] `spirit_id` 已存在时**不被覆盖**（C6）

### T-AL3-03 · `vec0` 能力探测（退化路径**已裁掉**）

- **里程碑**：M0
- **依赖**：T-AL3-01
- **产出**：`sqlite_backend.py::_probe_vec_capability()` / `_require_text_pk()`
- **要求**：
  1. 加载 `sqlite-vec` 扩展；探测 `vec0` 是否支持 **TEXT 主键**
  2. 支持 → 用 LLD 的 `vec_memories(mem_id TEXT PRIMARY KEY, embedding float[2560])`
  3. 探测结果写入 `meta`，**避免每次启动重复探测**
  4. **不支持 → 抛 `StorageFatalError` 并指明所需版本**（不是静默降级）
- **⚠️ 实现决策（偏离原规格，2026-09-14）**：
  原规格要求"不支持时退回 `INTEGER rowid` + 映射表 `mem_rowid`"。**该分支已裁掉**：
  - sqlite-vec 0.1.9 **实测支持 TEXT 主键**，退化分支**在当前依赖下无法被测试**
  - 而它位于**所有向量读写的关键路径**上——引入不可测的分支，比明确失败更危险
  - 改为：探测 → 不支持则**明确失败**（错误信息含探测结果与所需版本）
  - 若未来确需支持旧版扩展，**再实现并在有该版本的环境下补齐测试**
- **验收**：
  - [ ] 探测结果落 `meta`
  - [ ] TEXT 主键路径下 `vec_memories` 可建表、可写、可 KNN、可删
  - [ ] 二次启动不重复探测（断言仅探测一次）
  - [ ] **模拟不支持场景 → 抛 `StorageFatalError` 且不残留连接**（替代"两路径均成功"）

### T-AL3-04 · 生命周期骨架

- **里程碑**：M0
- **依赖**：T-AL3-01
- **产出**：`sqlite_backend.py::open` / `close`
- **要求**：`open` 建连 + `migrate()`；`close` 执行 WAL checkpoint 并关闭连接；`sqlite3` 连接**不得跨线程共享**（C1）
- **验收**：
  - [ ] `open` → `close` → `open` 正常
  - [ ] `close` 后连接已释放
  - [ ] `close` 幂等

---

## M1 · 真相源与检索（任务 T-AL3-05 ~ 16）

### T-AL3-05 · `memories` 表 CRUD

- **里程碑**：M1
- **依赖**：T-AL3-04
- **产出**：`MemoryRecord` 数据类、`put` / `get` / `query` / `update`
- **要求**：
  1. `MemoryRecord` 字段与 LLD-AL3 §2.1 **逐字一致**（含 `content_hash`）
  2. `put` 生成 ID：`{abbr}_{ulid}`，abbr ∈ `epi|sem|pro|cor`
  3. `query` 支持 `layer` / `status` / `types` / `since` / `until` / `limit` 组合过滤
  4. `row_factory = sqlite3.Row`；按列名取值（C2）
  5. **一律参数化查询**，禁止拼接（C11）
- **验收**：
  - [ ] ID 格式符合 `{abbr}_{ulid}` 且 ULID 单调递增
  - [ ] `put` → `get` 字段完全一致
  - [ ] `query(status='active')` 过滤生效
  - [ ] `query(types=[...])` 过滤生效

### T-AL3-06 · FTS5 表与三个同步触发器

- **里程碑**：M1
- **依赖**：T-AL3-05
- **产出**：`mem_fts` 虚拟表 + `trg_mem_ai` / `trg_mem_ad` / `trg_mem_au`
- **要求**：外部内容表模式；删除必须用 `'delete'` 语法（C5）
- **验收**：
  - [ ] 插入 `memories` 后 `mem_fts` 可检索到
  - [ ] 更新 `memories` 后 `mem_fts` 同步更新
  - [ ] 删除 `memories` 后 `mem_fts` 无残留
  - [ ] 中英文关键词均能命中

### T-AL3-07 · 向量索引读写

- **里程碑**：M1
- **依赖**：T-AL3-03 / T-AL3-05
- **产出**：向量写入/删除路径
- **要求**：写入前**校验维度 == `meta.embedding_dim`**，不符抛 `DimensionMismatchError`（C6 / F7）；记录 `embedding_model` 到行上
- **验收**：
  - [ ] 2560 维向量写入成功
  - [ ] 维度不符时抛 `DimensionMismatchError`，且**未产生半写**
  - [ ] `memories.embedding_model` 已落值

### T-AL3-08 · 检索原语

- **里程碑**：M1
- **依赖**：T-AL3-06 / T-AL3-07
- **产出**：`vector_search(vec, *, layer, top_k)` / `keyword_search(query, *, layer, top_k)` → `list[Hit]`
- **要求**：
  1. **只返回原生分**，不做融合排序（融合在 AL2）
  2. `Hit.content` 是**文本**（非向量）——交付给宿主的形态
  3. 检索**只读**，不开事务、不加写锁
- **验收**：
  - [ ] 合成向量下 top-k 顺序正确
  - [ ] `layer` 过滤生效
  - [ ] `Hit` 字段完整（`mem_id` / `layer` / `content` / `score` / `meta`）

### T-AL3-09 · `relations` 表与关联操作

- **里程碑**：M1
- **依赖**：T-AL3-05
- **产出**：`link` / `reinforce` / `neighbors`
- **要求**：`UNIQUE(src_kind, src_id, dst_kind, dst_id, rel_type)`；`reinforce` 做 upsert 并累加 `co_count`；`neighbors` 按 `weight DESC`
- **验收**：
  - [ ] `link` 重复调用不产生重复边
  - [ ] `reinforce` 后 `weight` 与 `co_count` 均增长
  - [ ] `neighbors` 按 weight 降序

### T-AL3-10 · `entities` 表

- **里程碑**：M1
- **依赖**：T-AL3-05
- **产出**：实体 upsert 与按名/别名查找
- **要求**：`UNIQUE(name, type)`；`aliases` 存 JSON 数组
- **验收**：
  - [ ] 同名同类型 upsert 不重复
  - [ ] 按别名可命中原实体

### T-AL3-11 · 工作记忆与意图槽

- **里程碑**：M1
- **依赖**：T-AL3-05
- **产出**：`wm_put` / `wm_list` / `wm_clear` / `intent_put` / `intent_list`
- **要求**：`working_memory` 按 `session_id` 隔离；外键 `ON DELETE CASCADE`
- **验收**：
  - [ ] `wm_put` × N 后 `wm_list` 按 `last_touched DESC` 返回
  - [ ] `wm_clear` 仅清当前 session
  - [ ] 删除 session 级联清除其工作记忆

### T-AL3-12 · `audit` 表与 append-only 保护

- **里程碑**：M1
- **依赖**：T-AL3-05
- **产出**：`audit()` / `audit_replay()` + `trg_audit_no_update` / `trg_audit_no_delete`
- **要求**：
  1. `actor` 区分 `extractor|dedup|consolidator|decay|optimizer|user|cli|system`
  2. **`op` 只能有一处权威**（P1-20）：历史上"权威列举"存在**三处**（LLD / `schema.sql` 注释 /
     `base.py` docstring）且互不相同——文档独有 5 个（`merge`/`invalidate`/`ignore`/`dormant`/`unlink`，
     其中 **4 个从未实现**）、实现独有 4 个（`set_status`/`touch`/`reinforce`/`reconcile`）。
     **M3 起 `op` 提为显式白名单**（`Literal` / frozen set），并让三处**逐字对齐**
  3. append-only 由触发器守（INV-8）
- **约束**：INV-8（append-only）/ DES-REV-006 P1-20
- **验收**：
  - [ ] `audit` 写入成功、`audit_replay` 按时间序返回
  - [ ] 对 `audit` 的 UPDATE / DELETE 被触发器拒绝
  - [ ] `audit_replay(since=...)` 过滤生效
  - [ ] **写入不在白名单里的 `op` → 被拒绝**（P1-20 的注入式验证；不再"写什么都收"）
  - [ ] **LLD / `schema.sql` / `base.py` 三处的 `op` 集合逐字相等**（文档对齐断言）

### T-AL3-13 · 分级加载存储（`overviews`）

- **里程碑**：M1（**P 档前置**）
- **依赖**：T-AL3-05
- **产出**：`overview_get` / `overview_put` / `overview_invalidate`
- **要求**：
  1. `overview_invalidate` 是**标记式**（O(1) 置 `stale=1`），不触发重算
  2. `overview_get` 需能读到 `stale` 状态（供上层决定降级）
  3. `overviews` 是**纯缓存**——清空后一切功能不受影响
- **约束**：INV-1
- **验收**：
  - [ ] `overview_put` / `overview_get` 往返正确，`level` 维度生效
  - [ ] `overview_invalidate` 返回受影响行数
  - [ ] **清空 `overviews` 后所有其他测试仍通过**

### T-AL3-14 · 内容寻址指纹

- **里程碑**：M1
- **依赖**：T-AL3-05
- **产出**：`content_hash_of(rec)`、`find_by_hash(h)`
- **要求**：按 LLD-AL3 §5 M11 的算法——三元组齐备时 hash `{subject,predicate,object,scope}`（`sort_keys=True`）；缺失时退化为 `normalize(content)` 的 hash。**非唯一索引**
- **约束**：INV-14
- **验收**：
  - [ ] 同一记录重复计算得到同一指纹
  - [ ] 三元组相同、`scope` 不同的记录**指纹不同**
  - [ ] `find_by_hash` 可返回多条（非唯一）

### T-AL3-15 · 可读档案导出与文本投影

- **里程碑**：M1（**P 档前置**）
- **依赖**：T-AL3-05 / T-AL3-14
- **产出**：`export_archive(path, fmt='markdown')`、`export()`（状态投影）
- **要求**：
  1. 档案**自描述头部**：`schema_version` / `spirit_name` / `spirit_id` / `exported_at` / `memory_count`
  2. 每条含：`content` / `abstract` / `created_at` / `valid_from` / `valid_to` / `confidence` / `salience` / `status` / `source_session` / `relations` / `content_hash`
  3. **不写向量**（体积大且不可读）
  4. `export()` 与 `export_archive()` 均为**纯函数**（同库状态同输出）
- **约束**：INV-13（开放、人类可读、不依赖运行时）、INV-1（纯函数）
- **验收**：
  - [ ] 档案用纯文本编辑器可读，头部字段齐全
  - [ ] 档案中**不含向量**
  - [ ] `export_archive` 连续两次输出**完全一致**
  - [ ] `export()` 连续两次输出完全一致

### T-AL3-16 · 写入的原子性

- **里程碑**：M1
- **依赖**：T-AL3-05 / T-AL3-06 / T-AL3-07 / T-AL3-12
- **产出**：`put()` 的事务包装
- **要求**：一条记忆的写入（`memories` + `vec_memories` + `mem_fts` + `audit`）在**同一事务**内提交
- **验收**：
  - [ ] 注入 audit 写入失败 → **记忆主体未落库**（无半写状态）
  - [ ] 注入向量写入失败 → 记忆主体未落库
  - [ ] 成功路径下四处数据一致
  - [ ] **注入 `hard_delete` 的 audit 写入失败 → 记忆未被删除**（P2-17：§8 早有这条要求，
     `test_store_m2.py` 全文却**没有 `monkeypatch`**——put 的同型用例有、delete 的没有）

---

## M2 · 治理、恢复与传承（任务 T-AL3-17 ~ 23）

### T-AL3-17 · `sessions` 生命周期

- **里程碑**：M2
- **依赖**：T-AL3-11
- **产出**：session 创建 / `turn_count` 递增 / `committed` 状态
- **验收**：
  - [ ] 会话结束置 `status='committed'` 并写 `ended_at`
  - [ ] `turn_count` 正确累加

### T-AL3-18 · 删除与恢复

- **里程碑**：M2
- **依赖**：T-AL3-12
- **产出**：`hard_delete(mem_id, *, reason, purge_snapshot=False)`、`restore_from_audit(audit_id)`
- **要求**：
  1. 删除前把**完整 `MemoryRecord` 快照**写入 `audit(op='forget').before`，**与删除同事务**
  2. 快照须**一并包含关联边**，以便恢复
  3. `restore_from_audit` 重建记录并**沿用原 ID**；原 ID 被占用时新建
  4. 恢复本身写 `audit(op='restore')`
  5. `purge_snapshot=True` 时清除该条内容快照（仅供合规）
  6. **不带 `reason` 拒绝执行**
- **约束**：INV-7 / INV-8 / D-22
- **验收**：
  - [ ] 删除后 `audit` 中该条历史仍在（不级联删除）
  - [ ] `restore_from_audit` 后记录与删除前**等价**（含 ID 与关联）
  - [ ] 恢复产生 `audit(op='restore')`
  - [ ] `purge_snapshot=True` 后**内容快照不可恢复**
  - [ ] 缺 `reason` 时抛错

### T-AL3-19 · 派生字段重放重建

- **里程碑**：M2
- **依赖**：T-AL3-12
- **产出**：`replay --rebuild-derived`
- **要求**：从 `audit` 重放 `strength` / `access_count` / `last_access_at`
- **约束**：INV-12
- **验收**：
  - [ ] 清空派生字段后 `replay` 能恢复到原值
  - [ ] 重放结果与原始值一致（逐条断言）

### T-AL3-20 · 重嵌入工具

- **里程碑**：M2
- **依赖**：T-AL3-07
- **产出**：`reindex.py`
- **要求**：支持**断点续跑**；更新 `meta.embedding_model` 与 `embedding_dim`
- **约束**：INV-2
- **验收**：
  - [ ] 换维度后 `reindex` 全库重算成功
  - [ ] 中断后重启能续跑（不重复处理已完成项）
  - [ ] `meta` 中模型与维度已更新

### T-AL3-21 · 启动对账

- **里程碑**：M2
- **依赖**：T-AL3-07
- **产出**：`reconcile()`
- **要求**：按 `mem_id` 差集——缺失向量补算、孤儿向量删除；支持 `--dry-run` 只报告
- **验收**：
  - [ ] 造"有记忆无向量"→ 对账修复
  - [ ] 造"孤儿向量"→ 对账删除
  - [ ] `--dry-run` 不产生写操作

### T-AL3-22 · 记忆包导出/导入与幂等

- **里程碑**：M2
- **依赖**：T-AL3-14 / T-AL3-15
- **产出**：`export_pack()` / `import_pack(data, merge=False)` / `import_archive(path)`
- **要求**：
  1. 导入时按 `content_hash` 判重：命中则跳过/合并，**不新建**
  2. 导入保留原 ID（可用时）
  3. 向量不必入档案，导入后由 `reindex` 补
- **约束**：INV-14
- **验收**：
  - [ ] `export_pack` → `import_pack` → **再次** `import_pack` **不产生重复记录**
  - [ ] `export → import → export` 两次导出等价
  - [ ] `import_archive` 后核心信息无损

### T-AL3-23 · 架构与合规测试

- **里程碑**：M2
- **依赖**：全部
- **产出**：`tests/test_arch.py`、依赖扫描脚本
- **要求**：AL3 不得 import `httpx` / `core` / `extract` / `model`
- **约束**：R2 / R7 / INV-10
- **验收**：
  - [ ] 代码扫描：AL3 无违规导入
  - [ ] 注入一行违禁 import → 依赖扫描使 CI 失败（**扫描有效性自证**）

---

## M3 · 时态存储与往返（任务 T-AL3-24 ~ 25）

### T-AL3-24 · 双时态写入路径与 as-of 查询

> **核实前置结论（2026-09-17，纠正原条款）**：`SCHEMA_VERSION` 当前是 **2**
> （v1 = 全量 DDL，v2 = `mem_fts` 重建为 CJK 归一化的普通 FTS5 表），
> 而 `valid_from` / `valid_to` / `superseded_by` **已在 v1 的全量 DDL 里**，
> `MemoryRecord` 与 `put` / `update` 的字段白名单也都覆盖了它们。
> **所以本任务不需要 schema 迁移**——原条款写的"v1 → v2 迁移（双时态字段）"是错的：
> 版本号错，前提也错（`schema.sql` 的注释本来就写明"MVP 只写 `valid_from`，`valid_to` 待 M3 的 INVALIDATE"）。
> 本任务做的是**启用**：把这三个字段从"占着位"变成"**被写、被查、被往返**"。

- **里程碑**：M3
- **依赖**：T-AL3-05 / T-AL3-14
- **产出**：`sqlite_backend.py` 的时态写入与查询入口、`archive.py` 的 pack 补字段
- **要求**：
  1. **写入**：`valid_from` 在 `put` 时**默认 = `created_at`**（不留空——空值会让 as-of 判定
     退化成"这条没有时态信息"）；`valid_to` 缺省 `NULL`（= 至今有效）
  2. **as-of 查询**：新增按时间点取数的入口，语义是
     `valid_from <= ts AND (valid_to IS NULL OR valid_to > ts)`；
     **判定必须落在 SQL 里**——捞回整表在 Python 里比时间是**静默的全表扫描**
  3. **`superseded_by` 的完整性**：写成环会让"谁取代谁"无解 → 写入时**拒绝自指**；
     且它与 `relations(rel_type='supersedes')` **必须同时写**（一个查得快、一个说得清）
  4. **pack 往返补字段**：`export_pack` / `import_pack` 现在带 `valid_from` / `valid_to`
     却**不带 `superseded_by`** → 往返会让"被谁取代"这条信息丢失，V3 的"不丢核心信息"就破了
- **约束**：INV-12（可重放）/ INV-13 / INV-14 / D-24 之后的时态口径
- **验收**：
  - [ ] `put` 之后 `valid_from` **等于 `created_at`**（不是 `NULL`）
  - [ ] `asof(ref, ts)` 在 `ts` 落在有效区间内时返回该条；区间外返回"无有效版本"
  - [ ] **as-of 判定的 SQL 执行次数与库大小无关**（计数断言，防全表捞）
  - [ ] `superseded_by` **自指被拒绝**
  - [ ] `supersedes` 关联边与 `superseded_by` **同时存在**（写下其一，另一个必在）
  - [ ] `export_pack → import_pack` 后 **`superseded_by` 逐字相等**

### T-AL3-25 · 完整包往返与幂等（增强 T-AL3-22）

- **里程碑**：M3
- **依赖**：T-AL3-22 / T-AL3-24
- **产出**：`export_pack()` / `import_pack(..., merge=False)` 支持 M3 新字段
- **要求**：
  1. 往返**必须带时态字段与关联边**（否则"导入后历史全平了"）
  2. 按 `content_hash` 判重（INV-14）——命中则**跳过或合并，不新建**
  3. 导入保留原 ID（可用时）
  4. **导入是"一次线性扫描 + 事务批量写"**（M3 的性能条款）：不得逐条开事务
- **约束**：INV-13 / INV-14
- **验收**：
  - [ ] `export_pack → import_pack → 再次 import_pack` → **零新增**
  - [ ] `export → import → export` 两次导出**等价**
  - [ ] 往返后 `valid_from` / `valid_to` / `superseded_by` **逐字相等**
  - [ ] 导入 N 条只开**一次**写事务（事务计数断言）

---

## M5 · 传承（任务 T-AL3-26）

### T-AL3-26 · 传承导出与重建（`export_archive` / `import_archive` 对称）

- **里程碑**：M5
- **依赖**：T-AL3-25
- **产出**：档案格式版本化 + `import_archive(path)`
- **要求**：
  1. 档案头部含 **`archive_version`**（与 `schema_version` 分开——档案格式会独立演化）
  2. **导出与导入必须对称**：`export_archive` 写出的每个字段，`import_archive` 都要能读回
  3. **档案在无器灵环境下可读懂**（INV-13）：纯文本、自描述、不出现只有本系统能解析的标记
  4. 删除类快照（`delete_snapshots`）**不进档案**——那是合规材料，不是记忆资产
- **约束**：INV-13（开放、人类可读、不依赖运行时）
- **验收**：
  - [ ] 档案头部含 `archive_version`
  - [ ] **用纯文本解析（不 import 器灵任何模块）即可读出全部记忆**（用例里真的不 import）
  - [ ] `export → import → export` 等价且**幂等**
  - [ ] 档案里**不含向量、不含删除快照**

---

## 完成检查表（AL3 层）

- [ ] M0 四个任务全通过
- [ ] M1 十二个任务全通过
- [ ] M2 七个任务全通过
- [ ] **M3 两个任务通过**（T-AL3-24 ~ 25）
- [ ] **M5 一个任务通过**（T-AL3-26）
- [ ] `test_arch.py` 全绿
- [ ] 全部降级路径有测试覆盖（`StorageBusyError` / `DimensionMismatchError` / `StorageFatalError`）
- [ ] 关键不变量逐条有测试：INV-1 / INV-2 / INV-8 / INV-12 / INV-13 / INV-14
- [ ] **`audit.op` 有显式白名单，且三处文档逐字对齐**（P1-20）
- [ ] **迁移可重复执行、失败不留半迁移**（M3 新增）
- [ ] **档案可在无器灵环境解析**（用"用例里不 import 器灵"机械证明）
