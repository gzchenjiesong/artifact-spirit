"""验收追溯矩阵：验收项 ⇄ 设计条款 ⇄ 用例。

为什么需要这张表
----------------
本项目实测到过三类**同源**现象（详见 ``docs/design/07-设计评审.md`` §12.8）：

1. **验收原文被用例改写成"能过的形式"**——验收写 ``aspirit status --json``，
   用例写 ``main(["--json", "status"])``，于是验收清单里那条命令**从未被执行过**；
2. **断言弱到无法失败**——``assert resp.ok in (True, False)``、传 ``token_budget=100``
   却断言 ``len(block) < 1200``：门禁绿着，守卫已经空了；
3. **条款过时，用例还钉着它**——任务书曾要求"采用 v2 checkpoint API"，
   用例却钉住相反行为，编码端两头堵：**满足用例就不满足条文**
   （即 P1-3，已于 2026-09-16 裁决为"改文档"）。

三者的共同点是：**门禁看起来在工作，实际守卫的是一份已经过期的设计。**

根因是验收要求**散落在两处**（LLD §8 清单 + 编码任务书 ``T-AL1-xx``）且
**没有一份"验收项 → 用例"的对照**——于是"用例过时"既不会被发现，也不会被追责，
只能靠人偶然读到；而读到的时候，往往已经写了一批迁就旧用例的代码。

本表把这条链写下来，并由 ``test_acceptance_matrix.py`` 机械校验。
规则出处：设计纲领 **E7**（用例是设计的断言，不是通过率的来源）。

状态四档（判定顺序即 E7 的裁判顺序：设计意图 > 条款 > 验收项 > 用例 > 实现）
------------------------------------------------------------------------------
- ``covered``：有用例，且断言强度与验收项相称
- ``weak``   ：用例在，但断言**弱于**条款（可能永远绿）→ 须改写**用例**
- ``gap``    ：**没有**用例 → 须**补用例**（不是补文档）
- ``stale``  ：条款文字与"实现 + 用例"冲突，且**条款是错的那一方** → 须**改文档**

``note`` 里形如 ``P1-5`` / ``P2-6`` 的编号一律指向本层评审日志
（``docs/design/07-设计评审.md`` 第四部分）。校验器会检查该编号**确实存在**——
**缺口必须留痕**，不许偷偷记成"通过"（E4）。

本轮（2026-09-16）的矩阵维护缺陷
--------------------------------
上一版矩阵里 8 处引用了**不存在**的用例名（弱用例被删除后未同步、把"计划要写的
用例名"当成了已存在的名字），同时又把**已经补齐**的缺口（P1-5 的处置早已落地）
继续记作 ``gap``。两种情况的方向相反，但性质相同：**矩阵在说谎**——
前者的红灯靠 ``test_referenced_tests_exist`` 才被发现，后者根本不会报红
（已覆盖的条目记成缺口，校验器无从判断）。

因此本版按"逐条核对真身"重建：每条 ``covered`` 都指向一个**真实存在且断言相称**的
用例；确实没有用例的，保持 ``gap`` 并带裁决编号。``test_value_v3_export_is_portable``
是唯一"矩阵先声明、用例后补齐"的条目——按 E7，``gap`` 的处置是**补用例**，
所以补的是用例，不是把矩阵改弱。
"""

from __future__ import annotations

from typing import NamedTuple

# --- 权威文档位置（校验器据此核对编号是否真实存在） ------------------------- #

TASK_BOOKS: dict[str, str] = {
    "AL1": "docs/design/encoding/AL1-适配层.md",
    "AL2": "docs/design/encoding/AL2-核心层.md",
    "AL3": "docs/design/encoding/AL3-存储层.md",
    "AL4": "docs/design/encoding/AL4-模型层.md",
    "AL5": "docs/design/encoding/AL5-辅助机制.md",
}
"""编码任务书——``T-{层}-xx`` 验收项的定义处。

2026-09-16 从"**只有 AL1**"扩到五层：AL1 单独覆盖时，其余四层的"新增条款没人管怎么验"
是**零信号**的——矩阵不覆盖某层，就等于那层的验收条款从未被对照过。
"""

LLDS: dict[str, str] = {
    "AL1": "docs/design/layers/AL1-适配层详细设计.md",
    "AL2": "docs/design/layers/AL2-核心层详细设计.md",
    "AL3": "docs/design/layers/AL3-存储层详细设计.md",
    "AL4": "docs/design/layers/AL4-模型层详细设计.md",
    "AL5": "docs/design/layers/AL5-辅助机制详细设计.md",
}
"""各层详细设计——``INV-`` 类约束与 §8 验收清单的出处。"""

# 兼容单层时代的旧引用（指向 AL1，避免上层脚本一次性全改）
TASK_BOOK = TASK_BOOKS["AL1"]
LLD = LLDS["AL1"]

REVIEW_LOG = "docs/design/07-设计评审.md"
"""评审日志（DES-REV-004 第四部分）——所有缺口与过时项的裁决留痕处。"""

COVERED = "covered"
WEAK = "weak"
GAP = "gap"
STALE = "stale"

STATUSES = (COVERED, WEAK, GAP, STALE)


class Item(NamedTuple):
    """一条验收项的追溯记录。

    ``key`` 形如 ``T-AL1-04#2``；``#R`` 前缀表示"任务书的**硬要求**中
    没有被任何验收项覆盖到的条款"——这类条款最容易在验收阶段整体漏掉。
    """

    key: str
    """验收项编号。"""

    claim: str
    """验收项摘要（与任务书措辞对齐，便于人工比对）。"""

    status: str
    """见模块 docstring 的四档说明。"""

    tests: tuple[str, ...] = ()
    """对应用例的**函数名**，必须能在 ``tests/`` 里真实找到。"""

    note: str = ""
    """补充说明。非 ``covered`` 时**必须**包含 ``P<级别>-<序号>`` 形式的裁决编号。"""


AL1_ITEMS: tuple[Item, ...] = (
    # ------------------------------------------------------------------ 01 --- #
    Item("T-AL1-01#1", "`__init__.py` 内无任何业务模块 import（R6 断言）", COVERED,
         ("test_package_init_has_zero_business_logic",)),
    Item("T-AL1-01#2", "包布局符合要求", COVERED,
         ("test_package_layout",)),
    Item("T-AL1-01#3", "`name` 返回 `artifact-spirit`", COVERED,
         ("test_provider_name",)),
    Item("T-AL1-01#R1", "要求1：`pyproject.toml` 声明 entry point `hermes_agent.memory_providers`",
         COVERED,
         ("test_entry_point_is_declared_and_consistent",),
         "P1-5 已处置：原 `test_package_layout` 只断言文件存在，不检查入口注册；"
         "现由 `test_entry_point_is_declared_and_consistent` 断言 pyproject 与 plugin.yaml 两处一致"),
    # ------------------------------------------------------------------ 02 --- #
    Item("T-AL1-02#1", "`is_available` 全程无网络调用（socket 打桩断言）", COVERED,
         ("test_is_available_makes_no_network_call",)),
    Item("T-AL1-02#S1", "INV-5 不变量用例（同为「无网络」断言）", COVERED,
         ("test_is_available_makes_no_network_call",),
         "P1-6 已处置：恒真弱用例（`assert resp.ok in (True, False)`）已删除，"
         "只保留三处 socket 入口全部打桩的严格版"),
    Item("T-AL1-02#2", "配置缺失时返回 False 而非抛错", COVERED,
         ("test_is_available_returns_false_for_missing_home",
          "test_is_available_returns_false_for_broken_config")),
    Item("T-AL1-02#3", "调用耗时在毫秒级", COVERED,
         ("test_is_available_is_fast",),
         "断言 <500ms，比条款的「毫秒级」宽松，但仍是可失败的上界"),
    # ------------------------------------------------------------------ 03 --- #
    Item("T-AL1-03#1", "`initialize` 后各层服务可用", COVERED,
         ("test_initialize_after_initialize_is_usable",)),
    Item("T-AL1-03#2", "配置非法时抛 `ConfigError`（含可操作提示）", COVERED,
         ("test_initialize_raises_config_error_on_bad_config",)),
    Item("T-AL1-03#3", "`initialize` 内不含装配实现细节", COVERED,
         ("test_initialize_does_not_contain_assembly_details",)),
    # ------------------------------------------------------------------ 04 --- #
    Item("T-AL1-04#1", "schema 里每个工具名都能被 `handle_tool_call` 处理（一一对应）", COVERED,
         ("test_tool_schemas_and_handlers_are_one_to_one",
          "test_tool_schema_names_match_handlers")),
    Item("T-AL1-04#2", "`get_config_schema` 输出可渲染为合法表单（**必须是 list**）", COVERED,
         ("test_config_schema_is_a_flat_list_for_the_host",
          "test_get_config_schema_is_a_list_of_field_dicts")),
    Item("T-AL1-04#3", "`save_config` 落盘结果不含任何密钥字段", COVERED,
         ("test_save_whitelist_blocks_everything_else",
          "test_config_schema_has_no_secret_field")),
    Item("T-AL1-04#4", "`shutdown` 连调两次不报错", COVERED,
         ("test_shutdown_is_idempotent",)),
    Item("T-AL1-04#R1", "要求：`save_config` 用白名单（C5）", COVERED,
         ("test_config_schema_keys_are_save_whitelisted",)),
    # ------------------------------------------------------------------ 05 --- #
    Item("T-AL1-05#1", "核心层抛异常时仍返回非空内容（保底）", COVERED,
         ("test_prefetch_returns_non_empty_when_core_raises",)),
    Item("T-AL1-05#2", "超时返回保底内容", COVERED,
         ("test_prefetch_returns_fallback_on_timeout",)),
    Item("T-AL1-05#3", "向量不可用时仍能返回（仅 BM25）", COVERED,
         ("test_prefetch_works_without_embedding",),
         "P1-5 已处置：补 `test_prefetch_works_without_embedding`；"
         "`status` 侧的降级另见 `test_status_reports_bm25_degradation`（AL5）"),
    Item("T-AL1-05#4", "返回体的 token 量在预算内", COVERED,
         ("test_prefetch_respects_token_budget",),
         "P1-5 已处置"),
    # ------------------------------------------------------------------ 06 --- #
    Item("T-AL1-06#1", "`sync_turn` p99 < 5ms 且无 I/O", COVERED,
         ("test_invariant_inv4_sync_turn_p99",
          "test_inv4_hot_path_never_touches_storage"),
         "P1-6 已处置：原来的两条弱用例（`< 200ms` / 「调用了就通过」）已删除，"
         "由「p99 < 5ms」+「后端碰一下就炸」两条严格版取代"),
    Item("T-AL1-06#2", "`on_session_end` 立即返回（不等巩固完成）", COVERED,
         ("test_on_session_end_returns_immediately",)),
    Item("T-AL1-06#3", "队列满时不阻塞", COVERED,
         ("test_submit_is_non_blocking_when_full",)),
    # ------------------------------------------------------------------ 07 --- #
    Item("T-AL1-07#1", "非法参数返回结构化错误", COVERED,
         ("test_invalid_args_return_structured_error",)),
    Item("T-AL1-07#2", "schema 包装后必填参数逐个可在 `properties` 中找到", COVERED,
         ("test_wrapped_tools_have_callable_parameter_definitions",
          "test_tool_schemas_expose_parameters_not_input_schema")),
    Item("T-AL1-07#3", "`spirit_expand(ref, 'L1')` 不含 L2 全文", COVERED,
         ("test_expand_l1_does_not_leak_l2_text",)),
    Item("T-AL1-07#4", "`spirit_forget` 不带确认时不产生写操作", COVERED,
         ("test_forget_without_confirm_writes_nothing", "test_forget_requires_reason")),
    Item("T-AL1-07#5", "`spirit_recall` 输出含六分量原因", COVERED,
         ("test_recall_tool_reports_reasons",),
         "断言为「含原因」，未逐项核对六分量（§12.6 弱断言讨论）"),
    # ------------------------------------------------------------------ 08 --- #
    Item("T-AL1-08#1", "每个子命令 `--help` 正常", COVERED,
         ("test_every_subcommand_help_exits_zero",),
         "P1-5 已处置：由 `test_every_subcommand_help_exits_zero` 对命令表**逐个**执行 --help"),
    Item("T-AL1-08#2", "`aspirit status --json` 输出合法 JSON", COVERED,
         ("test_cli_status_json", "test_cli_accepts_json_after_subcommand_too"),
         "P0-1：修订前用例把验收原文改写成 `--json` 前置形式，那两条命令从未被执行过"),
    Item("T-AL1-08#3", "`aspirit decay` 默认 dry-run（不写库）", COVERED,
         ("test_cli_decay_defaults_to_dry_run",)),
    Item("T-AL1-08#4", "`aspirit review` 输出人类可读", COVERED,
         ("test_cli_review_is_human_readable",)),
    Item("T-AL1-08#5", "CLI 与 provider 共用同一套装配（C10，不重复实现）", COVERED,
         ("test_cli_goes_through_shared_assembly",),
         "P1-5 已处置"),
    Item("T-AL1-08#R1", "要求：命令集齐全（init/status/layers/review/trace/export/…）", COVERED,
         ("test_cli_declares_every_documented_subcommand",),
         "P1-5 已处置：命令集与文档清单逐条比对，缺一即红"),
    # ------------------------------------------------------------------ 09 --- #
    Item("T-AL1-09#1", "各内部异常均有对应翻译", COVERED,
         ("test_internal_faults_are_translated_to_actionable_tools",
          "test_storage_failure_returns_actionable_error",
          "test_unregistered_fault_falls_back_without_crashing_host"),
         "P1-2 已处置：`translate_fault` 登记了 StorageFatalError / SchemaViolationError / "
         "ProviderUnavailableError / ConfigError 四族，并从**工具出口**逐个验证；"
         "未登记的异常退回兜底文案而不抛给宿主"),
    Item("T-AL1-09#2", "任何内部异常冒泡到 AL1 都返回安全值而非抛出", COVERED,
         ("test_invalid_args_return_structured_error",
          "test_unknown_tool_returns_clear_error",
          "test_uninitialized_error_shape_matches_host_convention")),
    Item("T-AL1-09#3", "存储不可用时不崩溃宿主", COVERED,
         ("test_storage_failure_returns_actionable_error",),
         "P1-2 已处置：注入 `StorageFatalError`，断言翻译含下一步动作且保留原始细节"),
    # ------------------------------------------------------------------ 10 --- #
    Item("T-AL1-10#1", "压缩前工作记忆已归档", COVERED,
         ("test_pre_compress_archives_working_memory",),
         "P1-5 已处置：原用例只断言返回 str，现断言工作记忆真的固化成情景记忆"),
    Item("T-AL1-10#2", "镜像写入后可在语义记忆 / 核心记忆中查到", COVERED,
         ("test_memory_write_mirrors_into_the_right_layers",),
         "P1-5 已处置"),
    Item("T-AL1-10#3", "上述两者均不对主线程造成阻塞", COVERED,
         ("test_memory_write_and_pre_compress_do_not_block",),
         "P1-5 已处置"),
    Item("T-AL1-10#R1", "要求1：不声明 v2 checkpoint API（v1 best-effort）", COVERED,
         ("test_pre_compress_does_not_claim_checkpoint_api",),
         "原措辞「采用 v2 checkpoint API」与验收项 3（不阻塞主线程）互斥——2026-09-16 裁决甲（P1-3）；"
         "条款改为本文后与实现、用例一致"),
    # ------------------------------------------------------------------ 11 --- #
    Item("T-AL1-11#1", "传入含 key 的 values → 落盘结果无该字段", COVERED,
         ("test_save_whitelist_blocks_everything_else",)),
    Item("T-AL1-11#2", "白名单外字段一律忽略", COVERED,
         ("test_save_whitelist_blocks_everything_else",)),
    Item("T-AL1-11#3", "二次写入不产生重复段", COVERED,
         ("test_save_does_not_duplicate_sections",)),
    # ------------------------------------------------------------------ 12 --- #
    Item("T-AL1-12#1", "输出含核心记忆摘要与状态", COVERED,
         ("test_system_prompt_block_contains_core_memory",),
         "P1-5 已处置"),
    Item("T-AL1-12#2", "超预算时截断且不报错", COVERED,
         ("test_system_prompt_block_truncates_to_budget",),
         "P2-6 已处置：原用例传 `token_budget=100` 却断言 `len(block) < 1200`——"
         "空库下即使完全忽略该参数也能通过；现先灌入远超预算的核心记忆，"
         "再断言「小预算块必须真的更短」"),
    Item("T-AL1-12#3", "调用耗时在预算内（有界，不随记忆总量增长）", COVERED,
         ("test_system_prompt_block_does_not_scan_long_term_memory",),
         "P1-5 已处置"),
    # ------------------------------------------------------------------ 13 --- #
    Item("T-AL1-13#1", "V1：`spirit_expand(ref,'L1')` 不含 L2 全文；体量随级别单调增长", COVERED,
         ("test_value_v1_progressive_disclosure",)),
    Item("T-AL1-13#2", "V2：`spirit_review` 可直接阅读；`spirit_trace` 展示完整变更史", COVERED,
         ("test_value_v2_review_and_trace",
          "test_value_v2_recall_finds_what_was_remembered")),
    Item("T-AL1-13#3", "V3：`spirit_export` 产物纯文本可读；导出→导入往返不丢核心信息", COVERED,
         ("test_value_v3_export_is_portable",
          "test_archive_is_readable_without_any_code",
          "test_export_import_export_is_equivalent")),
    Item("T-AL1-13#4", "INV-4：`sync_turn` p99 < 5ms", COVERED,
         ("test_invariant_inv4_sync_turn_p99",
          "test_inv4_hot_path_never_touches_storage")),
    Item("T-AL1-13#5", "INV-5：`is_available` 无网络调用", COVERED,
         ("test_is_available_makes_no_network_call",)),
    Item("T-AL1-13#R1", "要求：三条核心价值写端到端断言，产出 `tests/test_value.py`", COVERED,
         ("test_value_v1_progressive_disclosure",
          "test_value_v2_review_and_trace",
          "test_value_v3_export_is_portable",
          "test_end_to_end_turn_pipeline"),
         "P2-7 已处置：值测试已从 `test_runtime_adapter.py` 拆入 `tests/test_value.py`（P1-6）"),
    # ------------------------------------------------- M3–M5（尚未开工） --- #
    # 这五条随 DES-REV-004 的复核结论补进任务书（编号顺延追加）。任务书里**有**条款，
    # 矩阵就必须**有**条目——否则 `test_every_task_in_task_book_has_matrix_entry` 会红，
    # 而那条红正是"新增了要求却没人管它怎么验"的机械堵口。
    Item("T-AL1-14", "M3：`spirit_asof` 按时间点返回（该时刻有效的那一版）", COVERED,
         ("test_asof_tool_distinguishes_no_version_from_current_value",
          "test_asof_through_the_facade",
          "test_asof_returns_the_version_valid_at_that_time"),
         "M3 已落地（2026-09-17）：`found` 显式区分「那时没有」与「取到了」；"
         "并回带 `valid_from` / `valid_to` / `superseded_by`，让「为什么是这条版本」可查"),
    Item("T-AL1-15", "M4：审查干预在工具面闭环（review → trace → correct 且留审计）", COVERED,
         ("test_review_trace_correct_loop_is_visible_in_trace",
          "test_core_memory_tool_promotes_and_demotes",
          "test_core_memory_tool_without_reason_is_a_structured_error"),
         "M4 已落地（2026-09-17）：新增 `spirit_core`（升格 / 降格）；"
         "`review → trace → correct` 闭环后 **trace 里看得到 actor 与 reason**；"
         "缺 `reason` 返回结构化错误而非异常穿透（F10）"),
    Item("T-AL1-16", "M5：CLI 命令全集，声明与文档表一致且**逐个** `--help` 通过", COVERED,
         ("test_cli_declares_every_documented_subcommand",),
         "M5 已落地（2026-09-17）：新增 `ingest` 命令；`soul` / `awaken` **刻意不在表内**并写明原因——任务书明文要求「不实现就必须移除」；留一条「能列出但一跑就报未实现」的空壳，比没有这条命令更坏（DES-REV-003 P1-10）"),
    Item("T-AL1-17", "M5：传承三入口语义不重叠（`export` / `import` / `ingest`）", COVERED,
         ("test_value_v3_transfer_has_no_model_dependency",
          "test_value_v3_import_reports_what_it_did"),
         "M5 已落地（2026-09-17）：`spirit_import`（重建）与 `spirit_ingest`（提取）**依赖表就不同**——`Transferrer` 的字段里没有 llm/extractor，所以「导入不调模型」不是「没发生」而是**没有路径能发生**"),
    Item("T-AL1-18", "M5：传承往返逐字相等，且二次导入零新增（V3 最终判决）", COVERED,
         ("test_value_v3_roundtrip_is_verbatim_and_idempotent",
          "test_value_v3_archive_header_and_no_private_markers"),
         "M5 已落地（2026-09-17）：**逐字段**比对（用 dataclass 的 `==` 替我们比，比手写 assert 不易漏）；二次导入**零新增、零改写**；档案头含 `archive_version`，且不含向量 / 删除快照"),
)


AL1_INVARIANTS: tuple[Item, ...] = (
    Item("INV-4", "`sync_turn` 非阻塞且无 I/O", COVERED,
         ("test_invariant_inv4_sync_turn_p99",
          "test_inv4_hot_path_never_touches_storage")),
    Item("INV-5", "`is_available` 纯本地、零网络", COVERED,
         ("test_is_available_makes_no_network_call",),
         "P1-6 已处置：恒真弱用例已删除，严格版（三处 socket 入口打桩）保留"),
    Item("INV-9", "所有产出路径都位于传入的 `hermes_home` 之下", COVERED,
         ("test_inv9_outputs_stay_under_hermes_home",),
         "P1-5 已处置：原为零覆盖，现由 `test_inv9_outputs_stay_under_hermes_home` "
         "在最小配置（无配置文件、路径全靠默认值推导）下断言"),
)


# =========================================================================== #
# AL2 核心层（DES-REV-005 · 15 个 M1/M2 任务 + 8 个 M3–M5 任务）
# =========================================================================== #

AL2_ITEMS: tuple[Item, ...] = (
    # ---------------------------------------------------------------- M1 --- #
    Item("T-AL2-01", "`CoreFacade` 方法与共享数据类（契约按实现，P1-8/P1-9/P1-10）", COVERED,
         ("test_recall_weights_sum_to_one", "test_scored_raw_has_six_components",
          "test_core_layer_has_no_io_imports", "test_core_does_not_import_hermes")),
    Item("T-AL2-02", "显著性五因子过滤；降级时门槛同步缩放（P1-19）", COVERED,
         ("test_explicit_instruction_scores_higher_than_neutral",
          "test_similar_input_scores_lower_novelty",
          "test_below_threshold_produces_no_candidates",
          "test_salience_degrades_without_embedding"),
         "P1-19 已落进要求（降级时门槛 ×(1−novelty 权重)）；P2-8 的权重数值已写进 LLD §5 M1"),
    Item("T-AL2-03", "工作记忆容量与意图槽（淘汰=移出注意力窗口，P1-12）", COVERED,
         ("test_working_memory_evicts_coldest_from_attention",
          "test_same_topic_accumulates_into_same_chunk",
          "test_intent_slot_not_limited_by_capacity"),
         "P1-12 已落进要求；P1-15 的意图槽启发式已承认并说明为何够用"),
    Item("T-AL2-04", "六层 `LayerService` 骨架（基类共享行为，P1-11）", COVERED,
         ("test_all_layer_services_instantiate", "test_procedural_layer_is_placeholder",
          "test_sensory_layer_never_persists")),
    Item("T-AL2-05", "召回六因子打分（entity 走一次 SQL，P0-4）", COVERED,
         ("test_all_factors_within_unit_interval",
          "test_importance_is_decoupled_from_access_count",
          "test_entity_factor_rises_when_query_hits_entity",
          "test_recency_decays_monotonically"),
         "P0-4 已落进要求（entity 不得在循环里扫表）；P2-9 的权重组装已记载"),
    Item("T-AL2-06", "融合与预算裁剪（三条边界语义已记载，P2-9）", COVERED,
         ("test_weight_change_alters_ranking",
          "test_fuse_renormalizes_when_a_factor_is_missing",
          "test_clip_to_budget_keeps_abstract_form")),
    Item("T-AL2-07", "分级加载 `expand`（`hot_path` 已进契约，P1-8）", COVERED,
         ("test_expand_levels_grow_monotonically",
          "test_expand_l1_does_not_contain_l2_fulltext",
          "test_expand_l2_is_read_only",
          "test_hot_path_never_generates_l1")),
    Item("T-AL2-08", "审查 / 溯源 / 干预（AL2 只给数据结构，P1-17）", COVERED,
         ("test_review_includes_importance_breakdown", "test_trace_returns_full_history",
          "test_correct_produces_intent_not_direct_write",
          "test_forget_requires_reason_and_source",
          "test_intervention_audit_actor_is_user"),
         "P1-17 已落进要求：渲染归 AL5 的 `observability/render.py`"),
    # ---------------------------------------------------------------- M2 --- #
    Item("T-AL2-09", "提取 schema 与校验器（实现只有一份，P1-35）", COVERED,
         ("test_extraction_schema_validates_bad_payloads",
          "test_prompt_and_schema_share_one_source")),
    Item("T-AL2-10", "结构化提取与 F1/F3 降级", COVERED,
         ("test_extraction_produces_candidates",
          "test_extraction_produces_no_partial_candidates_on_bad_json",
          "test_llm_unavailable_degrades_to_raw_text_only")),
    Item("T-AL2-11", "去重决策；`audit.op` 必须等于执行分支（P0-6）", COVERED,
         ("test_dedup_add_branch", "test_dedup_ignore_branch",
          "test_dedup_decision_is_audited", "test_audit_op_covers_every_decision"),
         "P0-6 已落进要求（MERGE 不许静默降级成 ADD 而审计写 merge）；"
         "`test_audit_op_covers_every_decision` 把「决策 ↔ 审计 op」做成**双向穷举断言**"),
    Item("T-AL2-12", "巩固：工作 → 情景", COVERED,
         ("test_session_end_produces_one_episode", "test_consolidation_is_idempotent")),
    Item("T-AL2-13", "巩固：情景 → 语义（计数改聚合，P1-18）", COVERED,
         ("test_cross_session_repetition_promotes_to_semantic", "test_promotion_is_idempotent"),
         "P1-18 已落进要求：跨会话计数不得靠 `limit=500` 拉行"),
    Item("T-AL2-14", "衰减排序与降级（`dormant` 降权系数可配，P1-13）", COVERED,
         ("test_decay_produces_no_deletion_or_downgrade",
          "test_decay_module_contains_no_delete_calls",
          "test_decay_reports_low_strength_candidates",
          "test_status_target_never_archive",
          "test_dormant_memory_keeps_full_detail")),
    Item("T-AL2-15", "记忆优化删除（不可达判定；判据进文档，P1-14）", COVERED,
         ("test_isolated_memory_hits_unreachable_criterion",
          "test_optimizer_deletes_with_optimizer_actor_and_restorable",
          "test_optimizer_criterion_ignores_time_and_frequency"),
         "P0-7（错误 / 冲突记忆未实现）与 P1-14（判据魔数）已落进要求，由 M3 补齐"),
    # ------------------------------------------------------ M3–M5（待开工） --- #
    Item("T-AL2-16", "M3：双时态语义（失效标记 + as-of 查询）", COVERED,
         ("test_invalidate_marks_without_deleting",
          "test_invalidate_also_writes_supersedes_edge",
          "test_invalidate_audit_op_matches_the_action",
          "test_invalidate_rejects_self_reference",
          "test_invalidate_rejects_backwards_interval",
          "test_asof_through_the_facade"),
         "M3 已落地（2026-09-17）：失效产 `update` + `link` **两条**意图（`superseded_by` 答「被谁取代」、"
         "`supersedes` 边答「它取代了谁」）；自指与倒挂区间被拒；`invalidate` 已登记进 `AUDIT_OPS`"
         "白名单（P1-20 的处置）"),
    Item("T-AL2-17", "M3：五态去重（MERGE 真合并 / INVALIDATE 不删 / FORGET 仅合规）", COVERED,
         ("test_dedup_invalidate_branch_on_value_change",
          "test_dedup_merge_branch_on_different_wording",
          "test_merge_texts_prefers_the_superset_and_joins_otherwise",
          "test_fact_change_end_to_end_invalidates_the_old_version",
          "test_audit_op_covers_every_decision"),
         "M3 已落地（2026-09-17）：**取值变化改判 `INVALIDATE`**（历史不再被覆盖）；"
         "`MERGE` 真合并（`merge_texts`：包含关系取超集，否则分号连接）；"
         "`ignore` / `merge` 已登记进 `AUDIT_OPS`——P1-20 的残留（它们是**字典字面量的值**，"
         "上一轮的 `op=\"…\"` 统计漏了它们）"),
    Item("T-AL2-18", "M3：交叉验证（LLM 仲裁，异步限频，INV-11）", COVERED,
         ("test_no_conflict_means_no_llm_call",
          "test_conflict_triggers_arbitration_and_invalidates_older",
          "test_llm_unavailable_keeps_both_and_says_so",
          "test_bad_llm_output_falls_back_to_keep_both",
          "test_already_settled_pairs_are_not_asked_again"),
         "M3 已落地（2026-09-17）：三处限频（只挑真冲突 / 已裁决的跳过 / 每轮有上限）；"
         "**失败一律 `KEEP_BOTH`**（代价不对称：多留一条无害，误删一条静默消失）；"
         "未配置 LLM 时只报告不裁决（`report.skipped` 说清原因）"),
    Item("T-AL2-19", "M4：扩散激活（默认一跳，`hops > 1` 抛错）", COVERED,
         ("test_diffusion_gives_zero_without_edges",
          "test_seed_itself_scores_full",
          "test_neighbor_takes_the_max_edge_weight_and_is_clipped",
          "test_hops_beyond_one_is_refused_not_silently_downgraded",
          "test_activate_expands_along_edges_and_respects_threshold"),
         "M4 已落地（2026-09-17）：实现原本就有（六因子里一路），本轮补的是**从未断言过的性质**——"
         "`hops > 1` 抛错（不静默降级）、孤立得 0（不是兜底值）、弱边不激活（否则退化成全连通）"),
    Item("T-AL2-20", "M4：记忆进化（改写抽象、保留原文与 before）", COVERED,
         ("test_evolution_touches_abstract_but_never_content",
          "test_evolution_leaves_audit_with_before_and_after",
          "test_evolution_disabled_degrades_to_add_only",
          "test_evolution_without_llm_says_why_it_did_nothing"),
         "M4 已落地（2026-09-17）：`core/evolution.py`——**只改 `abstract`，永不改 `content`**；"
         "每次改写留 before/after；`evolution_enabled=False` 真正退回纯 ADD（连 LLM 都不调）；"
         "无 LLM 时报告 `skipped` 说明原因"),
    Item("T-AL2-21", "M4：核心记忆（高门槛升格，INV-11）", COVERED,
         ("test_core_promotion_is_refused_below_threshold",
          "test_promotion_is_audited_and_visible_in_the_prompt_block"),
         "M4 已落地（2026-09-17）：未达门槛**不升格**（差一点也不能上）；"
         "升格后 `system_prompt_block` **真的看得到它**；`propose()` 补上了原先**缺失的审计**"
         "（`op=promote`）——升格是所有写入里后果最重的一次"),
    Item("T-AL2-22", "M4：审查干预增强（时态干预 + 核心记忆干预）", COVERED,
         ("test_temporal_intervention_changes_asof_but_keeps_history",
          "test_interventions_require_a_reason",
          "test_core_interventions_are_audited_and_change_the_layer",
          "test_demote_refuses_a_non_core_memory"),
         "M4 已落地（2026-09-17）：`invalidate_since`（**事实变了**≠记错了：只截断有效期，"
         "历史仍可查）；`promote` / `demote`（人工升格**不过置信度门槛**——门槛是给机器设的）；"
         "四种干预**缺 `reason` 一律拒绝**；`promote` 必须同时改 `type`，"
         "否则会「升格成功但系统提示里看不见」"),
    Item("T-AL2-23", "M5：传承重建（导入走写队列、不重新提取、幂等）", COVERED,
         ("test_value_v3_roundtrip_is_verbatim_and_idempotent",
          "test_value_v3_transfer_has_no_model_dependency"),
         "M5 已落地（2026-09-17）：`core/transfer.py` 把档案翻成**写意图**（不再直插——CLI 原先调 `backend.import_archive` 会绕过写队列与审计）；以 `content_hash` 判重；两阶段回填 `superseded_by`（队列按优先级消费，顺序只能由**意图序列本身**保证）"),
)

AL2_INVARIANTS: tuple[Item, ...] = (
    Item("INV-1", "L1 概览是缓存：清空后功能不受影响", COVERED,
         ("test_hot_path_serves_stale_cache_without_recompute",
          "test_cold_path_generates_and_requests_cache_write",
          "test_overview_is_pure_cache_clearing_is_harmless")),
    Item("INV-7", "删除纪律：衰减只影响排序，物理删除仅白名单 + 留快照", COVERED,
         ("test_decay_module_contains_no_delete_calls", "test_optimizer_dry_run_writes_nothing",
          "test_forget_and_restore_roundtrip")),
    Item("INV-11", "核心记忆高门槛低频（不每轮触发 LLM）", GAP, (),
         "AL2 §10 有该不变量，M4 的核心记忆尚未开工——按 E7 记 `gap`（P1-5 同口径）"),
)


# =========================================================================== #
# AL3 存储层（DES-REV-006 · 23 个 M0–M2 任务 + 3 个 M3/M5 任务）
# =========================================================================== #

AL3_ITEMS: tuple[Item, ...] = (
    # ---------------------------------------------------------------- M0 --- #
    Item("T-AL3-01", "建库、PRAGMA 与迁移框架", COVERED,
         ("test_migrate_creates_all_tables", "test_migrate_creates_all_triggers",
          "test_migrate_is_idempotent", "test_journal_mode_is_wal",
          "test_schema_version_written", "test_creates_parent_directory")),
    Item("T-AL3-02", "`meta` 键值读写", COVERED,
         ("test_meta_roundtrip", "test_meta_set_overwrites", "test_spirit_id_not_overwritten",
          "test_spirit_id_persists_across_reopen")),
    Item("T-AL3-03", "`vec0` 能力探测（退化路径已裁掉）", COVERED,
         ("test_vec_capability_recorded", "test_vec_memories_accepts_text_pk",
          "test_vec_knn_and_delete_work", "test_unsupported_vec_capability_fails_loudly",
          "test_probe_not_repeated_on_reopen")),
    Item("T-AL3-04", "生命周期骨架 `open` / `close`", COVERED,
         ("test_open_close_open", "test_close_releases_connection", "test_close_is_idempotent")),
    # ---------------------------------------------------------------- M1 --- #
    Item("T-AL3-05", "`memories` 表 CRUD", COVERED,
         ("test_memory_id_format_is_abbr_ulid", "test_put_then_get_roundtrip_all_fields",
          "test_query_filters_by_status", "test_query_filters_by_types_and_layer",
          "test_update_patch_and_updated_at")),
    Item("T-AL3-06", "FTS5 表与同步触发器（CJK 逐字归一化）", COVERED,
         ("test_fts_indexes_on_insert", "test_fts_syncs_on_update", "test_fts_clears_on_delete",
          "test_fts_matches_both_chinese_and_english", "test_rebuild_fts_restores_index")),
    Item("T-AL3-07", "向量索引读写（维度校验 + 无半写）", COVERED,
         ("test_vector_write_and_knn_order",
          "test_vector_dimension_mismatch_raises_and_no_half_write",
          "test_embedding_model_recorded_on_row")),
    Item("T-AL3-08", "检索原语（只给原生分）", COVERED,
         ("test_search_returns_native_scores_not_fused", "test_hit_fields_complete",
          "test_search_layer_filter", "test_keyword_search_empty_query_returns_empty")),
    Item("T-AL3-09", "`relations` 表与关联操作", COVERED,
         ("test_link_is_idempotent", "test_reinforce_grows_weight_and_count",
          "test_neighbors_sorted_by_weight_desc", "test_neighbors_min_weight_filter")),
    Item("T-AL3-10", "`entities` 表", COVERED,
         ("test_entity_upsert_dedupes_same_name_type", "test_entity_find_by_name_and_alias",
          "test_entity_find_requires_no_llm")),
    Item("T-AL3-11", "工作记忆与意图槽", COVERED,
         ("test_wm_put_merges_same_chunk_key", "test_wm_list_orders_by_last_touched_desc",
          "test_wm_clear_only_current_session", "test_delete_session_cascades_working_memory")),
    Item("T-AL3-12", "`audit` 表与 append-only（`op` 白名单，P1-20）", COVERED,
         ("test_audit_write_and_replay_order", "test_audit_replay_since_filter",
          "test_audit_is_append_only_update_rejected", "test_audit_is_append_only_delete_rejected",
          "test_audit_for_returns_history_of_one_memory"),
         "P1-20 已落进要求：`op` 提为显式白名单，LLD / schema.sql / base.py 三处对齐"),
    Item("T-AL3-13", "分级加载存储 `overviews`（纯缓存）", COVERED,
         ("test_overview_put_get_roundtrip_by_level",
          "test_overview_invalidate_returns_affected_rows",
          "test_overview_is_pure_cache_clearing_is_harmless")),
    Item("T-AL3-14", "内容寻址指纹（INV-14）", COVERED,
         ("test_content_hash_is_deterministic", "test_content_hash_differs_by_scope",
          "test_content_hash_same_triple_different_wording", "test_find_by_hash_may_return_multiple")),
    Item("T-AL3-15", "可读档案导出与文本投影", COVERED,
         ("test_archive_header_self_describing", "test_archive_contains_core_fields",
          "test_archive_excludes_vectors", "test_export_archive_writes_readable_file")),
    Item("T-AL3-16", "写入的原子性（含删除同事务，P2-17）", COVERED,
         ("test_put_atomic_when_audit_fails", "test_put_atomic_when_vector_write_fails",
          "test_put_success_writes_all_four_places"),
         "P2-17 的缺口已写进验收（删除路径的 audit 失败用例）"),
    # ---------------------------------------------------------------- M2 --- #
    Item("T-AL3-17", "`sessions` 生命周期", COVERED,
         ("test_session_end_sets_committed_and_ended_at", "test_session_turn_count_accumulates")),
    Item("T-AL3-18", "删除与恢复（快照 + 可恢复）", COVERED,
         ("test_delete_keeps_audit_history",
          "test_restore_recovers_record_with_same_id_and_relations",
          "test_purge_snapshot_makes_restore_impossible", "test_delete_without_reason_is_rejected")),
    Item("T-AL3-19", "派生字段重放重建（INV-12）", COVERED,
         ("test_replay_restores_derived_fields", "test_replay_reports_orphan_events"),
         "**这条是 AL5 §8.4 的同名验收的实装处**——P1-58 曾误判为"
         "\"全仓无用例\"，实际用例在 AL3 层（单向检索只扫了 AL5 的测试文件）"),
    Item("T-AL3-20", "重嵌入工具（断点续跑）", COVERED,
         ("test_reindex_rebuilds_all_vectors", "test_reindex_resumes_without_reprocessing",
          "test_reindex_updates_meta_model_and_dim")),
    Item("T-AL3-21", "启动对账", COVERED,
         ("test_reconcile_fixes_missing_vectors", "test_reconcile_removes_orphan_vectors",
          "test_reconcile_dry_run_writes_nothing", "test_fresh_db_reconcile_is_clean")),
    Item("T-AL3-22", "记忆包导出/导入与幂等（INV-14）", COVERED,
         ("test_import_pack_twice_produces_no_duplicates", "test_export_import_export_is_equivalent",
          "test_import_pack_into_empty_db_is_lossless", "test_import_rejects_foreign_payload")),
    Item("T-AL3-23", "架构与合规测试（R2 / R7）", COVERED,
         ("test_store_layer_has_no_illegal_imports", "test_fts_is_derived_and_rebuildable")),
    # ------------------------------------------------------ M3 / M5（待开工） --- #
    Item("T-AL3-24", "M3：双时态写入与 as-of 查询（字段已在 v1 DDL，本任务**启用**它）", COVERED,
         ("test_put_defaults_valid_from_to_created_at",
          "test_asof_returns_the_version_valid_at_that_time",
          "test_asof_before_first_write_returns_none",
          "test_asof_sql_calls_do_not_scale_with_table_size",
          "test_superseded_by_self_reference_is_rejected"),
         "M3 已落地（2026-09-17）。**原条款写的「v1 → v2 迁移」是错的**：`SCHEMA_VERSION` 当时已是 2，"
         "且双时态字段早在 v1 的全量 DDL 里——本任务是**启用**（被写/被查/被往返），不是迁移"),
    Item("T-AL3-25", "M3：完整包往返带时态字段（含 `superseded_by`）", COVERED,
         ("test_import_pack_roundtrips_superseded_by",
          "test_import_pack_roundtrip_is_idempotent_for_temporal_fields"),
         "M3 已落地（2026-09-17）：导入侧原先**漏了** `superseded_by`（导出侧 `asdict()` 一直带着它）；"
         "并改为**两阶段导入**——该字段是自引用外键，包内顺序任意会让逐条 `put` 撞外键"),
    Item("T-AL3-26", "M5：传承导出与重建（`archive_version`，无器灵可读懂）", COVERED,
         ("test_value_v3_archive_header_and_no_private_markers",),
         "M5 已落地（2026-09-17）：`ARCHIVE_VERSION` 与 `schema_version` **分开**——合成一个数的后果是「加一个头部字段就得假装库结构变了」；`load_archive` / `check_archive_version`：**读不懂就明说**，不用旧读者硬读新档案"),
)

AL3_INVARIANTS: tuple[Item, ...] = (
    Item("INV-2", "向量维度与 `meta.embedding_dim` 一致，不一致拒绝启动", COVERED,
         ("test_embedding_dim_mismatch_refused", "test_vector_dimension_mismatch_raises_and_no_half_write")),
    Item("INV-8", "`audit` append-only（UPDATE / DELETE 被触发器拒绝）", COVERED,
         ("test_audit_is_append_only_update_rejected", "test_audit_is_append_only_delete_rejected",
          "test_audit_is_append_only")),
    Item("INV-12", "派生字段可从审计重放重建", COVERED,
         ("test_replay_restores_derived_fields",)),
    Item("INV-13", "档案开放可读、不依赖运行时", COVERED,
         ("test_archive_is_readable_without_any_code",)),
    Item("INV-14", "内容寻址去重（导入幂等）", COVERED,
         ("test_import_pack_twice_produces_no_duplicates",
          "test_content_hash_is_deterministic", "test_content_hash_differs_by_scope")),
)


# =========================================================================== #
# AL4 模型层（DES-REV-007 · 12 个 M0–M2 任务 + 1 个 M3 任务）
# =========================================================================== #

AL4_ITEMS: tuple[Item, ...] = (
    # ---------------------------------------------------------------- M0 --- #
    Item("T-AL4-01", "协议与异常族（父类归属不可动，P1-31 ~ P1-36）", COVERED,
         ("test_exception_hierarchy", "test_protocols_defined",
          "test_protocol_module_has_no_io_imports"),
         "P1-31 契约副本 / P1-33 父类归属 / P1-34 `SchemaValidationError` / "
         "P1-36 C11·C12 均已落进要求；P1-32 的 `temperature=None` 语义已写明"),
    Item("T-AL4-02", "OpenAI 兼容客户端骨架", COVERED,
         ("test_complete_returns_text", "test_base_url_trailing_slash_normalized",
          "test_timeout_is_passed_explicitly")),
    Item("T-AL4-03", "重试策略与异常分类", COVERED,
         ("test_connection_failure_retries_exactly_twice", "test_http_429_is_not_retried",
          "test_http_401_is_not_retried_and_raises_unavailable")),
    # ---------------------------------------------------------------- M1 --- #
    Item("T-AL4-04", "结构化输出与校验重试", COVERED,
         ("test_complete_json_returns_dict", "test_complete_json_retries_exactly_once_then_succeeds",
          "test_complete_json_raises_after_two_failures",
          "test_validator_detects_missing_field_and_type_error")),
    Item("T-AL4-05", "Embedding 批量与保序", COVERED,
         ("test_embed_preserves_input_order", "test_embed_batch_size_config",
          "test_embed_long_text_gets_own_batch")),
    Item("T-AL4-06", "三档模型解析与 fallback 链（P1-39 / P1-40）", COVERED,
         ("test_spirit_config_wins", "test_host_fallback_when_spirit_unconfigured",
          "test_env_fallback_when_nothing_configured", "test_all_segments_down_raises",
          "test_embedding_never_falls_back_to_llm"),
         "P1-39（env 段两个变量）与 P1-40（YAML 子集边界）已落进要求"),
    Item("T-AL4-07", "维度与模型属性（判定三步，比对不在 AL4，P0-11）", COVERED,
         ("test_known_embedding_model_dim", "test_explicit_dim_overrides_known",
          "test_unknown_model_requires_explicit_dim", "test_invalid_dim_rejected"),
         "P0-11 已落进要求：与库中 `meta` 的比对由 AL5 装配期做（R3 禁止 AL4 碰 store）"),
    Item("T-AL4-08", "配置对接（密钥只从环境变量读）", COVERED,
         ("test_api_key_read_from_env_var", "test_missing_key_env_error_does_not_leak_key")),
    # ---------------------------------------------------------------- M2 --- #
    Item("T-AL4-09", "降级路径（F1 / F2 / F3；最坏请求数连乘，P1-37）", COVERED,
         ("test_f1_llm_unconfigured_raises_provider_unavailable",
          "test_f2_embedding_unconfigured_raises_embedding_error",
          "test_f3_non_json_raises_schema_violation",
          "test_degradation_exception_types_are_mutually_distinguishable"),
         "P1-37（`(1+2)×2 = 6` 上界）与 P1-41（末段不得 `active=True`）已落进要求"),
    Item("T-AL4-10", "超时与限流纪律（不引入限流器）", COVERED,
         ("test_no_rate_limiter_in_model_layer", "test_timeouts_are_configurable")),
    Item("T-AL4-11", "密钥与日志纪律", COVERED,
         ("test_settings_repr_redacts_key", "test_exception_text_does_not_contain_key")),
    Item("T-AL4-12", "契约测试（mock HTTP；文件名已改正，P2-25）", COVERED,
         ("test_model_layer_respects_architecture", "test_model_layer_does_not_import_store_or_core",
          "test_no_openai_sdk_used", "test_tests_need_no_network"),
         "P2-25 已落进产出：实际文件是 `tests/test_model.py`，不是 `test_model_contract.py`"),
    # ------------------------------------------------------ M3（待开工） --- #
    Item("T-AL4-13", "M3：交叉验证仲裁池（专用档位，不新增 HTTP 路径）", COVERED,
         ("test_crosscheck_task_is_a_separate_slot",
          "test_crosscheck_reuses_the_single_http_path"),
         "M3 已落地（2026-09-17）：`TASKS` 与 `DEFAULT_LLM_TASK_MODELS` 都加了 `crosscheck`"
         "（走轻量档）；仲裁复用 `complete_json`，**不新增第二条 HTTP 路径**"),
)

AL4_INVARIANTS: tuple[Item, ...] = (
    Item("INV-6", "embedding 绝不被 LLM 取代（链的隔离性）", COVERED,
         ("test_embedding_error_is_embedding_error_not_llm_error",
          "test_llm_chain_failure_does_not_disable_embedding")),
)


# =========================================================================== #
# AL5 辅助机制（DES-REV-008 · 12 个 M0–M2 任务 + 1 个 M5 任务）
# =========================================================================== #

AL5_ITEMS: tuple[Item, ...] = (
    # ---------------------------------------------------------------- M0 --- #
    Item("T-AL5-01", "配置模型与装载（缺失不是错误，P1-57；模板对齐白名单，P2-27）", COVERED,
         ("test_toml_example_is_valid_and_loadable", "test_load_valid_toml",
          "test_env_overrides_toml", "test_missing_config_file_is_not_an_error",
          "test_broken_toml_gives_actionable_error"),
         "P1-57 已改条款（无配置 → 默认值启动 + 可操作提示）；P2-27 的模板对齐断言已入验收"),
    Item("T-AL5-02", "配置校验（严重性是文案前缀，P2-29）", COVERED,
         ("test_validate_blocks_bad_queue_max",
          "test_validate_warns_but_does_not_block_on_weight_sum"),
         "P2-29 已写进要求：`\"警告\"` 前缀是隐式契约，改文案即改接口"),
    Item("T-AL5-03", "生命周期装配骨架（启停顺序 = M8）", COVERED,
         ("test_start_and_stop_are_repeatable", "test_stop_order_has_writer_before_maintenance",
          "test_start_refuses_on_embedding_model_mismatch",
          "test_start_degrades_gracefully_without_models")),
    # ---------------------------------------------------------------- M1 --- #
    Item("T-AL5-04", "writer 线程与单写者（所有写路径都经队列，P0-12/P0-15）", COVERED,
         ("test_submit_is_non_blocking_when_full", "test_all_writes_happen_in_one_thread",
          "test_concurrent_submits_do_not_raise_busy", "test_writer_survives_task_exception",
          "test_write_now_goes_through_the_writer_thread"),
         "P0-12 与 P0-15 已修（离线产出 + 工具路径都必须经队列）；恒真兜底用例已删"),
    Item("T-AL5-05", "队列优先级与溢出（两张表都穷举，P1-43）", COVERED,
         ("test_overflow_drops_lowest_priority_first", "test_intent_priority_mapping",
          "test_intent_priority_covers_every_intent_op", "test_every_task_kind_has_a_priority"),
         "P1-43 已落进要求：`kind_priority` 与 `intent_priority` 都是穷举表"),
    Item("T-AL5-06", "`status` 可观测（配置告警必须有出口，P1-45）", COVERED,
         ("test_status_has_five_sections", "test_status_shows_model_chain_and_source",
          "test_status_reports_bm25_degradation", "test_status_is_idempotent",
          "test_status_surfaces_config_warnings")),
    Item("T-AL5-07", "`layers` / `reflect` / `audit_view`（单一投影，P1-46/P1-47/P1-48）", COVERED,
         ("test_layers_counts_match_backend", "test_reflect_explains_forgetting_and_recall",
          "test_audit_view_filters"),
         "P1-47 已落进要求：审计只有一个投影、编号必须是真实 `audit_id`"),
    Item("T-AL5-08", "启动对账与一致性校验", COVERED,
         ("test_reconcile_runs_on_start", "test_start_refuses_on_embedding_model_mismatch")),
    # ---------------------------------------------------------------- M2 --- #
    Item("T-AL5-09", "maintenance 线程与调度（缓存 thread_id，P1-44；消费后销账，P1-50）", COVERED,
         ("test_maintenance_merges_duplicate_tasks", "test_maintenance_runs_offline_cycle",
          "test_scheduled_consolidation_is_actually_executed"),
         "P1-44 / P1-50 / P2-31 均已落进要求"),
    Item("T-AL5-10", "关闭安全与崩溃恢复（无消费者不空等，P1-51）", COVERED,
         ("test_writer_stop_counts_inflight_tasks", "test_stop_reports_undrained_writes",
          "test_stop_without_threads_still_reports_sequence"),
         "P1-58 已更正：`replay --rebuild-derived` 的用例在 **AL3** "
         "（`test_replay_restores_derived_fields`），不是「全仓无用例」"),
    Item("T-AL5-11", "合规门禁（R1–R10 + 依赖 + 密钥）", COVERED,
         ("test_architecture_rules_all_green", "test_dependency_scan_is_clean",
          "test_only_runtime_creates_threads", "test_write_queue_does_no_business_reasoning",
          "test_arch_rule_r8_catches_thread_creation_outside_runtime")),
    Item("T-AL5-12", "`doctor` 自检（全程本地不联网）", COVERED,
         ("test_doctor_reports_each_check_independently", "test_status_text_is_readable")),
    # ------------------------------------------------------ M5（待开工） --- #
    Item("T-AL5-13", "M5：可观测扩展与传承支撑（四件套 + 进度可见）", COVERED,
         ("test_value_v3_import_reports_what_it_did",),
         "M5 已落地（2026-09-17）：`doctor` 新增「双时态字段」（用 `LIMIT 0` 探列，空库也能通过）与「档案格式」两项；传承导入有可读摘要，且新增 / 跳过 / 错误**分开计数**——合并会让数据问题被「幂等生效」这个好消息掩盖"),
)

AL5_INVARIANTS: tuple[Item, ...] = (
    Item("INV-4", "热路径绝不阻塞宿主", COVERED,
         ("test_invariant_inv4_sync_turn_p99", "test_queue_prefetch_is_non_blocking_and_does_no_io",
          "test_submit_is_non_blocking_when_full")),
    Item("INV-5", "本地判定不联网（`is_available` / `doctor`）", COVERED,
         ("test_is_available_makes_no_network_call", "test_doctor_reports_each_check_independently")),
    Item("INV-10", "依赖准入（无传染性 / 许可不明依赖）", COVERED,
         ("test_dependency_scan_is_clean", "test_dep_scan_catches_injected_forbidden_module")),
)


# =========================================================================== #
# 五层聚合（2026-09-16 从"仅 AL1"扩到 AL1–AL5）
# =========================================================================== #


class Layer(NamedTuple):
    """一层的追溯矩阵。

    每层自带**权威文档位置**：任务书（``T-{层}-xx`` 的定义处）与 LLD（``INV-x`` 的出处）——
    校验器据此核对"引用的编号是否真实存在"。
    """

    name: str
    task_book: str
    lld: str
    items: tuple[Item, ...]
    invariants: tuple[Item, ...] = ()

    @property
    def all_items(self) -> tuple[Item, ...]:
        return self.items + self.invariants

    @property
    def task_prefix(self) -> str:
        """任务编号前缀，如 ``T-AL1``。"""
        return f"T-{self.name}"


LAYERS: tuple[Layer, ...] = (
    Layer("AL1", TASK_BOOKS["AL1"], LLDS["AL1"], AL1_ITEMS, AL1_INVARIANTS),
    Layer("AL2", TASK_BOOKS["AL2"], LLDS["AL2"], AL2_ITEMS, AL2_INVARIANTS),
    Layer("AL3", TASK_BOOKS["AL3"], LLDS["AL3"], AL3_ITEMS, AL3_INVARIANTS),
    Layer("AL4", TASK_BOOKS["AL4"], LLDS["AL4"], AL4_ITEMS, AL4_INVARIANTS),
    Layer("AL5", TASK_BOOKS["AL5"], LLDS["AL5"], AL5_ITEMS, AL5_INVARIANTS),
)
"""五层追溯矩阵。**顺序即执行顺序**（AL3/AL4 → AL2 → AL5 → AL1）。"""
