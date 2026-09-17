"""价值判决：这套记忆系统**到底值不值**（V1 / V2 / V3 与不变式）。

本文件由 `test_runtime_adapter.py` 拆出（设计评审 P2-7）。它和其余测试文件的区别在
**提问方式**：`test_al1_*` / `test_al5_*` 问"实现是否符合契约"，这里问"用户因此得到了
什么"——省 token（V1）、找得回来（V2）、能删能救（V3）、不拖慢对话（INV-4）、
不偷偷联网（INV-5）。

所以在**真实链路**上跑（唯一被替换的是网络出口：mock 网关）。替身只用来隔离单层，
不用来制造"价值"——在假核心上量出来的省与快都不作数（ENC-000 §5）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from conftest import call_tool

from artifact_spirit.observability import AuditView

# =========================================================================== #
# V1 · 省：逐级展开不是口号
# =========================================================================== #


def test_value_v1_progressive_disclosure(home, client):
    """V1：先给摘要，确认有用再要全文——**价签是字符数**。

    "支持 L0/L1/L2"本身不是价值，**便宜**才是：模型每轮都带着注入的文本，
    如果 L1 和 L2 一样长，逐级展开就只是多了一层调用而已。
    """
    provider = client(home)
    content = "用户偏好深色主题，因为夜里写代码刺眼。" * 20
    mem = call_tool(
        provider, "spirit_remember", {"content": content, "abstract": "偏好深色主题"}
    )["data"]["id"]

    l1 = call_tool(provider, "spirit_expand", {"ref": mem, "level": "L1"})["text"]
    l2 = call_tool(provider, "spirit_expand", {"ref": mem, "level": "L2"})["text"]

    assert len(l2) > len(content) * 0.9, "L2 应当是全文"
    assert len(l1) * 2 < len(l2), (
        f"L1({len(l1)} 字符) 相对 L2({len(l2)} 字符) 不够省——逐级展开失去意义"
    )
    provider.shutdown()


# =========================================================================== #
# V2 · 找得回来：真实链路（对话 → 提取 → 落库 → 召回）
# =========================================================================== #


def test_value_v2_review_and_trace(home, client):
    """V2：记下来的东西**看得见、追得到**。

    `review` 让人在出事**之前**就能审阅记忆（而不是等它变味才发现），
    `trace` 让一条结论能追溯到来历——这是"你凭什么这么记得"的答案。
    """
    provider = client(home, start_threads=True)
    call_tool(provider, "spirit_remember", {"content": "用户偏好深色主题"})

    rows = provider.services.core.review(limit=10)
    assert rows, "写入的记忆在 review 里看不见"
    assert any("深色主题" in json.dumps(row, ensure_ascii=False, default=str) for row in rows), (
        "review 里看不到刚写的内容"
    )

    mem = rows[0]["id"] if isinstance(rows[0], dict) else rows[0].id
    events = provider.services.core.trace(mem)
    assert events, "追溯不到这条记忆的来历"
    provider.shutdown()


def test_end_to_end_turn_pipeline(home, client):
    """端到端：宿主调用 → 提取落库 → 下一轮注入（全链路都不崩、都不静默）。

    这一条把"这一轮对话"当输入、把"下一轮拿到的文本"当输出：中间任何一环断了
    （没提取 / 没落库 / 召回为空 / 注入抛异常）它都会红。

    输入带**显式指令信号**（"请记住："）——不是为了让用例好过，而是 AL2 的显著性
    门槛本来就是这样划的：`θ_salience = 0.35`，而"新颖"这一项最高只能贡献 0.30，
    所以**没有任何信号因子的普通陈述必然过不了线**（DES-REV-008 P0-14）。
    那条边界单独钉在 `test_plain_statement_stays_out_of_long_term_memory` 里。
    """
    provider = client(home, start_threads=True)
    provider.sync_turn("请记住：我更喜欢深色主题，夜里写代码刺眼", "记住了", session_id="s1")
    assert provider.services.flush(10.0), "写入管线没有在超时内消化这一轮"

    records = provider.services.backend.query(layer="semantic", status=None)
    assert records, "这一轮没有被消化成任何记忆"
    assert any("深色主题" in (record.content or "") for record in records)

    injected = provider.prefetch("用户喜欢什么主题", session_id="s2")
    assert injected and injected.strip(), "下一轮必须拿到注入内容"
    provider.on_session_end([], session_id="s1")
    provider.shutdown()


def test_plain_statement_stays_out_of_long_term_memory(home, client):
    """边界：**无任何信号因子的普通陈述进不了长期记忆**（DES-REV-008 P0-14）。

    这条用例记录的是当前默认值下的**真实边界**，不是"设计期望"——它的存在是为了
    让边界可见、可裁决，而不是被一句话带过。算术（AL2 LLD §5 M1 因子表）：

    - 因子与权重：novelty 0.30 / instruction 0.25 / entity 0.20 / emotion 0.15
      / core_deviation 0.10；门槛 `θ_salience = 0.35`；
    - 空库上"新颖"必然打满 ⇒ 单靠 novelty 最高 **0.30 < 0.35**；
    - 实测：`user prefers dark theme when coding at night` → `score = 0.300`，
      `passes_threshold = False` ⇒ **只进工作记忆**（AL2 的 M1 正是这么规定的，
      但 M1-a 只修了"降级态门槛不缩放"，没看出正常态下 novelty 单独也过不了线）。

    **裁决（P0-14 → 甲：承认这是设计边界，不动默认阈值/权重）**，依据是
    "没进长期记忆" ≠ "丢了"：内容照样进工作记忆，会话收尾时被固化成**情景记忆**。
    所以这里把那条逃生通道也一并钉住——否则"边界"就成了"静默丢数据"的体面说法。
    反例同样成立：带实体（`我住在杭州，平时用 iPhone。`）时 entity 因子把分数
    抬到 0.44，照常落语义记忆——门槛挡的是"什么信号都没有"的句子。
    """
    provider = client(home, start_threads=True)
    text = "user prefers dark theme when coding at night"
    provider.sync_turn(text, "noted", session_id="s1")
    assert provider.services.flush(10.0)

    assert provider.services.backend.query(layer="semantic", status=None) == [], (
        "门槛把普通陈述挡住了——如果这条红了，说明阈值/权重被调过，请同步 P0-14 的裁决"
    )
    assert provider.services.applier.failures == [], (
        "不是写失败，是被显著性门槛过滤：失败列表必须是空的"
    )

    # ① 被挡下的内容必须留在工作记忆——"不进长期记忆"不等于"当场丢掉"
    chunks = provider.services.backend.wm_list("s1")
    assert chunks, "被门槛挡下的内容没有进工作记忆——那才是真的丢数据"
    assert text in " ".join(chunk.content for chunk in chunks)

    # ② 会话收尾时它会被固化成情景记忆（不是"永远想不起来"）
    report = provider.services.core.consolidate(session_id="s1")
    episodes = [
        intent for intent in report.intents
        if intent.record is not None and intent.record.layer == "episodic"
    ]
    assert episodes, "工作记忆里的普通陈述必须在收尾时固化成情景记忆"
    assert text in episodes[0].record.content

    provider.shutdown()


def test_value_v2_recall_finds_what_was_remembered(home, client):
    """V2 的最低要求：**记下来的东西要能找回来**。

    找不回来的记忆等于没记——而且是以"看起来记了"的方式骗人：库里有、召回没有。
    """
    provider = client(home)
    call_tool(provider, "spirit_remember", {"content": "用户偏好深色主题"})

    result = call_tool(provider, "spirit_recall", {"query": "深色主题"})
    assert result["ok"] is True
    assert result["data"]["items"], "记下来的东西找不回来"
    provider.shutdown()


# =========================================================================== #
# V3 · 记忆属于使用者：能带走、能删能救
# =========================================================================== #


def test_value_v3_export_is_portable(home, client):
    """V3：档案**离开器灵也能读懂**，且能重建——记忆是资产，不属于工具。

    这是 V3 的权威定义（方案设计 §1.4 / LLD §8.5）在真实链路上的判决：
    ① 产物是**纯文本**、人眼可读，不是只有本系统能解析的序列化格式；
    ② 把它导进一个**全新的库**，内容不丢（"换个工具就记忆归零"正是要消灭的痛点）。
    """
    provider = client(home)
    content = "用户偏好深色主题，夜里写代码刺眼"
    call_tool(provider, "spirit_remember", {"content": content})
    assert provider.services.flush(10.0), "写入没有在超时内完成"

    exported = call_tool(provider, "spirit_export", {})
    assert exported["ok"] is True, exported
    path = Path(exported["data"]["path"])
    assert path.exists(), "导出没有落盘"

    # ① 纯文本可读：能按 UTF-8 解码，且原文逐字可寻
    text = path.read_text(encoding="utf-8")
    assert content in text, "档案里读不到原文——那不是能带走的资产"
    assert "\x00" not in text, "档案里出现 NUL 字节——它不是纯文本"

    # ② 往返不丢：导进一个**全新的库**，内容仍在
    from artifact_spirit.store import SQLiteBackend

    fresh = SQLiteBackend(str(Path(home) / "restored.db"), embedding_dim=8)
    fresh.open()
    try:
        fresh.import_archive(str(path))
        restored = fresh.query(layer="semantic", status=None)
    finally:
        fresh.close()

    assert any(content in (record.content or "") for record in restored), (
        "往返之后内容丢了——这个导出格式承载不了核心信息"
    )
    provider.shutdown()


def test_value_v3_roundtrip_is_verbatim_and_idempotent(home, client):
    """T-AL1-18：往返后核心信息**逐字相等**，且**二次导入零新增**（INV-14）。

    只断言"条数相同"是不够的——条数相同而字段错了（少了 `superseded_by`、
    `valid_from` 差一个时区）同样会过。所以这里**逐字段**比：

    `MemoryRecord` 是 dataclass，`==` 就是逐字段相等——**用类型系统替我们比**，
    比手写一串 assert 更不容易漏。
    """
    from conftest import write_intents

    from artifact_spirit.core import ArtifactSpiritCore
    from artifact_spirit.store import SQLiteBackend
    from artifact_spirit.store.archive import load_archive

    provider = client(home)
    content = "用户偏好深色主题，夜里写代码刺眼"
    call_tool(provider, "spirit_remember", {"content": content, "abstract": "偏深色主题"})
    assert provider.services.flush(10.0)

    exported = call_tool(provider, "spirit_export", {})
    path = Path(exported["data"]["path"])

    fresh = SQLiteBackend(str(Path(home) / "v3_roundtrip.db"), embedding_dim=8)
    fresh.open()
    try:
        core = ArtifactSpiritCore(backend=fresh)
        pack = load_archive(str(path))
        report = core.restore_pack(pack)
        write_intents(fresh, report.intents)

        assert report.imported == len(pack["memories"]), (
            f"应当整批落库：档案 {len(pack['memories'])} 条，实际导入 {report.imported} 条"
        )
        before = {r.id: r for r in fresh.query(status=None)}
        assert any(content in (r.content or "") for r in before.values()), "往返后内容丢了"

        # 二次导入：**零新增、零改写**
        again = core.restore_pack(load_archive(str(path)))
        assert again.imported == 0, "重复导入产生了新记录——幂等破了"
        assert again.skipped == report.imported
        after = {r.id: r for r in fresh.query(status=None)}
        assert set(after) == set(before), "二次导入改变了记录集合"
        for mem_id, record in before.items():
            assert after[mem_id] == record, f"二次导入改写了 {mem_id}（逐字段应当完全相等）"
    finally:
        fresh.close()
    provider.shutdown()


def test_value_v3_transfer_has_no_model_dependency():
    """传承重建**结构上不持有提取器与模型**——这就是"零 LLM 调用"的根因。

    用 monkeypatch 去炸提取器是运行期反证，但它有个前提：**得改得到那个属性**
    （`Extractor` 是 slots dataclass，改不了——这条用例第一版正是这么红的）。
    直接断言依赖表更硬：`Transferrer` 的字段里就没有 `llm` / `extractor`，
    于是"调用模型"不是"没发生"，而是**没有路径能发生**。

    这也是 `import` 与 `ingest` 语义不重叠的机械证据：
    语义之争说不清，依赖表说得清。
    """
    import dataclasses

    from artifact_spirit.core.transfer import Transferrer

    fields = {f.name for f in dataclasses.fields(Transferrer)}
    forbidden = sorted(fields & {"llm", "extractor", "model", "resolver", "embedding"})
    assert not forbidden, f"传承重建不该持有模型相关依赖，实际有：{forbidden}"


def test_value_v3_import_reports_what_it_did(home, client):
    """`spirit_import` 的结果**可读且分开计数**：新增 / 跳过 / 错误。

    把"跳过"与"错误"合并成一个数，会让真正的数据问题被"幂等生效"这个好消息掩盖——
    而用户此时最想知道的恰恰是"我的东西到底进来没有、没进来的那些为什么"。
    """
    provider = client(home)
    call_tool(provider, "spirit_remember", {"content": "用户偏好深色主题"})
    assert provider.services.flush(10.0)
    path = Path(call_tool(provider, "spirit_export", {})["data"]["path"])

    first = call_tool(provider, "spirit_import", {"path": str(path)})
    assert first["ok"] is True, first
    # 导回**同一个库**：内容都在 → 全部跳过。这正是"幂等"的定义域，
    # 也是用户最常见的用法（"我导错了，再导一次"）。
    assert first["data"]["imported"] == 0, "导回原库不该新增记录"
    assert first["data"]["skipped"] >= 1
    assert "跳过" in first["text"], f"结果摘要要能读懂：{first['text']}"

    second = call_tool(provider, "spirit_import", {"path": str(path)})
    assert second["data"]["imported"] == 0, "同一档案导两遍不该产生新记录"
    provider.shutdown()


def test_value_v3_archive_header_and_no_private_markers(home, client):
    """档案头部含 `archive_version`；**不含向量、不含删除快照**（T-AL3-26）。

    "不含删除快照"这条有个容易被忽略的理由：`delete_snapshots` 是**合规材料**
    （为 GDPR 式的"请删除我的数据"而留），不是记忆资产。把它写进要交给用户的档案里，
    等于把"用户要求删除的内容"又**还给了用户**——那是事故，不是特性。
    """
    provider = client(home)
    to_be_deleted = "这条会被用户显式删除"
    call_tool(provider, "spirit_remember", {"content": to_be_deleted})
    call_tool(provider, "spirit_remember", {"content": "这条会留下"})
    assert provider.services.flush(10.0)

    path = Path(call_tool(provider, "spirit_export", {})["data"]["path"])
    text = path.read_text(encoding="utf-8")

    assert "archive_version:" in text, "档案头部必须自描述格式版本（T-AL3-26）"
    assert "schema_version:" in text, "schema 版本也要在"
    assert "\x00" not in text, "出现了 NUL 字节——它不是纯文本"
    # 向量是派生数据（可由 reindex 重建），且 2560 维浮点既大又不可读
    assert "vector" not in text.lower(), "档案里不该有向量"
    assert "delete_snapshots" not in text, "删除快照是合规材料，不是记忆资产"

    provider.shutdown()


def test_value_v3_forget_removes_and_keeps_it_restorable(home, client):
    """V3：删除**真的生效**，且留下可救援的快照。

    只做到"能删"是不够的：一句误解被永久抹掉，用户会立刻退回"不敢让它记"。
    删除账本必须与救援面板同时存在（`aspirit audit --forgetting`）。
    """
    provider = client(home)
    mem = call_tool(provider, "spirit_remember", {"content": "用户偏好深色主题"})["data"]["id"]
    # `count_by_layer()` 是 `GROUP BY layer` 的**稀疏**结果（AL3 契约：没行就没有键）。
    # 修订前这里直接 `["semantic"]["active"]`：删掉最后一条语义记忆后键会**整个消失**，
    # 于是断言变成 KeyError——用例红，但它红在"查法错了"，而不是"功能坏了"。
    # 这类错误更坏的一面是它会让人误以为"删除功能不可靠"（DES-REV-008 P1-48）。
    before = provider.services.backend.count_by_layer().get("semantic", {}).get("active", 0)
    assert before >= 1, "刚写下的语义记忆必须能被数出来"

    call_tool(provider, "spirit_forget", {"mem_id": mem, "reason": "误删演练", "confirm": True})

    assert provider.services.backend.get(mem) is None
    assert provider.services.backend.count_by_layer().get("semantic", {}).get("active", 0) == before - 1

    restorable = AuditView(provider.services.backend).restorable()
    assert restorable, "删除没有留下可救援的快照"
    assert mem in json.dumps(restorable, ensure_ascii=False, default=str), (
        "快照里找不到刚删的那条——救援面板救不回来"
    )
    provider.shutdown()


# =========================================================================== #
# INV-4 · 不拖慢对话：宿主热路径
# =========================================================================== #
#
# "有记忆"和"没记忆"的对话体验差别，用户感觉最直接的地方就是**延迟**。
# 这组用例是 INV-4 的**严格版**：不是"没有明显卡顿"，而是"毫秒级 + 零存储访问"。
# 修订前它们分别写在两个弱断言里（`< 200ms`、`调用了就通过`），
# 把参数忽略掉也能过——被严格版取代后删除（P1-5）。


def test_invariant_inv4_sync_turn_p99(home, client):
    """INV-4：`sync_turn` 的 p99 必须在毫秒级——宿主**每轮**都要调它。"""
    provider = client(home)
    samples = []
    for index in range(200):
        started = time.perf_counter()
        provider.sync_turn(f"第 {index} 轮输入", "回复", session_id="s1")
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    p99 = samples[int(len(samples) * 0.99) - 1]

    provider.shutdown()
    assert p99 < 5, f"p99 = {p99:.2f}ms —— 热路径应当只是把事件投进队列"


def test_inv4_hot_path_never_touches_storage(home, client, monkeypatch):
    """INV-4：热路径**一次存储访问都不许有**（比"看起来很快"更硬的判据）。

    把后端整个换成"碰一下就炸"的替身：任何一次落库/查询都会立刻暴露。
    这比掐时间可靠——快机器上慢代码也能"通过"，而这里没有含糊空间。

    修订前这条**根本没跑起来**：`on_turn_start("用户输入", session_id="s1")` 用的是
    旧签名（宿主契约是 `(turn_number, message, **kwargs)`），第一行就 TypeError，
    于是它后面那三个真缺陷一次都没被看到（DES-REV-008 P0-13）。

    修复后（P0-13）：会话登记 / 收尾 / 清工作记忆都改成**写意图**，宿主线程只入队。
    所以这里没有 `xfail`——它是一条真断言。
    """
    provider = client(home)
    touched: list[str] = []

    class _ExplodingBackend:
        def __getattr__(self, name: str):
            def boom(*args, **kwargs):
                touched.append(name)
                raise AssertionError(f"热路径不得访问存储：{name}")

            return boom

    monkeypatch.setattr(provider.services, "backend", _ExplodingBackend())
    provider.on_turn_start(1, "用户输入", session_id="s1")
    provider.sync_turn("用户输入", "助手回复", session_id="s1")
    provider.on_memory_write("add", "USER.md", "用户叫老张")
    provider.on_session_end([], session_id="s1")
    monkeypatch.undo()

    provider.shutdown()
    assert touched == [], f"热路径访问了存储：{touched}"

