"""批处理与报告合并的契约。

全量评测上千题，**一条命令跑不完**（会超时），所以必须分批；而有了分批就必须有合并——
否则拿到的只是十份"某一组的表现"，**没有一份是"系统的表现"**。

这个文件钉住四件事：

1. **复用判据**：`库里有东西` ≠ `灌完了`（半库会被伪装成能力不足）；
2. **配置隔离**：`rule` 库与 `llm` 库同名不同物，必须落在不同目录；
3. **切片安全**：重叠时按题去重（重叠让分母虚高，而分数上看不出来）；
4. **合并的拒绝线**：口径不一致时**拒绝**，而不是挑一个当基准接着算。
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench.run import _bench_home, _ingest_state, _merge, _write_marker  # noqa: E402

_BASE_META = {
    "dataset": "locomo",
    "answer_mode": "llm",
    "ingest": "llm",
    "embedding": "on",
    "top_k": 10,
}


def _row(qid: str, *, gold: str = "Paris", pred: str | None = "Paris") -> dict:
    return {
        "id": qid,
        "category": "single_hop",
        "question": f"Q {qid}",
        "answer": gold,
        "prediction": pred,
        "retrieved_ids": ["m1"],
        "gold_ids": ["m1"],
        "hit": True,
    }


def _write_batch(tmp_path: Path, name: str, rows: list[dict], **meta) -> Path:
    payload_meta = {**_BASE_META, **meta}
    path = tmp_path / name
    path.write_text(
        json.dumps({"meta": payload_meta, "report": {}, "details": rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _args(**over) -> Namespace:
    base = {
        "report_from": [],
        "out": None,
        "baseline": None,
        "tolerance": 2.0,
        "min_f1": None,
        # 下面几个刻意给**与明细元数据不同**的默认值：
        # `_merge` 必须从元数据恢复口径，而不是沿用命令行默认值。
        "answer_mode": "extract",
        "ingest": "rule",
        "embedding": "off",
        "top_k": 10,
        "verbose": False,
        "dataset": "locomo",
    }
    base.update(over)
    return Namespace(**base)


# --------------------------------------------------------------------------- #
# 复用判据
# --------------------------------------------------------------------------- #


def test_ingest_marker_is_the_only_proof_a_library_is_complete(tmp_path):
    """**「库里有东西」不等于「灌完了」。**

    灌入跑到一半崩掉（网关抖动、进程被杀）会留下一个**半库**，
    而它和"灌好了"在检测上完全一样。用半库评测的后果不是"少几条记忆"，
    是**分数系统性偏低**——报告上却只读得到"检索不行"。
    """
    home = tmp_path / "g"
    home.mkdir()

    assert not _ingest_state(home, turns=19)[0], "没有标记就不能复用"

    _write_marker(home, turns=19)
    ok, why = _ingest_state(home, turns=19)
    assert ok, why

    # 数据集换了（段数变了）同样不复用：标记只证明"当时灌完了"，
    # 不证明"灌的就是这批素材"。
    assert not _ingest_state(home, turns=20)[0]


def test_ingest_state_survives_a_corrupt_marker(tmp_path):
    """标记坏掉时**判为不可复用**——坏掉的凭据不是凭据。"""
    home = tmp_path / "g"
    home.mkdir()
    (home / "_ingest_done.json").write_text("{ 不是 JSON", encoding="utf-8")
    assert not _ingest_state(home, turns=19)[0]


def test_bench_home_separates_configurations(tmp_path):
    """`rule` 库与 `llm` 库**同名不同物**，必须落在不同目录里。

    否则先用 `rule` 建库、之后换 `llm` 重跑时，复用检测会认领那个 rule 库——
    分数标着 `llm`，量的却是 `rule`，而报告上**没有任何信号**。
    """
    rule_kw = _bench_home(tmp_path, "conv-26", with_llm=False, with_embedding=False)
    llm_vec = _bench_home(tmp_path, "conv-26", with_llm=True, with_embedding=True)
    llm_kw = _bench_home(tmp_path, "conv-26", with_llm=True, with_embedding=False)

    assert len({rule_kw, llm_vec, llm_kw}) == 3, "同组不同配置必须分开"
    assert rule_kw.parent == llm_vec.parent, "组名仍然是上一级（便于按组清理）"


# --------------------------------------------------------------------------- #
# 合并
# --------------------------------------------------------------------------- #


def test_merge_sums_batches_like_a_single_run(tmp_path):
    """**分两批累加 == 一次跑完累加**——合并这一层不引入新口径。

    可行性来自 `Metrics.add` 是增量累加的（分子分母各自相加）。
    若它是"先算比例再平均"，分批平均与整体平均就会不等，合并也就不可信了。
    """
    first = _write_batch(tmp_path, "b1.json", [_row("q1"), _row("q2", pred=None)])
    second = _write_batch(tmp_path, "b2.json", [_row("q3")])
    out = tmp_path / "merged.json"

    assert _merge(_args(report_from=[str(first), str(second)], out=str(out))) == 0

    report = json.loads(out.read_text(encoding="utf-8"))["report"]
    assert report["total"] == 3
    assert report["answered"] == 2
    # **没召回到的题必须留在分母里**——否则召回越差、分母越小、指标越好看。
    assert report["answer_rate"] == round(2 / 3, 4), report["answer_rate"]


def test_merge_refuses_mixed_calibers(tmp_path):
    """口径不一致时**拒绝合并**：合并出来的分数**没有定义**。

    批 A 用 `top_k=10`、批 B 用 `top_k=20`，合并出的 `R@10` 既不量 A 也不量 B。
    这种数字比没有数字更坏——**它看起来像个结果**。
    """
    first = _write_batch(tmp_path, "a.json", [_row("q1")])
    second = _write_batch(tmp_path, "b.json", [_row("q2")], top_k=20)

    assert _merge(_args(report_from=[str(first), str(second)], out=None)) == 2


def test_merge_refuses_details_without_caliber_metadata(tmp_path):
    """没有元数据的旧明细要**拒绝**，而不是当作"匹配"放过去。

    静默接受等于把"不知道口径"当成"口径一致"——那正是拒绝线要拦的东西。
    """
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"report": {}, "details": []}), encoding="utf-8")

    assert _merge(_args(report_from=[str(path)], out=None)) == 2


def test_merge_takes_caliber_from_metadata_not_command_line(tmp_path):
    """合并时**从元数据恢复口径**，不用命令行默认值。

    报告的措辞依赖模式（`extract` 与 `llm` 的解读完全相反），
    沿用默认值会让一份 LLM 跑的合并报告印上"extract 模式的分数不可比"——把人吓一跳。
    """
    batch = _write_batch(tmp_path, "a.json", [_row("q1")])
    args = _args(report_from=[str(batch)], out=None)

    assert _merge(args) == 0
    assert args.answer_mode == "llm", "应当被元数据覆盖"
    assert args.top_k == 10


def test_merge_deduplicates_overlapping_batches(tmp_path):
    """批次切片重叠时**按题去重**。

    重叠不会报错，只会让分母虚高——而分数上看不出来。
    所以它必须被去重，并且**打印提示**（提示由 `_merge` 负责，这里只钉住计数）。
    """
    first = _write_batch(tmp_path, "a.json", [_row("q1"), _row("q2")])
    second = _write_batch(tmp_path, "b.json", [_row("q2"), _row("q3")])
    out = tmp_path / "m.json"

    assert _merge(_args(report_from=[str(first), str(second)], out=str(out))) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["report"]["total"] == 3


def test_merge_reports_a_missing_file_instead_of_crashing(tmp_path):
    """缺文件时给**退出码 2 与一句人话**，而不是抛栈。

    分批跑十批，任何一批漏跑都属常见——那时需要的是"缺哪份"，
    而不是一段 traceback（它会盖住真正的原因）。
    """
    assert _merge(_args(report_from=[str(tmp_path / "nope.json")], out=None)) == 2
