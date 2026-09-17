"""AL1 对外 CLI 验收（T-AL1-08 / C10）。

本文件由 `test_runtime_adapter.py` 拆出（设计评审 P1-6）。CLI 是**宿主之外**的第二个
入口，它对用户承担"脚本里稳定可控"的承诺：退出码、`--json`、命令集齐全。
"""

from __future__ import annotations

import argparse
import json
import sys

import pytest
from conftest import services_for

from artifact_spirit.cli import build_parser, main
from artifact_spirit.compliance import PACKAGE_ROOT
from artifact_spirit.config import config_path
from artifact_spirit.store.base import MemoryRecord

# LLD-AL1 §2.4 对外 CLI 的命令面。**逐字对照**：新增或删除子命令都必须回到文档，
# 不允许"代码里悄悄多一个、少一个"——脚本世界没有自动补全。
EXPECTED_SUBCOMMANDS = {
    "init",
    "status",
    "layers",
    "review",
    "reflect",
    "trace",
    "export",
    "import",
    "ingest",
    "consolidate",
    "decay",
    "forget",
    "restore",
    "correct",
    "audit",
    "doctor",
    "reindex",
    "replay",
    "optimize",
}

# `soul` / `awaken` **刻意不在表内**（T-AL1-16 的明文要求）：
# 它们在 v1.0 不实现，而任务书规定"不实现就必须从命令表移除并写明原因"——
# 留一条"能列出但一跑就报未实现"的空壳命令，比没有这条命令更坏：
# 用户会以为自己有一条可用的能力（DES-REV-003 P1-10 的教训）。
# 人格的**读**其实已经可用（`spirit_core` 升格 + `status` 展示核心记忆），
# 缺的是"独立命名实体"那一层，它排在 v1.0 之后。


def _subparsers():
    parser = build_parser()
    (sub,) = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    return sub


def test_cli_declares_every_documented_subcommand():
    """命令集**齐全性**（P1-5 补：此前只有"人工逐个跑过"）。"""
    sub = _subparsers()
    assert set(sub.choices) == EXPECTED_SUBCOMMANDS, (
        f"缺：{sorted(EXPECTED_SUBCOMMANDS - set(sub.choices))}；"
        f"多：{sorted(set(sub.choices) - EXPECTED_SUBCOMMANDS)}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_SUBCOMMANDS))
def test_every_subcommand_help_exits_zero(name, capsys):
    """T-AL1-08 验收 1 的**自动化**版本：每个子命令的 `--help` 都能正常退出。

    "18 个子命令逐个跑过"是一次性的证据；这条用例把同一件事变成每次都能重放的证据。
    """
    with pytest.raises(SystemExit) as excinfo:
        main([name, "--help"])
    assert excinfo.value.code == 0, f"`aspirit {name} --help` 退出码非 0"
    assert capsys.readouterr().out.strip(), "帮助文本不应为空"


def test_cli_status_json(home, capsys):
    assert main(["--home", home, "--json", "status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "identity" in payload and "layers" in payload


def test_cli_accepts_json_after_subcommand_too(home, capsys):
    """`aspirit status --json` 也要能用（T-AL1-08 验收就是这么写的）。

    修订前只支持 `aspirit --json status`，验收原文那条命令会直接
    "unrecognized arguments" 并以 2 退出——**测试当时写的是能过的那种顺序**，
    于是偏差被掩盖了。这里把两种位置都钉住。

    另外 `_SubParsersAction` 会把子解析器的默认值回写覆盖顶层值，
    所以子命令侧的 `--json` 必须是 `default=SUPPRESS`；写成 False 的话
    下面第一条断言（前置 flag 形式）会退回人类可读输出而失败。
    """
    for argv in (["--home", home, "--json", "status"], ["--home", home, "status", "--json"]):
        assert main(argv) == 0, f"{argv} 应正常退出"
        payload = json.loads(capsys.readouterr().out)
        assert "identity" in payload, f"{argv} 应输出 JSON"


def test_every_subcommand_declares_json_flag():
    """每个子命令都要自带 `--json`，否则后置写法会静默失效。

    `build_parser()` 结尾统一补 flag，这条用例防止将来新增子命令时漏掉
    （漏掉的表现是 `aspirit <新命令> --json` 报参数错误，而不是报错说"漏了 flag"）。
    """
    sub = _subparsers()
    missing = [
        name
        for name, sub_parser in sub.choices.items()
        if not any(action.dest == "json" for action in sub_parser._actions)
    ]
    assert not missing, f"这些子命令缺少 --json：{missing}"


def test_cli_goes_through_shared_assembly(home, capsys, monkeypatch):
    """C10：CLI 与宿主**共用同一套装配**，CLI 里不许自己拼后端/核心。

    之前的证据只有"`provider.py` 里没有 `SQLiteBackend(`"这一条文本断言——
    它管不到 CLI：cli.py 完全可以自己 new 一个后端，而测试不会红（P1-5）。
    """
    import artifact_spirit.runtime as runtime_module

    calls = {"n": 0}
    real_start = runtime_module.start

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real_start(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "start", spy)
    monkeypatch.setattr("artifact_spirit.cli.start", spy, raising=False)

    assert main(["--home", home, "status", "--json"]) == 0
    assert calls["n"] == 1, "CLI 必须经 AL5 的 start() 装配"

    text = (PACKAGE_ROOT / "cli.py").read_text(encoding="utf-8")
    assert "SQLiteBackend(" not in text, "CLI 不得自行构造存储"
    assert "ArtifactSpiritCore(" not in text, "CLI 不得自行构造核心层"
    capsys.readouterr()


def test_cli_json_is_not_hostage_to_human_renderer(home, capsys, monkeypatch):
    """P2-5：`--json` 必须**绕开人类渲染**。

    `_render_*` / `format_*` 会遍历整份报告，是命令里最容易抛异常的一段。
    JSON 恰恰是给脚本读的稳定契约——被人类渲染的异常拖崩，
    "脚本里稳定可控"就不成立了，而脚本比人更依赖 `--json`。
    """
    from artifact_spirit import cli

    def boom(*args, **kwargs):
        raise RuntimeError("人类渲染炸了")

    for renderer in ("status_text", "doctor_text", "format_review_text", "format_audit_text"):
        monkeypatch.setattr(cli, renderer, boom, raising=False)

    assert main(["--home", home, "status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["identity"], "JSON 模式不得被渲染异常绑架"


def test_print_out_survives_non_utf8_console():
    """控制台编码装不下 emoji 时，命令不能崩（Windows GBK 真机上复现过）。

    `status` / `doctor` 的输出里带 `✅` / `⚠️`，GBK 编不出来就会
    `UnicodeEncodeError` + traceback 退出——排障入口自己先倒在排障上。
    """
    from artifact_spirit.cli import _print_out

    class _GbkStdout:
        encoding = "gbk"

        def __init__(self):
            self.written: list[str] = []

        def write(self, text):
            text.encode("gbk")  # 装不下就抛，模拟真实控制台
            self.written.append(text)

        def flush(self):
            pass

    fake = _GbkStdout()
    real = sys.stdout
    sys.stdout = fake
    try:
        _print_out("体检：✅ 全部通过 ⚠️ 有警告")
    finally:
        sys.stdout = real
    assert fake.written, "应该有输出"
    assert "体检" in fake.written[0], "可编码的部分必须保留"


def test_cli_decay_defaults_to_dry_run(home, capsys):
    assert main(["--home", home, "decay"]) == 0
    out = capsys.readouterr().out
    assert "预演" in out or "未写回" in out


def test_cli_review_is_human_readable(home, capsys):
    main(["--home", home, "review"])
    out = capsys.readouterr().out
    assert "记忆审查" in out or "还没有任何记忆" in out


def test_cli_forget_requires_apply(home, capsys):
    services = services_for(home)

    record = MemoryRecord(id="", layer="semantic", type="fact", content="待删内容")
    services.backend.put(record, None)
    services.stop()

    assert main(["--home", home, "forget", record.id, "--reason", "测试"]) == 0
    assert "预演" in capsys.readouterr().out

    services = services_for(home)
    assert services.backend.get(record.id) is not None
    services.stop()


def test_cli_audit_and_doctor(home, capsys):
    assert main(["--home", home, "audit", "--limit", "5"]) == 0
    capsys.readouterr()
    main(["--home", home, "doctor"])
    assert "自检" in capsys.readouterr().out


def test_cli_init_creates_template(tmp_path, capsys):
    folder = tmp_path / "fresh"
    folder.mkdir()
    assert main(["--home", str(folder), "init"]) == 0
    assert config_path(folder).exists()
    assert "api_key_env" in config_path(folder).read_text(encoding="utf-8")


def test_cli_export_and_readable(home, tmp_path, capsys):
    services = services_for(home)
    services.backend.put(
        MemoryRecord(id="", layer="semantic", type="fact", content="要被导出的记忆"), None
    )
    services.stop()

    target = tmp_path / "archive.md"
    assert main(["--home", home, "export", str(target)]) == 0
    text = target.read_text(encoding="utf-8")
    assert "要被导出的记忆" in text
    assert "记忆档案" in text
