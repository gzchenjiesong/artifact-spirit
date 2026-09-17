"""可迁移档案与记忆包（T-AL3-15 / T-AL3-22 · INV-13 / INV-14）。

这里承载**核心价值 V3**：记忆是使用者的长期资产，不属于任何工具。
因此导出必须满足三个硬条件（INV-13）：

1. **开放** —— 不依赖器灵运行时，纯文本可解析
2. **人类可读** —— 文本编辑器打开就能读懂
3. **往返幂等** —— 导出→导入→再导出，结果等价，且**不产生重复记录**（INV-14）

**向量不入档案**：它是派生数据，可由 ``reindex`` 重建；且 2560 维浮点既大又不可读。
（这是对 LLD-AL3 "完整包含向量" 表述的一处收窄，依据是任务 T-AL3-22 的明文要求。）
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..common import now_iso
from .base import MemoryRecord
from .sqlite_backend import (
    META_EMBEDDING_DIM,
    META_SPIRIT_ID,
    META_SPIRIT_NAME,
    SQLiteBackend,
)
from .text import content_hash_of

__all__ = [
    "ARCHIVE_FORMAT",
    "ARCHIVE_HEADER_KEY",
    "ARCHIVE_VERSION",
    "PACK_FORMAT",
    "check_archive_version",
    "export_archive",
    "export_pack",
    "import_archive",
    "import_pack",
    "load_archive",
    "parse_markdown_archive",
    "render_archive",
]

PACK_FORMAT = "artifact-spirit-pack"
ARCHIVE_FORMAT = "artifact-spirit-archive"
ARCHIVE_HEADER_KEY = "artifact-spirit-archive"

ARCHIVE_VERSION = 1
"""**档案格式**的版本（T-AL3-26）。与 `schema_version` **分开**——两者独立演化：

- `schema_version`：**库结构**变了（表 / 列 / 迁移）；
- `archive_version`：**档案语法**变了（头部字段、记录块的行格式、分隔符）。

把它们合成一个数的后果很具体：**加一个头部字段，就得假装"库结构变了"**——
于是要么虚报 schema，要么不动版本号、让旧读者硬读新档案。
两者都不会当场出事，只会在"用户的档案放了三年之后"集中爆出来。
"""

_SUPPORTED_ARCHIVE_VERSIONS = (1,)
"""读得懂的档案格式版本。**只列读得懂的**——不写"尽力而为"的宽范围。"""

_HEADER_KEYS = (
    "format",
    "archive_version",
    "schema_version",
    "spirit_name",
    "spirit_id",
    "embedding_model",
    "embedding_dim",
    "data_as_of",
    "exported_at",
    "memory_count",
)

_REL_LINE = "- relations:"
_LIST_FIELDS = ("subject", "predicate", "object", "abstract", "scope")


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #


def _data_as_of(backend: SQLiteBackend) -> str:
    """档案反映的数据时间点。

    **刻意用"库内最新变更时间"而非墙钟时间**——这样导出是库状态的纯函数，
    连续两次导出才会逐字一致（T-AL3-15 验收）。需要真实导出时刻时由调用方传入。
    """
    row = backend.conn.execute("SELECT MAX(updated_at) AS m FROM memories").fetchone()
    latest = row["m"] if row else None
    return latest or backend.meta_get("created_at") or ""


def render_archive(
    backend: SQLiteBackend, *, fmt: str = "markdown", exported_at: str | None = None
) -> str:
    """把当前库状态渲染为档案文本。**纯函数**：同库状态、同参数 → 同输出。"""
    if fmt == "json":
        return json.dumps(_build_pack(backend, exported_at=exported_at), ensure_ascii=False, indent=2)

    # 按 **id** 排序而非 rowid：ULID 时间有序，且跨库稳定——
    # 这是"导出→导入→再导出结果等价"（INV-14 往返幂等）的前提。
    records = sorted(backend.query(status=None), key=lambda r: r.id)
    as_of = _data_as_of(backend)
    header = {
        "format": ARCHIVE_FORMAT,
        "archive_version": ARCHIVE_VERSION,
        "schema_version": backend.meta_get("schema_version") or "0",
        "spirit_name": backend.meta_get(META_SPIRIT_NAME) or "",
        "spirit_id": backend.meta_get(META_SPIRIT_ID) or "",
        "embedding_model": backend.meta_get("embedding_model") or "",
        "embedding_dim": backend.meta_get(META_EMBEDDING_DIM) or "",
        "data_as_of": as_of,
        "exported_at": exported_at or as_of,
        "memory_count": len(records),
    }

    lines: list[str] = ["---"]
    lines.extend(f"{key}: {header[key]}" for key in _HEADER_KEYS)
    lines.append("---")
    lines.append("")
    lines.append("# 器灵记忆档案")
    lines.append("")
    lines.append(
        "> 这是 Artifact Spirit 导出的记忆档案。**纯文本即可阅读**，"
        "不依赖任何程序；同名工具可据此重建记忆。"
    )
    lines.append("")

    relations_by_src: dict[str, list[dict]] = {}
    for rel in backend.all_relations():
        if rel["src_kind"] == "memory":
            relations_by_src.setdefault(rel["src_id"], []).append(rel)

    for rec in records:
        lines.extend(_render_record(rec, relations_by_src.get(rec.id, [])))

    entities = backend.entity_list()
    if entities:
        lines.append("## 实体索引")
        lines.append("")
        for ent in entities:
            alias = f"（别名：{'、'.join(ent.aliases)}）" if ent.aliases else ""
            lines.append(f"- `{ent.id}` **{ent.name}** · {ent.type}{alias}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_record(rec: MemoryRecord, relations: list[dict]) -> list[str]:
    scope = json.dumps(rec.scope, ensure_ascii=False) if rec.scope else ""
    lines = [
        f"## {rec.id}",
        "",
        f"- layer: {rec.layer}",
        f"- type: {rec.type}",
        f"- content: {_one_line(rec.content)}",
        f"- abstract: {_one_line(rec.abstract or '')}",
        f"- subject: {_one_line(rec.subject or '')}",
        f"- predicate: {_one_line(rec.predicate or '')}",
        f"- object: {_one_line(rec.object or '')}",
        f"- scope: {scope}",
        f"- confidence: {rec.confidence}",
        f"- salience: {rec.salience}",
        f"- status: {rec.status}",
        f"- valid_from: {rec.valid_from or ''}",
        f"- valid_to: {rec.valid_to or ''}",
        f"- superseded_by: {rec.superseded_by or ''}",
        f"- source_session: {rec.source_session or ''}",
        f"- created_at: {rec.created_at}",
        f"- updated_at: {rec.updated_at}",
        f"- content_hash: {rec.content_hash or ''}",
    ]
    if relations:
        lines.append(_REL_LINE)
        for rel in relations:
            lines.append(
                f"  - {rel['rel_type']} -> {rel['dst_id']} "
                f"(w={rel['weight']:.4f}, n={rel['co_count']})"
            )
    lines.append("")
    return lines


def _one_line(text: str) -> str:
    """字段值压成一行（避免破坏 markdown 列表结构）。"""
    return " ".join(str(text).split())


def export_archive(
    backend: SQLiteBackend, path: str, *, fmt: str = "markdown", exported_at: str | None = None
) -> None:
    """写出档案文件。父目录不存在时自动创建。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_archive(backend, fmt=fmt, exported_at=exported_at), encoding="utf-8"
    )


def _build_pack(backend: SQLiteBackend, *, exported_at: str | None = None) -> dict:
    as_of = _data_as_of(backend)
    return {
        "format": PACK_FORMAT,
        "pack_version": 1,
        "archive_version": ARCHIVE_VERSION,
        "schema_version": backend.meta_get("schema_version") or "0",
        "spirit_name": backend.meta_get(META_SPIRIT_NAME) or "",
        "spirit_id": backend.meta_get(META_SPIRIT_ID) or "",
        "embedding_model": backend.meta_get("embedding_model") or "",
        "embedding_dim": backend.meta_get(META_EMBEDDING_DIM) or "",
        "data_as_of": as_of,
        "exported_at": exported_at or as_of,
        "memories": [asdict(r) for r in sorted(backend.query(status=None), key=lambda r: r.id)],
        "relations": backend.all_relations(),
        "entities": [asdict(e) for e in backend.entity_list()],
    }


def export_pack(backend: SQLiteBackend) -> bytes:
    """可迁移资产包（JSON）。"""
    return json.dumps(_build_pack(backend), ensure_ascii=False, indent=2).encode("utf-8")


# --------------------------------------------------------------------------- #
# 导入
# --------------------------------------------------------------------------- #


def import_pack(
    backend: SQLiteBackend, data: bytes, *, merge: bool = False
) -> dict:
    """导入记忆包。**按 ``content_hash`` 判重**，因此重复导入不产生重复记录（INV-14）。

    Args:
        merge: 保留原 ID；仅在原 ID 已被占用时才重新分配。设为 ``False`` 亦保留原 ID
            ——"沿用原 ID"是 D-22/INV-14 的要求，两者在此一致。
    """
    payload: dict[str, Any] = json.loads(data.decode("utf-8"))
    if payload.get("format") != PACK_FORMAT:
        raise ValueError(
            f"不是器灵记忆包（format={payload.get('format')!r}，期望 {PACK_FORMAT!r}）"
        )
    check_archive_version(payload.get("archive_version"))

    stats = {"imported": 0, "skipped": 0, "relations": 0, "entities": 0}
    id_map: dict[str, str] = {}

    # 头部元信息：**仅在目标库尚未记录时**写入——不得覆盖既有身份或模型（C6 / INV-2）
    if payload.get("spirit_name") and not backend.meta_get(META_SPIRIT_NAME):
        backend.meta_set(META_SPIRIT_NAME, payload["spirit_name"])
    if payload.get("spirit_id") and not backend.meta_get(META_SPIRIT_ID):
        backend.meta_set(META_SPIRIT_ID, payload["spirit_id"])
    if payload.get("embedding_model") and not backend.meta_get("embedding_model"):
        backend.meta_set("embedding_model", payload["embedding_model"])

    # `superseded_by` 的外键指向 `memories` **自己**（`REFERENCES memories(id)`），
    # 而包里的顺序是任意的：被取代的**旧**记录完全可能排在取代它的**新**记录**前面**。
    # 逐条 `put` 各自开事务，`PRAGMA defer_foreign_keys` 帮不上忙（它只在事务内生效），
    # 所以这里走**两阶段**：先不带 `superseded_by` 落库，全部就位后再回填——
    # 顺序无关，且回填时外键一定满足（这是 T-AL3-24 实测撞出来的一条约束）。
    deferred_superseded: list[tuple[str, str]] = []

    for raw in payload.get("memories", []):
        record = MemoryRecord(**{k: v for k, v in raw.items() if k in _RECORD_FIELDS})
        record.content_hash = record.content_hash or content_hash_of(
            content=record.content,
            subject=record.subject,
            predicate=record.predicate,
            object_=record.object,
            scope=record.scope,
        )
        pending_target = record.superseded_by
        record.superseded_by = None  # 第二阶段回填，见上方说明

        hits = backend.find_by_hash(record.content_hash)
        if hits:
            stats["skipped"] += 1
            id_map[record.id] = hits[0].id
            if pending_target:
                deferred_superseded.append((hits[0].id, pending_target))
            continue

        original_id = record.id
        if backend.get(original_id) is not None:
            record.id = ""
        backend.put(record)
        id_map[original_id] = record.id
        if pending_target:
            deferred_superseded.append((record.id, pending_target))
        stats["imported"] += 1

    # 第二阶段：补上"被谁取代"（原 ID 可能被重分配，故经 `id_map` 翻译）
    for mem_id, target_id in deferred_superseded:
        backend.update(mem_id, {"superseded_by": id_map.get(target_id, target_id)})

    for ent in payload.get("entities", []):
        backend.entity_upsert(
            ent["name"],
            ent.get("type") or "",
            aliases=ent.get("aliases"),
            entity_id=ent.get("id"),
        )
        stats["entities"] += 1

    for rel in payload.get("relations", []):
        src_id = id_map.get(rel["src_id"], rel["src_id"])
        dst_id = id_map.get(rel["dst_id"], rel["dst_id"])
        if backend.get(src_id) is None or backend.get(dst_id) is None:
            continue
        backend.link(
            rel["src_kind"], src_id, rel["dst_kind"], dst_id, rel["rel_type"], rel["weight"]
        )
        stats["relations"] += 1

    return stats


def check_archive_version(raw: object) -> None:
    """校验档案格式版本。**读不懂就明说**，不用旧读者硬读新档案。

    这条检查很便宜，但它挡住的失败特别贵：旧版器灵读新版档案时，
    它不认识新增字段，于是**静默丢掉**——而用户以为导入成功了。
    「没报错」与「没丢东西」是两件事。
    """
    if raw in (None, ""):
        # 最早期的档案没有这个字段。那时只有一个版本，读得懂——
        # 所以"缺失"不等于"未知"，不能一律拒绝（否则老档案全部读不进来）。
        return
    try:
        version = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"档案的 archive_version 不是整数：{raw!r}") from exc
    if version not in _SUPPORTED_ARCHIVE_VERSIONS:
        raise ValueError(
            f"读不懂这个档案：archive_version={version}，"
            f"本版本支持 {_SUPPORTED_ARCHIVE_VERSIONS}。请用更新版本的器灵读取——"
            "**不要用旧版本硬读**，它会静默丢掉不认识的字段"
        )


def load_archive(path: str) -> dict:
    """读档案文件并解析为 pack **字典**（**不落库**）。

    与 :func:`import_archive` 分开，是因为"解析"与"落库"要被两条路径共用：

    - **在线导入**（T-AL2-23）走写意图 → 单写者；
    - **批量导入**走本模块自己的直插接口。

    把解析单独拎出来，两条路径用的是**同一个解析器**——否则
    "工具面读得懂、CLI 读不懂"这类分歧迟早会出现，而它只在跨版本时暴露。
    """
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() in (".json", ".pack"):
        pack: dict[str, Any] = json.loads(text)
        if pack.get("format") != PACK_FORMAT:
            raise ValueError(
                f"不是器灵记忆包（format={pack.get('format')!r}，期望 {PACK_FORMAT!r}）"
            )
        check_archive_version(pack.get("archive_version"))
        return pack
    return parse_markdown_archive(text)


def import_archive(backend: SQLiteBackend, path: str) -> dict:
    """从档案文件导入。按扩展名分派：``.json`` → 记忆包；其余 → markdown 档案。"""
    pack = load_archive(path)
    return import_pack(backend, json.dumps(pack, ensure_ascii=False).encode("utf-8"))


def parse_markdown_archive(text: str) -> dict:
    """解析 :func:`render_archive` 产出的 markdown 档案。

    格式故意保持"傻瓜级"——每行 ``- key: value``，便于人手改也便于程序读。
    """
    if ARCHIVE_HEADER_KEY not in text:
        raise ValueError("不是器灵档案：缺少自描述头部")

    header: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        _, head, body = text.split("---", 2)
        for line in head.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()

    memories: list[dict] = []
    entities: list[dict] = []
    current: dict | None = None
    section = "memories"

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            title = line[3:].strip()
            if title == "实体索引":
                if current is not None:
                    memories.append(current)
                    current = None
                section = "entities"
                continue
            if current is not None:
                memories.append(current)
            section = "memories"
            current = {"id": title, "relations": []}
            continue

        if section == "entities":
            parsed = _parse_entity_line(line)
            if parsed is not None:
                entities.append(parsed)
            continue

        if current is None:
            continue
        if line.strip() == _REL_LINE:
            continue
        if line.startswith("  - "):
            rel = _parse_relation(line[4:].strip())
            if rel is not None:
                current["relations"].append(rel)
            continue
        if line.startswith("- "):
            key, _, value = line[2:].partition(":")
            key = key.strip()
            value = value.strip()
            if key in _HEADER_KEYS:
                continue
            current[key] = value

    if current is not None:
        memories.append(current)

    records = [_archive_dict_to_record(m) for m in memories]
    return {
        "format": PACK_FORMAT,
        "pack_version": 1,
        "archive_version": header.get("archive_version", ""),
        "schema_version": header.get("schema_version", "0"),
        "spirit_name": header.get("spirit_name", ""),
        "spirit_id": header.get("spirit_id", ""),
        "embedding_model": header.get("embedding_model", ""),
        "embedding_dim": header.get("embedding_dim", ""),
        "data_as_of": header.get("data_as_of", ""),
        "exported_at": header.get("exported_at", ""),
        "memories": records,
        "relations": [rel for mem in memories for rel in _relations_for(mem)],
        "entities": entities,
    }


def _parse_entity_line(line: str) -> dict | None:
    """``- `ent_01ABC` **RAGFlow** · project（别名：ragflow, RAG）``"""
    stripped = line.strip()
    if not stripped.startswith("- `"):
        return None
    try:
        entity_id, rest = stripped[3:].split("`", 1)
    except ValueError:  # pragma: no cover
        return None
    rest = rest.strip()
    if not rest.startswith("**") or "**" not in rest[2:]:
        return None
    name, rest = rest[2:].split("**", 1)
    rest = rest.lstrip(" ·").strip()
    aliases: list[str] = []
    if "（别名：" in rest:
        type_part, alias_part = rest.split("（别名：", 1)
        aliases = [a.strip() for a in alias_part.rstrip("）").split("、") if a.strip()]
    else:
        type_part = rest
    return {"id": entity_id, "name": name.strip(), "type": type_part.strip(), "aliases": aliases}


def _parse_relation(text: str) -> dict | None:
    """``co_activation -> sem_01X (w=0.4000, n=3)``"""
    if "->" not in text:
        return None
    rel_type, rest = text.split("->", 1)
    dst_id = rest.split("(")[0].strip()
    weight, co_count = 1.0, 1
    if "w=" in rest:
        weight = float(rest.split("w=")[1].split(",")[0].strip(" )"))
    if "n=" in rest:
        co_count = int(rest.split("n=")[1].strip(" )"))
    return {
        "src_kind": "memory",
        "src_id": "",
        "dst_kind": "memory",
        "dst_id": dst_id,
        "rel_type": rel_type.strip(),
        "weight": weight,
        "co_count": co_count,
    }


def _relations_for(mem: dict) -> list[dict]:
    out = []
    for rel in mem.get("relations", []):
        rel = dict(rel)
        rel["src_id"] = mem["id"]
        out.append(rel)
    return out


def _archive_dict_to_record(data: dict) -> dict:
    def _f(key: str) -> float:
        try:
            return float(data.get(key) or 0.0)
        except ValueError:  # pragma: no cover
            return 0.0

    scope = data.get("scope") or ""
    parsed_scope = None
    if scope:
        try:
            parsed_scope = json.loads(scope)
        except json.JSONDecodeError:
            parsed_scope = None

    return {
        "id": data.get("id", ""),
        "layer": data.get("layer", "semantic") or "semantic",
        "type": data.get("type", "fact") or "fact",
        "content": data.get("content", ""),
        "abstract": _text_or_none(data.get("abstract")),
        "subject": _text_or_none(data.get("subject")),
        "predicate": _text_or_none(data.get("predicate")),
        "object": _text_or_none(data.get("object")),
        "scope": parsed_scope,
        "confidence": _f("confidence") or 0.7,
        "salience": _f("salience"),
        "status": data.get("status") or "active",
        "valid_from": _text_or_none(data.get("valid_from")),
        "valid_to": _text_or_none(data.get("valid_to")),
        # `superseded_by` 必须一起往返（T-AL3-24）：导出侧走 `asdict()`，**一直是带着它的**，
        # 而这里漏了——于是"被谁取代"这条信息在**导入后静默消失**，
        # 而 V3 承诺的正是"不丢核心信息"。
        "superseded_by": _text_or_none(data.get("superseded_by")),
        "source_session": _text_or_none(data.get("source_session")),
        "created_at": data.get("created_at") or now_iso(),
        "updated_at": data.get("updated_at") or now_iso(),
        "content_hash": _text_or_none(data.get("content_hash")),
    }


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_RECORD_FIELDS = frozenset(MemoryRecord.__dataclass_fields__)  # type: ignore[attr-defined]
