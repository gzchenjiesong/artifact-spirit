"""标识符生成：ULID 与记忆 ID。

零第三方依赖——ULID 用标准库自实现（LLD-AL3 §4）。
保证**同一毫秒内也单调递增**，以满足"ID 单调递增"的验收要求。
"""

from __future__ import annotations

import os
import threading
import time

__all__ = [
    "LAYER_ABBR",
    "new_entity_id",
    "new_intent_id",
    "new_memory_id",
    "new_overview_id",
    "new_ulid",
    "new_working_id",
]

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford Base32（无 I/L/O/U）
_TS_LEN = 10
_RND_LEN = 16
_ULID_LEN = _TS_LEN + _RND_LEN  # 26

# 记忆 ID 的层缩写（LLD-AL3 §5 M3）
LAYER_ABBR: dict[str, str] = {
    "episodic": "epi",
    "semantic": "sem",
    "procedural": "pro",
    "core": "cor",
}

_lock = threading.Lock()
_last_ts_ms = -1
_last_rnd = 0


def _encode(value: int, length: int) -> str:
    """小端 → 大端编码为 Crockford Base32，定长 ``length``。"""
    chars = []
    for _ in range(length):
        chars.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_ulid(ts_ms: int | None = None) -> str:
    """生成 26 字符 ULID。

    同一毫秒内通过**递增随机段**保证单调性（不依赖系统时钟回拨）。
    """
    global _last_ts_ms, _last_rnd

    if ts_ms is None:
        ts_ms = int(time.time() * 1000)

    with _lock:
        if ts_ms > _last_ts_ms:
            _last_ts_ms = ts_ms
            _last_rnd = int.from_bytes(os.urandom(10), "big")
        else:
            # 同一毫秒（或时钟回拨）：沿用上次时间戳并递增随机段
            ts_ms = _last_ts_ms
            _last_rnd += 1
            if _last_rnd >= (1 << 80):  # 溢出兜底，理论上不可达
                _last_ts_ms += 1
                ts_ms = _last_ts_ms
                _last_rnd = int.from_bytes(os.urandom(10), "big")

        ts_part = _encode(ts_ms & ((1 << 48) - 1), _TS_LEN)
        rnd_part = _encode(_last_rnd, _RND_LEN)

    return ts_part + rnd_part


def _prefixed(prefix: str) -> str:
    return f"{prefix}_{new_ulid()}"


def new_memory_id(layer: str) -> str:
    """``{abbr}_{ulid}``，abbr ∈ epi | sem | pro | cor（LLD-AL3 §5 M3）。"""
    try:
        abbr = LAYER_ABBR[layer]
    except KeyError as exc:  # pragma: no cover - 参数由内部保证
        raise ValueError(f"未知记忆层：{layer!r}") from exc
    return _prefixed(abbr)


def new_working_id() -> str:
    return _prefixed("wm")


def new_intent_id() -> str:
    return _prefixed("int")


def new_entity_id() -> str:
    return _prefixed("ent")


def new_overview_id() -> str:
    return _prefixed("ov")
