"""In-kernel trace record packing.

Each record is two 64-bit words: ``w0`` holds the ``clock64()`` timestamp and
``w1`` packs ``event_id`` / ``kind`` / ``lane`` / ``payload``.
"""

from enum import IntEnum

# Importing tilelang first populates sys.path for the bundled tvm package.
import tilelang.language as T
from tvm.ir import PrimExpr

__all__ = [
    "MAX_EVENTS_DEFAULT",
    "EventKind",
    "pack_w1",
    "pack_w1_tir",
    "unpack_w1",
]


# Per-slot record capacity (config default; callers may override).
MAX_EVENTS_DEFAULT = 768
# Field widths within w1; each field starts where the previous one ends.
_EVENT_ID_BITS = 24
_KIND_BITS = 4
_LANE_BITS = 4
_PAYLOAD_BITS = 32
_EVENT_ID_SHIFT = 0
_KIND_SHIFT = _EVENT_ID_SHIFT + _EVENT_ID_BITS
_LANE_SHIFT = _KIND_SHIFT + _KIND_BITS
_PAYLOAD_SHIFT = _LANE_SHIFT + _LANE_BITS
_EVENT_ID_MASK = (1 << _EVENT_ID_BITS) - 1
_KIND_MASK = (1 << _KIND_BITS) - 1
_LANE_MASK = (1 << _LANE_BITS) - 1
_PAYLOAD_MASK = (1 << _PAYLOAD_BITS) - 1
# Max distinct lanes that fit the 4-bit lane field.
MAX_LANES = 1 << _LANE_BITS


class EventKind(IntEnum):
    """Event kind packed into ``w1`` bits 24..27."""

    RANGE_BEGIN = 0
    RANGE_END = 1
    INSTANT = 2


def _header_bits(event_id: int, kind: int, lane: int) -> int:
    """The low 32 bits of ``w1``; raises ``ValueError`` when a field exceeds its width."""
    kind = int(kind)
    if not 0 <= event_id <= _EVENT_ID_MASK:
        raise ValueError(f"event_id {event_id} does not fit {_EVENT_ID_BITS} bits")
    if not 0 <= kind <= _KIND_MASK:
        raise ValueError(f"kind {kind} does not fit {_KIND_BITS} bits")
    if not 0 <= lane <= _LANE_MASK:
        raise ValueError(f"lane {lane} does not fit {_LANE_BITS} bits")
    return (event_id << _EVENT_ID_SHIFT) | (kind << _KIND_SHIFT) | (lane << _LANE_SHIFT)


def pack_w1(event_id: int, kind: int, lane: int, payload: int) -> int:
    """Pack the four ``w1`` fields into one unsigned 64-bit word.

    ``payload`` is the raw unsigned 32-bit pattern of an i32/f32 value. Raises
    ``ValueError`` when any field is negative or exceeds its bit width.
    """
    base = _header_bits(event_id, kind, lane)
    if not 0 <= payload <= _PAYLOAD_MASK:
        raise ValueError(f"payload {payload} does not fit {_PAYLOAD_BITS} bits")
    return base | (payload << _PAYLOAD_SHIFT)


def unpack_w1(w1: int) -> tuple[int, int, int, int]:
    """Unpack a ``w1`` word into ``(event_id, kind, lane, payload)``; inverse of ``pack_w1``."""
    event_id = (w1 >> _EVENT_ID_SHIFT) & _EVENT_ID_MASK
    kind = (w1 >> _KIND_SHIFT) & _KIND_MASK
    lane = (w1 >> _LANE_SHIFT) & _LANE_MASK
    payload = (w1 >> _PAYLOAD_SHIFT) & _PAYLOAD_MASK
    return event_id, kind, lane, payload


def pack_w1_tir(event_id: int, kind: int, lane: int, payload_expr: PrimExpr) -> PrimExpr:
    """The device-side ``w1`` word as an int64 TIR expression.

    ``event_id`` / ``kind`` / ``lane`` are compile-time ints folded into the low 32 bits;
    ``payload_expr`` is a runtime i32 shifted into bits 32..63. The i32 is reinterpreted
    as uint32 before widening so a negative payload does not sign-extend into the high half.
    """
    base = _header_bits(event_id, kind, lane)
    payload_u32 = T.reinterpret(payload_expr, "uint32")
    payload_i64 = T.Cast("int64", payload_u32)
    payload_hi = T.shift_left(payload_i64, T.Cast("int64", _PAYLOAD_SHIFT))
    return T.bitwise_or(T.Cast("int64", base), payload_hi)
