"""In-kernel trace ``EventRecord`` packing (16-byte / 2x u64 layout).

Each record is two 64-bit words: ``w0`` holds the ``clock64()`` timestamp and
``w1`` packs ``event_id`` / ``kind`` / ``lane`` / ``payload``. This module owns
the host-side pack/unpack helpers plus a device-side (TIR) packer for the emit
path.
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


class EventKind(IntEnum):
    """Event kind packed into ``w1`` bits 24..27."""

    RANGE_BEGIN = 0
    RANGE_END = 1
    INSTANT = 2
    # Reserved: ``trace.dag`` is now a build-time declaration (no runtime record),
    # so no DAG record is ever emitted. Kept to keep the enum value stable.
    DAG = 3


# Per-slot record capacity (config default; callers may override).
MAX_EVENTS_DEFAULT = 768

# Field widths and bit offsets within w1.
_EVENT_ID_BITS = 24
_KIND_BITS = 4
_LANE_BITS = 4
_PAYLOAD_BITS = 32

_EVENT_ID_SHIFT = 0
_KIND_SHIFT = 24
_LANE_SHIFT = 28
_PAYLOAD_SHIFT = 32

_EVENT_ID_MASK = (1 << _EVENT_ID_BITS) - 1
_KIND_MASK = (1 << _KIND_BITS) - 1
_LANE_MASK = (1 << _LANE_BITS) - 1
_PAYLOAD_MASK = (1 << _PAYLOAD_BITS) - 1

# Max distinct lanes that fit the 4-bit lane field.
MAX_LANES = 1 << _LANE_BITS


def pack_w1(event_id: int, kind: int, lane: int, payload: int) -> int:
    """Pack the four ``w1`` fields into a single unsigned 64-bit word.

    Args:
        event_id: Interned event id, must fit 24 bits (0..2**24-1).
        kind: Event kind, must fit 4 bits (an :class:`EventKind` or its int).
        lane: Interned lane id, must fit 4 bits.
        payload: Raw unsigned 32-bit bit-pattern of an i32/f32 payload. Must fit
            32 bits; no silent truncation.

    Returns:
        The packed 64-bit word as a Python int.

    Raises:
        ValueError: If any field is negative or exceeds its bit width.
    """
    kind = int(kind)
    if not 0 <= event_id <= _EVENT_ID_MASK:
        raise ValueError(f"event_id {event_id} does not fit {_EVENT_ID_BITS} bits")
    if not 0 <= kind <= _KIND_MASK:
        raise ValueError(f"kind {kind} does not fit {_KIND_BITS} bits")
    if not 0 <= lane <= _LANE_MASK:
        raise ValueError(f"lane {lane} does not fit {_LANE_BITS} bits")
    if not 0 <= payload <= _PAYLOAD_MASK:
        raise ValueError(f"payload {payload} does not fit {_PAYLOAD_BITS} bits")

    return ((event_id << _EVENT_ID_SHIFT) | (kind << _KIND_SHIFT) | (lane << _LANE_SHIFT) |
            (payload << _PAYLOAD_SHIFT))


def unpack_w1(w1: int) -> tuple[int, int, int, int]:
    """Unpack a ``w1`` word into ``(event_id, kind, lane, payload)``.

    Inverse of :func:`pack_w1` over the full field ranges.

    Args:
        w1: A packed 64-bit word.

    Returns:
        Tuple of ``(event_id, kind, lane, payload)``.
    """
    event_id = (w1 >> _EVENT_ID_SHIFT) & _EVENT_ID_MASK
    kind = (w1 >> _KIND_SHIFT) & _KIND_MASK
    lane = (w1 >> _LANE_SHIFT) & _LANE_MASK
    payload = (w1 >> _PAYLOAD_SHIFT) & _PAYLOAD_MASK
    return event_id, kind, lane, payload


def pack_w1_tir(event_id: int, kind: int, lane: int, payload_expr: PrimExpr) -> PrimExpr:
    """Build the device-side ``w1`` packed word as an int64 TIR expression.

    The ``event_id`` / ``kind`` / ``lane`` fields are compile-time Python ints
    and fold into a constant base occupying the low 32 bits. ``payload_expr`` is a
    runtime i32 PrimExpr whose raw bits are shifted into bits 32..63. The i32 is
    reinterpreted as uint32 before widening so a negative i32 does not
    sign-extend into the high half.

    Args:
        event_id: Interned event id, must fit 24 bits.
        kind: Event kind, must fit 4 bits (an :class:`EventKind` or its int).
        lane: Interned lane id, must fit 4 bits.
        payload_expr: TIR i32 PrimExpr carrying the raw payload bits.

    Returns:
        An int64 PrimExpr equal to the packed ``w1`` word.

    Raises:
        ValueError: If ``event_id`` / ``kind`` / ``lane`` exceed their widths.
    """
    kind = int(kind)
    if not 0 <= event_id <= _EVENT_ID_MASK:
        raise ValueError(f"event_id {event_id} does not fit {_EVENT_ID_BITS} bits")
    if not 0 <= kind <= _KIND_MASK:
        raise ValueError(f"kind {kind} does not fit {_KIND_BITS} bits")
    if not 0 <= lane <= _LANE_MASK:
        raise ValueError(f"lane {lane} does not fit {_LANE_BITS} bits")

    base = (event_id << _EVENT_ID_SHIFT) | (kind << _KIND_SHIFT) | (lane << _LANE_SHIFT)
    payload_u32 = T.reinterpret(payload_expr, "uint32")
    payload_i64 = T.Cast("int64", payload_u32)
    payload_hi = T.shift_left(payload_i64, T.Cast("int64", _PAYLOAD_SHIFT))
    return T.bitwise_or(T.Cast("int64", base), payload_hi)
