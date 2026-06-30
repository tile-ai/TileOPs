"""Internal marker-emission machinery backing the ``trace`` annotation methods.

Holds the device-side ``tl_trace_marker`` emitter plus the scope / token types
that ``trace.group`` and ``trace.range`` return. Not part of the public API —
import :data:`tileops.trace.trace`, never this module.
"""

import tilelang.language as T

from .record import EventKind
from .state import MARKER, build_state

# Default render sub-lane for records that do not name one.
DEFAULT_LANE = "main"


def _emit(event_id: int, kind: int, lane: int, payload) -> None:
    """Emit one ``tl_trace_marker`` placeholder for the active group.

    ``event_id`` / ``kind`` / ``lane`` / ``gid`` are compile-time-constant ints;
    ``payload`` is the lone runtime arg, carried through as a PrimExpr so a loop
    index survives into the lowered ``pack_w1_tir`` call.
    """
    state = build_state()
    if payload is None:
        payload_expr = T.int32(0)
    elif isinstance(payload, int):
        payload_expr = T.int32(payload)
    else:
        payload_expr = payload
    T.evaluate(
        T.call_extern("handle", MARKER, event_id, int(kind), lane, state.active_gid,
                      payload_expr))


class AnnoToken:
    """Opaque handle naming an open range; pairs a RANGE_END with its RANGE_BEGIN.

    Returned by ``trace.range_start`` and consumed by ``trace.range_end``. Holds
    no device state.
    """

    __slots__ = ("name", "event_id", "lane")

    def __init__(self, name: str, event_id: int, lane: int):
        self.name = name
        self.event_id = event_id
        self.lane = lane


def open_range(name: str, lane: str) -> AnnoToken:
    """Intern ``name`` / ``lane``, emit a RANGE_BEGIN, and return its token."""
    state = build_state()
    event_id = state.intern_event(name)
    lane_id = state.intern_lane(lane)
    _emit(event_id, EventKind.RANGE_BEGIN, lane_id, 0)
    return AnnoToken(name, event_id, lane_id)


def close_range(tok: AnnoToken | None) -> None:
    """Emit the RANGE_END matching ``tok`` (``None`` no-ops)."""
    if tok is None:
        return
    _emit(tok.event_id, EventKind.RANGE_END, tok.lane, 0)


def emit_instant(name: str, payload, lane: str) -> None:
    """Intern ``name`` / ``lane`` and emit a single INSTANT marker."""
    state = build_state()
    event_id = state.intern_event(name)
    lane_id = state.intern_lane(lane)
    _emit(event_id, EventKind.INSTANT, lane_id, payload)


def declare_flow(src_name: str, dst_name: str) -> None:
    """Record a build-time ``(src_name, dst_name)`` flow pair (no device marker)."""
    build_state().declare_flow(src_name, dst_name)


class GroupScope:
    """Context manager that makes a declared group live for its body."""

    def __init__(self, name: str, lead: int):
        self._name = name
        self._lead = lead

    def __enter__(self) -> "GroupScope":
        build_state().push_group(self._name, self._lead)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        build_state().pop_group()
        return False


class RangeScope:
    """Context manager emitting RANGE_BEGIN on enter, RANGE_END on exit."""

    def __init__(self, name: str, lane: str):
        self._name = name
        self._lane = lane
        self._tok: AnnoToken | None = None

    def __enter__(self) -> AnnoToken | None:
        self._tok = open_range(self._name, self._lane)
        return self._tok

    def __exit__(self, exc_type, exc, tb) -> bool:
        close_range(self._tok)
        return False
