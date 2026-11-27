"""Per-thread trace build state shared by the annotation API and ``trace.lower``.

The ``trace.*`` markers emit ``tl_trace_marker`` placeholders into the kernel body and
intern the names they reference into this thread's ``TraceState``; ``trace.lower`` reads
those tables after the build to make the host decode maps, then retires the state
(``begin_build_epoch``) so the next build starts clean.

TileLang builds a kernel on the calling thread, so one epoch on a thread brackets exactly
one build. Nested builds on one thread share its state and are not supported.
"""

import threading

from tileops.trace._record import MAX_LANES

__all__ = [
    "MARKER",
    "TraceState",
    "begin_build_epoch",
    "build_state",
]

# A placeholder is ``T.evaluate(T.call_extern("handle", MARKER, event_id, kind, lane, gid,
# payload))``: compile-time ids and one runtime payload. ``trace.lower`` matches arg0.
MARKER = "tl_trace_marker"


class TraceState:
    """The intern tables and the active group stack of one build.

    Event, group and lane names intern in separate namespaces, each numbered from 0 in
    first-use order: an event name to the 24-bit ``event_id`` and a lane name to the
    4-bit lane id packed into a record, a group name to the ``gid`` that selects the
    slot row. The ``*_to_name`` maps, ``group_id_to_lead`` and ``declared_flows`` are
    what ``trace.lower`` copies into the host decode maps.
    """

    def __init__(self) -> None:
        self.id_to_name: dict[int, str] = {}
        self.group_id_to_name: dict[int, str] = {}
        self.group_id_to_lead: dict[int, int] = {}
        self.lane_id_to_name: dict[int, str] = {}
        self.declared_flows: list[tuple[str, str]] = []
        # The live group is the top. A gid is interned on the group's first marker, so a
        # kernel that declares its own groups gets a contiguous gid range. The default
        # group's writer is thread 0.
        self._group_stack: list[tuple[str, int]] = [("default", 0)]

    @staticmethod
    def _intern(id_to_name: dict[int, str], name: str) -> "tuple[int, bool]":
        """The id of *name* in *id_to_name*, and whether it was added now."""
        for known_id, known in id_to_name.items():
            if known == name:
                return known_id, False
        new_id = len(id_to_name)
        id_to_name[new_id] = name
        return new_id, True

    @property
    def active_gid(self) -> int:
        """Group id of the live group, interning its name on first use."""
        name, lead = self._group_stack[-1]
        gid, added = self._intern(self.group_id_to_name, name)
        if added:
            self.group_id_to_lead[gid] = lead
        return gid

    def push_group(self, name: str, lead: int) -> None:
        self._group_stack.append((name, lead))

    def pop_group(self) -> None:
        self._group_stack.pop()

    def intern_event(self, name: str) -> int:
        return self._intern(self.id_to_name, name)[0]

    def intern_lane(self, name: str) -> int:
        """Intern a lane name; raises ``ValueError`` past the 4-bit lane field's capacity."""
        if name not in self.lane_id_to_name.values() and len(self.lane_id_to_name) >= MAX_LANES:
            raise ValueError(
                f"too many distinct lanes: the packed lane field holds at most "
                f"{MAX_LANES}; cannot intern {name!r} (have "
                f"{sorted(self.lane_id_to_name.values())})"
            )
        return self._intern(self.lane_id_to_name, name)[0]

    def declare_flow(self, src_name: str, dst_name: str) -> None:
        """Record a ``(src_name, dst_name)`` flow; ``compute_flows`` renders it."""
        self.declared_flows.append((src_name, dst_name))


_BUILD = threading.local()


def begin_build_epoch() -> None:
    """Retire this thread's state; its next marker starts a fresh one."""
    _BUILD.state = None


def build_state() -> TraceState:
    """This thread's ``TraceState``, created on the first marker of a build epoch."""
    state = getattr(_BUILD, "state", None)
    if state is None:
        state = _BUILD.state = TraceState()
    return state
