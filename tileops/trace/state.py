"""Build-global trace state shared between the annotation API and the transform.

The trace tool splits responsibilities:

* The module-level ``tileops.trace.api.trace`` namespace emits lightweight
  *placeholders* (``tl_trace_marker`` ``call_extern`` nodes) into the kernel body
  and interns the human-readable names referenced by those placeholders into id
  tables.
* The ``tileops.trace.passes.lower`` transform reads those same intern tables
  back out (to build the host-side decode maps) after the kernel is built.

The two halves never call each other directly; they communicate through the
build registry installed here. The ``trace.*`` markers populate a fresh
``TraceState`` per build epoch (``build_state``); ``trace.lower`` reads
it, then retires the epoch (``begin_build_epoch``) so the next build starts
clean.

This is process-global, single-threaded build state: TileLang kernels are built
eagerly on the calling thread, so one build epoch brackets exactly one build.
"""

from .record import MAX_LANES

__all__ = [
    "MARKER",
    "TraceState",
    "begin_build_epoch",
    "build_state",
]

# Placeholder symbol name. A trace placeholder is
# ``T.evaluate(T.call_extern("handle", MARKER, event_id, kind, lane, gid, payload))``.
# The id args are compile-time-constant interned ids; ``payload`` is the one
# runtime arg (a PrimExpr). The transform locates the placeholder by matching
# ``StringImm`` arg0 == MARKER.
MARKER = "tl_trace_marker"

_DEFAULT_LEAD = 0
_DEFAULT_LANE = "main"


class TraceState:
    """Build-global intern tables + the active group stack for one build.

    A fresh instance backs each build epoch (see ``build_state``). The
    annotation API mutates it as it emits placeholders; the transform reads the
    reverse maps to label the host-side decode.

    Event names, group names, and lane names each live in a **separate** intern
    namespace: an event name interns to a 24-bit ``event_id`` packed into the
    record, a group name interns to a ``gid`` that selects the slot row, and a
    lane name interns to a 4-bit ``lane_id`` packed into the record.

    Attributes:
        enabled: Whether placeholders are emitted.
        id_to_name: Reverse event-id map (``event_id -> name``).
        group_id_to_name: Reverse group-id map (``gid -> name``).
        group_id_to_lead: Group-id to elected lead lane (``gid -> lead``).
        lane_id_to_name: Reverse lane-id map (``lane_id -> name``).
        declared_flows: Build-time-declared ``(src_name, dst_name)`` flow pairs.
    """

    def __init__(self) -> None:
        self.enabled = False

        # event-name -> event_id (sequential from 0) + reverse for the host.
        self._name_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {}

        # group-name -> gid (sequential from 0) + reverses for host/transform.
        self._group_to_id: dict[str, int] = {}
        self.group_id_to_name: dict[int, str] = {}
        self.group_id_to_lead: dict[int, int] = {}

        # lane-name -> lane_id (sequential from 0) + reverse for the host.
        self._lane_to_id: dict[str, int] = {}
        self.lane_id_to_name: dict[int, str] = {}

        # Build-time-declared flows: each a (src_name, dst_name) pair. The host
        # pairs same-named slices per CTA in timestamp order to draw arrows.
        self.declared_flows: list[tuple[str, str]] = []

        # Active-group stack of (name, lead); the live group is the top. gids are
        # assigned lazily on first emit (see active_gid) so a kernel that declares
        # its own groups gets a contiguous gid range.
        self._group_stack: list[tuple[str, int]] = [("default", _DEFAULT_LEAD)]

    # -- group stack ------------------------------------------------------

    @property
    def active_gid(self) -> int:
        """Group id of the live group, interning its name on first use."""
        name, lead = self._group_stack[-1]
        return self._intern_group(name, lead)

    @property
    def active_lead(self) -> int:
        """Elected lead lane of the live group (top of the stack)."""
        return self._group_stack[-1][1]

    def push_group(self, name: str, lead: int) -> None:
        """Make ``(name, lead)`` the live group until the matching ``pop_group``."""
        self._group_stack.append((name, lead))

    def pop_group(self) -> None:
        """Restore the previously live group (undo one ``push_group``)."""
        self._group_stack.pop()

    # -- intern tables ----------------------------------------------------

    def intern_event(self, name: str) -> int:
        """Intern an event ``name`` to a stable 24-bit event id.

        Args:
            name: Human-readable event name.

        Returns:
            The interned event id; stable across repeated lookups of ``name``.
        """
        eid = self._name_to_id.get(name)
        if eid is None:
            eid = len(self._name_to_id)
            self._name_to_id[name] = eid
            self.id_to_name[eid] = name
        return eid

    def intern_lane(self, name: str) -> int:
        """Intern a lane ``name`` to a stable 4-bit lane id.

        Args:
            name: Render sub-lane name (e.g. ``"main"``, ``"tma"``).

        Returns:
            The interned lane id; stable across repeated lookups of ``name``.

        Raises:
            ValueError: If interning ``name`` would exceed the 16-lane cap of the
                4-bit packed lane field.
        """
        lane_id = self._lane_to_id.get(name)
        if lane_id is None:
            if len(self._lane_to_id) >= MAX_LANES:
                raise ValueError(
                    f"too many distinct lanes: the packed lane field holds at most "
                    f"{MAX_LANES}; cannot intern {name!r} (have "
                    f"{sorted(self._lane_to_id)})")
            lane_id = len(self._lane_to_id)
            self._lane_to_id[name] = lane_id
            self.lane_id_to_name[lane_id] = name
        return lane_id

    def intern_group(self, name: str, lead: int) -> int:
        """Intern a group ``name`` (with its ``lead``) to a stable gid.

        Args:
            name: Work-group name.
            lead: Elected lead lane for the group's writer (``tx == lead``).

        Returns:
            The interned gid; stable across repeated lookups of ``name``.
        """
        return self._intern_group(name, lead)

    def _intern_group(self, name: str, lead: int) -> int:
        gid = self._group_to_id.get(name)
        if gid is None:
            gid = len(self._group_to_id)
            self._group_to_id[name] = gid
            self.group_id_to_name[gid] = name
            self.group_id_to_lead[gid] = lead
        return gid

    def declare_flow(self, src_name: str, dst_name: str) -> None:
        """Record a build-time flow declaration between two named ranges.

        The declaration emits no runtime record; the host renders it by pairing
        each CTA's ``src_name`` slices with its ``dst_name`` slices in timestamp
        order (see ``tileops.trace.decode.compute_flows``).

        Args:
            src_name: Source (producer) range name.
            dst_name: Destination (consumer) range name.
        """
        self.declared_flows.append((src_name, dst_name))


# ---------------------------------------------------------------------------
# Always-emit build registry.
#
# ``trace.*`` markers always emit their placeholder into the kernel body and
# intern the referenced names into a per-build registry. TileLang builds
# eagerly and single-threaded, so one build = one registry epoch: ``trace.lower``
# reads the registry right after the build it transforms, then retires it via
# begin_build_epoch(). The next marker lazily installs a fresh registry.
# ---------------------------------------------------------------------------

_BUILD_STATE: TraceState | None = None


def begin_build_epoch() -> None:
    """Retire the current build registry so the next marker starts a fresh one.

    Called by ``tileops.trace.passes.lower`` after it has consumed a build's
    interned maps. The next ``build_state`` lookup installs a new empty
    ``TraceState``.
    """
    global _BUILD_STATE
    _BUILD_STATE = None


def build_state() -> TraceState:
    """Return the live build registry, installing a fresh one if none is active.

    Lazily creates the registry on the first marker of a build epoch so a kernel
    which emits no markers never allocates one.

    Returns:
        The current build epoch's ``TraceState``.
    """
    global _BUILD_STATE
    if _BUILD_STATE is None:
        _BUILD_STATE = TraceState()
        _BUILD_STATE.enabled = True
    return _BUILD_STATE
