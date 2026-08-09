"""Host-side decode of the single ``slots`` buffer into ``Slice`` / ``Instant``.

After readback the device emits one ``slots`` int64 tensor of shape
``(num_cta, num_groups, (max_events + 1) * 2)`` (the layout
``tileops.trace.passes.lower`` writes). Each ``(cta, gid)`` slot row
is a flat run of int64 words:

* ``word[0]`` = event count (header), saturated at ``max_events``.
* ``word[1]`` = reserved (garbage from ``torch.empty``; ignored).
* event ``e`` in ``[0, count)`` occupies words ``[(e + 1) * 2, (e + 1) * 2 + 1]``::

      word[(e + 1) * 2]      = clock64 timestamp (int64)
      word[(e + 1) * 2 + 1]  = pack_w1(event_id, kind, lane, payload)

This module turns those raw words into render-ready objects: per ``(cta, gid)``
slot it reads the header count, unpacks each ``w1`` word, zeroes timestamps per
**CTA**, and stack-matches RANGE_BEGIN/RANGE_END pairs per ``(cta, gid, lane)``
track into ``Slice`` objects. INSTANT records become ``Instant``
objects. The ``lane`` in each track is the interned lane id; exporters resolve it
to a name via the ``lane_id_to_name`` map.

Flow arrows are NOT decoded from records: ``trace.dag`` is a build-time
declaration, so no flow record exists on the device. Instead ``compute_flows``
takes the decoded ``Slice`` list plus the declared ``(src_name, dst_name)``
pairs and, per CTA, pairs the i-th ``src_name`` slice with the i-th ``dst_name``
slice (both in timestamp order) into ``FlowEdge`` objects.

Per-CTA timestamp zeroing:
    ``clock64`` is a per-SM counter, so timestamps are only comparable within a
    single CTA (all of a CTA's groups run on the same SM). Each CTA subtracts the
    minimum timestamp across all of its groups and events, giving the producer and
    consumer warpgroups of one CTA a shared origin while keeping different CTAs
    independent.
"""

from dataclasses import dataclass

from .record import EventKind, unpack_w1

__all__ = ["FlowEdge", "Instant", "Slice", "compute_flows", "cycles_to_us", "decode"]


@dataclass
class Slice:
    """A completed RANGE_BEGIN/RANGE_END span on one render track.

    Attributes:
        track: Render track ``(cta, gid, lane)``.
        name: Human-readable event name from ``id_to_name``.
        ts_cy: Per-CTA-zeroed start timestamp in ``clock64`` cycles.
        dur_cy: Span duration in cycles (end minus start, non-negative).
        payload: Raw 32-bit payload from the RANGE_BEGIN record, or ``None``.
    """

    track: tuple
    name: str
    ts_cy: int
    dur_cy: int
    payload: int | None


@dataclass
class Instant:
    """A point-in-time INSTANT marker on one render track.

    Attributes:
        track: Render track ``(cta, gid, lane)``.
        name: Human-readable event name from ``id_to_name``.
        ts_cy: Per-CTA-zeroed timestamp in ``clock64`` cycles.
        payload: Raw 32-bit payload from the record, or ``None``.
    """

    track: tuple
    name: str
    ts_cy: int
    payload: int | None


@dataclass
class FlowEdge:
    """One resolved flow arrow connecting a source slice to a destination slice.

    Built by ``compute_flows`` from a declared ``(src_name, dst_name)`` flow:
    per CTA, the i-th ``src_name`` slice (by start timestamp) pairs with the i-th
    ``dst_name`` slice. The arrow runs from the source slice's END to the
    destination slice's START.

    Attributes:
        src_track: Source render track ``(cta, gid, lane)``.
        src_ts_cy: Per-CTA-zeroed source endpoint (source slice end) in cycles.
        dst_track: Destination render track ``(cta, gid, lane)``.
        dst_ts_cy: Per-CTA-zeroed destination endpoint (dest slice start) in cycles.
        src_name: Source range name.
        dst_name: Destination range name.
    """

    src_track: tuple
    src_ts_cy: int
    dst_track: tuple
    dst_ts_cy: int
    src_name: str
    dst_name: str


def cycles_to_us(cy: int, sm_clock_ghz: float = 1.5) -> float:
    """Convert a ``clock64`` cycle count to microseconds at a fixed SM clock.

    Args:
        cy: Cycle count (e.g. a ``Slice.dur_cy`` or zeroed ``ts_cy``).
        sm_clock_ghz: Locked SM clock in GHz (1.5 GHz on the bench rig).

    Returns:
        The duration in microseconds (``cy / (sm_clock_ghz * 1e3)``).
    """
    return cy / (sm_clock_ghz * 1e3)


def _to_numpy(arr):
    """Return ``arr`` as a host numpy array, accepting torch tensors or numpy.

    Args:
        arr: A torch tensor (any device) or a numpy array.

    Returns:
        The same data as a CPU numpy array.
    """
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return arr


def decode(slots, *, id_to_name: dict[int, str], group_id_to_name: dict[int, str] | None = None,
           lane_id_to_name: dict[int, str] | None = None, max_events: int, num_groups: int,
           sm_clock_ghz: float = 1.5) -> list:
    """Decode the ``slots`` buffer into a flat list of ``Slice`` / ``Instant``.

    Walks every ``(cta, gid)`` slot, reads the header count, unpacks each event's
    ``w1`` word, zeroes timestamps against the owning **CTA**'s minimum, and
    stack-matches range pairs per ``(cta, gid, lane)`` track in stored program
    order. Flow arrows are not decoded here; ``compute_flows`` derives them
    from the returned slices and the declared flow pairs.

    Tracks are always the full 3-tuple ``(cta, gid, lane)`` with ``lane`` the
    interned lane id; exporters resolve it to a name via ``lane_id_to_name``.

    Robustness:
        * RANGE_END without a matching open begin on its track is skipped.
        * If the track stack top mismatches the end's ``event_id``, the matching
          begin is searched deeper in the stack; intervening begins are kept open.
        * Begins still open at end of a slot are dropped (e.g. when ``max_events``
          clamps the row before a RANGE_END is recorded).

    Args:
        slots: ``(num_cta, num_groups, (max_events + 1) * 2)`` int64 buffer.
            Accepts a torch tensor or numpy array.
        id_to_name: Maps interned event id to a human-readable name.
        group_id_to_name: Unused for decode (gid stays numeric in the track);
            accepted for symmetry with the host maps. ``None`` is fine.
        lane_id_to_name: Unused for decode (lane stays numeric in the track);
            accepted for symmetry with the host maps. ``None`` is fine.
        max_events: Per-slot event capacity; the header count is clamped to it.
        num_groups: Logical groups per CTA (the ``gid`` axis length).
        sm_clock_ghz: Locked SM clock in GHz; carried for downstream consumers.

    Returns:
        A flat list of ``Slice`` and ``Instant`` objects across all
        ``(cta, gid)`` slots.
    """
    del group_id_to_name  # gid stays numeric in the track; accepted for symmetry.
    del lane_id_to_name  # lane stays numeric in the track; accepted for symmetry.
    del sm_clock_ghz  # carried for API symmetry; conversion happens downstream.

    slots_np = _to_numpy(slots)
    num_cta = slots_np.shape[0]

    events: list = []

    # Per-CTA clock origin: a CTA's group rows share one SM, so they zero against
    # a common minimum taken across all of that CTA's groups and events.
    cta_ts_min: dict[int, int] = {}
    for cta in range(num_cta):
        for gid in range(num_groups):
            row = slots_np[cta, gid]
            count = min(int(row[0]), max_events)
            if count <= 0:
                continue
            ts_words = [int(row[(e + 1) * 2]) for e in range(count)]
            slot_min = min(ts_words)
            prev = cta_ts_min.get(cta)
            cta_ts_min[cta] = slot_min if prev is None else min(prev, slot_min)

    for cta in range(num_cta):
        if cta not in cta_ts_min:
            continue
        ts_min = cta_ts_min[cta]
        for gid in range(num_groups):
            row = slots_np[cta, gid]
            count = min(int(row[0]), max_events)
            if count <= 0:
                continue

            # Open RANGE_BEGIN stacks keyed by lane; each entry is (event_id, ts0,
            # payload). Program order is the stored event order within the slot.
            open_stacks: dict[int, list[tuple[int, int, int]]] = {}

            for e in range(count):
                ts = int(row[(e + 1) * 2]) - ts_min
                event_id, kind, lane, payload = unpack_w1(int(row[(e + 1) * 2 + 1]))
                name = id_to_name.get(event_id, str(event_id))

                if kind == EventKind.RANGE_BEGIN:
                    open_stacks.setdefault(lane, []).append((event_id, ts, payload))

                elif kind == EventKind.RANGE_END:
                    stack = open_stacks.get(lane)
                    if not stack:
                        continue  # end without any open begin on this track.
                    # Match by event_id; tolerate a mismatched top by searching down.
                    match_idx = None
                    for i in range(len(stack) - 1, -1, -1):
                        if stack[i][0] == event_id:
                            match_idx = i
                            break
                    if match_idx is None:
                        continue  # no matching begin; skip this end.
                    begin_id, ts0, begin_payload = stack.pop(match_idx)
                    events.append(
                        Slice(
                            track=(cta, gid, lane),
                            name=id_to_name.get(begin_id, str(begin_id)),
                            ts_cy=ts0,
                            dur_cy=ts - ts0,
                            payload=begin_payload,
                        ))

                elif kind == EventKind.INSTANT:
                    events.append(
                        Instant(track=(cta, gid, lane), name=name, ts_cy=ts, payload=payload))

            # Begins still open at slot end (e.g. clamped RANGE_END) are dropped.

    return events


def compute_flows(events: list, flows: list) -> list:
    """Resolve declared flow pairs into per-CTA ``FlowEdge`` arrows.

    For each declared ``(src_name, dst_name)`` flow and each CTA, collects that
    CTA's ``Slice`` objects named ``src_name`` and named ``dst_name`` (each
    sorted by start timestamp), then pairs the i-th source with the i-th
    destination for ``i`` in ``[0, min(len_src, len_dst))``. Each pair becomes one
    ``FlowEdge`` whose arrow runs from the source slice END to the
    destination slice START. Distinct occurrences therefore yield distinct edges
    with distinct endpoints — never collapsing onto a single anchor.

    Args:
        events: Flat event list from ``decode`` (only ``Slice`` objects
            participate; other kinds are ignored).
        flows: Declared ``(src_name, dst_name)`` pairs (``host_maps["flows"]``).

    Returns:
        A flat list of ``FlowEdge`` objects across all flows and CTAs.
    """
    slices = [e for e in events if isinstance(e, Slice)]
    ctas = sorted({s.track[0] for s in slices})

    edges: list = []
    for src_name, dst_name in flows:
        for cta in ctas:
            srcs = sorted((s for s in slices if s.track[0] == cta and s.name == src_name),
                          key=lambda s: s.ts_cy)
            dsts = sorted((s for s in slices if s.track[0] == cta and s.name == dst_name),
                          key=lambda s: s.ts_cy)
            for src, dst in zip(srcs, dsts, strict=False):
                edges.append(
                    FlowEdge(
                        src_track=src.track,
                        src_ts_cy=src.ts_cy + src.dur_cy,
                        dst_track=dst.track,
                        dst_ts_cy=dst.ts_cy,
                        src_name=src_name,
                        dst_name=dst_name,
                    ))
    return edges
