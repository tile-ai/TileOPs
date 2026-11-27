"""Host-side decode of the ``slots`` buffer into ``Slice`` / ``Instant`` events.

The layout is the one ``tileops.trace._passes`` documents and writes. ``clock64`` is a
per-SM counter, so timestamps compare only within one CTA: each CTA's events are zeroed
against the minimum timestamp across all of its groups. Flow arrows are not on the device;
``compute_flows`` derives them from the decoded slices and the declared flow pairs.
"""

from dataclasses import dataclass

from tileops.trace._record import EventKind, unpack_w1

__all__ = ["FlowEdge", "Instant", "Slice", "compute_flows", "decode"]


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
    """One flow arrow, from a source slice's end to a destination slice's start.

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


def decode(
    slots,
    *,
    id_to_name: dict[int, str],
    group_id_to_name: dict[int, str] | None = None,
    lane_id_to_name: dict[int, str] | None = None,
    max_events: int,
    num_groups: int,
    sm_clock_ghz: float = 1.5,
) -> list:
    """Decode the ``slots`` buffer into a flat list of ``Slice`` / ``Instant``.

    Range pairs are stack-matched per ``(cta, gid, lane)`` track in stored order. A
    RANGE_END with no open begin of its ``event_id`` on its track is skipped; one that
    matches below the stack top leaves the begins above it open; begins still open at the
    end of a slot (``max_events`` clamped the row) are dropped.

    Args:
        slots: The ``slots`` int64 buffer, a torch tensor or a numpy array.
        id_to_name: Maps interned event id to its name.
        group_id_to_name: Unused; the track keeps the numeric gid.
        lane_id_to_name: Unused; the track keeps the numeric lane id.
        max_events: Per-slot event capacity; the header count is clamped to it.
        num_groups: Logical groups per CTA.
        sm_clock_ghz: Unused; exporters convert cycles.
    """
    del group_id_to_name, lane_id_to_name, sm_clock_ghz

    slots_np = slots.cpu().numpy() if hasattr(slots, "cpu") else slots
    events: list = []
    for cta in range(slots_np.shape[0]):
        rows = []
        for gid in range(num_groups):
            row = slots_np[cta, gid]
            count = min(int(row[0]), max_events)
            if count > 0:
                rows.append((gid, row, count))
        if not rows:
            continue
        ts_min = min(int(row[(e + 1) * 2]) for _, row, count in rows for e in range(count))
        for gid, row, count in rows:
            # Open begins per lane: (event_id, ts, payload).
            open_stacks: dict[int, list[tuple[int, int, int]]] = {}
            for e in range(count):
                ts = int(row[(e + 1) * 2]) - ts_min
                event_id, kind, lane, payload = unpack_w1(int(row[(e + 1) * 2 + 1]))
                if kind == EventKind.RANGE_BEGIN:
                    open_stacks.setdefault(lane, []).append((event_id, ts, payload))
                elif kind == EventKind.RANGE_END:
                    stack = open_stacks.get(lane, [])
                    match = next(
                        (i for i in range(len(stack) - 1, -1, -1) if stack[i][0] == event_id),
                        None,
                    )
                    if match is None:
                        continue
                    _, ts0, begin_payload = stack.pop(match)
                    events.append(
                        Slice(
                            track=(cta, gid, lane),
                            name=id_to_name.get(event_id, str(event_id)),
                            ts_cy=ts0,
                            dur_cy=ts - ts0,
                            payload=begin_payload,
                        )
                    )
                elif kind == EventKind.INSTANT:
                    events.append(
                        Instant(
                            track=(cta, gid, lane),
                            name=id_to_name.get(event_id, str(event_id)),
                            ts_cy=ts,
                            payload=payload,
                        )
                    )
    return events


def compute_flows(events: list, flows: list) -> list:
    """Resolve declared ``(src_name, dst_name)`` flows into per-CTA ``FlowEdge`` arrows.

    Per CTA, the i-th ``src_name`` slice pairs with the i-th ``dst_name`` slice, both by
    start timestamp; an arrow runs from the source's end to the destination's start.
    """
    slices = [e for e in events if isinstance(e, Slice)]
    ctas = sorted({s.track[0] for s in slices})

    edges: list = []
    for src_name, dst_name in flows:
        for cta in ctas:
            srcs = sorted(
                (s for s in slices if s.track[0] == cta and s.name == src_name),
                key=lambda s: s.ts_cy,
            )
            dsts = sorted(
                (s for s in slices if s.track[0] == cta and s.name == dst_name),
                key=lambda s: s.ts_cy,
            )
            for src, dst in zip(srcs, dsts, strict=False):
                edges.append(
                    FlowEdge(
                        src_track=src.track,
                        src_ts_cy=src.ts_cy + src.dur_cy,
                        dst_track=dst.track,
                        dst_ts_cy=dst.ts_cy,
                        src_name=src_name,
                        dst_name=dst_name,
                    )
                )
    return edges
