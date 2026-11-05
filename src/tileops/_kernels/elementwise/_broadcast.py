"""Broadcast shape lowering for elementwise kernels: the plan, its name, and its offsets."""

from dataclasses import dataclass

import torch

from tileops._kernels.constants import MAX_GRID_YZ_BLOCKS


def flat(t):
    """The flat view every PrimFunc here takes."""
    return t.reshape(-1)


def broadcast_target(*tensors):
    """The output shape a multi-operand kernel writes, from the operands it got."""
    return torch.broadcast_shapes(*(tuple(t.shape) for t in tensors if t is not None))


def expand_flat(t, shape):
    """Broadcast *t* to *shape*, then flatten it."""
    if tuple(t.shape) != tuple(shape):
        t = t.expand(shape)
    return t.contiguous().reshape(-1)


def coalesce_broadcast_dims(a_shape, b_shape):
    """Coalesce N-dim broadcast into minimal effective dimensions."""
    out_shape, coalesced_shape, (a_strides, b_strides) = coalesce_broadcast_operands(
        a_shape, b_shape
    )
    return out_shape, coalesced_shape, a_strides, b_strides


def coalesce_broadcast_operands(*shapes):
    """Coalesce the broadcast of any number of operand shapes into minimal dimensions.

    Returns ``(out_shape, coalesced_shape, strides)``: ``strides`` holds one tuple per
    operand, its element stride along each coalesced dimension, 0 where it broadcasts.
    Adjacent dimensions merge when every operand walks them as one.
    """
    shapes = [tuple(s) or (1,) for s in shapes]
    out_shape = torch.broadcast_shapes(*shapes)
    ndim = len(out_shape)
    padded = [(1,) * (ndim - len(s)) + s for s in shapes]

    def _make_strides(padded_shape):
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * padded_shape[i + 1]
        return [0 if padded_shape[i] == 1 and out_shape[i] > 1 else strides[i] for i in range(ndim)]

    raw = [_make_strides(p) for p in padded]

    groups = [(out_shape[0], [r[0] for r in raw])]
    for i in range(1, ndim):
        prev_out, prev = groups[-1]
        here = [r[i] for r in raw]
        if all(
            (c == 0 and p == 0) or (c != 0 and p == c * out_shape[i])
            for c, p in zip(here, prev, strict=True)
        ):
            groups[-1] = (prev_out * out_shape[i], here)
        else:
            groups.append((out_shape[i], here))

    groups = [g for g in groups if g[0] > 1] or [(1, [0] * len(shapes))]
    coalesced_shape = tuple(g[0] for g in groups)
    strides = tuple(tuple(g[1][k] for g in groups) for k in range(len(shapes)))
    return out_shape, coalesced_shape, strides


def strided_offset(flat_idx, divisors, strides):
    """One operand's element offset for output element *flat_idx*: a compile-time divmod chain.

    *divisors* are the coalesced shape's row-major strides, *strides* the operand's own.
    """
    offset = 0
    remaining = flat_idx
    for d in range(len(divisors) - 1):
        offset = offset + (remaining // divisors[d]) * strides[d]
        remaining = remaining % divisors[d]
    return offset + remaining * strides[-1]


def compute_broadcast_offsets(flat_idx, ndim, divisors, a_strides, b_strides):
    """Compute a_off and b_off from flat_idx using compile-time unrolled divmod chain."""
    a_off = 0
    b_off = 0
    remaining = flat_idx
    for d in range(ndim - 1):
        coord = remaining // divisors[d]
        remaining = remaining % divisors[d]
        a_off = a_off + coord * a_strides[d]
        b_off = b_off + coord * b_strides[d]
    a_off = a_off + remaining * a_strides[ndim - 1]
    b_off = b_off + remaining * b_strides[ndim - 1]
    return a_off, b_off


def row_tile_leaves_tail(inner, rows, threads, num_per_thread, staged):
    """Whether a row-broadcast block leaves columns to the one-element-per-thread path.

    A tile that divides the row leaves none. Otherwise the grid indexes vectors,
    which leaves none either, so long as a vector stays inside a row and, for the
    staged form, every block is full.
    """
    if inner % (threads * num_per_thread) == 0:
        return False
    if inner % num_per_thread:
        return True
    return bool(staged and (rows * (inner // num_per_thread)) % threads)


def is_contiguous_same_shape(coalesced_shape, a_strides, b_strides):
    """Return True when both inputs are contiguous with the same shape (no broadcast)."""
    return (
        len(coalesced_shape) == 1
        and all(s == 1 for s in a_strides)
        and all(s == 1 for s in b_strides)
    )


def row_broadcast_split(coalesced_shape, a_strides, b_strides):
    """``(rows, inner)`` when the innermost coalesced dim reads at stride 0 or 1.

    That layout gives every row a constant offset into each operand, so a
    kernel can pay the divmod chain once per block and keep the inner loop
    affine — which is what lets it vectorize. Returns ``None`` for layouts
    this does not cover (an inner dim read at a wider stride, or more rows
    than a grid axis holds), which keep the per-element offset path.
    """
    if len(coalesced_shape) < 2:
        return None
    if a_strides[-1] not in (0, 1) or b_strides[-1] not in (0, 1):
        return None
    inner = coalesced_shape[-1]
    rows = 1
    for d in coalesced_shape[:-1]:
        rows *= d
    # The rows run along grid y.
    if rows > MAX_GRID_YZ_BLOCKS:
        return None
    return rows, inner


@dataclass(frozen=True)
class BroadcastPlan:
    """The coalesced shape and per-operand strides a binary kernel indexes with."""

    coalesced_shape: tuple[int, ...]
    a_strides: tuple[int, ...]
    b_strides: tuple[int, ...]


_BROADCAST_PLANS: dict[str, "BroadcastPlan"] = {}


def register_broadcast_plan(plan: BroadcastPlan) -> str:
    """Bind *plan* to a name derived from its own contents, and return that name."""
    name = f"{plan.coalesced_shape}|{plan.a_strides}|{plan.b_strides}"
    _BROADCAST_PLANS[name] = plan
    return name


def broadcast_plan_for(name: str) -> BroadcastPlan:
    """The plan registered under *name*."""
    return _BROADCAST_PLANS[name]
