"""Broadcast shape lowering for elementwise kernels: the plan, its name, and its offsets."""

from dataclasses import dataclass

import torch


def _flat(t):
    """The flat view every PrimFunc here takes."""
    return t.reshape(-1)


def _broadcast_target(*tensors):
    """The output shape a multi-operand kernel writes, from the operands it got."""
    return torch.broadcast_shapes(*(tuple(t.shape) for t in tensors if t is not None))


def _expand_flat(t, shape):
    """Broadcast *t* to *shape*, then flatten it."""
    if tuple(t.shape) != tuple(shape):
        t = t.expand(shape)
    return t.contiguous().reshape(-1)


def coalesce_broadcast_dims(a_shape, b_shape):
    """Coalesce N-dim broadcast into minimal effective dimensions."""
    if len(a_shape) == 0:
        a_shape = (1,)
    if len(b_shape) == 0:
        b_shape = (1,)

    out_shape = torch.broadcast_shapes(a_shape, b_shape)
    ndim = len(out_shape)
    a_pad = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b_pad = (1,) * (ndim - len(b_shape)) + tuple(b_shape)

    def _make_strides(padded_shape):
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * padded_shape[i + 1]
        return [0 if padded_shape[i] == 1 and out_shape[i] > 1 else strides[i] for i in range(ndim)]

    a_raw = _make_strides(a_pad)
    b_raw = _make_strides(b_pad)

    groups = [(out_shape[0], a_raw[0], b_raw[0])]
    for i in range(1, ndim):
        prev_out, prev_as, prev_bs = groups[-1]
        a_can = (a_raw[i] == 0 and prev_as == 0) or (
            a_raw[i] != 0 and prev_as == a_raw[i] * out_shape[i]
        )
        b_can = (b_raw[i] == 0 and prev_bs == 0) or (
            b_raw[i] != 0 and prev_bs == b_raw[i] * out_shape[i]
        )
        if a_can and b_can:
            groups[-1] = (prev_out * out_shape[i], a_raw[i], b_raw[i])
        else:
            groups.append((out_shape[i], a_raw[i], b_raw[i]))

    groups = [g for g in groups if g[0] > 1] or [(1, 0, 0)]
    coalesced_shape = tuple(g[0] for g in groups)
    a_strides = tuple(g[1] for g in groups)
    b_strides = tuple(g[2] for g in groups)
    return out_shape, coalesced_shape, a_strides, b_strides


def _compute_broadcast_offsets(flat_idx, ndim, divisors, a_strides, b_strides):
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


def _is_contiguous_same_shape(coalesced_shape, a_strides, b_strides):
    """Return True when both inputs are contiguous with the same shape (no broadcast)."""
    return (
        len(coalesced_shape) == 1
        and all(s == 1 for s in a_strides)
        and all(s == 1 for s in b_strides)
    )


# CUDA caps grid.y at 65535, and one grid axis carries the rows.
_CUDA_MAX_GRID_Y = 65535


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
    if rows > _CUDA_MAX_GRID_Y:
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
