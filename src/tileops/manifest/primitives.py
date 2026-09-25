"""Built-in primitives of the manifest expression language (docs/design/manifest.md § Signature).

The set is fixed. Each entry maps a primitive's name, as an expression writes it, to its concrete
implementation on Python values.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

__all__ = ["PRIMITIVES", "PRIMITIVE_KINDS", "namespace", "normalize_axis"]


def normalize_axis(axis: int, rank: int) -> int:
    """At rank 0, `0` and `-1` name the scalar axis; otherwise an axis lies in [-rank, rank)."""
    lo, hi = (-1, 1) if rank == 0 else (-rank, rank)
    if not lo <= axis < hi:
        raise ValueError(f"axis {axis} is out of range for rank {rank}")
    return 0 if rank == 0 else axis % rank


def _axes(dim) -> list[int]:
    return [dim] if isinstance(dim, int) else list(dim)


def broadcast(*shapes):
    rank = max(len(s) for s in shapes)
    out = []
    for i in range(rank):
        dims = {s[len(s) - rank + i] for s in shapes if len(s) - rank + i >= 0} - {1}
        if len(dims) > 1:
            raise ValueError(f"shapes {shapes} are not broadcastable")
        out.append(dims.pop() if dims else 1)
    return tuple(out)


def reduced(shape, dim, keepdim, mode):
    rank = len(shape)
    if dim is None:
        axes = set(range(rank))
    elif not isinstance(dim, int) and len(dim) == 0:
        if mode == "reject":
            raise ValueError("an empty dim is rejected")
        axes = set(range(rank)) if mode == "full" else set()
    else:
        axes = {normalize_axis(d, rank) for d in _axes(dim)}
    if rank == 0:
        return ()
    return tuple(1 if i in axes else s for i, s in enumerate(shape) if keepdim or i not in axes)


def valid_axes(dim, rank):
    try:
        if dim is not None:
            [normalize_axis(d, rank) for d in _axes(dim)]
    except ValueError:
        return False
    return True


def unique_axes(dim, rank):
    if dim is None or isinstance(dim, int):
        return True
    return len({normalize_axis(d, rank) for d in dim}) == len(dim)


def per_axis(value, i, n, fallback=None):
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"expected {n} values, got {value}")
        value = value[i]
    if value is None:
        if fallback is None:
            raise ValueError("per_axis: the value and its fallback are both None")
        return fallback
    return value


def ceil_div(a, b):
    if b <= 0:
        raise ValueError(f"ceil_div needs a positive divisor, got {b}")
    return -(-a // b)


def _seq_extreme(fn):
    def extreme(values, default=None):
        values = list(values)
        if not values:
            if default is None:
                raise ValueError(f"{fn.__name__} of an empty sequence needs a default")
            return default
        return fn(values)

    return extreme


def conv_out(length, kernel, stride, padding, dilation):
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def pool_out(length, kernel, stride, padding, dilation, ceil_mode):
    span = dilation * (kernel - 1) + 1
    out = (length + 2 * padding - span + (stride - 1 if ceil_mode else 0)) // stride + 1
    if ceil_mode and (out - 1) * stride >= length + padding:
        out -= 1
    return max(out, 0)


def moe_capacity(layout, rows, experts):
    if layout.kind == "masked":
        return experts * layout.max_m
    if layout.metadata_kind == "per_row" and layout.packing == "aligned":
        a = layout.alignment
        return ceil_div(rows + experts * (a - 1), a) * a
    return rows


def promote_int_to_float(dtype):
    return "float32" if dtype in ("uint8", "int8", "int16", "int32", "int64") else dtype


def coalesce_dtype(value, dtype):
    return dtype if value is None else value


def mhc_expansion(q):
    n = math.isqrt(q + 1) - 1
    if n <= 0 or n * n + 2 * n != q:
        raise ValueError(f"no positive n satisfies n * n + 2 * n == {q}")
    return n


PRIMITIVES = {
    "broadcast": broadcast,
    "reduced": reduced,
    "valid_axes": valid_axes,
    "unique_axes": unique_axes,
    "per_axis": per_axis,
    "ceil_div": ceil_div,
    "len": len,
    "prod": math.prod,
    "sum": sum,
    "max": _seq_extreme(max),
    "min": _seq_extreme(min),
    "all": all,
    "conv.out": conv_out,
    "pool.out": pool_out,
    "moe.capacity": moe_capacity,
    "mhc.expansion": mhc_expansion,
    "promote_int_to_float": promote_int_to_float,
    "coalesce_dtype": coalesce_dtype,
}


def namespace() -> dict:
    """The names an expression evaluates against: every primitive, dotted ones as attributes."""
    scope: dict = {"__builtins__": {}, "inf": math.inf}
    for name, fn in PRIMITIVES.items():
        head, _, attr = name.partition(".")
        if attr:
            scope.setdefault(head, SimpleNamespace()).__dict__[attr] = fn
        else:
            scope[name] = fn
    return scope


# Argument kinds and result kind of each primitive.
# `Axes` is `Int | Seq[Int] | None`; `name=Kind` is an optional argument passed by position or
# name; a quoted union restricts a string argument to its members.
PRIMITIVE_KINDS: dict[str, tuple[tuple[str, ...], str]] = {
    "broadcast": (("Shape*",), "Shape"),
    "reduced": (("Shape", "Axes", "Bool", "'full' | 'noop' | 'reject'"), "Shape"),
    "valid_axes": (("Axes", "Int"), "Bool"),
    "unique_axes": (("Axes", "Int"), "Bool"),
    "per_axis": (("Int | Seq[Maybe[Int]] | None", "Int", "Int", "fallback=Maybe[Int]"), "Int"),
    "ceil_div": (("Int", "Int"), "Int"),
    "len": (("Seq",), "Dim"),
    "prod": (("Seq[Int]",), "Int"),
    "sum": (("Seq[Int]",), "Int"),
    "max": (("Seq[Int]", "default=Maybe[Int]"), "Int"),
    "min": (("Seq[Int]", "default=Maybe[Int]"), "Int"),
    "all": (("Seq[Bool]",), "Bool"),
    "conv.out": (("Int", "Int", "Int", "Int", "Int"), "Int"),
    "pool.out": (("Int", "Int", "Int", "Int", "Int", "Bool"), "Dim"),
    "moe.capacity": (("ADT", "Int", "Int"), "Dim"),
    "mhc.expansion": (("Int",), "Dim"),
    "promote_int_to_float": (("DType",), "DType"),
    "coalesce_dtype": (("Maybe[DType]", "DType"), "DType"),
}
