"""Built-in primitives, generators and `requires` predicates (docs/design/manifest.md).

The set is fixed. Each entry maps a primitive's name, as an expression writes it, to its concrete
implementation on Python values.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

# The seed both conftests give the global RNG; every private workload RNG derives from it.
WORKLOAD_SEED = 1235

__all__ = [
    "GENERATORS",
    "GENERATOR_KINDS",
    "GENERATOR_RANKS",
    "GENERATOR_SHAPES",
    "PREDICATES",
    "PREDICATE_KINDS",
    "PREDICATE_RANKS",
    "PRIMITIVES",
    "PRIMITIVE_KINDS",
    "RANDOM_GENERATORS",
    "WORKLOAD_SEED",
    "namespace",
    "normalize_axis",
]


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
    if layout.packing == "aligned":
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


def balanced_sizes(total, count):
    if count <= 0 or total < 0:
        raise ValueError(f"balanced_sizes needs count > 0 and total >= 0, got {total}, {count}")
    return [total // count + (i < total % count) for i in range(count)]


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
    "balanced_sizes": balanced_sizes,
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
    "balanced_sizes": (("Int", "Int"), "Seq[Int]"),
}


# ---------------------------------------------------------------- generators


def as_tensor(values):
    if any(v < 0 for v in values):
        raise ValueError(f"as_tensor needs a non-negative list, got {values}")
    return list(values)


def prefix_sum(lengths):
    out = [0]
    for n in as_tensor(lengths):
        out.append(out[-1] + n)
    return out


def exclusive_prefix_sum(lengths):
    if not lengths:
        raise ValueError("exclusive_prefix_sum needs a non-empty list")
    return prefix_sum(lengths)[:-1]


def padded_exclusive_prefix_sum(lengths, pad):
    if pad <= 0:
        raise ValueError(f"pad must be positive, got {pad}")
    return exclusive_prefix_sum([ceil_div(n + 1, pad) * pad for n in as_tensor(lengths)])


def chunk_indices(lengths, chunk):
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    return [[i, j] for i, n in enumerate(as_tensor(lengths)) for j in range(ceil_div(n, chunk))]


def token_indices(lengths):
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError(f"token_indices needs a non-empty positive list, got {lengths}")
    return [[i, j] for i, n in enumerate(lengths) for j in range(n)]


def chunk_offsets(lengths, chunk):
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    return prefix_sum([ceil_div(n, chunk) for n in as_tensor(lengths)])


def paged_block_table(rng, batch, width, pool):
    if not 0 < width <= pool:
        raise ValueError(f"paged_block_table needs 0 < width <= pool, got {width}, {pool}")
    if pool >= batch * width:
        pages = rng.sample(range(pool), batch * width)
        return [pages[b * width : (b + 1) * width] for b in range(batch)]
    return [rng.sample(range(pool), width) for _ in range(batch)]


def nsa_block_indices(rng, lengths, block_size, selected, heads_kv):
    if not lengths or any(n <= 0 for n in lengths) or min(block_size, selected, heads_kv) <= 0:
        raise ValueError("nsa_block_indices needs positive arguments and a non-empty positive list")
    sentinel = sum(lengths)
    rows = []
    for n in lengths:
        for j in range(n):
            visible = max(ceil_div(j, block_size), 1)
            per_head = []
            for _ in range(heads_kv):
                picked = sorted(rng.sample(range(visible), min(selected, visible)))
                per_head.append(picked + [sentinel] * (selected - len(picked)))
            rows.append(per_head)
    return rows


def nsa_block_counts(rng, tokens, heads_kv, selected):
    if min(tokens, heads_kv, selected) <= 0:
        raise ValueError("nsa_block_counts needs positive arguments")
    return [[rng.randint(1, selected) for _ in range(heads_kv)] for _ in range(tokens)]


def topk_ids(rng, rows, k, experts):
    if not 0 < k <= experts:
        raise ValueError(f"topk_ids needs 0 < K <= E, got K={k}, E={experts}")
    return [rng.sample(range(experts), k) for _ in range(rows)]


# Deterministic generators take no RNG; pseudo-random ones take it as their first argument.
GENERATORS = {
    "as_tensor": as_tensor,
    "prefix_sum": prefix_sum,
    "exclusive_prefix_sum": exclusive_prefix_sum,
    "padded_exclusive_prefix_sum": padded_exclusive_prefix_sum,
    "chunk_indices": chunk_indices,
    "token_indices": token_indices,
    "chunk_offsets": chunk_offsets,
    "paged_block_table": paged_block_table,
    "nsa_block_indices": nsa_block_indices,
    "nsa_block_counts": nsa_block_counts,
    "topk_ids": topk_ids,
}
# Argument kinds of each generator; a pseudo-random one's RNG is not an argument.
GENERATOR_KINDS: dict[str, tuple[tuple[str, ...], str]] = {
    "as_tensor": (("Seq[Int]",), "Value"),
    "prefix_sum": (("Seq[Int]",), "Value"),
    "exclusive_prefix_sum": (("Seq[Int]",), "Value"),
    "padded_exclusive_prefix_sum": (("Seq[Int]", "Int"), "Value"),
    "paged_block_table": (("Int", "Int", "Int"), "Value"),
    "chunk_indices": (("Seq[Int]", "Int"), "Value"),
    "token_indices": (("Seq[Int]",), "Value"),
    "chunk_offsets": (("Seq[Int]", "Int"), "Value"),
    "nsa_block_indices": (("Seq[Int]", "Int", "Int", "Int"), "Value"),
    "nsa_block_counts": (("Int", "Int", "Int"), "Value"),
    "topk_ids": (("Int", "Int", "Int"), "Value"),
}
# The rank of each generator's result.
GENERATOR_RANKS = {
    "as_tensor": 1,
    "prefix_sum": 1,
    "exclusive_prefix_sum": 1,
    "padded_exclusive_prefix_sum": 1,
    "paged_block_table": 2,
    "chunk_indices": 2,
    "token_indices": 2,
    "chunk_offsets": 1,
    "nsa_block_indices": 3,
    "nsa_block_counts": 2,
    "topk_ids": 2,
}
# The shape of each generator's result, from its arguments.
GENERATOR_SHAPES = {
    "as_tensor": lambda L: (len(L),),
    "prefix_sum": lambda L: (len(L) + 1,),
    "exclusive_prefix_sum": lambda L: (len(L),),
    "padded_exclusive_prefix_sum": lambda L, pad: (len(L),),
    "paged_block_table": lambda batch, width, pool: (batch, width),
    "chunk_indices": lambda L, c: (sum(ceil_div(n, c) for n in L), 2),
    "token_indices": lambda L: (sum(L), 2),
    "chunk_offsets": lambda L, c: (len(L) + 1,),
    "nsa_block_indices": lambda L, block, selected, heads: (sum(L), heads, selected),
    "nsa_block_counts": lambda tokens, heads, selected: (tokens, heads),
    "topk_ids": lambda rows, k, experts: (rows, k),
}
RANDOM_GENERATORS = frozenset(
    {"paged_block_table", "nsa_block_indices", "nsa_block_counts", "topk_ids"}
)


# ---------------------------------------------------------------- requires predicates


def _flat(values):
    return [y for x in values for y in _flat(x)] if isinstance(values, list) else [values]


def prefix_offsets(x, total):
    return (
        bool(x)
        and x[0] == 0
        and all(a <= b for a, b in zip(x, x[1:], strict=False))
        and x[-1] == total
    )


def max_segment(x, bound):
    return all(b - a <= bound for a, b in zip(x, x[1:], strict=False))


def in_range(x, lo, hi):
    return all(lo <= v < hi for v in _flat(x))


def paged_fits(x, cu, cap):
    flat = all(isinstance(v, int) for v in (*x, *cu))
    return (
        flat
        and len(cu) == len(x) + 1
        and all(c + (cu[i + 1] - cu[i]) <= cap for i, c in enumerate(x))
    )


# Argument kinds of each predicate after the constrained tensor's contents.
PREDICATE_KINDS: dict[str, tuple[tuple[str, ...], str]] = {
    "prefix_offsets": (("Int",), "Bool"),
    "max_segment": (("Int",), "Bool"),
    "in_range": (("Int", "Int"), "Bool"),
    "attn.paged_fits": (("Seq[Int]", "Int"), "Bool"),
}
# The rank a predicate reads its constrained tensor at; `in_range` reads any.
PREDICATE_RANKS = {"prefix_offsets": 1, "max_segment": 1, "attn.paged_fits": 1}
# The constrained tensor's contents are each predicate's first argument.
PREDICATES = {
    "prefix_offsets": prefix_offsets,
    "max_segment": max_segment,
    "in_range": in_range,
    "attn.paged_fits": paged_fits,
}
