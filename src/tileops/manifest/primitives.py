"""Built-in primitives, generators and `requires` predicates (docs/design/manifest.md).

This module is the list of each closed set. Each table maps a member's name, as an expression
writes it, to its implementation on Python values, whose docstring states what it computes.
"""

from __future__ import annotations

import math
import numbers
from types import SimpleNamespace

from tileops.manifest.dtype_rules import DTYPE_BITS, DTYPE_CATEGORY

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


# The largest finite value of each floating dtype; the lowest is its negation.
_FLOAT_MAX = {
    "float16": 65504.0,
    "bfloat16": 3.3895313892515355e38,
    "float32": 3.4028234663852886e38,
    "float64": 1.7976931348623157e308,
    "float8_e4m3fn": 448.0,
    "float8_e4m3": 240.0,
    "float8_e5m2": 57344.0,
    "float8_e4m3fnuz": 240.0,
    "float8_e5m2fnuz": 57344.0,
}
_COMPLEX_PART = {"complex64": "float32", "complex128": "float64"}
# Floating formats without an infinity.
_NO_INF = frozenset({"float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz"})


def normalize_axis(axis: int, rank: int) -> int:
    """At rank 0, `0` and `-1` name the scalar axis; otherwise an axis lies in [-rank, rank)."""
    lo, hi = (-1, 1) if rank == 0 else (-rank, rank)
    if not lo <= axis < hi:
        raise ValueError(f"axis {axis} is out of range for rank {rank}")
    return 0 if rank == 0 else axis % rank


def _axes(dim) -> list[int]:
    return [dim] if isinstance(dim, int) else list(dim)


def broadcast(*shapes):
    """PyTorch broadcasting of the shapes; raises when they do not broadcast."""
    rank = max(len(s) for s in shapes)
    out = []
    for i in range(rank):
        dims = {s[len(s) - rank + i] for s in shapes if len(s) - rank + i >= 0} - {1}
        if len(dims) > 1:
            raise ValueError(f"shapes {shapes} are not broadcastable")
        out.append(dims.pop() if dims else 1)
    return tuple(out)


def reduced(shape, dim, keepdim, mode):
    """The shape after reducing `dim`; `None` reduces every axis, an empty sequence follows `mode` (all, none, raise)."""
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
    """Whether every axis normalizes at `rank`; `None` is valid."""
    try:
        if dim is not None:
            [normalize_axis(d, rank) for d in _axes(dim)]
    except ValueError:
        return False
    return True


def unique_axes(dim, rank):
    """Whether the normalized axes are distinct."""
    if dim is None or isinstance(dim, int):
        return True
    return len({normalize_axis(d, rank) for d in dim}) == len(dim)


def per_axis(value, i, n, fallback=None):
    """Item `i` of a length-`n` sequence, a scalar itself, or `fallback` for `None`."""
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
    """`a / b` rounded up, for a positive `b`."""
    if b <= 0:
        raise ValueError(f"ceil_div needs a positive divisor, got {b}")
    return -(-a // b)


def _prod(values):
    """The product of the values; 1 for none."""
    return math.prod(values)


def _sum(values):
    """The sum of the values; 0 for none."""
    return sum(values)


def _all(values):
    """Whether every value holds; true for none."""
    return all(values)


def _seq_extreme(fn):
    def extreme(values, default=None):
        """The extreme of the values, or `default` when there are none."""
        values = list(values)
        if not values:
            if default is None:
                raise ValueError(f"{fn.__name__} of an empty sequence needs a default")
            return default
        return fn(values)

    return extreme


def conv_out(length, kernel, stride, padding, dilation):
    """The convolution output length."""
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def pool_out(length, kernel, stride, padding, dilation, ceil_mode):
    """The pooling output length, never negative."""
    span = dilation * (kernel - 1) + 1
    out = (length + 2 * padding - span + (stride - 1 if ceil_mode else 0)) // stride + 1
    if ceil_mode and (out - 1) * stride >= length + padding:
        out -= 1
    return max(out, 0)


def moe_capacity(layout, rows, experts):
    """Rows the layout materializes: `E * max_m` masked, `R + E * (alignment - 1)` rounded up to `alignment` aligned, else `R`."""
    if layout.kind == "masked":
        return experts * layout.max_m
    if layout.packing == "aligned":
        a = layout.alignment
        return ceil_div(rows + experts * (a - 1), a) * a
    return rows


def promote_int_to_float(dtype):
    """float32 for an integral dtype, else the dtype itself."""
    return "float32" if category(dtype) == "int" else dtype


def coalesce_dtype(value, dtype):
    """`value` when present, else `dtype`."""
    return dtype if value is None else value


def mhc_expansion(q):
    """The positive `n` with `n * n + 2 * n == q`."""
    n = math.isqrt(q + 1) - 1
    if n <= 0 or n * n + 2 * n != q:
        raise ValueError(f"no positive n satisfies n * n + 2 * n == {q}")
    return n


def balanced_sizes(total, count):
    """`count` sizes summing to `total`, each `total // count`, the first `total % count` one larger."""
    if count <= 0 or total < 0:
        raise ValueError(f"balanced_sizes needs count > 0 and total >= 0, got {total}, {count}")
    return [total // count + (i < total % count) for i in range(count)]


def repeat(value, count):
    """`count` copies of the integer `value`."""
    if count < 0 or not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"repeat needs an integer value and count >= 0, got {value!r}, {count}")
    return [value] * count


def category(x):
    """`'bool'`, `'int'`, `'float'` or `'complex'`: the category of a number or a dtype name."""
    if isinstance(x, str):
        return DTYPE_CATEGORY.get(x, "int")
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, numbers.Integral):
        return "int"
    return "float" if isinstance(x, numbers.Real) else "complex"


def _fits_float(v, dtype):
    if math.isinf(v):
        return dtype not in _NO_INF
    return math.isnan(v) or -_FLOAT_MAX[dtype] <= v <= _FLOAT_MAX[dtype]


def representable(v, dtype):
    """Whether `v` converts to `dtype` without overflow, by PyTorch's scalar conversion rule."""
    if dtype == "bool":
        return True
    if dtype in _COMPLEX_PART:
        part = _COMPLEX_PART[dtype]
        return _fits_float(complex(v).real, part) and _fits_float(complex(v).imag, part)
    if isinstance(v, numbers.Complex) and not isinstance(v, numbers.Real):
        return False
    if dtype in _FLOAT_MAX:
        return _fits_float(v, dtype)
    bits = DTYPE_BITS[dtype]
    lo, hi = (
        (0, 2**bits - 1) if dtype.startswith("uint") else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    )
    if isinstance(v, numbers.Integral):
        # An unsigned dtype also takes a negative int it can wrap.
        return (-hi if lo == 0 else lo) <= v <= hi
    return math.isfinite(v) and lo <= v <= hi


PRIMITIVES = {
    "broadcast": broadcast,
    "reduced": reduced,
    "valid_axes": valid_axes,
    "unique_axes": unique_axes,
    "per_axis": per_axis,
    "ceil_div": ceil_div,
    "len": len,
    "prod": _prod,
    "sum": _sum,
    "max": _seq_extreme(max),
    "min": _seq_extreme(min),
    "all": _all,
    "conv.out": conv_out,
    "pool.out": pool_out,
    "moe.capacity": moe_capacity,
    "mhc.expansion": mhc_expansion,
    "promote_int_to_float": promote_int_to_float,
    "coalesce_dtype": coalesce_dtype,
    "balanced_sizes": balanced_sizes,
    "repeat": repeat,
    "category": category,
    "representable": representable,
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
    "repeat": (("Int", "Int"), "Seq[Int]"),
    "category": (("Value",), "'bool' | 'int' | 'float' | 'complex'"),
    "representable": (("Value", "DType"), "Bool"),
}


# ---------------------------------------------------------------- generators


def as_tensor(values):
    """The non-negative list itself."""
    if any(v < 0 for v in values):
        raise ValueError(f"as_tensor needs a non-negative list, got {values}")
    return list(values)


def prefix_sum(lengths):
    """`[0]` followed by the running sums of the lengths."""
    out = [0]
    for n in as_tensor(lengths):
        out.append(out[-1] + n)
    return out


def exclusive_prefix_sum(lengths):
    """Item `i` is the sum of the lengths before it."""
    if not lengths:
        raise ValueError("exclusive_prefix_sum needs a non-empty list")
    return prefix_sum(lengths)[:-1]


def chunk_indices(lengths, chunk):
    """One `(request, chunk)` row per `chunk`-sized piece of each length."""
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    return [[i, j] for i, n in enumerate(as_tensor(lengths)) for j in range(ceil_div(n, chunk))]


def token_indices(lengths):
    """One `(sequence, position)` row per token."""
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError(f"token_indices needs a non-empty positive list, got {lengths}")
    return [[i, j] for i, n in enumerate(lengths) for j in range(n)]


def packed_positions(lengths):
    """Positions restarting at 0 for each sequence."""
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError(f"packed_positions needs a non-empty positive list, got {lengths}")
    return [j for n in lengths for j in range(n)]


def chunk_offsets(lengths, chunk):
    """The prefix sum of each length's chunk count."""
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    return prefix_sum([ceil_div(n, chunk) for n in as_tensor(lengths)])


def paged_block_table(rng, batch, width, pool):
    """Disjoint random pages per request when the pool holds them all, else the first `width` of a permutation per request."""
    if not 0 < width <= pool:
        raise ValueError(f"paged_block_table needs 0 < width <= pool, got {width}, {pool}")
    if pool >= batch * width:
        pages = rng.sample(range(pool), batch * width)
        return [pages[b * width : (b + 1) * width] for b in range(batch)]
    return [rng.sample(range(pool), width) for _ in range(batch)]


def nsa_block_indices(rng, lengths, block_size, selected, heads_kv):
    """Position `j` sees `max(ceil_div(j, block_size), 1)` blocks and draws up to `selected` distinct ones per head, ascending, padded with the sentinel `sum(lengths)`."""
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
    """Each count uniform in `[1, selected]`."""
    if min(tokens, heads_kv, selected) <= 0:
        raise ValueError("nsa_block_counts needs positive arguments")
    return [[rng.randint(1, selected) for _ in range(heads_kv)] for _ in range(tokens)]


def topk_ids(rng, rows, k, experts):
    """`rows` rows of `k` distinct random values in `[0, experts)`."""
    if not 0 < k <= experts:
        raise ValueError(f"topk_ids needs 0 < K <= E, got K={k}, E={experts}")
    return [rng.sample(range(experts), k) for _ in range(rows)]


def random_ids(rng, rows, width, upper):
    """Rows of independent indices in [0, upper), allowing repeated tokens."""
    if min(rows, width, upper) <= 0:
        raise ValueError("random_ids needs positive arguments")
    return [[rng.randrange(upper) for _ in range(width)] for _ in range(rows)]


def sample_indices(rng, n, hi):
    """`n` distinct random values in `[0, hi)`."""
    if not 0 <= n <= hi:
        raise ValueError(f"sample_indices needs 0 <= n <= hi, got n={n}, hi={hi}")
    return rng.sample(range(hi), n)


def causal_topk_indices(rng, batch, seq, heads_kv, k, extent, start, stride):
    """Per token and head, up to `k` distinct keys it can see, padded with `extent`."""
    if min(batch, seq, heads_kv, k, extent, stride) <= 0 or start < 0:
        raise ValueError(
            "causal_topk_indices needs positive B, S, H_kv, K, E, stride and start >= 0"
        )
    rows = []
    for _ in range(batch):
        per_token = []
        for t in range(seq):
            visible = min(max(1, (t + start) // stride), extent)
            per_head = []
            for _ in range(heads_kv):
                picked = rng.sample(range(visible), min(k, visible))
                per_head.append(picked + [extent] * (k - len(picked)))
            per_token.append(per_head)
        rows.append(per_token)
    return rows


def sparse_topk_positions(rng, lengths, queries, k):
    """Per request of length `n` and query `s`, up to `k` distinct positions in `[0, n - queries + s]`, padded with -1."""
    if queries <= 0 or k <= 0 or any(n < queries for n in lengths):
        raise ValueError(
            f"attn.sparse_topk_positions needs S_q > 0, K > 0 and every length >= S_q, "
            f"got lengths={lengths}, S_q={queries}, K={k}"
        )
    rows = []
    for n in lengths:
        per_query = []
        for s in range(queries):
            visible = n - queries + s + 1
            picked = rng.sample(range(visible), min(k, visible))
            per_query.append(picked + [-1] * (k - len(picked)))
        rows.append(per_query)
    return rows


def key_windows(lengths, first, count, side):
    """Per query position, the first key of its sequence (`start`) or one past itself (`end`)."""
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError(f"key_windows needs a non-empty positive list, got {lengths}")
    if first < 0 or count < 0 or first + count > sum(lengths):
        raise ValueError(
            f"key_windows needs 0 <= first and first + count <= {sum(lengths)}, "
            f"got first={first}, count={count}"
        )
    if side not in ("start", "end"):
        raise ValueError(f"key_windows side must be 'start' or 'end', got {side!r}")
    starts = prefix_sum(lengths)
    out, segment = [], 0
    for p in range(first, first + count):
        while starts[segment + 1] <= p:
            segment += 1
        out.append(starts[segment] if side == "start" else p + 1)
    return out


def full(shape, value):
    """A nested list of `shape` holding `value`."""
    if any(n < 0 for n in shape):
        raise ValueError(f"full needs non-negative extents, got {shape}")

    def fill(axes):
        return value if not axes else [fill(axes[1:]) for _ in range(axes[0])]

    return fill(list(shape))


def _segment_ids(sizes, scale=1):
    return [i for i, n in enumerate(sizes) for _ in range(n * scale)]


def moe_layout_metadata(layout, rows, experts):
    """Metadata of `layout` for `rows` rows split across `experts` as evenly as the layout allows."""
    if experts <= 0 or rows < 0:
        raise ValueError(f"moe.layout_metadata needs E > 0 and R >= 0, got R={rows}, E={experts}")
    if layout.kind == "masked":
        if rows != experts * layout.max_m:
            raise ValueError(f"masked metadata needs R == E * max_m, got R={rows}")
        return [layout.max_m if i % 2 == 0 else layout.max_m // 2 for i in range(experts)]
    per_row = layout.metadata_kind == "per_row"
    if layout.packing == "tight":
        sizes = balanced_sizes(rows, experts)
        return _segment_ids(sizes) if per_row else prefix_sum(sizes)[1:]
    a = layout.alignment
    if rows % a:
        raise ValueError(f"aligned metadata needs R % alignment == 0, got R={rows}, alignment={a}")
    tiles = balanced_sizes(rows // a, experts)
    if per_row:
        return _segment_ids(tiles, a)
    starts = prefix_sum(tiles)
    return [a * starts[i] + (a * t - a // 2 if t > 0 else 0) for i, t in enumerate(tiles)]


# Deterministic generators take no RNG; pseudo-random ones take it as their first argument.
GENERATORS = {
    "as_tensor": as_tensor,
    "prefix_sum": prefix_sum,
    "exclusive_prefix_sum": exclusive_prefix_sum,
    "chunk_indices": chunk_indices,
    "token_indices": token_indices,
    "packed_positions": packed_positions,
    "chunk_offsets": chunk_offsets,
    "paged_block_table": paged_block_table,
    "nsa_block_indices": nsa_block_indices,
    "nsa_block_counts": nsa_block_counts,
    "topk_ids": topk_ids,
    "random_ids": random_ids,
    "sample_indices": sample_indices,
    "moe.layout_metadata": moe_layout_metadata,
    "causal_topk_indices": causal_topk_indices,
    "attn.sparse_topk_positions": sparse_topk_positions,
    "key_windows": key_windows,
    "full": full,
}
# Argument kinds of each generator; a pseudo-random one's RNG is not an argument.
GENERATOR_KINDS: dict[str, tuple[tuple[str, ...], str]] = {
    "as_tensor": (("Seq[Int]",), "Value"),
    "prefix_sum": (("Seq[Int]",), "Value"),
    "exclusive_prefix_sum": (("Seq[Int]",), "Value"),
    "paged_block_table": (("Int", "Int", "Int"), "Value"),
    "chunk_indices": (("Seq[Int]", "Int"), "Value"),
    "token_indices": (("Seq[Int]",), "Value"),
    "packed_positions": (("Seq[Int]",), "Value"),
    "chunk_offsets": (("Seq[Int]", "Int"), "Value"),
    "nsa_block_indices": (("Seq[Int]", "Int", "Int", "Int"), "Value"),
    "nsa_block_counts": (("Int", "Int", "Int"), "Value"),
    "topk_ids": (("Int", "Int", "Int"), "Value"),
    "random_ids": (("Int", "Int", "Int"), "Value"),
    "sample_indices": (("Int", "Int"), "Value"),
    "moe.layout_metadata": (("ADT", "Int", "Int"), "Value"),
    "causal_topk_indices": (("Int", "Int", "Int", "Int", "Int", "Int", "Int"), "Value"),
    "attn.sparse_topk_positions": (("Seq[Int]", "Int", "Int"), "Value"),
    "key_windows": (("Seq[Int]", "Int", "Int", "'start' | 'end'"), "Value"),
    "full": (("Seq[Int]", "Int"), "Value"),
}
# The rank of each generator's result where it is fixed; `full`'s is the length of its shape.
GENERATOR_RANKS = {
    "as_tensor": 1,
    "prefix_sum": 1,
    "exclusive_prefix_sum": 1,
    "paged_block_table": 2,
    "chunk_indices": 2,
    "token_indices": 2,
    "packed_positions": 1,
    "chunk_offsets": 1,
    "nsa_block_indices": 3,
    "nsa_block_counts": 2,
    "topk_ids": 2,
    "random_ids": 2,
    "sample_indices": 1,
    "moe.layout_metadata": 1,
    "causal_topk_indices": 4,
    "attn.sparse_topk_positions": 3,
    "key_windows": 1,
}
# The shape of each generator's result, from its arguments.
GENERATOR_SHAPES = {
    "as_tensor": lambda L: (len(L),),
    "prefix_sum": lambda L: (len(L) + 1,),
    "exclusive_prefix_sum": lambda L: (len(L),),
    "paged_block_table": lambda batch, width, pool: (batch, width),
    "chunk_indices": lambda L, c: (sum(ceil_div(n, c) for n in L), 2),
    "token_indices": lambda L: (sum(L), 2),
    "packed_positions": lambda L: (sum(L),),
    "chunk_offsets": lambda L, c: (len(L) + 1,),
    "nsa_block_indices": lambda L, block, selected, heads: (sum(L), heads, selected),
    "nsa_block_counts": lambda tokens, heads, selected: (tokens, heads),
    "topk_ids": lambda rows, k, experts: (rows, k),
    "random_ids": lambda rows, width, upper: (rows, width),
    "sample_indices": lambda n, hi: (n,),
    "moe.layout_metadata": lambda layout, rows, experts: (
        (rows,) if layout.kind == "contiguous" and layout.metadata_kind == "per_row" else (experts,)
    ),
    "causal_topk_indices": lambda batch, seq, heads, k, extent, start, stride: (
        batch,
        seq,
        heads,
        k,
    ),
    "attn.sparse_topk_positions": lambda L, queries, k: (len(L), queries, k),
    "key_windows": lambda L, first, count, side: (count,),
    "full": lambda shape, value: tuple(shape),
}
RANDOM_GENERATORS = frozenset(
    {
        "paged_block_table",
        "nsa_block_indices",
        "nsa_block_counts",
        "topk_ids",
        "random_ids",
        "sample_indices",
        "causal_topk_indices",
        "attn.sparse_topk_positions",
    }
)


# ---------------------------------------------------------------- requires predicates


def _flat(values):
    return [y for x in values for y in _flat(x)] if isinstance(values, list) else [values]


def prefix_offsets(x, total):
    """`x` starts at 0, is non-decreasing and ends at `total`."""
    return (
        bool(x)
        and x[0] == 0
        and all(a <= b for a, b in zip(x, x[1:], strict=False))
        and x[-1] == total
    )


def max_segment(x, bound):
    """Adjacent differences of `x` are at most `bound`."""
    return all(b - a <= bound for a, b in zip(x, x[1:], strict=False))


def in_range(x, lo, hi):
    """Every element of `x` lies in `[lo, hi)`."""
    return all(lo <= v < hi for v in _flat(x))


def sums_to(x, total):
    """The elements of `x` sum to `total`."""
    return sum(x) == total


def exclusive_prefix_of(x, lengths):
    """`x` has one element per length; element `i` is the sum of the lengths before it."""
    return len(x) == len(lengths) and all(v == sum(lengths[:i]) for i, v in enumerate(x))


def chunk_offsets_of(x, offsets, chunk):
    """`x` counts, cumulatively, the `chunk`-sized pieces of each segment of `offsets`."""
    if chunk <= 0:
        return False
    sizes = [ceil_div(b - a, chunk) for a, b in zip(offsets, offsets[1:], strict=False)]
    return all(n >= 0 for n in sizes) and x == prefix_sum(sizes)


def indices_within(x, offsets):
    """Each row `(i, j)` of `x` names segment `i` of `offsets` and position `j` inside it."""
    segments = len(offsets) - 1
    return all(
        len(row) == 2
        and 0 <= row[0] < segments
        and 0 <= row[1] < offsets[row[0] + 1] - offsets[row[0]]
        for row in x
    )


def paged_fits(x, cu, cap):
    """Element `i` of `x` plus segment `i` of `cu` is at most `cap`."""
    flat = all(isinstance(v, int) for v in (*x, *cu))
    return (
        flat
        and len(cu) == len(x) + 1
        and all(c + (cu[i + 1] - cu[i]) <= cap for i, c in enumerate(x))
    )


def moe_layout_valid(x, layout, rows, experts):
    """`x` is valid metadata of `layout` for `rows` rows and `experts` experts."""
    ordered = all(a <= b for a, b in zip(x, x[1:], strict=False))
    if layout.kind == "masked":
        return len(x) == experts and all(0 <= v <= layout.max_m for v in x)
    a = layout.alignment
    if layout.metadata_kind == "per_row":
        top = experts + (layout.packing == "aligned")
        changes = all(i % a == 0 for i in range(1, len(x)) if x[i] != x[i - 1])
        return len(x) == rows and ordered and all(0 <= v < top for v in x) and changes
    if len(x) != experts:
        return False
    if layout.packing == "tight":
        return ordered and (not x or (x[0] >= 0 and x[-1] == rows))
    ends = [0, *x]
    return all(ends[i + 1] >= ceil_div(ends[i], a) * a for i in range(len(x))) and (
        not x or x[-1] <= rows
    )


# Argument kinds of each predicate after the constrained tensor's contents.
PREDICATE_KINDS: dict[str, tuple[tuple[str, ...], str]] = {
    "prefix_offsets": (("Int",), "Bool"),
    "max_segment": (("Int",), "Bool"),
    "in_range": (("Int", "Int"), "Bool"),
    "sums_to": (("Int",), "Bool"),
    "exclusive_prefix_of": (("Seq[Int]",), "Bool"),
    "indices_within": (("Seq[Int]",), "Bool"),
    "chunk_offsets_of": (("Seq[Int]", "Int"), "Bool"),
    "attn.paged_fits": (("Seq[Int]", "Int"), "Bool"),
    "moe.layout_valid": (("ADT", "Int", "Int"), "Bool"),
}
# The rank a predicate reads its constrained tensor at; `in_range` reads any.
PREDICATE_RANKS = {
    "prefix_offsets": 1,
    "max_segment": 1,
    "sums_to": 1,
    "exclusive_prefix_of": 1,
    "indices_within": 2,
    "chunk_offsets_of": 1,
    "attn.paged_fits": 1,
    "moe.layout_valid": 1,
}
# The constrained tensor's contents are each predicate's first argument.
PREDICATES = {
    "prefix_offsets": prefix_offsets,
    "max_segment": max_segment,
    "in_range": in_range,
    "sums_to": sums_to,
    "exclusive_prefix_of": exclusive_prefix_of,
    "indices_within": indices_within,
    "chunk_offsets_of": chunk_offsets_of,
    "attn.paged_fits": paged_fits,
    "moe.layout_valid": moe_layout_valid,
}
