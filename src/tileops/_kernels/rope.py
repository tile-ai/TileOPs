"""Rotary position embedding kernels.

A kernel receives cos/sin tables and rotates each token's heads at a table row:
    y = x * cos + rotate(x) * sin

- ``RoPENeoxPositionIdsKernel``: NeoX halves, ``(x[k], x[k + rotary_dim / 2])``.
- ``RoPENonNeoxKernel``: RoFormer (GPT-J) adjacent pairs, ``(x[2k], x[2k + 1])``.

Both rotate the query and the key at the row each token's position names, in one launch,
rotate the leading ``rotary_dim`` columns of each head and copy the rest, and take the token
count from the call: one built kernel serves every count.

- ``RoPECosSinKernel``: the tables are the caller's, one row per sequence position; either
  pairing, dense or packed tokens, at any strides, in place or not.
"""

import dataclasses
import math
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

__all__ = [
    "RoPECall",
    "RoPECosSinCall",
    "RoPECosSinFwdInterface",
    "RoPECosSinKernel",
    "RoPENeoxPositionIdsFwdInterface",
    "RoPENeoxPositionIdsKernel",
    "RoPENonNeoxFwdInterface",
    "RoPENonNeoxKernel",
]


@dataclasses.dataclass(frozen=True)
class RoPECall(CallSpec):
    """One rotation of a query and a key whose heads are flattened, at each token's row.

    The token count is not a fact of the call: a kernel serves any. ``key_heads`` is 0 for a
    call without a key.
    """

    query_heads: int = 0
    key_heads: int = 0
    head_size: int = 0
    rotary_dim: int = 0
    max_position: int = 0
    dtype: torch.dtype | None = None
    position_dtype: torch.dtype = torch.int64
    # None for a call without offsets.
    offset_dtype: torch.dtype | None = None


class RoPEPositionsFwdInterface(KernelInterface):
    """The contract both rotation conventions share; only the pairing differs."""

    request = RoPECall

    @abstractmethod
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        offsets: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Rotate each token's leading rotary columns; nothing is written in place.

        Token ``t`` reads table row ``positions[t] + offsets[t]`` (``positions[t]`` without
        offsets) and rotates the first
        ``call.rotary_dim`` columns of each head of ``call.head_size`` columns; the remaining
        columns are copied. A position outside ``[0, call.max_position)`` is clamped to the
        table rather than read out of bounds. Nothing is written in place, and no output
        aliases an input.

        Args:
            positions: ``[*S]`` in ``call.position_dtype``, ``int32`` or ``int64``, on
                ``call.device``, contiguous.
            query: ``[*S, call.query_heads * call.head_size]`` in ``call.dtype`` on
                ``call.device``, contiguous.
            key: ``[*S, call.key_heads * call.head_size]`` in the same dtype, device and
                layout, or None where ``call.key_heads`` is 0.
            offsets: ``[*S]`` in ``call.offset_dtype``, ``int32`` or ``int64``, on
                ``call.device``, contiguous, or None where ``call.offset_dtype`` is. A call
                with offsets has a key.
            cos: The cosine table, ``(call.max_position, call.rotary_dim // 2)`` in
                ``call.dtype`` on ``call.device``, contiguous, row-major.
            sin: The sine table, in the same shape, dtype, device and layout.

        Returns:
            ``(query_out, key_out)``: new contiguous tensors on ``call.device``, shaped and
            typed as ``query`` and ``key``; ``key_out`` is None where ``key`` is.
        """


class RoPENeoxPositionIdsFwdInterface(RoPEPositionsFwdInterface):
    """The GPT-NeoX rotation: column ``k`` of a head pairs with ``k + rotary_dim / 2``.

    Pair ``k`` becomes ``(x[k] * cos[k] - x[k + h] * sin[k], x[k + h] * cos[k] + x[k] *
    sin[k])`` with ``h = rotary_dim / 2``.
    """


class RoPENonNeoxFwdInterface(RoPEPositionsFwdInterface):
    """The RoFormer (GPT-J) rotation: columns ``2k`` and ``2k + 1`` of a head pair up.

    Pair ``k`` becomes ``(x[2k] * cos[k] - x[2k + 1] * sin[k], x[2k + 1] * cos[k] + x[2k] *
    sin[k])``.
    """


def _head_units(
    neox: bool, head_size: int, rotary_dim: int, rot_vec: int, tail_vec: int
) -> tuple[int, int, int]:
    """A rotating unit's width in columns, and a head's rotating and copying unit counts.

    A GPT-J unit holds whole pairs; one unaligned to a pair-wide vector reads elementwise.
    """
    width = rot_vec if neox else max(rot_vec, 2)
    rotating = (rotary_dim // 2 if neox else rotary_dim) // width
    return width, rotating, (head_size - rotary_dim) // tail_vec


@target_cache(maxsize=32)
def _rope_positions_kernel(
    query_heads: int,
    key_heads: int,
    head_size: int,
    rotary_dim: int,
    max_position: int,
    dtype: str,
    position_dtype: str,
    offset_dtype: str | None,
    neox: bool,
    rot_vec: int,
    tail_vec: int,
) -> object:
    """The rotation of a query and a key of ``tokens`` rows, ``tokens`` dynamic.

    A unit is part of one head: NeoX, ``rot_vec`` columns of the low half and the
    ``rot_vec`` facing them in the high half; GPT-J, ``rot_vec`` adjacent columns, whole
    pairs; or ``tail_vec`` columns past ``rotary_dim``, copied. A token's query units come
    before its key units, so one launch serves both. A block holds ``threads * units`` units
    of one token, ``units`` a thread, and reads the token's position once; consecutive
    blocks walk a token's parts, then the next token's. ``rot_vec`` and ``tail_vec`` divide
    every column offset they start at, so each unit moves as vectors. The clamp sits in the
    table index, so the loads carry no branch on the position. The arithmetic is f32 and
    rounds once.
    """
    half = rotary_dim // 2
    width, rot_units, tail_units = _head_units(neox, head_size, rotary_dim, rot_vec, tail_vec)
    table_width = width if neox else width // 2
    vector_pairs = neox or rot_vec >= 2
    head_units = rot_units + tail_units
    query_units = query_heads * head_units
    token_units = (query_heads + key_heads) * head_units

    @tilelang.jit(out_idx=[6, 7] if offset_dtype else [5, 6] if key_heads else [4])
    def kernel(threads, units):
        num_tokens = T.dynamic("num_tokens")
        query_shape = (num_tokens, query_heads * head_size)
        key_shape = (num_tokens, key_heads * head_size)
        table_shape = (max_position, half)
        parts = -(-token_units // (threads * units))

        @T.macro
        def rotate_unit(row, x, y, token, head, slot, cos_table, sin_table):
            lo = T.alloc_local([width], dtype)
            hi = T.alloc_local([width], dtype)
            cs = T.alloc_local([table_width], dtype)
            sn = T.alloc_local([table_width], dtype)
            kept_run = T.alloc_local([tail_vec], dtype)
            if slot < rot_units:
                col = slot * width
                low = head * head_size + col
                first = col if neox else col // 2
                if vector_pairs:
                    for j in T.vectorized(width):
                        lo[j] = x[token, low + j]
                else:
                    for j in T.serial(width):
                        lo[j] = x[token, low + j]
                if neox:
                    for j in T.vectorized(width):
                        hi[j] = x[token, low + half + j]
                for j in T.vectorized(table_width):
                    cs[j] = cos_table[row, first + j]
                for j in T.vectorized(table_width):
                    sn[j] = sin_table[row, first + j]
                if neox:
                    for j in T.unroll(width):
                        c = T.Cast("float32", cs[j])
                        s = T.Cast("float32", sn[j])
                        x_low = T.Cast("float32", lo[j])
                        x_high = T.Cast("float32", hi[j])
                        lo[j] = T.Cast(dtype, x_low * c - x_high * s)
                        hi[j] = T.Cast(dtype, x_high * c + x_low * s)
                    for j in T.vectorized(width):
                        y[token, low + half + j] = hi[j]
                else:
                    for p in T.unroll(table_width):
                        c = T.Cast("float32", cs[p])
                        s = T.Cast("float32", sn[p])
                        x_even = T.Cast("float32", lo[2 * p])
                        x_odd = T.Cast("float32", lo[2 * p + 1])
                        lo[2 * p] = T.Cast(dtype, x_even * c - x_odd * s)
                        lo[2 * p + 1] = T.Cast(dtype, x_odd * c + x_even * s)
                if vector_pairs:
                    for j in T.vectorized(width):
                        y[token, low + j] = lo[j]
                else:
                    for j in T.serial(width):
                        y[token, low + j] = lo[j]
            elif tail_units > 0:
                kept = head * head_size + rotary_dim + (slot - rot_units) * tail_vec
                for j in T.vectorized(tail_vec):
                    kept_run[j] = x[token, kept + j]
                for j in T.vectorized(tail_vec):
                    y[token, kept + j] = kept_run[j]

        @T.macro
        def rotate_token(
            positions, offsets, query, key, cos_table, sin_table, query_out, key_out, token, part
        ):
            # The position first: the table loads wait on it, and the input loads issued
            # behind it overlap that wait.
            row = T.alloc_local([1], "int64" if offset_dtype else position_dtype)
            if offset_dtype:
                seen = T.Cast("int64", positions[token]) + T.Cast("int64", offsets[token])
                row[0] = T.max(0, T.min(seen, max_position - 1))
            else:
                row[0] = T.max(0, T.min(positions[token], max_position - 1))
            for r in T.unroll(units):
                within = (part * units + r) * threads + T.get_thread_binding()
                if within < query_units:
                    rotate_unit(
                        row[0],
                        query,
                        query_out,
                        token,
                        within // head_units,
                        within % head_units,
                        cos_table,
                        sin_table,
                    )
                elif within < token_units:
                    rotate_unit(
                        row[0],
                        key,
                        key_out,
                        token,
                        (within - query_units) // head_units,
                        (within - query_units) % head_units,
                        cos_table,
                        sin_table,
                    )

        if offset_dtype:

            @T.prim_func
            def main(
                positions: T.Tensor((num_tokens,), position_dtype),
                query: T.Tensor(query_shape, dtype),
                key: T.Tensor(key_shape, dtype),
                offsets: T.Tensor((num_tokens,), offset_dtype),
                cos_table: T.Tensor(table_shape, dtype),
                sin_table: T.Tensor(table_shape, dtype),
                query_out: T.Tensor(query_shape, dtype),
                key_out: T.Tensor(key_shape, dtype),
            ):
                with T.Kernel(num_tokens * parts, threads=threads) as bx:
                    rotate_token(
                        positions,
                        offsets,
                        query,
                        key,
                        cos_table,
                        sin_table,
                        query_out,
                        key_out,
                        bx // parts,
                        bx % parts,
                    )

        elif key_heads:

            @T.prim_func
            def main(
                positions: T.Tensor((num_tokens,), position_dtype),
                query: T.Tensor(query_shape, dtype),
                key: T.Tensor(key_shape, dtype),
                cos_table: T.Tensor(table_shape, dtype),
                sin_table: T.Tensor(table_shape, dtype),
                query_out: T.Tensor(query_shape, dtype),
                key_out: T.Tensor(key_shape, dtype),
            ):
                # Consecutive blocks walk consecutive memory: a token's parts, then the next.
                with T.Kernel(num_tokens * parts, threads=threads) as bx:
                    rotate_token(
                        positions,
                        positions,
                        query,
                        key,
                        cos_table,
                        sin_table,
                        query_out,
                        key_out,
                        bx // parts,
                        bx % parts,
                    )

        else:

            @T.prim_func
            def main(
                positions: T.Tensor((num_tokens,), position_dtype),
                query: T.Tensor(query_shape, dtype),
                cos_table: T.Tensor(table_shape, dtype),
                sin_table: T.Tensor(table_shape, dtype),
                query_out: T.Tensor(query_shape, dtype),
            ):
                with T.Kernel(num_tokens * parts, threads=threads) as bx:
                    rotate_token(
                        positions,
                        positions,
                        query,
                        query,
                        cos_table,
                        sin_table,
                        query_out,
                        query_out,
                        bx // parts,
                        bx % parts,
                    )

        return main

    return kernel


class _RoPEPositionsKernel(Kernel):
    """The rotation at each token's position, one convention per subclass.

    Each thread moves whole vectors: the widest power of two up to 16 bytes dividing every
    column offset its unit starts at.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES
    _NEOX: bool

    @classmethod
    def entry_for(cls, call: RoPECall) -> Entry:
        return call, lambda: cls(
            call.query_heads,
            call.key_heads,
            call.head_size,
            call.rotary_dim,
            call.max_position,
            call.dtype,
            position_dtype=call.position_dtype,
            offset_dtype=call.offset_dtype,
            resident_threads=call.sm_count * call.threads_per_sm,
        )

    def __init__(
        self,
        query_heads: int,
        key_heads: int,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        dtype: torch.dtype,
        config: dict | None = None,
        *,
        position_dtype: torch.dtype = torch.int64,
        offset_dtype: torch.dtype | None = None,
        resident_threads: int = 0,
    ):
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > head_size:
            raise ValueError("rotary_dim must be positive, even, and <= head_size")
        if query_heads <= 0 or key_heads < 0:
            raise ValueError(
                "query_heads must be positive and key_heads not negative, got "
                f"{query_heads} and {key_heads}"
            )
        if max_position <= 0:
            raise ValueError(f"max_position must be positive, got {max_position}")
        if position_dtype not in (torch.int32, torch.int64):
            raise ValueError(f"position_dtype must be int32 or int64, got {position_dtype}")
        if offset_dtype not in (None, torch.int32, torch.int64):
            raise ValueError(f"offset_dtype must be int32, int64 or None, got {offset_dtype}")
        self.query_heads = query_heads
        self.key_heads = key_heads
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        self.dtype = dtype
        self.position_dtype = position_dtype
        self.offset_dtype = offset_dtype
        lanes = VECTOR_ACCESS_BYTES // dtype.itemsize
        span = rotary_dim // 2 if self._NEOX else rotary_dim
        rot_vec = math.gcd(lanes, head_size, span)
        tail_vec = math.gcd(lanes, head_size, rotary_dim)
        _, rotating, copying = _head_units(self._NEOX, head_size, rotary_dim, rot_vec, tail_vec)
        self._token_units = (query_heads + key_heads) * (rotating + copying)
        self._resident_threads = resident_threads
        self.kernel = _rope_positions_kernel(
            query_heads,
            key_heads,
            head_size,
            rotary_dim,
            max_position,
            self.dtype_str,
            self.dtype_to_str(position_dtype),
            None if offset_dtype is None else self.dtype_to_str(offset_dtype),
            self._NEOX,
            rot_vec,
            tail_vec,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Measured, H200, 32 query and 8 key heads of 128: bfloat16 at 8192 tokens 40.96 us
        # with four units a thread against 45.41 with one; float32 at 4096 tokens 39.49
        # with one against 40.29 with four.
        return {"threads": 128, "units": 1 if self.dtype == torch.float32 else 4}

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        offsets: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if positions.numel() == 0:
            return torch.empty_like(query), None if key is None else torch.empty_like(key)
        # A call whose units fill at most half the device's resident threads is bound by the
        # latency of one unit, so each thread takes one; a larger one keeps several in
        # flight a thread. H200, 32 query and 8 key bfloat16 heads of 128: 2.24 us against
        # 2.90 at 64 tokens, 21.02 against 22.18 at 4096.
        rows = positions.view(-1)
        wide = 2 * rows.numel() * self._token_units <= self._resident_threads
        program = self.program(self._compile, wide)
        query_rows = query.view(-1, query.shape[-1])
        if key is None:
            return program(rows, query_rows, cos, sin).view(query.shape), None
        tensors = (rows, query_rows, key.view(-1, key.shape[-1]))
        if offsets is not None:
            tensors += (offsets.view(-1),)
        query_out, key_out = program(*tensors, cos, sin)
        return query_out.view(query.shape), key_out.view(key.shape)

    def _compile(self, cfg: dict, wide: bool):
        """The program ``forward`` launches, compiled from *cfg*; *wide* gives each thread one
        unit and spreads a token over about ``units / 256`` blocks of whole warps."""
        if wide:
            # Eight warps give each of an SM's four schedulers a second warp to issue while
            # the first waits on memory. The nearest whole count of such blocks a token splits
            # a token only at twelve warps or more: a smaller token is bound by one unit's
            # latency, and a larger one's loads queue on the one SM its single block runs on.
            block = 8 * 32
            parts = max(1, (self._token_units + block // 2) // block)
            return self.kernel(-(-self._token_units // (parts * 32)) * 32, 1)
        return self.kernel(cfg["threads"], cfg["units"])


class RoPENeoxPositionIdsKernel(_RoPEPositionsKernel, RoPENeoxPositionIdsFwdInterface):
    """GPT-NeoX rotation of each token's heads at the table row its position names."""

    _NEOX = True


class RoPENonNeoxKernel(_RoPEPositionsKernel, RoPENonNeoxFwdInterface):
    """RoFormer (GPT-J) adjacent-pair rotation of each token's heads at its position's row.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """

    _NEOX = False


@dataclasses.dataclass(frozen=True)
class RoPECosSinCall(CallSpec):
    """One rotation of ``x`` by caller-given per-row ``cos`` and ``sin`` tables.

    The token count, the batch and the strides of ``x`` are not facts of the call: a kernel
    serves any. ``seqlens_dtype`` is None for dense ``x``; ``offset_dtype`` is None where
    ``seqlen_offsets`` is an int.
    """

    heads: int = 0
    head_dim: int = 0
    rotary_half: int = 0
    interleaved: bool = False
    inplace: bool = False
    dtype: torch.dtype | None = None
    cos_dtype: torch.dtype | None = None
    sin_dtype: torch.dtype | None = None
    seqlens_dtype: torch.dtype | None = None
    offset_dtype: torch.dtype | None = None


class RoPECosSinFwdInterface(KernelInterface):
    """The rotation of each token's heads by the table row its sequence position names.

    Pair ``k`` of a head's first ``2 * call.rotary_half`` columns is ``(x[k], x[k + Rh])``
    (NeoX) or ``(x[2k], x[2k + 1])`` (interleaved) and becomes ``(a * c - b * s,
    b * c + a * s)`` with ``c, s = cos[row, k], sin[row, k]``.
    """

    request = RoPECosSinCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        seqlen_offsets: int | torch.Tensor,
    ) -> torch.Tensor:
        """Rotate the leading ``2 * call.rotary_half`` columns of each head of ``x``.

        Token ``s`` of sequence ``b`` reads row ``s + seqlen_offsets[b]`` of the tables, an
        int offset serving every sequence; a row outside ``[0, P)`` is clamped to the
        tables rather than read out of bounds.

        Args:
            x: ``[B, S, call.heads, call.head_dim]`` dense, or ``[Tt, call.heads,
                call.head_dim]`` packed, in ``call.dtype`` on ``call.device``, at any
                strides. Written in place where ``call.inplace``.
            cos: ``[P, call.rotary_half]`` in ``call.cos_dtype``, contiguous.
            sin: ``[P, call.rotary_half]`` in ``call.sin_dtype``, contiguous.
            cu_seqlens: ``[B + 1]`` sequence starts in ``call.seqlens_dtype``, contiguous,
                for packed ``x``; None for dense.
            seqlen_offsets: An int, or ``[B]`` in ``call.offset_dtype``, contiguous.

        Returns:
            ``x`` itself where ``call.inplace``, its untouched columns as they were; else a
            new contiguous tensor shaped and typed as ``x``.
        """


@target_cache(maxsize=32)
def _rope_cos_sin_kernel(
    heads: int,
    head_dim: int,
    rotary_half: int,
    interleaved: bool,
    inplace: bool,
    dtype: str,
    cos_dtype: str,
    sin_dtype: str,
    seqlens_dtype: str | None,
    offset_dtype: str | None,
    vec: int,
    unit_columns: bool,
) -> object:
    """The rotation of ``x`` into ``y``, both flat views addressed at runtime strides.

    Element ``(b, i, h, c)`` of ``x`` sits at ``(b * qb + i * qs + h * qh) * vec + c * qc``,
    ``i`` being ``s`` dense and the token packed, and ``y``'s likewise with ``p*``: each
    stride but the column's is passed divided by ``vec``, so TileLang proves every unit's
    start vector-aligned. Where ``unit_columns`` the column strides are 1; otherwise ``vec``
    is 1 and they are runtime scalars. In place, ``y`` is ``x`` and the copied columns are
    not touched.

    A unit is two ``vec``-column vectors of one head: ``vec`` columns of the rotary low half
    and the ``vec`` facing them in the high half, whose pairs are NeoX's ``(x[k], x[k + Rh])``
    or, interleaved, the adjacent pairs inside each vector; or one ``vec``-column vector past
    the rotary columns, copied. A thread takes one unit position of ``per_thread`` heads of
    one token. It loads those heads' x first, then finds the token's table row (dense: grid
    row ``by`` is the sequence; packed: a binary search over ``cu_seqlens``) and reads the
    row's slots once for all its heads. The arithmetic is f32 and rounds once.
    """
    rotary = 2 * rotary_half
    rot_units = rotary_half // vec
    # One column a vector holds no pair: the unit is then one adjacent pair.
    adjacent = interleaved and vec == 1
    head_units = rot_units + (0 if inplace else (head_dim - rotary) // vec)

    # The flat index proves no access in bounds; the bound check would split every vector.
    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True})
    def kernel(threads, per_thread):
        # A flat view, a table or the sequences may number 2^31 or more.
        rows = T.dynamic("rows", "int64")
        span = T.dynamic("span", "int64")
        out_span = T.dynamic("out_span", "int64")
        table_shape = (rows, rotary_half)
        # Thread g of a token takes heads g, g + groups, ...: one warp access covers
        # neighbouring heads.
        groups = -(-heads // per_thread)
        token_items = groups * head_units

        @T.macro
        def rotate(
            x, y, cos_table, sin_table, cu_seqlens, offsets, offset, tokens, extent, q, bx, by
        ):
            # Dense, tokens is S and extent B: grid row by is sequence b, so no thread divides
            # by a runtime extent. Packed, tokens is Tt and extent the steps of the search for
            # a token's sequence. cu_seqlens and offsets are read only where the call has them.
            qb, qs, qh, qc, pb, ps, ph, pc = q
            if unit_columns:
                qc = 1
                pc = 1
            found = T.alloc_local([2], "int64")
            v = T.alloc_local([2 * vec * per_thread], dtype)
            cs = T.alloc_local([vec], cos_dtype)
            sn = T.alloc_local([vec], sin_dtype)
            units = tokens * token_items
            item = T.Cast("int64", bx) * threads + T.get_thread_binding()
            token = item // token_items
            group = T.Cast("int32", item % token_items) // head_units
            slot = T.Cast("int32", item % token_items) % head_units
            if item < units:
                # Packed x has no sequence stride, so its address needs no search.
                if seqlens_dtype is not None:
                    x_token = token * qs
                    y_token = token * ps
                else:
                    x_token = T.Cast("int64", by) * qb + token * qs
                    y_token = T.Cast("int64", by) * pb + token * ps
                if slot < rot_units:
                    low = slot * (2 if adjacent else vec)
                    high = low + (1 if adjacent else rotary_half)
                    # Every head's x first: its loads wait on no table row, so they are
                    # in flight while the row is found and its slots are read.
                    for k in T.unroll(per_thread):
                        if k * groups + group < heads:
                            x_at = (x_token + (k * groups + group) * qh) * vec
                            for j in T.vectorized(vec):
                                v[2 * k * vec + j] = x[x_at + (low + j) * qc]
                            for j in T.vectorized(vec):
                                v[(2 * k + 1) * vec + j] = x[x_at + (high + j) * qc]
                    if seqlens_dtype is not None:
                        # The last sequence starting at or before the token: an empty
                        # sequence's start equals the next one's, which wins.
                        found[0] = 0
                        found[1] = cu_seqlens.shape[0] - 2
                        for _ in T.serial(extent):
                            if found[0] < found[1]:
                                mid = found[0] + (found[1] - found[0] + 1) // 2
                                if cu_seqlens[mid] <= token:
                                    found[0] = mid
                                else:
                                    found[1] = mid - 1
                        b = found[0]
                        s = token - T.Cast("int64", cu_seqlens[b])
                    else:
                        b = by
                        s = token
                    if offset_dtype is not None:
                        seen = s + T.Cast("int64", offsets[b])
                    else:
                        seen = s + offset
                    row = T.max(0, T.min(seen, T.Cast("int64", rows) - 1))
                    if interleaved and not adjacent:
                        # Pairs slot * vec / 2 on, then rotary_half / 2 past them.
                        for half in T.unroll(2):
                            for j in T.vectorized(vec // 2):
                                cs[half * (vec // 2) + j] = cos_table[
                                    row, (half * rotary_half + slot * vec) // 2 + j
                                ]
                            for j in T.vectorized(vec // 2):
                                sn[half * (vec // 2) + j] = sin_table[
                                    row, (half * rotary_half + slot * vec) // 2 + j
                                ]
                    else:
                        for j in T.vectorized(vec):
                            cs[j] = cos_table[row, slot * vec + j]
                        for j in T.vectorized(vec):
                            sn[j] = sin_table[row, slot * vec + j]
                    for k in T.unroll(per_thread):
                        if k * groups + group < heads:
                            y_at = (y_token + (k * groups + group) * ph) * vec
                            for p in T.unroll(vec):
                                a = 2 * k * vec + (2 * p if interleaved else p)
                                e = a + (1 if interleaved else vec)
                                x_a = T.Cast("float32", v[a])
                                x_e = T.Cast("float32", v[e])
                                c = T.Cast("float32", cs[p])
                                sv = T.Cast("float32", sn[p])
                                v[a] = T.Cast(dtype, x_a * c - x_e * sv)
                                v[e] = T.Cast(dtype, x_e * c + x_a * sv)
                            for j in T.vectorized(vec):
                                y[y_at + (low + j) * pc] = v[2 * k * vec + j]
                            for j in T.vectorized(vec):
                                y[y_at + (high + j) * pc] = v[(2 * k + 1) * vec + j]
                elif not inplace:
                    kept = rotary + (slot - rot_units) * vec
                    for k in T.unroll(per_thread):
                        if k * groups + group < heads:
                            x_at = (x_token + (k * groups + group) * qh) * vec
                            y_at = (y_token + (k * groups + group) * ph) * vec
                            for j in T.vectorized(vec):
                                v[j] = x[x_at + (kept + j) * qc]
                            for j in T.vectorized(vec):
                                y[y_at + (kept + j) * pc] = v[j]

        # One entry point per form of cu_seqlens and seqlen_offsets.
        if seqlens_dtype is not None and offset_dtype is not None:

            @T.prim_func
            def main(
                x: T.Tensor((span,), dtype),
                cos_table: T.Tensor(table_shape, cos_dtype),
                sin_table: T.Tensor(table_shape, sin_dtype),
                cu_seqlens: T.Tensor((T.dynamic("bounds", "int64"),), seqlens_dtype),
                offsets: T.Tensor((T.dynamic("batch", "int64"),), offset_dtype),
                y: T.Tensor((out_span,), dtype),
                tokens: T.int64,
                extent: T.int64,
                qb: T.int64,
                qs: T.int64,
                qh: T.int64,
                qc: T.int64,
                pb: T.int64,
                ps: T.int64,
                ph: T.int64,
                pc: T.int64,
            ):
                q = (qb, qs, qh, qc, pb, ps, ph, pc)
                with T.Kernel(
                    T.Cast("int32", T.ceildiv(tokens * token_items, threads)),
                    1,
                    threads=threads,
                ) as (bx, by):
                    rotate(
                        x,
                        y,
                        cos_table,
                        sin_table,
                        cu_seqlens,
                        offsets,
                        0,
                        tokens,
                        extent,
                        q,
                        bx,
                        by,
                    )

        elif seqlens_dtype is not None:

            @T.prim_func
            def main(
                x: T.Tensor((span,), dtype),
                cos_table: T.Tensor(table_shape, cos_dtype),
                sin_table: T.Tensor(table_shape, sin_dtype),
                cu_seqlens: T.Tensor((T.dynamic("bounds", "int64"),), seqlens_dtype),
                y: T.Tensor((out_span,), dtype),
                offset: T.int64,
                tokens: T.int64,
                extent: T.int64,
                qb: T.int64,
                qs: T.int64,
                qh: T.int64,
                qc: T.int64,
                pb: T.int64,
                ps: T.int64,
                ph: T.int64,
                pc: T.int64,
            ):
                q = (qb, qs, qh, qc, pb, ps, ph, pc)
                with T.Kernel(
                    T.Cast("int32", T.ceildiv(tokens * token_items, threads)),
                    1,
                    threads=threads,
                ) as (bx, by):
                    rotate(
                        x, y, cos_table, sin_table, cu_seqlens, x, offset, tokens, extent, q, bx, by
                    )

        elif offset_dtype is not None:

            @T.prim_func
            def main(
                x: T.Tensor((span,), dtype),
                cos_table: T.Tensor(table_shape, cos_dtype),
                sin_table: T.Tensor(table_shape, sin_dtype),
                offsets: T.Tensor((T.dynamic("batch", "int64"),), offset_dtype),
                y: T.Tensor((out_span,), dtype),
                tokens: T.int64,
                extent: T.int64,
                qb: T.int64,
                qs: T.int64,
                qh: T.int64,
                qc: T.int64,
                pb: T.int64,
                ps: T.int64,
                ph: T.int64,
                pc: T.int64,
            ):
                q = (qb, qs, qh, qc, pb, ps, ph, pc)
                with T.Kernel(
                    T.Cast("int32", T.ceildiv(tokens * token_items, threads)),
                    T.Cast("int32", extent),
                    threads=threads,
                ) as (bx, by):
                    rotate(x, y, cos_table, sin_table, x, offsets, 0, tokens, extent, q, bx, by)

        else:

            @T.prim_func
            def main(
                x: T.Tensor((span,), dtype),
                cos_table: T.Tensor(table_shape, cos_dtype),
                sin_table: T.Tensor(table_shape, sin_dtype),
                y: T.Tensor((out_span,), dtype),
                offset: T.int64,
                tokens: T.int64,
                extent: T.int64,
                qb: T.int64,
                qs: T.int64,
                qh: T.int64,
                qc: T.int64,
                pb: T.int64,
                ps: T.int64,
                ph: T.int64,
                pc: T.int64,
            ):
                q = (qb, qs, qh, qc, pb, ps, ph, pc)
                with T.Kernel(
                    T.Cast("int32", T.ceildiv(tokens * token_items, threads)),
                    T.Cast("int32", extent),
                    threads=threads,
                ) as (bx, by):
                    rotate(x, y, cos_table, sin_table, x, x, offset, tokens, extent, q, bx, by)

        return main

    return kernel


def _flat(t: torch.Tensor) -> torch.Tensor:
    """A one-dimensional view of *t*'s storage from its first element to its last."""
    span = sum((n - 1) * s for n, s in zip(t.shape, t.stride(), strict=True)) + 1
    return t.as_strided((span,), (1,))


class RoPECosSinKernel(Kernel, RoPECosSinFwdInterface):
    """The rotation by caller-given tables, either pairing, dense or packed, at any strides.

    Each thread moves whole vectors: the widest power of two up to 16 bytes dividing ``Rh``,
    the head size and every stride of ``x`` and the output but the column's, or one column
    where a column stride is not 1.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @classmethod
    def entry_for(cls, call: RoPECosSinCall) -> Entry:
        return call, lambda: cls(
            call.heads,
            call.head_dim,
            call.rotary_half,
            call.dtype,
            call.cos_dtype,
            call.sin_dtype,
            interleaved=call.interleaved,
            inplace=call.inplace,
            seqlens_dtype=call.seqlens_dtype,
            offset_dtype=call.offset_dtype,
            resident_threads=call.sm_count * call.threads_per_sm,
        )

    def __init__(
        self,
        heads: int,
        head_dim: int,
        rotary_half: int,
        dtype: torch.dtype,
        cos_dtype: torch.dtype,
        sin_dtype: torch.dtype,
        config: dict | None = None,
        *,
        interleaved: bool = False,
        inplace: bool = False,
        seqlens_dtype: torch.dtype | None = None,
        offset_dtype: torch.dtype | None = None,
        resident_threads: int = 0,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.rotary_half = rotary_half
        self.dtype = dtype
        self.cos_dtype = cos_dtype
        self.sin_dtype = sin_dtype
        self.interleaved = interleaved
        self.inplace = inplace
        self.seqlens_dtype = seqlens_dtype
        self.offset_dtype = offset_dtype
        self._resident_threads = resident_threads
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Two heads a thread, each sharing the thread's cos and sin loads.
        return {"threads": 128, "heads": 2}

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        seqlen_offsets: int | torch.Tensor,
    ) -> torch.Tensor:
        y = x if self.inplace else torch.empty(x.shape, dtype=x.dtype, device=x.device)
        packed = cu_seqlens is not None
        # Dense x is (b, s, h, c); packed x is (t, h, c), addressed as b = 0, i = t.
        x_strides = (0, *x.stride()) if packed else x.stride()
        y_strides = (0, *y.stride()) if packed else y.stride()
        unit_columns = x_strides[3] == 1 and y_strides[3] == 1
        vec = 1
        if unit_columns:
            lanes = VECTOR_ACCESS_BYTES // self.dtype.itemsize
            vec = math.gcd(lanes, self.rotary_half, self.head_dim, *x_strides[:3], *y_strides[:3])
        # A call whose units, one head a thread, fill at most half the device's resident
        # threads is bound by the latency of one unit, so each thread takes one head; a
        # larger one keeps several heads' loads in flight a thread.
        copied = 0 if self.inplace else self.head_dim - 2 * self.rotary_half
        head_units = self.rotary_half // vec + copied // vec
        tokens = x.shape[0] if packed else x.shape[0] * x.shape[1]
        wide = 2 * tokens * self.heads * head_units <= self._resident_threads
        program = self.program(self._compile, vec, unit_columns, wide)
        # The offsets are a tensor argument, or an int scalar after the tensors.
        offsets_tensor = isinstance(seqlen_offsets, torch.Tensor)
        strides = [stride // vec for stride in x_strides[:3]] + [x_strides[3]]
        strides += [stride // vec for stride in y_strides[:3]] + [y_strides[3]]
        if packed:
            # The steps of a binary search over the sequences.
            steps = (cu_seqlens.shape[0] - 2).bit_length()
            launches = [(x, y, seqlen_offsets, [x.shape[0], steps])]
        else:
            # A dense launch runs one grid row a sequence, and CUDA's grid has at most 65535
            # rows: more sequences launch in parts.
            rows = 65535
            launches = [
                (
                    x[b : b + rows],
                    y[b : b + rows],
                    seqlen_offsets[b : b + rows] if offsets_tensor else seqlen_offsets,
                    [x.shape[1], min(rows, x.shape[0] - b)],
                )
                for b in range(0, x.shape[0], rows)
            ]
        for x_part, y_part, offsets, sizes in launches:
            tensors = [_flat(x_part), cos, sin]
            if packed:
                tensors.append(cu_seqlens)
            if offsets_tensor:
                tensors.append(offsets)
            tensors.append(_flat(y_part))
            scalars = [] if offsets_tensor else [offsets]
            program(*tensors, *scalars, *sizes, *strides)
        return y

    def _compile(self, cfg: dict, vec: int, unit_columns: bool, wide: bool):
        """The program ``forward`` launches for *vec*-wide vectors, compiled from *cfg*;
        *wide* gives each thread one head."""
        kernel = _rope_cos_sin_kernel(
            self.heads,
            self.head_dim,
            self.rotary_half,
            self.interleaved,
            self.inplace,
            self.dtype_str,
            self.dtype_to_str(self.cos_dtype),
            self.dtype_to_str(self.sin_dtype),
            None if self.seqlens_dtype is None else self.dtype_to_str(self.seqlens_dtype),
            None if self.offset_dtype is None else self.dtype_to_str(self.offset_dtype),
            vec,
            unit_columns,
        )
        return kernel(cfg["threads"], 1 if wide else cfg["heads"])
