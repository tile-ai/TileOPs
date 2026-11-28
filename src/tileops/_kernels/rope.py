"""Rotary position embedding kernels: ``y = x * cos + rotate(x) * sin`` at a table row.

- ``RoPENeoxPositionIdsKernel``: NeoX halves, ``(x[k], x[k + rotary_dim / 2])``.
- ``RoPENonNeoxKernel``: RoFormer (GPT-J) adjacent pairs, ``(x[2k], x[2k + 1])``.

Both rotate the query and the key in one launch at the row each token's position names,
rotate each head's leading ``rotary_dim`` columns and copy the rest, or in place leave the rest
untouched; one built kernel serves every token count.

- ``RoPECosSinKernel``: the caller's tables, one row a sequence position; either pairing,
  dense or packed tokens, at any strides, in place or not.
"""

import dataclasses
import math
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import (
    MAX_BLOCK_THREADS,
    MAX_GRID_YZ_BLOCKS,
    VECTOR_ACCESS_BYTES,
)
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops._utils import WARP_LANES

__all__ = [
    "RoPECall",
    "RoPECosSinCall",
    "RoPECosSinFwdInterface",
    "RoPECosSinKernel",
    "RoPENeoxPositionIdsFwdInterface",
    "RoPENeoxPositionIdsKernel",
    "RoPENeoxStagedRowsKernel",
    "RoPENonNeoxFwdInterface",
    "RoPENonNeoxKernel",
    "RoPENonNeoxStagedRowsKernel",
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
    # In place, the elements from one token's query row to the next and from one key row to
    # the next (0 without a key); None for a call that writes new tensors.
    row_strides: tuple[int, int] | None = None


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
        """Rotate each token's leading rotary columns, in place where ``call.row_strides`` is set.

        Token ``t`` reads table row ``positions[t] + offsets[t]`` (``positions[t]`` without
        offsets) and rotates the first
        ``call.rotary_dim`` columns of each head of ``call.head_size`` columns; the remaining
        columns are copied, or in place left as they are. A position outside
        ``[0, call.max_position)`` is clamped to the table rather than read out of bounds.

        Args:
            positions: ``[*S]`` in ``call.position_dtype``, ``int32`` or ``int64``, on
                ``call.device``, contiguous.
            query: ``[*S, call.query_heads * call.head_size]`` in ``call.dtype`` on
                ``call.device``; contiguous, or in place viewable as rows
                ``call.row_strides[0]`` elements apart of contiguous columns.
            key: ``[*S, call.key_heads * call.head_size]`` in the same dtype and device, laid
                out as ``query`` with rows ``call.row_strides[1]`` apart in place, or None;
                with ``call.key_heads`` 0 it may still be present, and empty.
            offsets: ``[*S]`` in ``call.offset_dtype``, ``int32`` or ``int64``, on
                ``call.device``, contiguous, or None where ``call.offset_dtype`` is. A call
                with offsets has a key.
            cos: The cosine table, ``(call.max_position, call.rotary_dim // 2)`` in
                ``call.dtype`` on ``call.device``, contiguous, row-major.
            sin: The sine table, in the same shape, dtype, device and layout.

        Returns:
            ``(query_out, key_out)``: ``query`` and ``key`` themselves in place, otherwise new
            contiguous tensors on ``call.device`` shaped and typed as them; ``key_out`` is None
            where ``key`` is.
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


def _staged_sides(key_heads: int) -> int:
    """The staged blocks a token takes: one for its query row, one for its key row if any."""
    return 2 if key_heads else 1


def _head_units(
    neox: bool, head_size: int, rotary_dim: int, rot_vec: int, tail_vec: int
) -> tuple[int, int, int]:
    """A rotating unit's width in columns, and a head's rotating and copying unit counts.

    A GPT-J unit holds whole pairs; one unaligned to a pair-wide vector reads elementwise.
    """
    width = rot_vec if neox else max(rot_vec, 2)
    rotating = (rotary_dim // 2 if neox else rotary_dim) // width
    return width, rotating, (head_size - rotary_dim) // tail_vec


@T.macro
def _rotate_pair(lo, i, hi, j, c, s):
    """Rotate the pair ``(lo[i], hi[j])`` by the angle of float32 cosine *c* and sine *s*, in
    float32, each value rounding once to its buffer's dtype."""
    x_lo = T.Cast("float32", lo[i])
    x_hi = T.Cast("float32", hi[j])
    lo[i] = T.Cast(lo.dtype, x_lo * c - x_hi * s)
    hi[j] = T.Cast(hi.dtype, x_hi * c + x_lo * s)


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
    row_vec: int,
    inplace: bool = False,
    query_stride: int = 0,
    key_stride: int = 0,
) -> object:
    """The rotation of a query and a key of ``tokens`` rows, ``tokens`` dynamic.

    A unit is part of one head: NeoX, ``rot_vec`` columns of the low half and the ``rot_vec``
    facing them in the high half; GPT-J, ``rot_vec`` adjacent columns; or ``tail_vec`` columns
    past ``rotary_dim``, copied. ``rot_vec`` and ``tail_vec`` divide every column offset they
    start at. A token's query units come before its key units, so one launch serves both.
    ``inplace`` writes the rotated units back into the query and the key, whose rows lie
    ``query_stride`` and ``key_stride`` elements apart, and no unit copies.

    Three programs:

    - Default: a block holds ``threads * units`` units of one token, ``units`` a thread;
      consecutive blocks walk a token's parts, then the next token's.
    - ``wide``: a ``(head units, threads / head units)`` block holds whole heads on a
      ``(tokens, parts)`` grid.
    - Staged: a block loads one token's query or key row into shared memory, rotates it there
      and copies it back, on a ``(tokens, sides)`` grid.

    Each block clamps its token's position into the table once, so a position outside the table
    reads the nearest row. The arithmetic is f32 and rounds once.
    """
    half = rotary_dim // 2
    width, rot_units, tail_units = _head_units(neox, head_size, rotary_dim, rot_vec, tail_vec)
    if inplace:
        tail_units = 0
    table_width = width if neox else width // 2
    # A unit's stride in a thread's registers: a rotating unit's low columns or a copied run.
    run = width if inplace else max(width, tail_vec)
    vector_pairs = neox or rot_vec >= 2
    head_units = rot_units + tail_units
    token_units = (query_heads + key_heads) * head_units

    sides = _staged_sides(key_heads)

    @tilelang.jit(
        out_idx=[] if inplace else [6, 7] if offset_dtype else [5, 6] if key_heads else [4]
    )
    def kernel(threads, units, staged, wide):
        num_tokens = T.dynamic("num_tokens")
        query_shape = (num_tokens, query_heads * head_size)
        key_shape = (num_tokens, key_heads * head_size)
        table_shape = (max_position, half)
        parts = sides if staged else -(-token_units // (threads * units))
        block_heads = threads // head_units
        # A wide program's token takes its parts a grid row apart, so they run on SMs apart. A
        # staged block's token is its block index: TileLang writes the row back with one bulk
        # store only where it proves the row in bounds, which it does not for a quotient of
        # the block index.
        split = wide or staged
        grid = (num_tokens, parts) if split else (num_tokens * parts, 1)
        block = (head_units, block_heads) if wide else threads

        @T.macro
        def table_slots(cos_table, sin_table, row, first, cs, sn, at):
            # The cos and sin of a unit's pairs from table row row into cs and sn at index at.
            for j in T.vectorized(table_width):
                cs[at + j] = cos_table[row, first + j]
            for j in T.vectorized(table_width):
                sn[at + j] = sin_table[row, first + j]

        @T.macro
        def rotate_unit(lo, lo_at, hi, hi_at, cs, sn, at):
            # Rotate a unit held at lo[lo_at:] (and hi[hi_at:] for NeoX) by cs and sn at index at.
            if neox:
                for j in T.unroll(width):
                    c = T.Cast("float32", cs[at + j])
                    s = T.Cast("float32", sn[at + j])
                    _rotate_pair(lo, lo_at + j, hi, hi_at + j, c, s)
            else:
                for p in T.unroll(table_width):
                    c = T.Cast("float32", cs[at + p])
                    s = T.Cast("float32", sn[at + p])
                    _rotate_pair(lo, lo_at + 2 * p, lo, lo_at + 2 * p + 1, c, s)

        @T.macro
        def move_unit(r, row, x, y, token, head, slot, cos_table, sin_table, regs, load, store):
            # Thread unit r: load reads its columns into regs, a row reads its cos/sin row, store
            # rotates and writes them; with regs None the unit takes registers of its own, at
            # index 0.
            if regs is None:
                r = 0
                regs = (
                    T.alloc_local([width], dtype),
                    T.alloc_local([width], dtype),
                    T.alloc_local([table_width], dtype),
                    T.alloc_local([table_width], dtype),
                    T.alloc_local([tail_vec], dtype),
                )
            # Bound before lo: a buffer is named after the last variable bound to it, and the
            # copied run may alias lo.
            kept_run = regs[4]
            lo, hi, cs, sn = regs[:4]
            if slot < rot_units:
                col = slot * width
                low = head * head_size + col
                first = col if neox else col // 2
                if load:
                    if vector_pairs:
                        for j in T.vectorized(width):
                            lo[r * run + j] = x[token, low + j]
                    else:
                        for j in T.serial(width):
                            lo[r * run + j] = x[token, low + j]
                    if neox:
                        for j in T.vectorized(width):
                            hi[r * width + j] = x[token, low + half + j]
                if row is not None:
                    table_slots(cos_table, sin_table, row, first, cs, sn, r * table_width)
                if store:
                    rotate_unit(lo, r * run, hi, r * width, cs, sn, r * table_width)
                    if neox:
                        for j in T.vectorized(width):
                            y[token, low + half + j] = hi[r * width + j]
                    if vector_pairs:
                        for j in T.vectorized(width):
                            y[token, low + j] = lo[r * run + j]
                    else:
                        for j in T.serial(width):
                            y[token, low + j] = lo[r * run + j]
            elif tail_units > 0:
                kept = head * head_size + rotary_dim + (slot - rot_units) * tail_vec
                if load:
                    for j in T.vectorized(tail_vec):
                        kept_run[r * run + j] = x[token, kept + j]
                if store:
                    for j in T.vectorized(tail_vec):
                        y[token, kept + j] = kept_run[r * run + j]

        @T.macro
        def each_unit(
            row,
            regs,
            load,
            store,
            query,
            key,
            cos_table,
            sin_table,
            query_out,
            key_out,
            token,
            part,
        ):
            # move_unit on each of the thread's units, in the query or the key; row is the
            # token's table row buffer, or None where no unit reads the tables.
            shared = (cos_table, sin_table, regs, load, store)
            for r in T.unroll(units):
                within = (part * units + r) * threads + T.get_thread_binding()
                # A wide thread's head and slot are its thread indices: no index is divided.
                head = (
                    part * block_heads + T.get_thread_binding(1) if wide else within // head_units
                )
                slot = T.get_thread_binding(0) if wide else within % head_units
                if head < query_heads:
                    move_unit(
                        r,
                        None if row is None else row[0],
                        query,
                        query_out,
                        token,
                        head,
                        slot,
                        *shared,
                    )
                elif head < query_heads + key_heads:
                    move_unit(
                        r,
                        None if row is None else row[0],
                        key,
                        key_out,
                        token,
                        head - query_heads,
                        slot,
                        *shared,
                    )

        def token_position(positions, offsets, token):
            # The token's position, plus its offset where the call has offsets.
            if offset_dtype:
                return T.Cast("int64", positions[token]) + T.Cast("int64", offsets[token])
            return positions[token]

        def clamped(position):
            # The table row a position reads: the nearest row where it is outside the table.
            return T.max(0, T.min(position, max_position - 1))

        @T.macro
        def table_row(positions, offsets, token, row):
            if offset_dtype:
                seen = token_position(positions, offsets, token)
                row[0] = clamped(seen)
            else:
                row[0] = clamped(positions[token])

        @T.macro
        def stage_row(x, y, heads, positions, offsets, token, cos_table, sin_table, buf):
            # Row token of x, heads heads, into buf; its rotating units rotate there and the row
            # goes to y. Thread t takes vectors and units t, t + threads, ...
            tid = T.get_thread_binding()
            row_vecs = heads * head_size // row_vec
            rotating = heads * rot_units
            units = -(-rotating // threads)
            lo = T.alloc_local([units * width], dtype)
            hi = T.alloc_local([units * width], dtype)
            cs = T.alloc_local([units * table_width], dtype)
            sn = T.alloc_local([units * table_width], dtype)
            for i in T.unroll(-(-row_vecs // threads)):
                if i * threads + tid < row_vecs:
                    for j in T.vectorized(row_vec):
                        buf[(i * threads + tid) * row_vec + j] = x[
                            token, (i * threads + tid) * row_vec + j
                        ]
            # Each table load clamps the bound position itself: TileLang proves a load in bounds
            # from a clamp in its index, not from one stored to a buffer, and guards it otherwise.
            seen = token_position(positions, offsets, token)
            for k in T.unroll(units):
                if k * threads + tid < rotating:
                    slot = (k * threads + tid) % rot_units
                    first = slot * width if neox else slot * width // 2
                    table_slots(cos_table, sin_table, clamped(seen), first, cs, sn, k * table_width)
            T.sync_threads()
            # Every unit is read before any is written back: TileLang places the barrier a
            # loop of both needs inside the per-unit branch, which not every thread enters.
            for k in T.unroll(units):
                unit = k * threads + tid
                if unit < rotating:
                    low = (unit // rot_units) * head_size + (unit % rot_units) * width
                    for j in T.vectorized(width):
                        lo[k * width + j] = buf[low + j]
                    if neox:
                        for j in T.vectorized(width):
                            hi[k * width + j] = buf[low + half + j]
            for k in T.unroll(units):
                unit = k * threads + tid
                if unit < rotating:
                    low = (unit // rot_units) * head_size + (unit % rot_units) * width
                    rotate_unit(lo, k * width, hi, k * width, cs, sn, k * table_width)
                    if neox:
                        for j in T.vectorized(width):
                            buf[low + half + j] = hi[k * width + j]
                    for j in T.vectorized(width):
                        buf[low + j] = lo[k * width + j]
            T.sync_threads()
            T.copy(buf[0 : heads * head_size], y[token, 0 : heads * head_size])

        @T.macro
        def rotate_token(positions, offsets, query, key, cos_table, sin_table, query_out, key_out):
            with T.Kernel(grid[0], grid[1], threads=block) as (bx, by):
                # The block's token and its part of the token, from its grid indices.
                token = bx if split else bx // parts
                part = by if split else bx % parts
                # A staged block's table loads clamp the position in their index, so it holds no
                # row.
                row_dtype = "int64" if offset_dtype else position_dtype
                row = None if staged else T.alloc_local([1], row_dtype)
                tensors = (query, key, cos_table, sin_table, query_out, key_out, token, part)
                if staged:
                    buf = T.alloc_shared([max(query_heads, key_heads) * head_size], dtype)
                    rows = (positions, offsets, token, cos_table, sin_table, buf)
                    if part == 0:
                        stage_row(query, query_out, query_heads, *rows)
                    else:
                        stage_row(key, key_out, key_heads, *rows)
                elif tail_units == 0 and not wide and not inplace:
                    # Other resident blocks of a throughput-bound call hide each block's wait on the
                    # position.
                    table_row(positions, offsets, token, row)
                    each_unit(row, None, True, True, *tensors)
                else:
                    # A copied run takes a rotating unit's low-column registers.
                    lo = T.alloc_local([units * run], dtype)
                    hi = T.alloc_local([units * width], dtype)
                    cs = T.alloc_local([units * table_width], dtype)
                    sn = T.alloc_local([units * table_width], dtype)
                    regs = (lo, hi, cs, sn, lo)
                    if units == 1 or inplace:
                        # A copying lane needs no position, and in a wide call no other block hides
                        # the position's load, so the columns load first: ptxas waits out that load
                        # at the clamp, ahead of any later load. In place the output buffer is
                        # the input, so every unit loads before the first store.
                        each_unit(None, regs, True, False, *tensors)
                        table_row(positions, offsets, token, row)
                        each_unit(row, regs, False, True, *tensors)
                    else:
                        # A warp can hold rotating and copying units, each path waiting out a
                        # load-to-store round trip a unit.
                        table_row(positions, offsets, token, row)
                        each_unit(row, regs, True, False, *tensors)
                        each_unit(None, regs, False, True, *tensors)

        if inplace:
            query_rows = T.StridedTensor(query_shape, (query_stride, 1), dtype)
            key_rows = T.StridedTensor(key_shape, (key_stride, 1), dtype)
            if key_heads:

                @T.prim_func
                def main(
                    positions: T.Tensor((num_tokens,), position_dtype),
                    query: query_rows,
                    key: key_rows,
                    cos_table: T.Tensor(table_shape, dtype),
                    sin_table: T.Tensor(table_shape, dtype),
                ):
                    rotate_token(positions, positions, query, key, cos_table, sin_table, query, key)

            else:

                @T.prim_func
                def main(
                    positions: T.Tensor((num_tokens,), position_dtype),
                    query: query_rows,
                    cos_table: T.Tensor(table_shape, dtype),
                    sin_table: T.Tensor(table_shape, dtype),
                ):
                    rotate_token(
                        positions, positions, query, query, cos_table, sin_table, query, query
                    )

        elif offset_dtype:

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
                rotate_token(
                    positions, offsets, query, key, cos_table, sin_table, query_out, key_out
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
                rotate_token(
                    positions, positions, query, key, cos_table, sin_table, query_out, key_out
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
                rotate_token(
                    positions, positions, query, query, cos_table, sin_table, query_out, query_out
                )

        return main

    return kernel


class _RoPEPositionsKernel(Kernel):
    """The rotation at each token's position, one convention per subclass.

    Each thread moves whole vectors: the widest power of two up to 16 bytes dividing every
    column offset its unit starts at.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
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
            row_strides=call.row_strides,
            **cls._device_terms(call),
        )

    @classmethod
    def _device_terms(cls, call: RoPECall) -> dict:
        """The constructor's keyword arguments that come from *call*'s device facts."""
        return {
            "resident_threads": call.sm_count * call.threads_per_sm,
            "sm_count": call.sm_count,
        }

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
        row_strides: tuple[int, int] | None = None,
        resident_threads: int = 0,
        sm_count: int = 0,
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
        self.key_heads = key_heads
        self.dtype = dtype
        self.inplace = row_strides is not None
        lanes = VECTOR_ACCESS_BYTES // dtype.itemsize
        span = rotary_dim // 2 if self._NEOX else rotary_dim
        # In place a vector also divides the row strides, so every row starts aligned to it.
        rot_vec = math.gcd(lanes, head_size, span, *(row_strides or ()))
        tail_vec = math.gcd(lanes, head_size, rotary_dim)
        _, rotating, copying = _head_units(self._NEOX, head_size, rotary_dim, rot_vec, tail_vec)
        if self.inplace:
            copying = 0
        heads, head_units = query_heads + key_heads, rotating + copying
        self._token_units = heads * head_units
        self._resident_threads = resident_threads
        self._sm_count = sm_count
        # The parts a wide program can split a token into: whole heads within one block, and
        # no more parts than the grid's second dimension holds.
        self._wide_parts = [
            p
            for p in range(1, min(heads, MAX_GRID_YZ_BLOCKS) + 1)
            if heads % p == 0 and head_units * heads // p <= MAX_BLOCK_THREADS
        ]
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
            math.gcd(lanes, head_size),
            self.inplace,
            *(row_strides or (0, 0)),
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Four units a thread keep more 16-bit loads in flight; a float32 unit already moves
        # twice the bytes, and one a thread is faster.
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
            if self.inplace:
                return query, key
            return torch.empty_like(query), None if key is None else torch.empty_like(key)
        rows = positions.view(-1)
        program = self.program(self._compile, self._shape(rows.numel()))
        query_rows = query.view(rows.numel(), query.shape[-1])
        if not self.key_heads and offsets is None:
            # A key of no heads is built into the query-only program.
            if self.inplace:
                program(rows, query_rows, cos, sin)
                return query, key
            query_out = program(rows, query_rows, cos, sin).view(query.shape)
            return query_out, None if key is None else torch.empty_like(key)
        tensors = (rows, query_rows, key.view(rows.numel(), key.shape[-1]))
        if offsets is not None:
            tensors += (offsets.view(-1),)
        if self.inplace:
            program(*tensors, cos, sin)
            return query, key
        query_out, key_out = program(*tensors, cos, sin)
        return query_out.view(query.shape), key_out.view(key.shape)

    def _shape(self, tokens: int) -> int:
        """The parts a call of *tokens* tokens splits each token into, 0 where it is not wide.

        A wide call splits each token into the most parts of whole heads that leave every block an
        SM of its own, else into the fewest parts a block holds. A token with no split into blocks of
        at most ``MAX_BLOCK_THREADS`` threads and ``MAX_GRID_YZ_BLOCKS`` parts takes the program that
        is not wide.
        """
        if not self._wide_parts or 2 * tokens * self._token_units > self._resident_threads:
            return 0
        spread = [p for p in self._wide_parts if tokens * p <= self._sm_count]
        return spread[-1] if spread else self._wide_parts[0]

    def _compile(self, cfg: dict, parts: int):
        """The program ``forward`` launches, compiled from *cfg*; *parts* nonzero gives a block
        a thread per unit of ``1 / parts`` of its token's heads."""
        if parts:
            return self.kernel(self._token_units // parts, 1, False, True)
        return self.kernel(cfg["threads"], cfg["units"], False, False)


class RoPENeoxPositionIdsKernel(_RoPEPositionsKernel, RoPENeoxPositionIdsFwdInterface):
    """GPT-NeoX rotation of each token's heads at the table row its position names."""

    _NEOX = True


class RoPENonNeoxKernel(_RoPEPositionsKernel, RoPENonNeoxFwdInterface):
    """RoFormer (GPT-J) adjacent-pair rotation of each token's heads at its position's row.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """

    _NEOX = False


class _RoPEStagedRowsKernel(_RoPEPositionsKernel):
    """The rotation where heads copy columns, each token's query row and key row staged in shared
    memory and written back once the whole row has rotated.

    A call takes it only where its rows outnumber the staged blocks the device holds at once
    (an upper bound by shared memory and threads), so blocks wait and stores overlap loads. A
    block takes one thread a share of its row, the SM's shared memory over its resident threads
    in whole 16-byte vectors, rounded up to whole warps, at most 1024. The config's ``threads``
    and ``units`` shape only the positions kernel's programs.
    """

    @staticmethod
    def _row_bytes(call: RoPECall) -> int:
        """The bytes of the longer of a token's query row and key row."""
        return max(call.query_heads, call.key_heads) * call.head_size * call.dtype.itemsize

    @classmethod
    def refusal(cls, call: RoPECall) -> str | None:
        if call.rotary_dim == call.head_size or call.row_strides is not None:
            return "the call copies no columns"
        if cls._row_bytes(call) > call.smem_budget:
            return (
                f"a row takes {cls._row_bytes(call)} bytes, past the {call.smem_budget} a block has"
            )
        return super().refusal(call)

    @classmethod
    def _device_terms(cls, call: RoPECall) -> dict:
        row = cls._row_bytes(call)
        share = -(-call.smem_per_sm // call.threads_per_sm // VECTOR_ACCESS_BYTES)
        warp_bytes = share * VECTOR_ACCESS_BYTES * WARP_LANES
        threads = min(MAX_BLOCK_THREADS, -(-row // warp_bytes) * WARP_LANES)
        per_sm = min(call.threads_per_sm // threads, call.smem_per_sm // row)
        return {
            **super()._device_terms(call),
            "staged_threads": threads,
            "wave_rows": call.sm_count * per_sm,
        }

    def __init__(self, *args, staged_threads: int, wave_rows: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._staged_threads = staged_threads
        self._wave_rows = wave_rows

    def _shape(self, tokens: int) -> int | None:
        """``None`` for a call whose rows (a query row and a key row a token) outnumber one
        wave of staged blocks, else the positions kernel's parts."""
        if tokens * _staged_sides(self.key_heads) > self._wave_rows:
            return None
        return super()._shape(tokens)

    def _compile(self, cfg: dict, parts: int | None):
        """The staged program for *parts* ``None``, else the positions kernel's."""
        if parts is None:
            return self.kernel(self._staged_threads, 1, True, False)
        return super()._compile(cfg, parts)


class RoPENeoxStagedRowsKernel(_RoPEStagedRowsKernel, RoPENeoxPositionIdsFwdInterface):
    """GPT-NeoX rotation of each token's heads, rows staged where heads copy columns."""

    _NEOX = True
    preferred_over = frozenset({"rope_neox_positions"})


class RoPENonNeoxStagedRowsKernel(_RoPEStagedRowsKernel, RoPENonNeoxFwdInterface):
    """RoFormer (GPT-J) rotation of each token's heads, rows staged where heads copy columns."""

    _NEOX = False
    preferred_over = frozenset({"rope_gptj"})


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
    contiguous: bool,
) -> object:
    """The rotation of ``x`` into ``y``, both flat views addressed at runtime strides.

    Element ``(b, i, h, c)`` of ``x`` sits at ``(b * qb + i * qs + h * qh) * vec + c * qc``
    (``i`` the sequence position dense, the token packed), ``y``'s likewise with ``p*``. Every
    stride but the column's is passed divided by ``vec`` so TileLang proves each unit's start
    vector-aligned. Where ``unit_columns`` the column strides are 1, otherwise ``vec`` is 1;
    where ``contiguous`` the strides follow from ``heads``, ``head_dim`` and the token count. In
    place, ``y`` is ``x`` and the copied columns are not touched.

    A unit is ``vec`` columns of the rotary low half and the ``vec`` facing them in the high half
    (NeoX, or the adjacent pairs inside each vector, interleaved), or one ``vec``-column vector
    past the rotary columns, copied. A thread takes one unit position of ``per_thread`` heads of
    one token; packed tokens find their table row by a binary search over ``cu_seqlens``.
    """
    rotary = 2 * rotary_half
    rot_units = rotary_half // vec
    # One column a vector holds no pair: the unit is then one adjacent pair.
    adjacent = interleaved and vec == 1
    head_units = rot_units + (0 if inplace else (head_dim - rotary) // vec)

    # TileLang proves no flat-index access in bounds, and its inserted check would split every
    # vector; `item < units` guards each thread of a grid that exceeds the units.
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
        def rotate(x, y, cos_table, sin_table, cu_seqlens, offsets, offset, tokens, extent, q):
            # Packed, one grid row; dense, a grid row a sequence.
            with T.Kernel(
                T.Cast("int32", T.ceildiv(tokens * token_items, threads)),
                1 if seqlens_dtype is not None else T.Cast("int32", extent),
                threads=threads,
            ) as (bx, by):
                # Dense, tokens is S and extent B: grid row by is sequence b, so no thread divides
                # by a runtime extent. Packed, tokens is Tt and extent the steps of the search for
                # a token's sequence. cu_seqlens and offsets are read only where the call has them.
                if contiguous:
                    head = head_dim // vec
                    q = (tokens * heads * head, heads * head, head, 1) * 2
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
                # Whole blocks of a token's units make the grid exact; programs of several heads a
                # thread keep the check.
                if (per_thread == 1 and token_items % threads == 0) or item < units:
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
                        if offset_dtype == "int64":
                            # An offset past the table first drops to its last row, so the sum
                            # cannot wrap.
                            seen = s + T.min(offsets[b], T.Cast("int64", rows) - 1)
                        elif offset_dtype is not None:
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
                                    c = T.Cast("float32", cs[p])
                                    sv = T.Cast("float32", sn[p])
                                    _rotate_pair(v, a, v, e, c, sv)
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
                rotate(x, y, cos_table, sin_table, cu_seqlens, offsets, 0, tokens, extent, q)

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
                rotate(x, y, cos_table, sin_table, cu_seqlens, x, offset, tokens, extent, q)

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
                rotate(x, y, cos_table, sin_table, x, offsets, 0, tokens, extent, q)

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
                rotate(x, y, cos_table, sin_table, x, x, offset, tokens, extent, q)

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
        # One head a thread where the units fill at most half the resident threads, as in
        # `_RoPEPositionsKernel.forward`.
        copied = 0 if self.inplace else self.head_dim - 2 * self.rotary_half
        head_units = self.rotary_half // vec + copied // vec
        tokens = x.shape[0] if packed else x.shape[0] * x.shape[1]
        wide = 2 * tokens * self.heads * head_units <= self._resident_threads
        # Such a call's time is mostly one thread's chain of dependent instructions, which strides
        # known at compile time shorten.
        contiguous = wide and x.is_contiguous()
        program = self.program(self._compile, vec, unit_columns, wide, contiguous)
        launch_tokens = x.shape[0] if packed else x.shape[1]
        # The offsets are a tensor argument, or an int scalar after the tensors.
        offsets_tensor = isinstance(seqlen_offsets, torch.Tensor)
        if not offsets_tensor:
            # Clamped on the host, an int offset reaches the same rows and its sum with a
            # position cannot wrap in the kernel.
            seqlen_offsets = max(-launch_tokens, min(seqlen_offsets, cos.shape[0] - 1))
        strides = [stride // vec for stride in x_strides[:3]] + [x_strides[3]]
        strides += [stride // vec for stride in y_strides[:3]] + [y_strides[3]]
        if packed:
            # The steps of a binary search over the sequences.
            steps = (cu_seqlens.shape[0] - 2).bit_length()
            launches = [(x, y, seqlen_offsets, [x.shape[0], steps])]
        else:
            # A dense launch runs one grid row a sequence; more sequences launch in parts.
            rows = MAX_GRID_YZ_BLOCKS
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

    def _compile(self, cfg: dict, vec: int, unit_columns: bool, wide: bool, contiguous: bool):
        """The program ``forward`` launches for *vec*-wide vectors, compiled from *cfg*;
        *wide* gives each thread one head and *contiguous* derives its strides from the call."""
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
            contiguous,
        )
        return kernel(cfg["threads"], 1 if wide else cfg["heads"])
