"""Rotary position embedding kernels.

The ops compute their cos/sin tables; a kernel receives them and rotates each token's query
and key heads at the table row its position names:
    y = x * cos + rotate(x) * sin

- ``RoPENeoxPositionIdsKernel``: NeoX halves, ``(x[k], x[k + rotary_dim / 2])``.
- ``RoPENonNeoxKernel``: RoFormer (GPT-J) adjacent pairs, ``(x[2k], x[2k + 1])``.

Both rotate the leading ``rotary_dim`` columns of each head and copy the rest, rotate the
query and the key in one launch, and take the token count from the call: one built kernel
serves every count.
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
        """The program ``forward`` launches, compiled from *cfg*; *wide* gives a block a thread
        per unit of its token, whole warps, up to the block limit."""
        if wide:
            return self.kernel(min(1024, -(-self._token_units // 32) * 32), 1)
        return self.kernel(cfg["threads"], cfg["units"])


class RoPENeoxPositionIdsKernel(_RoPEPositionsKernel, RoPENeoxPositionIdsFwdInterface):
    """GPT-NeoX rotation of each token's heads at the table row its position names."""

    _NEOX = True


class RoPENonNeoxKernel(_RoPEPositionsKernel, RoPENonNeoxFwdInterface):
    """RoFormer (GPT-J) adjacent-pair rotation of each token's heads at its position's row.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """

    _NEOX = False
