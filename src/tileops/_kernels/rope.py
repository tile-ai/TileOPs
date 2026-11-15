"""Rotary position embedding kernels.

The ops compute their cos/sin tables; a kernel receives them and rotates:
    y = x * cos + rotate(x) * sin

- ``RoPENeoxPositionIdsKernel``: NeoX halves, each token at the table row its position
  names, the leading ``rotary_dim`` columns of each head rotated and the rest copied.
- ``RoPENonNeoxKernel``: RoFormer adjacent pairs over ``(batch, seq_len, heads, dim)``,
  token ``s`` at row ``s``; an op gathers the rows its positions name first.
"""

import dataclasses
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

__all__ = [
    "RoPECall",
    "RoPENeoxPositionIdsCall",
    "RoPENeoxPositionIdsFwdInterface",
    "RoPENeoxPositionIdsKernel",
    "RoPENonNeoxFwdInterface",
    "RoPENonNeoxKernel",
]


@dataclasses.dataclass(frozen=True)
class RoPECall(CallSpec):
    """One rotation over a tensor the caller also supplies the frequency tables for."""

    seq_len: int = 0
    head_dim: int = 0
    batch: int = 1
    num_heads: int = 1
    dtype: torch.dtype | None = None


@dataclasses.dataclass(frozen=True)
class RoPENeoxPositionIdsCall(CallSpec):
    """One rotation over packed tokens, each reading the table row its position names."""

    num_tokens: int = 0
    num_heads: int = 0
    head_dim: int = 0
    rotary_dim: int = 0
    max_position: int = 0
    dtype: torch.dtype | None = None
    position_dtype: torch.dtype = torch.int32


class RoPENonNeoxFwdInterface(KernelInterface):
    """The RoFormer adjacent-pair rotation of a tensor by caller-supplied cos/sin tables."""

    request = RoPECall

    @abstractmethod
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate each adjacent pair of a head; nothing is written in place.

        Pair ``k`` of a head, ``(x[2k], x[2k + 1])``, becomes
        ``(x[2k] * cos[k] - x[2k + 1] * sin[k], x[2k + 1] * cos[k] + x[2k] * sin[k])``, the
        table row being the token's position along the sequence axis. Nothing is written in
        place, and the output aliases no input.

        Args:
            x: The input, ``(call.batch, call.seq_len, call.num_heads, call.head_dim)`` in
                ``call.dtype`` on ``call.device``, contiguous in that axis order. The op
                makes it contiguous before the call.
            cos: The cosine table, ``(call.seq_len, call.head_dim // 2)`` in ``call.dtype``
                on ``call.device``, contiguous, row-major.
            sin: The sine table, in the same shape, dtype, device and layout.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *x*.
        """


class RoPENeoxPositionIdsFwdInterface(KernelInterface):
    """The GPT-NeoX rotation of packed tokens, each at the position it names."""

    request = RoPENeoxPositionIdsCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """Rotate each token's leading rotary columns; nothing is written in place.

        Token ``t`` reads table row ``position_ids[t]`` and rotates the first
        ``call.rotary_dim`` columns of each of its heads by the half-split convention; the
        remaining columns are copied. A position outside ``[0, call.max_position)`` is
        clamped to the table rather than read out of bounds, and counted, which
        ``take_out_of_range`` reports. Nothing is written in place except that counter, and
        the output aliases no input.

        Args:
            x: The input, ``(call.num_tokens, call.num_heads, call.head_dim)`` in
                ``call.dtype`` on ``call.device``, contiguous in that axis order.
            cos: The cosine table, ``(call.max_position, call.rotary_dim // 2)`` in
                ``call.dtype`` on ``call.device``, contiguous, row-major.
            sin: The sine table, in the same shape, dtype, device and layout.
            position_ids: ``(call.num_tokens,)`` in ``call.position_dtype``, ``int32`` or
                ``int64``, on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *x*.
        """

    @abstractmethod
    def take_out_of_range(self) -> bool:
        """Whether a call since the previous ask saw a position outside the table.

        The op asks after every call and raises when the answer is true, so an
        implementation that clamps silently still reports here.
        """


@target_cache(maxsize=32)
def _make_rope_neox_position_ids_thd(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    dtype: str,
    position_dtype: str = "int32",
    one_wave: bool = False,
) -> object:
    """THD neox RoPE kernel with explicit absolute position ids.

    ``x`` and ``y`` are flat over ``(num_tokens, num_heads, head_dim)``. Columns past
    ``rotary_dim`` are copied through unrotated. The arithmetic is f32 and rounds
    once.

    A position outside ``[0, max_position)`` is clamped to the table rather than
    faulting, and ``status`` counts how many were seen. The count only grows and is
    never reset, so one buffer serves every call and a caller learns of an out-of-range
    position by the count moving.

    ``one_wave`` builds the body for a grid of at most one wave, whose time is the chain of
    a position load then a table load. A thread there loads its ``num_per_thread`` pairs
    as vectors, the position and the table first; the clamp sits in the table index, where
    the bounds analysis sees it, so the loads carry no branch on the position; and the
    thread holding a token's first pair counts that token's position, adding no load.
    """
    half = rotary_dim // 2
    token_stride = num_heads * head_dim
    n_total = num_tokens * token_stride
    n_pairs = num_tokens * num_heads * half
    n_tail = num_tokens * num_heads * (head_dim - rotary_dim)
    n_walked = max(n_pairs, n_tail)

    @tilelang.jit(out_idx=[5])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((n_total,), dtype),
            cos_table: T.Tensor((max_position, half), dtype),
            sin_table: T.Tensor((max_position, half), dtype),
            position_ids: T.Tensor((num_tokens,), position_dtype),
            status: T.Tensor((1,), "int32"),
            y: T.Tensor((n_total,), dtype),
        ):
            with T.Kernel(T.ceildiv(n_walked, threads_arg * npt_arg), threads=threads_arg) as bx:
                # A thread's pairs share its first pair's row only where they divide the row.
                if one_wave and half % npt_arg == 0:
                    tx = T.get_thread_binding()
                    cs = T.alloc_local([npt_arg], dtype)
                    sn = T.alloc_local([npt_arg], dtype)
                    lo = T.alloc_local([npt_arg], dtype)
                    hi = T.alloc_local([npt_arg], dtype)
                    first = (bx * threads_arg + tx) * npt_arg
                    if first < n_pairs:
                        row = first // half
                        col = first % half
                        seen = position_ids[row // num_heads]
                        if (
                            row % num_heads == 0
                            and col == 0
                            and seen != T.max(0, T.min(seen, max_position - 1))
                        ):
                            T.atomic_add(status[0], 1)
                        for j in T.vectorized(npt_arg):
                            cs[j] = cos_table[T.max(0, T.min(seen, max_position - 1)), col + j]
                        for j in T.vectorized(npt_arg):
                            sn[j] = sin_table[T.max(0, T.min(seen, max_position - 1)), col + j]
                        low = row * head_dim + col
                        for j in T.vectorized(npt_arg):
                            lo[j] = x[low + j]
                        for j in T.vectorized(npt_arg):
                            hi[j] = x[low + half + j]
                        for j in T.unroll(npt_arg):
                            c = T.Cast("float32", cs[j])
                            s = T.Cast("float32", sn[j])
                            x_low = T.Cast("float32", lo[j])
                            x_high = T.Cast("float32", hi[j])
                            lo[j] = T.Cast(dtype, x_low * c - x_high * s)
                            hi[j] = T.Cast(dtype, x_high * c + x_low * s)
                        for j in T.vectorized(npt_arg):
                            y[low + j] = lo[j]
                        for j in T.vectorized(npt_arg):
                            y[low + half + j] = hi[j]
                else:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        token = (bx * threads_arg + i) * npt_arg + j
                        if token < num_tokens:
                            seen = position_ids[token]
                            if seen != T.max(0, T.min(seen, max_position - 1)):
                                T.atomic_add(status[0], 1)
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        pair_idx = (bx * threads_arg + i) * npt_arg + j
                        if pair_idx < n_pairs:
                            row = pair_idx // half
                            col = pair_idx % half
                            pos = T.max(0, T.min(position_ids[row // num_heads], max_position - 1))
                            low = row * head_dim + col
                            x_low = T.Cast("float32", x[low])
                            x_high = T.Cast("float32", x[low + half])
                            c = T.Cast("float32", cos_table[pos, col])
                            s = T.Cast("float32", sin_table[pos, col])
                            y[low] = T.Cast(dtype, x_low * c - x_high * s)
                            y[low + half] = T.Cast(dtype, x_high * c + x_low * s)
                if n_tail > 0:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        tail_idx = (bx * threads_arg + i) * npt_arg + j
                        if tail_idx < n_tail:
                            row = tail_idx // (head_dim - rotary_dim)
                            col = tail_idx % (head_dim - rotary_dim)
                            kept = row * head_dim + rotary_dim + col
                            y[kept] = x[kept]

        return main

    return kernel


@target_cache(maxsize=32)
def _make_rope_non_neox_2d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: str,
) -> object:
    """2D non-neox (RoFormer) RoPE kernel: (batch, seq_len, num_heads, head_dim).

    The arithmetic is f32 and rounds once.
    """
    half = head_dim // 2
    n_total = batch * seq_len * num_heads * head_dim
    n_pairs = n_total // 2

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        block = threads_arg * npt_arg
        staged = npt_arg % 2 == 0 and n_total % block == 0 and block % head_dim == 0

        @T.prim_func
        def main(
            x: T.Tensor((n_total,), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((n_total,), dtype),
        ):
            if staged:
                with T.Kernel(n_total // block, threads=threads_arg) as bx:
                    xs = T.alloc_shared((block,), dtype)
                    ys = T.alloc_shared((block,), dtype)
                    T.copy(x[bx * block : (bx + 1) * block], xs)
                    for i, j in T.Parallel(threads_arg, npt_arg // 2):
                        slot = i * npt_arg + j * 2
                        idx = bx * block + slot
                        head = idx // head_dim
                        pair = (idx % head_dim) // 2
                        s_idx = (head // num_heads) % seq_len
                        c = T.Cast("float32", cos_table[s_idx, pair])
                        s = T.Cast("float32", sin_table[s_idx, pair])
                        x_even = T.Cast("float32", xs[slot])
                        x_odd = T.Cast("float32", xs[slot + 1])
                        ys[slot] = T.Cast(dtype, x_even * c - x_odd * s)
                        ys[slot + 1] = T.Cast(dtype, x_odd * c + x_even * s)
                    T.copy(ys, y[bx * block : (bx + 1) * block])
            else:
                with T.Kernel(T.ceildiv(n_pairs, block), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        pair_idx = (bx * threads_arg + i) * npt_arg + j
                        if pair_idx < n_pairs:
                            head = pair_idx // (head_dim // 2)
                            pair = pair_idx % (head_dim // 2)
                            s_idx = (head // num_heads) % seq_len
                            even = head * head_dim + pair * 2
                            c = T.Cast("float32", cos_table[s_idx, pair])
                            s = T.Cast("float32", sin_table[s_idx, pair])
                            x_even = T.Cast("float32", x[even])
                            x_odd = T.Cast("float32", x[even + 1])
                            y[even] = T.Cast(dtype, x_even * c - x_odd * s)
                            y[even + 1] = T.Cast(dtype, x_odd * c + x_even * s)

        return main

    return kernel


class RoPENeoxPositionIdsKernel(Kernel, RoPENeoxPositionIdsFwdInterface):
    """GPT-NeoX style RoPE kernel for packed THD tensors with explicit positions."""

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @classmethod
    def entry_for(cls, call: RoPENeoxPositionIdsCall) -> Entry:
        return call, lambda: cls(
            num_tokens=call.num_tokens,
            num_heads=call.num_heads,
            head_dim=call.head_dim,
            rotary_dim=call.rotary_dim,
            max_position=call.max_position,
            dtype=call.dtype,
            position_dtype=call.position_dtype,
            sm_count=call.sm_count,
        )

    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        dtype: torch.dtype,
        position_dtype: torch.dtype = torch.int32,
        sm_count: int | None = None,
        config: dict | None = None,
    ):
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > head_dim:
            raise ValueError("rotary_dim must be positive, even, and <= head_dim")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if max_position <= 0:
            raise ValueError(f"max_position must be positive, got {max_position}")

        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        self.dtype = dtype
        if position_dtype not in (torch.int32, torch.int64):
            raise ValueError(f"position_dtype must be int32 or int64, got {position_dtype}")
        self.position_dtype = position_dtype
        self.sm_count = sm_count
        # Grows by one per position seen outside [0, max_position), and is never
        # reset; take_out_of_range answers whether it moved.
        self._status: torch.Tensor | None = None
        self._seen_out_of_range = 0
        self.kernel = self._build_kernel()
        self.init_config(config)

    def _build_kernel(self) -> object:
        return _make_rope_neox_position_ids_thd(
            self.num_tokens,
            self.num_heads,
            self.head_dim,
            self.rotary_dim,
            self.max_position,
            self.dtype_to_str(self.dtype),
            self.dtype_to_str(self.position_dtype),
            one_wave=self._one_wave,
        )

    @property
    def _one_wave(self) -> bool:
        """Whether a 512-thread grid of two pairs a thread fits one wave of the device.

        Such a grid's time is the latency of a position load then a table load; sixteen
        warps a block keep four times the chains in flight that four warps of eight pairs do.
        """
        half = self.rotary_dim // 2
        pairs = self.num_tokens * self.num_heads * half
        return (
            self.sm_count is not None
            and half % 2 == 0
            and self.head_dim == self.rotary_dim
            and -(-pairs // (512 * 2)) <= self.sm_count
        )

    @property
    def default_config(self) -> dict:
        if self._one_wave:
            return {"threads": 512, "num_per_thread": 2}
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def take_out_of_range(self) -> bool:
        """Whether a call since the previous ask saw a position outside the table.

        Costs one device synchronisation per ask.
        """
        if self._status is None:
            return False
        count = int(self._status.item())
        moved = count != self._seen_out_of_range
        self._seen_out_of_range = count
        return moved

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        orig_shape = x.shape
        if self._status is None:
            self._status = torch.zeros(1, device=x.device, dtype=torch.int32)
        result = self.program(self._compile)(
            x.contiguous().reshape(-1),
            cos,
            sin,
            position_ids.contiguous(),
            self._status,
        )
        return result.reshape(orig_shape)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(cfg["threads"], cfg["num_per_thread"])


class RoPENonNeoxKernel(Kernel, RoPENonNeoxFwdInterface):
    """Original RoFormer RoPE kernel with adjacent-pair rotation.

    Rotation: pairs (x_even, x_odd) -> (-x_odd, x_even), on a
    ``(batch, seq_len, num_heads, head_dim)`` input whose token ``s`` reads table row ``s``.
    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    @classmethod
    def entry_for(cls, call: RoPECall) -> Entry:
        return call, lambda: cls(
            call.seq_len, call.head_dim, call.dtype, batch=call.batch, num_heads=call.num_heads
        )

    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        batch: int = 1,
        num_heads: int = 1,
        config: dict | None = None,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.batch = batch
        self.num_heads = num_heads
        self.kernel = _make_rope_non_neox_2d(
            batch, seq_len, num_heads, head_dim, self.dtype_to_str(dtype)
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default threads and columns per thread."""
        npt = 8 if self.dtype == torch.float32 else 16
        return {"threads": 128, "num_per_thread": npt}

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        result = self.program(self._compile)(x.contiguous().reshape(-1), cos, sin)
        return result.reshape(x.shape)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(cfg["threads"], cfg["num_per_thread"])
