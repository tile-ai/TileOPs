"""Rotary Position Embedding (RoPE) kernels — 5 variants x 2 layouts.

All 5 variants share the same core rotation logic applied at the kernel level:
    y = x * cos + rotate(x) * sin

The variants differ only in how the frequency tables (cos, sin) are computed,
which is handled in the Op layer (Python). The kernel receives pre-computed
cos/sin tables.

Two rotation styles exist at the kernel level:
- **Neox-style** (used by neox, llama31, yarn, longrope): split x into halves,
  rotate_half = concat(-x2, x1).
- **Non-neox** (original RoFormer): adjacent-pair rotation,
  rotate_pairs = interleave(-x_odd, x_even).

Layouts:
- 1D: (seq_len, head_dim) — single-head or pre-reshaped
- 2D: (batch, seq_len, num_heads, head_dim) — multi-head batched

Each kernel class uses the ``explicit_parallel`` strategy:
    Global → Register → Compute → Register → Global
"""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

__all__ = [
    "RopeLlama31Kernel",
    "RopeLongRopeKernel",
    "RopeNeoxKernel",
    "RopeNeoxPositionIdsKernel",
    "RopeNonNeoxKernel",
    "RopeYarnKernel",
]


# Kernel factories for 1D and 2D layouts


@functools.lru_cache(maxsize=32)
def _make_rope_neox_1d(
    seq_len: int, head_dim: int, dtype: str, threads: int = 256, num_per_thread: int = 8
) -> object:
    """1D neox RoPE kernel: (seq_len, head_dim) x cos(seq_len, half) x sin(seq_len, half).

    cos/sin are of shape (seq_len, head_dim // 2), one entry per rotated pair.
    One thread per pair ``(c, c + half)``; the arithmetic is f32 and rounds once.
    """
    half = head_dim // 2
    n_pairs = seq_len * half
    block_size = threads * num_per_thread

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((seq_len, head_dim), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((seq_len, head_dim), dtype),
        ):
            with T.Kernel(T.ceildiv(n_pairs, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    pair_idx = (bx * threads_arg + i) * npt_arg + j
                    if pair_idx < n_pairs:
                        row = pair_idx // half
                        col = pair_idx % half
                        c = T.Cast("float32", cos_table[row, col])
                        s = T.Cast("float32", sin_table[row, col])
                        x_low = T.Cast("float32", x[row, col])
                        x_high = T.Cast("float32", x[row, col + half])
                        y[row, col] = T.Cast(dtype, x_low * c - x_high * s)
                        y[row, col + half] = T.Cast(dtype, x_high * c + x_low * s)

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_rope_neox_2d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: str,
    threads: int = 256,
    num_per_thread: int = 8,
) -> object:
    """2D neox RoPE kernel: (batch, seq_len, num_heads, head_dim).

    cos/sin are of shape (seq_len, head_dim // 2), broadcast over batch and heads.
    One thread per pair ``(c, c + half)``; the arithmetic is f32 and rounds once.
    """
    half = head_dim // 2
    n_total = batch * seq_len * num_heads * head_dim
    n_pairs = batch * seq_len * num_heads * half
    block_size = threads * num_per_thread

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((n_total,), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((n_total,), dtype),
        ):
            with T.Kernel(T.ceildiv(n_pairs, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    pair_idx = (bx * threads_arg + i) * npt_arg + j
                    if pair_idx < n_pairs:
                        head = pair_idx // half
                        col = pair_idx % half
                        s_idx = (head // num_heads) % seq_len
                        low = head * head_dim + col
                        c = T.Cast("float32", cos_table[s_idx, col])
                        s = T.Cast("float32", sin_table[s_idx, col])
                        x_low = T.Cast("float32", x[low])
                        x_high = T.Cast("float32", x[low + half])
                        y[low] = T.Cast(dtype, x_low * c - x_high * s)
                        y[low + half] = T.Cast(dtype, x_high * c + x_low * s)

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_rope_non_neox_1d(
    seq_len: int, head_dim: int, dtype: str, threads: int = 256, num_per_thread: int = 8
) -> object:
    """1D non-neox (RoFormer) RoPE kernel: adjacent-pair rotation.

    cos/sin shape: (seq_len, head_dim // 2), one entry per pair.
    The arithmetic is f32 and rounds once.
    """
    half = head_dim // 2
    n_pairs = seq_len * half
    block_size = threads * num_per_thread
    rows_per_block = block_size // head_dim
    staged = (
        num_per_thread % 2 == 0
        and rows_per_block > 0
        and block_size % head_dim == 0
        and seq_len % rows_per_block == 0
    )

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((seq_len, head_dim), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((seq_len, head_dim), dtype),
        ):
            if staged:
                with T.Kernel(seq_len // rows_per_block, threads=threads_arg) as bx:
                    xs = T.alloc_shared((rows_per_block, head_dim), dtype)
                    ys = T.alloc_shared((rows_per_block, head_dim), dtype)
                    row0 = bx * rows_per_block
                    T.copy(x[row0 : row0 + rows_per_block, :], xs)
                    for i, j in T.Parallel(threads_arg, npt_arg // 2):
                        slot = i * npt_arg + j * 2
                        row = slot // head_dim
                        col = slot % head_dim
                        pair = col // 2
                        c = T.Cast("float32", cos_table[row0 + row, pair])
                        s = T.Cast("float32", sin_table[row0 + row, pair])
                        x_even = T.Cast("float32", xs[row, col])
                        x_odd = T.Cast("float32", xs[row, col + 1])
                        ys[row, col] = T.Cast(dtype, x_even * c - x_odd * s)
                        ys[row, col + 1] = T.Cast(dtype, x_odd * c + x_even * s)
                    T.copy(ys, y[row0 : row0 + rows_per_block, :])
            else:
                with T.Kernel(T.ceildiv(n_pairs, block_size), threads=threads_arg) as bx:
                    for i, j in T.Parallel(threads_arg, npt_arg):
                        pair_idx = (bx * threads_arg + i) * npt_arg + j
                        if pair_idx < n_pairs:
                            row = pair_idx // half
                            pair = pair_idx % half
                            c = T.Cast("float32", cos_table[row, pair])
                            s = T.Cast("float32", sin_table[row, pair])
                            x_even = T.Cast("float32", x[row, pair * 2])
                            x_odd = T.Cast("float32", x[row, pair * 2 + 1])
                            y[row, pair * 2] = T.Cast(dtype, x_even * c - x_odd * s)
                            y[row, pair * 2 + 1] = T.Cast(dtype, x_odd * c + x_even * s)

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_rope_neox_position_ids_thd(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    dtype: str,
    threads: int = 256,
    num_per_thread: int = 8,
) -> object:
    """THD neox RoPE kernel with explicit absolute position ids.

    A thread owns the pair ``(c, c + half)`` a neox rotation couples, so ``x`` is
    read once and both of its outputs leave in the same step: the walked space is
    the rotated half, not the head. Where ``rotary_dim < head_dim`` a second walk
    copies the columns past it. The rotation runs in f32 and rounds once, at the
    store into ``y``.

    ``status`` counts the positions seen outside ``[0, max_position)``. It is
    reported from a walk over the token axis, which is ``num_heads * half`` times
    shorter than the rotation's, so the rotation stays branch-free; the rotation
    clamps its own table index so an out-of-range position cannot fault before the
    caller reads the count back. The count only grows, which is what lets one
    buffer serve every call without a reset: a caller raises when it moves.
    """
    half = rotary_dim // 2
    token_stride = num_heads * head_dim
    n_total = num_tokens * token_stride
    n_pairs = num_tokens * num_heads * half
    n_tail = num_tokens * num_heads * (head_dim - rotary_dim)
    block_size = threads * num_per_thread
    # One grid covers both walks, so the copied columns get blocks of their own
    # where there are more of them than there are rotated pairs.
    n_walked = max(n_pairs, n_tail)

    @tilelang.jit(out_idx=[5])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((n_total,), dtype),
            cos_table: T.Tensor((max_position, half), dtype),
            sin_table: T.Tensor((max_position, half), dtype),
            position_ids: T.Tensor((num_tokens,), "int32"),
            status: T.Tensor((1,), "int32"),
            y: T.Tensor((n_total,), dtype),
        ):
            with T.Kernel(T.ceildiv(n_walked, block_size), threads=threads_arg) as bx:
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


@functools.lru_cache(maxsize=32)
def _make_rope_non_neox_2d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: str,
    threads: int = 256,
    num_per_thread: int = 8,
) -> object:
    """2D non-neox (RoFormer) RoPE kernel: (batch, seq_len, num_heads, head_dim).

    The arithmetic is f32 and rounds once.
    """
    half = head_dim // 2
    n_total = batch * seq_len * num_heads * head_dim
    n_pairs = n_total // 2
    block_size = threads * num_per_thread

    staged = num_per_thread % 2 == 0 and n_total % block_size == 0 and block_size % head_dim == 0

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((n_total,), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((n_total,), dtype),
        ):
            if staged:
                with T.Kernel(T.ceildiv(n_total, block_size), threads=threads_arg) as bx:
                    xs = T.alloc_shared((block_size,), dtype)
                    ys = T.alloc_shared((block_size,), dtype)
                    T.copy(x[bx * block_size : (bx + 1) * block_size], xs)
                    for i, j in T.Parallel(threads_arg, npt_arg // 2):
                        slot = i * npt_arg + j * 2
                        idx = bx * block_size + slot
                        head = idx // head_dim
                        pair = (idx % head_dim) // 2
                        s_idx = (head // num_heads) % seq_len
                        c = T.Cast("float32", cos_table[s_idx, pair])
                        s = T.Cast("float32", sin_table[s_idx, pair])
                        x_even = T.Cast("float32", xs[slot])
                        x_odd = T.Cast("float32", xs[slot + 1])
                        ys[slot] = T.Cast(dtype, x_even * c - x_odd * s)
                        ys[slot + 1] = T.Cast(dtype, x_odd * c + x_even * s)
                    T.copy(ys, y[bx * block_size : (bx + 1) * block_size])
            else:
                with T.Kernel(T.ceildiv(n_pairs, block_size), threads=threads_arg) as bx:
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


# Kernel base class for RoPE


class _RopeKernelBase(Kernel):
    """Base class for all RoPE kernel variants.

    The core rotation is performed in the TileLang kernel.
    Variant-specific frequency computation is done in the Op layer.

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension (must be even).
        dtype: Torch dtype.
        layout: "1d" for (seq_len, head_dim) or "2d" for
            (batch, seq_len, num_heads, head_dim).
        batch: Batch size (required for 2d layout).
        num_heads: Number of heads (required for 2d layout).
        config: Optional config dict.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES
    ROTATION_STYLE: str = "neox"  # "neox" or "non_neox"

    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        layout: str = "1d",
        batch: int = 1,
        num_heads: int = 1,
        config: dict | None = None,
        tune: bool = False,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if layout not in ("1d", "2d"):
            raise ValueError(f"layout must be '1d' or '2d', got '{layout}'")

        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.layout = layout
        self.batch = batch
        self.num_heads = num_heads

        self.kernel = self._build_kernel()
        self.init_config(config, tune)

    def _build_kernel(self) -> object:
        cfg = self.default_config
        dtype_str = self.dtype_to_str(self.dtype)

        if self.ROTATION_STYLE == "neox":
            if self.layout == "1d":
                return _make_rope_neox_1d(
                    self.seq_len,
                    self.head_dim,
                    dtype_str,
                    threads=cfg["threads"],
                    num_per_thread=cfg["num_per_thread"],
                )
            else:
                return _make_rope_neox_2d(
                    self.batch,
                    self.seq_len,
                    self.num_heads,
                    self.head_dim,
                    dtype_str,
                    threads=cfg["threads"],
                    num_per_thread=cfg["num_per_thread"],
                )
        elif self.ROTATION_STYLE == "non_neox":
            if self.layout == "1d":
                return _make_rope_non_neox_1d(
                    self.seq_len,
                    self.head_dim,
                    dtype_str,
                    threads=cfg["threads"],
                    num_per_thread=cfg["num_per_thread"],
                )
            else:
                return _make_rope_non_neox_2d(
                    self.batch,
                    self.seq_len,
                    self.num_heads,
                    self.head_dim,
                    dtype_str,
                    threads=cfg["threads"],
                    num_per_thread=cfg["num_per_thread"],
                )
        else:
            raise ValueError(f"Unknown rotation style: {self.ROTATION_STYLE}")

    @property
    def default_config(self) -> dict:
        if self.ROTATION_STYLE == "non_neox":
            npt = 8 if self.dtype == torch.float32 else 16
            return {"threads": 128, "num_per_thread": npt}
        npt = 2 if self.dtype == torch.float32 else 4
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation.

        Args:
            x: Input tensor. 1D: (seq_len, head_dim), 2D: (batch, seq_len, num_heads, head_dim).
            cos: Cosine table of shape (seq_len, head_dim // 2).
            sin: Sine table of shape (seq_len, head_dim // 2).

        Returns:
            Rotated tensor of same shape as x.
        """
        cfg = self.config
        orig_shape = x.shape
        if self.layout == "2d":
            x_flat = x.contiguous().reshape(-1)
            result = self.kernel(cfg["threads"], cfg["num_per_thread"])(x_flat, cos, sin)
            return result.reshape(orig_shape)
        else:
            return self.kernel(cfg["threads"], cfg["num_per_thread"])(x, cos, sin)


# Concrete kernel classes (5 variants)


class RopeNeoxKernel(_RopeKernelBase):
    """GPT-NeoX style RoPE kernel.

    Rotation: split dimension at midpoint, rotate_half = concat(-x2, x1).
    Reference: GPT-NeoX / HuggingFace transformers RotaryEmbedding.
    """

    ROTATION_STYLE = "neox"


class RopeNeoxPositionIdsKernel(Kernel):
    """GPT-NeoX style RoPE kernel for packed THD tensors with explicit positions."""

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        head_dim: int,
        rotary_dim: int,
        max_position: int,
        dtype: torch.dtype,
        config: dict | None = None,
        tune: bool = False,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
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
        #: Grows by one per position seen outside ``[0, max_position)``. One buffer
        #: serves every call because the count is never reset; ``out_of_range_since``
        #: answers whether it moved.
        self._status: torch.Tensor | None = None
        self._seen_out_of_range = 0
        self.kernel = self._build_kernel()
        self.init_config(config, tune)

    def _build_kernel(self) -> object:
        cfg = self.default_config
        return _make_rope_neox_position_ids_thd(
            self.num_tokens,
            self.num_heads,
            self.head_dim,
            self.rotary_dim,
            self.max_position,
            self.dtype_to_str(self.dtype),
            threads=cfg["threads"],
            num_per_thread=cfg["num_per_thread"],
        )

    @property
    def default_config(self) -> dict:
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def take_out_of_range(self) -> bool:
        """Whether a call since the previous ask saw a position outside the table.

        One device read per ask, and the count it compares against is held here, so
        a caller pays one synchronisation rather than one before and one after.
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
        cfg = self.config
        orig_shape = x.shape
        if self._status is None:
            self._status = torch.zeros(1, device=x.device, dtype=torch.int32)
        result = self.kernel(cfg["threads"], cfg["num_per_thread"])(
            x.contiguous().reshape(-1),
            cos,
            sin,
            position_ids.contiguous(),
            self._status,
        )
        return result.reshape(orig_shape)


class RopeNonNeoxKernel(_RopeKernelBase):
    """Original RoFormer RoPE kernel with adjacent-pair rotation.

    Rotation: pairs (x_even, x_odd) -> (-x_odd, x_even).
    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """

    ROTATION_STYLE = "non_neox"


class RopeLlama31Kernel(_RopeKernelBase):
    """Llama 3.1 RoPE kernel.

    Same neox rotation as standard RoPE; differs in frequency computation
    (handled by Op layer with piecewise scaling).
    Reference: Meta Llama 3.1 model implementation.
    """

    ROTATION_STYLE = "neox"


class RopeYarnKernel(_RopeKernelBase):
    """YaRN RoPE kernel.

    Same neox rotation; differs in frequency computation (YaRN linear ramp
    interpolation, handled by Op layer).
    Reference: Peng et al., "YaRN: Efficient Context Window Extension of LLMs".
    """

    ROTATION_STYLE = "neox"


class RopeLongRopeKernel(_RopeKernelBase):
    """LongRoPE kernel.

    Same neox rotation; differs in frequency computation (per-dimension
    rescale factors, handled by Op layer).
    Reference: Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2M Tokens".
    """

    ROTATION_STYLE = "neox"
