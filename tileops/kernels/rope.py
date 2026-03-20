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

from tileops.kernels.kernel import Kernel

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

__all__ = [
    "RopeNeoxKernel",
    "RopeNonNeoxKernel",
    "RopeLlama31Kernel",
    "RopeYarnKernel",
    "RopeLongRopeKernel",
]


# ---------------------------------------------------------------------------
# Kernel factories for 1D and 2D layouts
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _make_rope_neox_1d(seq_len: int, head_dim: int, dtype: str,
                       threads: int = 256, num_per_thread: int = 8) -> object:
    """1D neox RoPE kernel: (seq_len, head_dim) x cos(seq_len, half) x sin(seq_len, half).

    cos/sin are of shape (seq_len, head_dim // 2).
    The kernel internally constructs full-dim cos/sin by duplication.
    """
    half = head_dim // 2
    N_total = seq_len * head_dim
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
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat_idx = (bx * threads_arg + i) * npt_arg + j
                    row = flat_idx // head_dim
                    col = flat_idx % head_dim
                    # Neox: first half uses cos[col], second half uses cos[col - half]
                    freq_idx = col % half
                    c = cos_table[row, freq_idx]
                    s = sin_table[row, freq_idx]
                    val = x[row, col]
                    # rotate_half: if col < half -> paired with x[row, col + half] (negated)
                    # if col >= half -> paired with x[row, col - half]
                    rotated = T.if_then_else(
                        col < half,
                        -x[row, col + half],
                        x[row, col - half],
                    )
                    y[row, col] = val * c + rotated * s

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_rope_neox_2d(batch: int, seq_len: int, num_heads: int, head_dim: int,
                       dtype: str, threads: int = 256, num_per_thread: int = 8) -> object:
    """2D neox RoPE kernel: (batch, seq_len, num_heads, head_dim).

    cos/sin are of shape (seq_len, head_dim // 2), broadcast over batch and heads.
    """
    half = head_dim // 2
    N_total = batch * seq_len * num_heads * head_dim
    block_size = threads * num_per_thread
    stride_b = seq_len * num_heads * head_dim
    stride_s = num_heads * head_dim

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((N_total,), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((N_total,), dtype),
        ):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat_idx = (bx * threads_arg + i) * npt_arg + j
                    rem = flat_idx % stride_b
                    s_idx = rem // stride_s
                    rem2 = rem % stride_s
                    col = rem2 % head_dim

                    freq_idx = col % half
                    c = cos_table[s_idx, freq_idx]
                    s = sin_table[s_idx, freq_idx]
                    val = x[flat_idx]

                    # Compute the paired index for rotate_half
                    paired_col = T.if_then_else(col < half, col + half, col - half)
                    head_base = flat_idx - col
                    paired_idx = head_base + paired_col
                    paired_val = x[paired_idx]

                    rotated = T.if_then_else(col < half, -paired_val, paired_val)
                    y[flat_idx] = val * c + rotated * s

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_rope_non_neox_1d(seq_len: int, head_dim: int, dtype: str,
                           threads: int = 256, num_per_thread: int = 8) -> object:
    """1D non-neox (RoFormer) RoPE kernel: adjacent-pair rotation.

    cos/sin shape: (seq_len, head_dim // 2). Applied with interleaving to match
    the adjacent-pair pattern.
    """
    half = head_dim // 2
    N_total = seq_len * head_dim
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
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat_idx = (bx * threads_arg + i) * npt_arg + j
                    row = flat_idx // head_dim
                    col = flat_idx % head_dim
                    freq_idx = col // 2  # pair index
                    c = cos_table[row, freq_idx]
                    s = sin_table[row, freq_idx]
                    val = x[row, col]
                    # For even col: paired with x[row, col+1] (negated)
                    # For odd col: paired with x[row, col-1]
                    is_even = (col % 2) == 0
                    rotated = T.if_then_else(
                        is_even,
                        -x[row, col + 1],
                        x[row, col - 1],
                    )
                    y[row, col] = val * c + rotated * s

        return main

    return kernel


@functools.lru_cache(maxsize=32)
def _make_rope_non_neox_2d(batch: int, seq_len: int, num_heads: int, head_dim: int,
                           dtype: str, threads: int = 256,
                           num_per_thread: int = 8) -> object:
    """2D non-neox (RoFormer) RoPE kernel: (batch, seq_len, num_heads, head_dim)."""
    half = head_dim // 2
    N_total = batch * seq_len * num_heads * head_dim
    block_size = threads * num_per_thread
    stride_b = seq_len * num_heads * head_dim
    stride_s = num_heads * head_dim

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((N_total,), dtype),
            cos_table: T.Tensor((seq_len, half), dtype),
            sin_table: T.Tensor((seq_len, half), dtype),
            y: T.Tensor((N_total,), dtype),
        ):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat_idx = (bx * threads_arg + i) * npt_arg + j
                    rem = flat_idx % stride_b
                    s_idx = rem // stride_s
                    rem2 = rem % stride_s
                    col = rem2 % head_dim

                    freq_idx = col // 2
                    c = cos_table[s_idx, freq_idx]
                    s = sin_table[s_idx, freq_idx]
                    val = x[flat_idx]

                    # Adjacent pair: even pairs with next, odd pairs with prev
                    is_even = (col % 2) == 0
                    paired_col = T.if_then_else(is_even, col + 1, col - 1)
                    head_base = flat_idx - col
                    paired_idx = head_base + paired_col
                    paired_val = x[paired_idx]

                    rotated = T.if_then_else(is_even, -paired_val, paired_val)
                    y[flat_idx] = val * c + rotated * s

        return main

    return kernel


# ---------------------------------------------------------------------------
# Kernel base class for RoPE
# ---------------------------------------------------------------------------


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

    def __init__(self, seq_len: int, head_dim: int, dtype: torch.dtype,
                 layout: str = "1d", batch: int = 1, num_heads: int = 1,
                 config: dict | None = None, tune: bool = False):
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
                    self.seq_len, self.head_dim, dtype_str,
                    threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
                )
            else:
                return _make_rope_neox_2d(
                    self.batch, self.seq_len, self.num_heads, self.head_dim,
                    dtype_str, threads=cfg["threads"],
                    num_per_thread=cfg["num_per_thread"],
                )
        elif self.ROTATION_STYLE == "non_neox":
            if self.layout == "1d":
                return _make_rope_non_neox_1d(
                    self.seq_len, self.head_dim, dtype_str,
                    threads=cfg["threads"], num_per_thread=cfg["num_per_thread"],
                )
            else:
                return _make_rope_non_neox_2d(
                    self.batch, self.seq_len, self.num_heads, self.head_dim,
                    dtype_str, threads=cfg["threads"],
                    num_per_thread=cfg["num_per_thread"],
                )
        else:
            raise ValueError(f"Unknown rotation style: {self.ROTATION_STYLE}")

    @property
    def default_config(self) -> dict:
        npt = 4 if self.dtype == torch.float32 else 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Concrete kernel classes (5 variants)
# ---------------------------------------------------------------------------


class RopeNeoxKernel(_RopeKernelBase):
    """GPT-NeoX style RoPE kernel.

    Rotation: split dimension at midpoint, rotate_half = concat(-x2, x1).
    Reference: GPT-NeoX / HuggingFace transformers RotaryEmbedding.
    """

    ROTATION_STYLE = "neox"


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
