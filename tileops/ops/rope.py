"""Rotary Position Embedding (RoPE) ops — 5 variants x 2 layouts.

Each Op computes variant-specific frequency tables (cos, sin) at construction
time and delegates the actual rotation to the corresponding kernel.

Variants and frequency computation:
- **RopeNeoxOp**: standard theta = 10000^(-2k/d) frequencies
- **RopeNonNeoxOp**: same frequencies, different rotation pattern (adjacent pairs)
- **RopeLlama31Op**: piecewise-scaled frequencies for Llama 3.1
- **RopeYarnOp**: YaRN linear-ramp interpolated frequencies
- **RopeLongRopeOp**: per-dimension rescaled frequencies

Layouts:
- ``"1d"``: input shape ``(seq_len, head_dim)``
- ``"2d"``: input shape ``(batch, seq_len, num_heads, head_dim)``
"""

import math
from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.rope import (
    RopeLlama31Kernel,
    RopeLongRopeKernel,
    RopeNeoxKernel,
    RopeNonNeoxKernel,
    RopeYarnKernel,
)

from .op import Op

__all__ = [
    "RopeNeoxOp",
    "RopeNonNeoxOp",
    "RopeLlama31Op",
    "RopeYarnOp",
    "RopeLongRopeOp",
]


# ---------------------------------------------------------------------------
# Frequency computation helpers (pure Python / PyTorch, run on host)
# ---------------------------------------------------------------------------


def _base_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                dtype: torch.dtype = torch.float32,
                device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Standard RoPE cos/sin tables.

    Args:
        head_dim: Head dimension (must be even).
        seq_len: Sequence length.
        base: Frequency base (default 10000).
        dtype: Output dtype.
        device: Torch device.

    Returns:
        (cos, sin) each of shape (seq_len, head_dim // 2).
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _llama31_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                   scale_factor: float = 8.0, low_freq_factor: float = 1.0,
                   high_freq_factor: float = 4.0,
                   original_max_position: int = 8192,
                   dtype: torch.dtype = torch.float32,
                   device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Llama 3.1 piecewise-scaled frequency computation.

    Args:
        head_dim: Head dimension.
        seq_len: Sequence length.
        base: Frequency base.
        scale_factor: Scaling factor for low frequencies.
        low_freq_factor: Threshold for low-frequency wavelengths.
        high_freq_factor: Threshold for high-frequency wavelengths.
        original_max_position: Original maximum position length.
        dtype: Output dtype.
        device: Torch device.

    Returns:
        (cos, sin) each of shape (seq_len, head_dim // 2).
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))

    low_freq_wavelen = original_max_position / low_freq_factor
    high_freq_wavelen = original_max_position / high_freq_factor

    scaled_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq.item()
        if wavelen < high_freq_wavelen:
            scaled_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            scaled_freqs.append(freq / scale_factor)
        else:
            smooth = (original_max_position / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor)
            scaled_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    freqs = torch.stack(scaled_freqs)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _yarn_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                scale: float = 16.0, original_max_position: int = 4096,
                beta_fast: float = 32.0, beta_slow: float = 1.0,
                attn_factor: float = 1.0,
                dtype: torch.dtype = torch.float32,
                device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """YaRN linear-ramp interpolated frequency computation.

    Args:
        head_dim: Head dimension.
        seq_len: Sequence length.
        base: Frequency base.
        scale: Context extension scale factor.
        original_max_position: Original max context length.
        beta_fast: Fast decay boundary parameter.
        beta_slow: Slow decay boundary parameter.
        attn_factor: Attention scaling factor.
        dtype: Output dtype.
        device: Torch device.

    Returns:
        (cos, sin) each of shape (seq_len, head_dim // 2).
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))

    low_dim = max(int(half * math.log(original_max_position / (beta_fast * 2 * math.pi))
                      / math.log(base)), 0)
    high_dim = min(int(half * math.log(original_max_position / (beta_slow * 2 * math.pi))
                       / math.log(base)), half - 1)

    ramp = torch.zeros(half, device=device, dtype=torch.float32)
    if high_dim > low_dim:
        ramp_range = torch.arange(half, device=device, dtype=torch.float32)
        ramp = torch.clamp((ramp_range - low_dim) / (high_dim - low_dim), 0.0, 1.0)

    freqs_scaled = freqs / scale
    freqs_interpolated = (1.0 - ramp) * freqs_scaled + ramp * freqs

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs_interpolated)
    return (torch.cos(angles) * attn_factor).to(dtype), (torch.sin(angles) * attn_factor).to(dtype)


def _longrope_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                    rescale_factors: Optional[torch.Tensor] = None,
                    dtype: torch.dtype = torch.float32,
                    device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """LongRoPE per-dimension rescaled frequency computation.

    Args:
        head_dim: Head dimension.
        seq_len: Sequence length.
        base: Frequency base.
        rescale_factors: Per-dimension rescale factors of shape (head_dim // 2,).
        dtype: Output dtype.
        device: Torch device.

    Returns:
        (cos, sin) each of shape (seq_len, head_dim // 2).
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    if rescale_factors is not None:
        rescale_factors = rescale_factors.to(device=device, dtype=torch.float32)
        freqs = freqs / rescale_factors

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


# ---------------------------------------------------------------------------
# Base Op class for RoPE
# ---------------------------------------------------------------------------


class _RopeOpBase(Op):
    """Base class for all RoPE ops.

    Subclass must set ``kernel_cls``, ``_op_name``, and implement
    ``_compute_cos_sin()`` to generate variant-specific frequency tables.

    At construction time, cos/sin tables are computed from variant parameters
    and cached. The ``forward()`` method accepts only the input tensor ``x``.

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension (must be even).
        dtype: Torch dtype.
        layout: "1d" or "2d".
        batch: Batch size (for 2d layout).
        num_heads: Number of attention heads (for 2d layout).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    kernel_cls: type
    _op_name: str

    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        layout: str = "1d",
        batch: int = 1,
        num_heads: int = 1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.layout = layout
        self.batch = batch
        self.num_heads = num_heads

        # Compute variant-specific cos/sin tables from stored parameters
        self._cos, self._sin = self._compute_cos_sin()

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            seq_len=seq_len, head_dim=head_dim, dtype=dtype,
            layout=layout, batch=batch, num_heads=num_heads, tune=tune,
        )

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute variant-specific cos/sin frequency tables.

        Must be overridden by each concrete subclass to use its stored
        variant parameters.

        Returns:
            (cos, sin) each of shape (seq_len, head_dim // 2).
        """
        raise NotImplementedError(
            "Subclass must implement _compute_cos_sin()"
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._op_name: self.kernel_cls}

    @property
    def total_memory(self) -> float:
        """Read x + cos + sin + write y."""
        half = self.head_dim // 2
        elem = self.dtype.itemsize
        cos_sin_elems = self.seq_len * half * 2
        if self.layout == "1d":
            x_elems = self.seq_len * self.head_dim
        else:
            x_elems = self.batch * self.seq_len * self.num_heads * self.head_dim
        return (2 * x_elems + cos_sin_elems) * elem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation using internally computed cos/sin tables.

        Args:
            x: Input tensor. Shape depends on layout:
                - 1D: ``(seq_len, head_dim)``
                - 2D: ``(batch, seq_len, num_heads, head_dim)``

        Returns:
            Rotated output tensor with same shape as x.
        """
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        return self.kernel(x, self._cos, self._sin)


# ---------------------------------------------------------------------------
# Concrete Op classes (5 variants)
# ---------------------------------------------------------------------------


class RopeNeoxOp(_RopeOpBase):
    """GPT-NeoX style RoPE op with standard theta frequencies.

    Computes cos/sin tables at construction using standard theta = base^(-2k/d).

    Reference: GPT-NeoX / HuggingFace transformers RotaryEmbedding.

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension (must be even).
        dtype: Torch dtype.
        layout: "1d" or "2d".
        batch: Batch size (for 2d).
        num_heads: Number of heads (for 2d).
        base: Frequency base (default 10000).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "rope_neox"
    kernel_cls = RopeNeoxKernel

    def __init__(self, seq_len: int, head_dim: int, dtype: torch.dtype,
                 layout: str = "1d", batch: int = 1, num_heads: int = 1,
                 base: float = 10000.0,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.base = base
        super().__init__(seq_len, head_dim, dtype, layout, batch, num_heads,
                         kernel_map, tune)

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _base_freqs(self.head_dim, self.seq_len, base=self.base,
                           dtype=self.dtype)


class RopeNonNeoxOp(_RopeOpBase):
    """Original RoFormer RoPE op with adjacent-pair rotation.

    Computes cos/sin tables at construction using standard theta = base^(-2k/d).

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension (must be even).
        dtype: Torch dtype.
        layout: "1d" or "2d".
        batch: Batch size (for 2d).
        num_heads: Number of heads (for 2d).
        base: Frequency base (default 10000).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "rope_non_neox"
    kernel_cls = RopeNonNeoxKernel

    def __init__(self, seq_len: int, head_dim: int, dtype: torch.dtype,
                 layout: str = "1d", batch: int = 1, num_heads: int = 1,
                 base: float = 10000.0,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.base = base
        super().__init__(seq_len, head_dim, dtype, layout, batch, num_heads,
                         kernel_map, tune)

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _base_freqs(self.head_dim, self.seq_len, base=self.base,
                           dtype=self.dtype)


class RopeLlama31Op(_RopeOpBase):
    """Llama 3.1 RoPE op with piecewise frequency scaling.

    Computes cos/sin tables at construction using Llama 3.1 piecewise-scaled
    frequencies based on wavelength thresholds.

    Reference: Meta Llama 3.1 model implementation.

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension (must be even).
        dtype: Torch dtype.
        layout: "1d" or "2d".
        batch: Batch size (for 2d).
        num_heads: Number of heads (for 2d).
        base: Frequency base (default 10000).
        scale_factor: Scaling factor for low frequencies (default 8.0).
        low_freq_factor: Low-frequency wavelen threshold (default 1.0).
        high_freq_factor: High-frequency wavelen threshold (default 4.0).
        original_max_position: Original max position (default 8192).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "rope_llama31"
    kernel_cls = RopeLlama31Kernel

    def __init__(self, seq_len: int, head_dim: int, dtype: torch.dtype,
                 layout: str = "1d", batch: int = 1, num_heads: int = 1,
                 base: float = 10000.0, scale_factor: float = 8.0,
                 low_freq_factor: float = 1.0, high_freq_factor: float = 4.0,
                 original_max_position: int = 8192,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.base = base
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position = original_max_position
        super().__init__(seq_len, head_dim, dtype, layout, batch, num_heads,
                         kernel_map, tune)

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _llama31_freqs(
            self.head_dim, self.seq_len, base=self.base,
            scale_factor=self.scale_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position=self.original_max_position,
            dtype=self.dtype,
        )


class RopeYarnOp(_RopeOpBase):
    """YaRN RoPE op with linear-ramp frequency interpolation.

    Computes cos/sin tables at construction using YaRN linear-ramp
    interpolation between scaled and original frequencies.

    Reference: Peng et al., "YaRN: Efficient Context Window Extension of LLMs".

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension (must be even).
        dtype: Torch dtype.
        layout: "1d" or "2d".
        batch: Batch size (for 2d).
        num_heads: Number of heads (for 2d).
        base: Frequency base (default 10000).
        scale: Context extension scale (default 16.0).
        original_max_position: Original max position (default 4096).
        beta_fast: Fast decay boundary (default 32.0).
        beta_slow: Slow decay boundary (default 1.0).
        attn_factor: Attention scaling factor (default 1.0).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "rope_yarn"
    kernel_cls = RopeYarnKernel

    def __init__(self, seq_len: int, head_dim: int, dtype: torch.dtype,
                 layout: str = "1d", batch: int = 1, num_heads: int = 1,
                 base: float = 10000.0, scale: float = 16.0,
                 original_max_position: int = 4096,
                 beta_fast: float = 32.0, beta_slow: float = 1.0,
                 attn_factor: float = 1.0,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.base = base
        self.scale = scale
        self.original_max_position = original_max_position
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attn_factor = attn_factor
        super().__init__(seq_len, head_dim, dtype, layout, batch, num_heads,
                         kernel_map, tune)

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _yarn_freqs(
            self.head_dim, self.seq_len, base=self.base,
            scale=self.scale, original_max_position=self.original_max_position,
            beta_fast=self.beta_fast, beta_slow=self.beta_slow,
            attn_factor=self.attn_factor, dtype=self.dtype,
        )


class RopeLongRopeOp(_RopeOpBase):
    """LongRoPE op with per-dimension frequency rescaling.

    Computes cos/sin tables at construction using per-dimension rescale
    factors applied to the base frequencies.

    Reference: Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2M Tokens".

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension (must be even).
        dtype: Torch dtype.
        layout: "1d" or "2d".
        batch: Batch size (for 2d).
        num_heads: Number of heads (for 2d).
        base: Frequency base (default 10000).
        rescale_factors: Per-dimension rescale factors of shape (head_dim // 2,).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "rope_longrope"
    kernel_cls = RopeLongRopeKernel

    def __init__(self, seq_len: int, head_dim: int, dtype: torch.dtype,
                 layout: str = "1d", batch: int = 1, num_heads: int = 1,
                 base: float = 10000.0,
                 rescale_factors: Optional[torch.Tensor] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.base = base
        self.rescale_factors = rescale_factors
        super().__init__(seq_len, head_dim, dtype, layout, batch, num_heads,
                         kernel_map, tune)

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _longrope_freqs(
            self.head_dim, self.seq_len, base=self.base,
            rescale_factors=self.rescale_factors, dtype=self.dtype,
        )
