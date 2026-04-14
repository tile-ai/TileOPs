"""Rotary Position Embedding (RoPE) ops — 5 variants x 2 layouts.

Each Op computes variant-specific frequency tables (cos, sin) lazily at
forward time (on the same device as the input tensor) and delegates the
actual rotation to the corresponding kernel.

Variants and frequency computation:
- **RopeNeoxOp**: standard theta = 10000^(-2k/d) frequencies
- **RopeNonNeoxOp**: same frequencies, different rotation pattern (adjacent pairs)
- **RopeLlama31Op**: piecewise-scaled frequencies for Llama 3.1
- **RopeYarnOp**: YaRN linear-ramp interpolated frequencies
- **RopeLongRopeOp**: per-dimension rescaled frequencies

Layouts:
- ``"1d"``: input shape ``(seq_len, head_dim)``
- ``"2d"``: input shape ``(batch, seq_len, num_heads, head_dim)``

torch.compile support:
- All 5 concrete ops are registered via @torch.library.custom_op at module
  load time.  A factory function (_register_rope_custom_op) registers every
  op; instances are looked up at runtime via _OP_REGISTRY keyed by
  id(instance).
"""

import math
import weakref
from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.rope import (
    RopeLlama31Kernel,
    RopeLongRopeKernel,
    RopeNeoxKernel,
    RopeNonNeoxKernel,
    RopeYarnKernel,
)

from .op_base import Op

# ---------------------------------------------------------------------------
# torch.compile registration factory
#
# Creates a @torch.library.custom_op + register_fake pair for each RoPE op.
# Instances register themselves in _OP_REGISTRY keyed by integer id.
# The custom_op receives this key and looks up the instance to call the
# pre-built tilelang kernel.
# ---------------------------------------------------------------------------

_OP_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _register_rope_custom_op(op_cls):
    """Register a RoPE op for torch.compile.

    Args:
        op_cls: The Op subclass to register (must have ``_op_name``).
    """
    op_name = op_cls._op_name

    @torch.library.custom_op(f"top::rope_{op_name}", mutates_args=())
    def _wrapped(x: torch.Tensor, instance_key: int) -> torch.Tensor:
        instance = _OP_REGISTRY[instance_key]
        return instance._eager_forward(x)

    @_wrapped.register_fake
    def _(x: torch.Tensor, instance_key: int) -> torch.Tensor:
        return torch.empty_like(x)

    op_cls._wrapped = _wrapped


__all__ = [
    "RopeLlama31Op",
    "RopeLongRopeOp",
    "RopeNeoxOp",
    "RopeNonNeoxOp",
    "RopeYarnOp",
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


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float,
                              max_position_embeddings: int) -> float:
    """Inverse dim formula to find dim based on number of rotations.

    Matches the canonical TVM/vLLM ``yarn_find_correction_dim`` formula.
    """
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(beta_fast: float, beta_slow: float, dim: int,
                                base: float,
                                max_position_embeddings: int) -> tuple[int, int]:
    """Find low/high correction dims from rotation boundary parameters."""
    low = math.floor(
        _yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def _yarn_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                scale: float = 16.0, original_max_position: int = 4096,
                beta_fast: float = 32.0, beta_slow: float = 1.0,
                attn_factor: float = 1.0,
                dtype: torch.dtype = torch.float32,
                device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """YaRN frequency computation with NTK-aware interpolation.

    Implements the canonical YaRN formula:
    1. ``freq_extra`` = original inverse frequencies (for extrapolation dims)
    2. ``freq_inter`` = NTK-aware scaled inverse frequencies where
       ``scale`` is applied to the base: ``1/(scale*base)^(2k/d)``
    3. Linear ramp mask between correction dims blends the two
    4. ``inv_freq = freq_inter * (1 - mask) + freq_extra * mask``

    Reference: TVM ``rope_freq_yarn`` in position_embedding.py;
    Peng et al., "YaRN: Efficient Context Window Extension of LLMs".

    Args:
        head_dim: Head dimension.
        seq_len: Sequence length.
        base: Frequency base (theta).
        scale: Context extension scale factor (scaling_factor).
        original_max_position: Original max context length.
        beta_fast: Fast rotation boundary (passed as low_rot).
        beta_slow: Slow rotation boundary (passed as high_rot).
        attn_factor: Attention scaling factor (applied to cos/sin output).
        dtype: Output dtype.
        device: Torch device.

    Returns:
        (cos, sin) each of shape (seq_len, head_dim // 2).
    """
    half = head_dim // 2
    dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)

    # Original inverse frequencies (extrapolation)
    freq_extra = 1.0 / (base ** (dim_indices / half))

    # NTK-aware scaled inverse frequencies (interpolation):
    # scale is applied to the base, not as a divisor on freq
    freq_inter = 1.0 / ((scale * base) ** (dim_indices / half))

    # Find correction range
    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, half, base, original_max_position,
    )
    # Avoid division by zero when low == high
    if low == high:
        high = high + 1

    # Linear ramp mask: 1 near low dims (extrapolation), 0 near high dims (interpolation)
    inv_freq_mask = 1.0 - torch.clamp(
        (dim_indices - low) / (high - low), 0.0, 1.0,
    )

    # Blend: mask=1 -> freq_extra, mask=0 -> freq_inter
    inv_freq = freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    return (torch.cos(angles) * attn_factor).to(dtype), (torch.sin(angles) * attn_factor).to(dtype)


def _longrope_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                    rescale_factors: Optional[torch.Tensor] = None,
                    max_position_embeddings: int = 4096,
                    original_max_position_embeddings: int = 4096,
                    dtype: torch.dtype = torch.float32,
                    device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """LongRoPE per-dimension rescaled frequency computation.

    Implements the canonical LongRoPE formula:
    1. ``divisor = ext_factors[k] * base^(2k/d)`` (ext_factors multiply the
       divisor in the inverse-frequency computation)
    2. ``scaling_factor = sqrt(1 + log(scale) / log(orig_max_pos))``
       where ``scale = max_pos / orig_max_pos`` (amplitude factor applied
       to cos/sin output when scale > 1)

    Reference: TVM ``rope_freq_longrope`` in position_embedding.py;
    Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2M Tokens".

    Args:
        head_dim: Head dimension.
        seq_len: Sequence length.
        base: Frequency base.
        rescale_factors: Per-dimension rescale factors (ext_factors) of
            shape (head_dim // 2,). These multiply the divisor in the
            inverse-frequency formula.
        max_position_embeddings: Extended max position length.
        original_max_position_embeddings: Original max position length.
        dtype: Output dtype.
        device: Torch device.

    Returns:
        (cos, sin) each of shape (seq_len, head_dim // 2).
    """
    half = head_dim // 2
    dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)
    divisor = base ** (dim_indices / half)

    # ext_factors multiply the divisor (matching canonical formula)
    if rescale_factors is not None:
        rf = rescale_factors.to(device=device, dtype=torch.float32)
        divisor = rf * divisor

    freqs = 1.0 / divisor

    # Compute amplitude scaling factor
    scale = max_position_embeddings / original_max_position_embeddings
    if scale > 1.0:
        scaling_factor = math.sqrt(
            1.0 + math.log(scale) / math.log(original_max_position_embeddings)
        )
    else:
        scaling_factor = 1.0

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return (
        (torch.cos(angles) * scaling_factor).to(dtype),
        (torch.sin(angles) * scaling_factor).to(dtype),
    )


# ---------------------------------------------------------------------------
# Base Op class for RoPE
# ---------------------------------------------------------------------------


class _RopeOpBase(Op):
    """Base class for all RoPE ops.

    Subclass must set ``kernel_cls``, ``_op_name``, and implement
    ``_compute_cos_sin(device)`` to generate variant-specific frequency tables.
    Subclass should also set ``_wrapped`` via ``_register_rope_custom_op``
    to enable torch.compile support.

    Cos/sin tables are computed lazily at forward time on the same device as
    the input tensor, avoiding device-mismatch issues in multi-GPU settings.

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
    _wrapped = None  # Set by _register_rope_custom_op at class definition

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

        # Lazy cos/sin cache: computed on first forward call per device
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cache_device: Optional[torch.device] = None

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            seq_len=seq_len, head_dim=head_dim, dtype=dtype,
            layout=layout, batch=batch, num_heads=num_heads, tune=tune,
        )

        # Register in global registry for torch.compile dispatch
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    def _get_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached cos/sin tables, recomputing if device changed."""
        if self._cos_cache is None or self._cache_device != device:
            self._cos_cache, self._sin_cache = self._compute_cos_sin(device=device)
            self._cache_device = device
        return self._cos_cache, self._sin_cache

    def _compute_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute variant-specific cos/sin frequency tables.

        Must be overridden by each concrete subclass to use its stored
        variant parameters.

        Args:
            device: Torch device to create tensors on.

        Returns:
            (cos, sin) each of shape (seq_len, head_dim // 2).
        """
        raise NotImplementedError(
            "Subclass must implement _compute_cos_sin(device)"
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

    def _validate_and_prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Validate input shape/dtype/device and return a contiguous tensor.

        Raises:
            ValueError: If x is not CUDA, has wrong dtype, or wrong shape.
        """
        if not x.is_cuda:
            raise ValueError("Input must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected x.dtype {self.dtype}, got {x.dtype}")
        if self.layout == "1d":
            expected = (self.seq_len, self.head_dim)
        else:
            expected = (self.batch, self.seq_len, self.num_heads, self.head_dim)
        if tuple(x.shape) != expected:
            raise ValueError(
                f"Expected input shape {expected} for layout "
                f"'{self.layout}', got {tuple(x.shape)}"
            )
        return x.contiguous()

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Direct kernel call for use inside custom_op implementation.

        Called from the custom_op wrapper after validation has already
        been performed in ``forward()``.
        """
        cos, sin = self._get_cos_sin(x.device)
        return self.kernel(x, cos, sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation using internally computed cos/sin tables.

        Args:
            x: Input tensor. Shape depends on layout:
                - 1D: ``(seq_len, head_dim)``
                - 2D: ``(batch, seq_len, num_heads, head_dim)``

        Returns:
            Rotated output tensor with same shape as x.
        """
        x = self._validate_and_prepare(x)
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(x, self._instance_key)
        return self._eager_forward(x)


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

    def _compute_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return _base_freqs(self.head_dim, self.seq_len, base=self.base,
                           dtype=self.dtype, device=device)


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

    def _compute_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return _base_freqs(self.head_dim, self.seq_len, base=self.base,
                           dtype=self.dtype, device=device)


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

    def _compute_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return _llama31_freqs(
            self.head_dim, self.seq_len, base=self.base,
            scale_factor=self.scale_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position=self.original_max_position,
            dtype=self.dtype, device=device,
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

    def _compute_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return _yarn_freqs(
            self.head_dim, self.seq_len, base=self.base,
            scale=self.scale, original_max_position=self.original_max_position,
            beta_fast=self.beta_fast, beta_slow=self.beta_slow,
            attn_factor=self.attn_factor, dtype=self.dtype, device=device,
        )


class RopeLongRopeOp(_RopeOpBase):
    """LongRoPE op with per-dimension frequency rescaling.

    Computes cos/sin tables at construction using per-dimension rescale
    factors (ext_factors) that multiply the divisor, plus a scale-dependent
    amplitude factor applied to cos/sin output.

    Reference: TVM ``rope_freq_longrope`` in position_embedding.py;
    Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2M Tokens".

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension (must be even).
        dtype: Torch dtype.
        layout: "1d" or "2d".
        batch: Batch size (for 2d).
        num_heads: Number of heads (for 2d).
        base: Frequency base (default 10000).
        rescale_factors: Per-dimension rescale factors (ext_factors) of shape
            (head_dim // 2,). These multiply the divisor.
        max_position_embeddings: Extended max position length (default 4096).
        original_max_position_embeddings: Original max position length
            (default 4096).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "rope_longrope"
    kernel_cls = RopeLongRopeKernel

    def __init__(self, seq_len: int, head_dim: int, dtype: torch.dtype,
                 layout: str = "1d", batch: int = 1, num_heads: int = 1,
                 base: float = 10000.0,
                 rescale_factors: Optional[torch.Tensor] = None,
                 max_position_embeddings: int = 4096,
                 original_max_position_embeddings: int = 4096,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False):
        self.base = base
        self.rescale_factors = rescale_factors
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        super().__init__(seq_len, head_dim, dtype, layout, batch, num_heads,
                         kernel_map, tune)

    def _compute_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return _longrope_freqs(
            self.head_dim, self.seq_len, base=self.base,
            rescale_factors=self.rescale_factors,
            max_position_embeddings=self.max_position_embeddings,
            original_max_position_embeddings=self.original_max_position_embeddings,
            dtype=self.dtype, device=device,
        )


# ---------------------------------------------------------------------------
# torch.compile registration for all 5 RoPE ops
# ---------------------------------------------------------------------------

for _cls in [RopeNeoxOp, RopeNonNeoxOp, RopeLlama31Op, RopeYarnOp, RopeLongRopeOp]:
    _register_rope_custom_op(_cls)

# Clean up loop variable
del _cls
