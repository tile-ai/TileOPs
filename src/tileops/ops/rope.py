"""Rotary Position Embedding (RoPE) ops — 4 frequency schemes x 2 input layouts.

Each Op computes variant-specific frequency tables (cos, sin) lazily at
forward time (on the same device as the input tensor) and delegates the
actual rotation to the corresponding kernel.

Variants and frequency computation:
- **RoPEFwdOp**: standard theta = 10000^(-2k/d) frequencies, either rotation convention
- **RoPELlama31FwdOp**: piecewise-scaled frequencies for Llama 3.1
- **YaRNFwdOp**: YaRN linear-ramp interpolated frequencies
- **LongRoPEFwdOp**: per-dimension rescaled frequencies

Input layouts:
- ``"1d"``: input shape $[seq\\_len \\times head\\_dim]$
- ``"2d"``: input shape $[batch \\times seq\\_len \\times num\\_heads \\times head\\_dim]$

torch.compile support: each op's compile-boundary operator is generated from its
manifest entry.
"""

import math
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.rope import (
    RoPECall,
    RoPENeoxFwdInterface,
    RoPENeoxKernel,
    RoPENeoxPositionIdsCall,
    RoPENeoxPositionIdsFwdInterface,
    RoPENeoxPositionIdsKernel,
    RoPENonNeoxFwdInterface,
    RoPENonNeoxKernel,
)
from tileops.ops.op_base import Op

__all__ = [
    "LongRoPEFwdOp",
    "RoPEFwdOp",
    "RoPELlama31FwdOp",
    "RoPENeoxPositionIdsFwdOp",
    "YaRNFwdOp",
    "base_freqs",
]


# Frequency computation helpers (pure Python / PyTorch, run on host)


def base_freqs(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    """Inverse dim formula to find dim based on number of rotations.

    Matches the canonical TVM/vLLM ``yarn_find_correction_dim`` formula.
    """
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_find_correction_range(
    beta_fast: float, beta_slow: float, dim: int, base: float, max_position_embeddings: int
) -> tuple[int, int]:
    """Find low/high correction dims from rotation boundary parameters."""
    low = math.floor(_yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


class _RoPEOpBase(Op):
    """Base class for the four frequency-scheme RoPE ops.

    Subclass sets ``kernel_types`` and ``interfaces`` and implements ``_compute_cos_sin``
    to generate its variant-specific frequency tables.

    Cos/sin tables are computed lazily at forward time on the same device as
    the input tensor, avoiding device-mismatch issues in multi-GPU settings.
    """

    # The rotation convention keys the interface; the scheme variants serve NeoX only.
    rope_layout: str = "neox"

    def __init__(
        self,
        input_layout: str = "1d",
        base: float = 10000.0,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            input_layout: "1d" for ``[seq_len, head_dim]`` or "2d" for
                ``[batch, seq_len, num_heads, head_dim]``.
            base: Frequency base (default 10000).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.input_layout = input_layout
        self.base = base
        self.target = target
        self.tune = tune
        self._freq_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    def _get_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached cos/sin tables for this extent, dtype and device."""
        key = (seq_len, head_dim, dtype, device)
        if key not in self._freq_cache:
            self._freq_cache[key] = self._compute_cos_sin(seq_len, head_dim, dtype, device)
        return self._freq_cache[key]

    def _compute_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Variant-specific (cos, sin) tables, each of shape ``(seq_len, head_dim // 2)``."""
        raise NotImplementedError("Subclass must implement _compute_cos_sin")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation using internally computed cos/sin tables.

        Args:
            x: Input tensor. Shape depends on the input layout:
                - 1D: ``(seq_len, head_dim)``
                - 2D: ``(batch, seq_len, num_heads, head_dim)``

        Returns:
            Rotated output tensor with same shape as x.
        """
        return self._call_boundary(x)

    def _eager_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        if self.input_layout == "1d":
            (seq_len, head_dim), batch, num_heads = x.shape, 1, 1
        else:
            batch, seq_len, num_heads, head_dim = x.shape
        call = RoPECall(
            seq_len=seq_len,
            head_dim=head_dim,
            input_layout=self.input_layout,
            batch=batch,
            num_heads=num_heads,
            dtype=x.dtype,
            device=x.device,
        )
        cos, sin = self._get_cos_sin(seq_len, head_dim, x.dtype, x.device)
        x = x.contiguous()
        self.kernel = self.kernel_for(self.rope_layout, call)
        return self.kernel(x, cos, sin)


# Concrete Op classes (4 frequency schemes)


class RoPEFwdOp(_RoPEOpBase):
    """RoPE with standard theta frequencies, in either rotation convention.

    ``rope_layout="neox"`` splits a head at its midpoint and rotates the halves against
    each other; ``"interleaved"`` rotates each adjacent pair. The frequency table is the
    same either way, so the two differ only in which elements pair up.

    References: GPT-NeoX / HuggingFace transformers RotaryEmbedding (NeoX); Su et al.,
    "RoFormer: Enhanced Transformer with Rotary Position Embedding" (interleaved).
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "rope_neox": RoPENeoxKernel,
        "rope_non_neox": RoPENonNeoxKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "neox": RoPENeoxFwdInterface,
        "interleaved": RoPENonNeoxFwdInterface,
    }

    def __init__(
        self,
        rope_layout: str = "neox",
        input_layout: str = "1d",
        base: float = 10000.0,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            rope_layout: "neox" to rotate the two halves of a head against each other,
                "interleaved" to rotate each adjacent pair.
            input_layout: "1d" for ``[seq_len, head_dim]`` or "2d" for
                ``[batch, seq_len, num_heads, head_dim]``.
            base: Frequency base (default 10000).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        if rope_layout not in ("neox", "interleaved"):
            raise ValueError(f"rope_layout must be 'neox' or 'interleaved', got '{rope_layout}'")
        self.rope_layout = rope_layout
        super().__init__(input_layout, base, target=target, kernel_map=kernel_map, tune=tune)

    def _compute_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return base_freqs(head_dim, seq_len, base=self.base, dtype=dtype, device=device)


class RoPENeoxPositionIdsFwdOp(Op):
    """GPT-NeoX style RoPE for packed THD tensors with explicit positions.

    The first ``rotary_dim`` columns of each head rotate (all of them when
    ``rotary_dim`` is None) and the rest are copied.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "rope_neox_position_ids": RoPENeoxPositionIdsKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "rope_neox_position_ids": RoPENeoxPositionIdsFwdInterface
    }

    def __init__(
        self,
        max_position: int,
        base: float = 10000.0,
        rotary_dim: Optional[int] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            max_position: Manifest ``params.max_position``, ``int``.
            base: Manifest ``params.base``, ``float``, default ``10000.0``.
            rotary_dim: Manifest ``params.rotary_dim``, ``int | None``, default ``None``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.max_position = max_position
        self.base = base
        self.rotary_dim = rotary_dim
        self.target = target
        self.tune = tune
        self._freq_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    def _get_cos_sin(
        self, rotary_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (rotary_dim, dtype, device)
        if key not in self._freq_cache:
            self._freq_cache[key] = base_freqs(
                rotary_dim, self.max_position, base=self.base, dtype=dtype, device=device
            )
        return self._freq_cache[key]

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            x: ``[tokens, heads, head_dim]``, dtype ``float16 | bfloat16 | float32``.
            position_ids: ``[tokens]``, dtype ``int32 | int64``, each in
                ``[0, max_position)``.

        Returns:
            ``output``, shaped as ``x``.
        """
        return self._call_boundary(x, position_ids)

    def _eager_forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        num_tokens, num_heads, head_dim = x.shape
        rotary_dim = head_dim if self.rotary_dim is None else self.rotary_dim
        call = RoPENeoxPositionIdsCall(
            num_tokens=num_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=self.max_position,
            dtype=x.dtype,
            device=x.device,
        )
        cos, sin = self._get_cos_sin(rotary_dim, x.dtype, x.device)
        x, position_ids = x.contiguous(), position_ids.to(torch.int32).contiguous()
        self.kernel = self.kernel_for("rope_neox_position_ids", call)
        output = self.kernel(x, cos, sin, position_ids)
        # The kernel counts the positions it found outside the table rather than the
        # op proving they are inside it first: two reductions and two launches in
        # front of every call cost more device time than the rotation they guard.
        # It clamps its own table index, so this call read nothing out of bounds.
        if self.kernel.take_out_of_range():
            raise ValueError("position_ids must be in [0, max_position)")
        return output


class RoPELlama31FwdOp(_RoPEOpBase):
    """Llama 3.1 RoPE op with piecewise frequency scaling.

    Computes cos/sin tables at construction using Llama 3.1 piecewise-scaled
    frequencies based on wavelength thresholds.

    Reference: Meta Llama 3.1 model implementation.

    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"rope_llama31": RoPENeoxKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"neox": RoPENeoxFwdInterface}

    @staticmethod
    def _llama31_freqs(
        head_dim: int,
        seq_len: int,
        base: float = 10000.0,
        scale_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position: int = 8192,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                    high_freq_factor - low_freq_factor
                )
                scaled_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

        freqs = torch.stack(scaled_freqs)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(t, freqs)
        return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    def __init__(
        self,
        input_layout: str = "1d",
        base: float = 10000.0,
        scale_factor: float = 8.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
        original_max_position: int = 8192,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            input_layout: "1d" or "2d".
            base: Frequency base (default 10000).
            scale_factor: Scaling factor for low frequencies (default 8.0).
            low_freq_factor: Low-frequency wavelen threshold (default 1.0).
            high_freq_factor: High-frequency wavelen threshold (default 4.0).
            original_max_position: Original max position (default 8192).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position = original_max_position
        super().__init__(input_layout, base, target=target, kernel_map=kernel_map, tune=tune)

    def _compute_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return RoPELlama31FwdOp._llama31_freqs(
            head_dim,
            seq_len,
            base=self.base,
            scale_factor=self.scale_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position=self.original_max_position,
            dtype=dtype,
            device=device,
        )


class YaRNFwdOp(_RoPEOpBase):
    """YaRN RoPE op with linear-ramp frequency interpolation.

    Computes cos/sin tables at construction using YaRN linear-ramp
    interpolation between scaled and original frequencies.

    Reference: Peng et al., "YaRN: Efficient Context Window Extension of LLMs".

    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"yarn": RoPENeoxKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"neox": RoPENeoxFwdInterface}

    @staticmethod
    def _yarn_freqs(
        head_dim: int,
        seq_len: int,
        base: float = 10000.0,
        scale: float = 16.0,
        original_max_position: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attn_factor: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            beta_fast,
            beta_slow,
            half,
            base,
            original_max_position,
        )
        # Avoid division by zero when low == high
        if low == high:
            high = high + 1

        # Linear ramp mask: 1 near low dims (extrapolation), 0 near high dims (interpolation)
        inv_freq_mask = 1.0 - torch.clamp(
            (dim_indices - low) / (high - low),
            0.0,
            1.0,
        )

        # Blend: mask=1 -> freq_extra, mask=0 -> freq_inter
        inv_freq = freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(t, inv_freq)
        return (torch.cos(angles) * attn_factor).to(dtype), (torch.sin(angles) * attn_factor).to(
            dtype
        )

    def __init__(
        self,
        input_layout: str = "1d",
        base: float = 10000.0,
        scale: float = 16.0,
        original_max_position: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attn_factor: float = 1.0,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            input_layout: "1d" or "2d".
            base: Frequency base (default 10000).
            scale: Context extension scale (default 16.0).
            original_max_position: Original max position (default 4096).
            beta_fast: Fast decay boundary (default 32.0).
            beta_slow: Slow decay boundary (default 1.0).
            attn_factor: Attention scaling factor (default 1.0).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.
        """
        self.scale = scale
        self.original_max_position = original_max_position
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attn_factor = attn_factor
        super().__init__(input_layout, base, target=target, kernel_map=kernel_map, tune=tune)

    def _compute_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return YaRNFwdOp._yarn_freqs(
            head_dim,
            seq_len,
            base=self.base,
            scale=self.scale,
            original_max_position=self.original_max_position,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            attn_factor=self.attn_factor,
            dtype=dtype,
            device=device,
        )


class LongRoPEFwdOp(_RoPEOpBase):
    """LongRoPE op with per-dimension frequency rescaling.

    Computes cos/sin tables at construction using per-dimension rescale
    factors (ext_factors) that multiply the divisor, plus a scale-dependent
    amplitude factor applied to cos/sin output. Construction rejects a zero or NaN
    rescale factor, since the frequency divides by it.

    Reference: TVM ``rope_freq_longrope`` in position_embedding.py;
    Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2M Tokens".

    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"rope_longrope": RoPENeoxKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"neox": RoPENeoxFwdInterface}

    @staticmethod
    def _longrope_freqs(
        head_dim: int,
        seq_len: int,
        base: float = 10000.0,
        rescale_factors: Optional[torch.Tensor] = None,
        max_position_embeddings: int = 4096,
        original_max_position_embeddings: int = 4096,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(
        self,
        input_layout: str = "1d",
        base: float = 10000.0,
        rescale_factors: Optional[torch.Tensor] = None,
        max_position_embeddings: int = 4096,
        original_max_position_embeddings: int = 4096,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            input_layout: "1d" or "2d".
            base: Frequency base (default 10000).
            rescale_factors: Per-dimension rescale factors (ext_factors) of shape
                (head_dim // 2,). These multiply the divisor.
            max_position_embeddings: Extended max position length (default 4096).
            original_max_position_embeddings: Original max position length
                (default 4096).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune.

        Raises:
            ValueError: A rescale factor is zero or NaN; the frequency divides by it.
        """
        # A meta tensor holds no values to check.
        if rescale_factors is not None and not rescale_factors.is_meta:
            defined = (rescale_factors != 0) & ~torch.isnan(rescale_factors)
            if not bool(defined.all()):
                raise ValueError("rescale_factors must be nonzero and not NaN")
        self.rescale_factors = rescale_factors
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        super().__init__(input_layout, base, target=target, kernel_map=kernel_map, tune=tune)

    def _compute_cos_sin(
        self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return LongRoPEFwdOp._longrope_freqs(
            head_dim,
            seq_len,
            base=self.base,
            rescale_factors=self.rescale_factors,
            max_position_embeddings=self.max_position_embeddings,
            original_max_position_embeddings=self.original_max_position_embeddings,
            dtype=dtype,
            device=device,
        )
