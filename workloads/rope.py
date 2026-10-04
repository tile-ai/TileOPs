"""Workload definitions for the RoPE op family."""

import math

import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase


def rope_verification():
    from workloads.numerics import Exact

    return Exact()


class RopeWorkload(WorkloadBase):
    def verification(self, *inputs):
        return rope_verification()

    def __init__(
        self,
        variant: str,
        input_layout: str,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        extra_kwargs: dict | None = None,
    ):
        self.variant = variant
        self.input_layout = input_layout
        self.batch = batch
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.extra_kwargs = extra_kwargs or {}

    def gen_inputs(self) -> tuple[torch.Tensor]:
        """Generate only x; cos/sin are computed by the op internally."""
        if self.input_layout == "1d":
            x = torch.randn(self.seq_len, self.head_dim, device=run_device(), dtype=self.dtype)
        else:
            x = torch.randn(
                self.batch,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                device=run_device(),
                dtype=self.dtype,
            )
        return (x,)


def rope_frequency_tables(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute standard RoPE cos/sin tables.

    Returns:
        (cos, sin) each of shape (seq_len, head_dim // 2).
    """
    device = device or run_device()
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_vals = torch.cos(angles).to(dtype)
    sin_vals = torch.sin(angles).to(dtype)
    return cos_vals, sin_vals


def llama31_frequency_tables(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    scale_factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_position: int = 8192,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Llama 3.1 scaled frequency computation."""
    device = device or run_device()
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
    cos_vals = torch.cos(angles).to(dtype)
    sin_vals = torch.sin(angles).to(dtype)
    return cos_vals, sin_vals


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    """Canonical yarn_find_correction_dim from TVM position_embedding.py."""
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_find_correction_range(
    beta_fast: float, beta_slow: float, dim: int, base: float, max_position_embeddings: int
) -> tuple[int, int]:
    """Canonical yarn_find_correction_range from TVM position_embedding.py."""
    low = math.floor(_yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_frequency_tables(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    scale: float = 16.0,
    original_max_position: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attn_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical YaRN frequency computation with NTK-aware interpolation.

    Reference: TVM ``rope_freq_yarn`` in position_embedding.py.

    Key formula:
    - freq_extra = 1 / (base ^ (2k/d))  (original, for extrapolation)
    - freq_inter = 1 / ((scale * base) ^ (2k/d))  (NTK-aware, for interpolation)
    - Linear ramp mask between correction dims
    - inv_freq = freq_inter * (1 - mask) + freq_extra * mask
    """
    device = device or run_device()
    half = head_dim // 2
    dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)

    freq_extra = 1.0 / (base ** (dim_indices / half))
    freq_inter = 1.0 / ((scale * base) ** (dim_indices / half))

    low, high = _yarn_find_correction_range(
        beta_fast,
        beta_slow,
        half,
        base,
        original_max_position,
    )
    if low == high:
        high = high + 1

    inv_freq_mask = 1.0 - torch.clamp(
        (dim_indices - low) / (high - low),
        0.0,
        1.0,
    )
    inv_freq = freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    cos_vals = (torch.cos(angles) * attn_factor).to(dtype)
    sin_vals = (torch.sin(angles) * attn_factor).to(dtype)
    return cos_vals, sin_vals


def longrope_frequency_tables(
    head_dim: int,
    seq_len: int,
    base: float = 10000.0,
    rescale_factors: torch.Tensor | None = None,
    max_position_embeddings: int = 4096,
    original_max_position_embeddings: int = 4096,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical LongRoPE frequency computation with amplitude scaling.

    Reference: TVM ``rope_freq_longrope`` in position_embedding.py.

    Key formula:
    - divisor = ext_factors[k] * base^(2k/d)  (ext_factors multiply divisor)
    - scaling_factor = sqrt(1 + log(scale) / log(orig_max_pos)) if scale > 1
    - cos/sin are multiplied by scaling_factor (amplitude factor)
    """
    device = device or run_device()
    half = head_dim // 2
    dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)
    divisor = base ** (dim_indices / half)

    if rescale_factors is not None:
        rf = rescale_factors.to(device=device, dtype=torch.float32)
        divisor = rf * divisor

    freqs = 1.0 / divisor

    scale = max_position_embeddings / original_max_position_embeddings
    if scale > 1.0:
        scaling_factor = math.sqrt(
            1.0 + math.log(scale) / math.log(original_max_position_embeddings)
        )
    else:
        scaling_factor = 1.0

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_vals = (torch.cos(angles) * scaling_factor).to(dtype)
    sin_vals = (torch.sin(angles) * scaling_factor).to(dtype)
    return cos_vals, sin_vals


def ref_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference neox RoPE: full-dim cos/sin broadcast with half-rotation."""
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    if x.ndim == 2:
        return (x.float() * cos_full.float() + _rotate_half_neox(x).float() * sin_full.float()).to(
            x.dtype
        )
    elif x.ndim == 4:
        cos_full = cos_full.unsqueeze(0).unsqueeze(2)
        sin_full = sin_full.unsqueeze(0).unsqueeze(2)
        return (x.float() * cos_full.float() + _rotate_half_neox(x).float() * sin_full.float()).to(
            x.dtype
        )
    elif x.ndim == 3:
        return (x.float() * cos_full.float() + _rotate_half_neox(x).float() * sin_full.float()).to(
            x.dtype
        )
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")


def ref_rope_non_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference non-neox (RoFormer) RoPE: adjacent pair rotation."""
    cos_interleaved = cos.repeat_interleave(2, dim=-1)
    sin_interleaved = sin.repeat_interleave(2, dim=-1)
    if x.ndim == 2:
        return (
            x.float() * cos_interleaved.float()
            + _rotate_half_non_neox(x).float() * sin_interleaved.float()
        ).to(x.dtype)
    elif x.ndim == 4:
        cos_interleaved = cos_interleaved.unsqueeze(0).unsqueeze(2)
        sin_interleaved = sin_interleaved.unsqueeze(0).unsqueeze(2)
        return (
            x.float() * cos_interleaved.float()
            + _rotate_half_non_neox(x).float() * sin_interleaved.float()
        ).to(x.dtype)
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")


class RopeCase(RopeWorkload):
    """Generic test fixture for RoPE ops.

    The op computes cos/sin internally; the test generates only x as input
    and computes the reference rotation using independently generated
    frequency tables.
    """

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Independently compute cos/sin for the reference implementation."""
        if self.variant in ("neox", "non_neox"):
            return rope_frequency_tables(self.head_dim, self.seq_len, dtype=self.dtype)
        elif self.variant == "rope_llama31":
            return llama31_frequency_tables(
                self.head_dim, self.seq_len, dtype=self.dtype, **self.extra_kwargs
            )
        elif self.variant == "yarn_rope":
            return yarn_frequency_tables(
                self.head_dim, self.seq_len, dtype=self.dtype, **self.extra_kwargs
            )
        elif self.variant == "longrope":
            return longrope_frequency_tables(
                self.head_dim, self.seq_len, dtype=self.dtype, **self.extra_kwargs
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        """Pure-PyTorch reference: independently computes cos/sin and applies rotation."""
        cos, sin = self._compute_cos_sin()
        if self.variant in ("neox", "rope_llama31", "yarn_rope", "longrope"):
            return ref_rope_neox(x, cos, sin)
        elif self.variant == "non_neox":
            return ref_rope_non_neox(x, cos, sin)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    """Neox-style rotation: split at midpoint and negate first half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _rotate_half_non_neox(x: torch.Tensor) -> torch.Tensor:
    """Non-neox (RoFormer) rotation: adjacent pairs."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack([-x_odd, x_even], dim=-1)
    return rotated.flatten(-2)


class RopeCall(CallWorkload):
    """A manifest rotation checked against independently constructed frequencies."""

    def verification(self, *inputs):
        return rope_verification()

    def ref_program(self, x, position_ids=None):
        name = self.call.signature.name
        p = self.call.params
        if name == "RopeNeoxPositionIdsFwdOp":
            cos, sin = rope_frequency_tables(
                p.get("rotary_dim") or x.shape[-1],
                p["max_position"],
                base=p["base"],
                dtype=x.dtype,
                device=x.device,
            )
            return ref_rope_neox_position_ids(x, cos, sin, position_ids, p.get("rotary_dim"))
        tables = {
            "RopeFwdOp": rope_frequency_tables,
            "RopeLlama31FwdOp": llama31_frequency_tables,
            "RopeYarnFwdOp": yarn_frequency_tables,
            "RopeLongRopeFwdOp": longrope_frequency_tables,
        }[name]
        seq_len = x.shape[0] if p["input_layout"] == "1d" else x.shape[1]
        kwargs = {k: v for k, v in p.items() if k not in ("input_layout", "rope_layout")}
        if "rescale_factors" in self.call.specs:
            kwargs["rescale_factors"] = self.tensors["rescale_factors"]
        cos, sin = tables(x.shape[-1], seq_len, dtype=x.dtype, device=x.device, **kwargs)
        rotate = (
            ref_rope_non_neox if p.get("rope_layout", "neox") == "interleaved" else ref_rope_neox
        )
        return rotate(x, cos, sin)

    def __init__(self, call, device=None):
        CallWorkload.__init__(self, call, device)
        self.tensors = call.materialize(self.device)

    def gen_inputs(self):
        return tuple(self.tensors[name] for name in self.call.signature.inputs)

    def arguments(self):
        return self.call.arguments(self.tensors)


def ref_rope_neox_position_ids(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """Reference neox RoPE for packed THD tensors with explicit positions."""
    rotary_dim = x.shape[-1] if rotary_dim is None else rotary_dim
    cos_full = torch.cat([cos, cos], dim=-1)[position_ids].unsqueeze(1)
    sin_full = torch.cat([sin, sin], dim=-1)[position_ids].unsqueeze(1)
    x_rot = x[..., :rotary_dim]
    y_rot = (
        x_rot.float() * cos_full.float() + _rotate_half_neox(x_rot).float() * sin_full.float()
    ).to(x.dtype)
    if rotary_dim == x.shape[-1]:
        return y_rot
    return torch.cat([y_rot, x[..., rotary_dim:]], dim=-1)
