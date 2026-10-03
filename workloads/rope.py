"""Workload definitions for the RoPE op family."""

import math

import torch

from workloads.device import run_device
from workloads.workload_base import WorkloadBase


class RopeWorkload(WorkloadBase):
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
