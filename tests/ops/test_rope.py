"""Tests for Rotary Position Embedding (RoPE) ops — 5 variants x 2 layouts.

Variants:
- neox: GPT-NeoX interleaved rotation (ref: GPT-NeoX / HuggingFace transformers)
- non_neox: original RoFormer adjacent-pair rotation (ref: RoFormer paper)
- rope_llama31: Llama 3.1 with frequency scaling (ref: Meta Llama 3.1)
- yarn_rope: YaRN with attention-factor scaling (ref: YaRN paper)
- longrope: LongRoPE with per-dimension rescale factors (ref: LongRoPE paper)

Each variant supports 1D layout (seq_len, head_dim) and
2D layout (batch, seq_len, num_heads, head_dim).

The op computes cos/sin internally from variant parameters; tests call
``op(x)`` directly and compare against a pure-PyTorch reference that
independently computes the same frequency tables.
"""

import math

import pytest
import torch

from tests.test_base import FixtureBase, TestBase

# ---------------------------------------------------------------------------
# Reference implementations (pure PyTorch)
# ---------------------------------------------------------------------------


def _compute_freqs_cis_base(head_dim: int, seq_len: int, base: float = 10000.0,
                            dtype: torch.dtype = torch.float32,
                            device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Compute standard RoPE cos/sin tables.

    Returns:
        (cos, sin) each of shape (seq_len, head_dim // 2).
    """
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_vals = torch.cos(angles).to(dtype)
    sin_vals = torch.sin(angles).to(dtype)
    return cos_vals, sin_vals


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


def ref_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference neox RoPE: full-dim cos/sin broadcast with half-rotation."""
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    if x.ndim == 2:
        return (x.float() * cos_full.float()
                + _rotate_half_neox(x).float() * sin_full.float()).to(x.dtype)
    elif x.ndim == 4:
        cos_full = cos_full.unsqueeze(0).unsqueeze(2)
        sin_full = sin_full.unsqueeze(0).unsqueeze(2)
        return (x.float() * cos_full.float()
                + _rotate_half_neox(x).float() * sin_full.float()).to(x.dtype)
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")


def ref_rope_non_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference non-neox (RoFormer) RoPE: adjacent pair rotation."""
    cos_interleaved = cos.repeat_interleave(2, dim=-1)
    sin_interleaved = sin.repeat_interleave(2, dim=-1)
    if x.ndim == 2:
        return (x.float() * cos_interleaved.float()
                + _rotate_half_non_neox(x).float() * sin_interleaved.float()).to(x.dtype)
    elif x.ndim == 4:
        cos_interleaved = cos_interleaved.unsqueeze(0).unsqueeze(2)
        sin_interleaved = sin_interleaved.unsqueeze(0).unsqueeze(2)
        return (x.float() * cos_interleaved.float()
                + _rotate_half_non_neox(x).float() * sin_interleaved.float()).to(x.dtype)
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")


def _compute_llama31_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                           scale_factor: float = 8.0, low_freq_factor: float = 1.0,
                           high_freq_factor: float = 4.0,
                           original_max_position: int = 8192,
                           dtype: torch.dtype = torch.float32,
                           device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Llama 3.1 scaled frequency computation."""
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
    cos_vals = torch.cos(angles).to(dtype)
    sin_vals = torch.sin(angles).to(dtype)
    return cos_vals, sin_vals


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float,
                              max_position_embeddings: int) -> float:
    """Canonical yarn_find_correction_dim from TVM position_embedding.py."""
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(beta_fast: float, beta_slow: float, dim: int,
                                base: float,
                                max_position_embeddings: int) -> tuple[int, int]:
    """Canonical yarn_find_correction_range from TVM position_embedding.py."""
    low = math.floor(
        _yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def _compute_yarn_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                        scale: float = 16.0, original_max_position: int = 4096,
                        beta_fast: float = 32.0, beta_slow: float = 1.0,
                        attn_factor: float = 1.0,
                        dtype: torch.dtype = torch.float32,
                        device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical YaRN frequency computation with NTK-aware interpolation.

    Reference: TVM ``rope_freq_yarn`` in position_embedding.py.

    Key formula:
    - freq_extra = 1 / (base ^ (2k/d))  (original, for extrapolation)
    - freq_inter = 1 / ((scale * base) ^ (2k/d))  (NTK-aware, for interpolation)
    - Linear ramp mask between correction dims
    - inv_freq = freq_inter * (1 - mask) + freq_extra * mask
    """
    half = head_dim // 2
    dim_indices = torch.arange(0, half, device=device, dtype=torch.float32)

    freq_extra = 1.0 / (base ** (dim_indices / half))
    freq_inter = 1.0 / ((scale * base) ** (dim_indices / half))

    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, half, base, original_max_position,
    )
    if low == high:
        high = high + 1

    inv_freq_mask = 1.0 - torch.clamp(
        (dim_indices - low) / (high - low), 0.0, 1.0,
    )
    inv_freq = freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)
    cos_vals = (torch.cos(angles) * attn_factor).to(dtype)
    sin_vals = (torch.sin(angles) * attn_factor).to(dtype)
    return cos_vals, sin_vals


def _compute_longrope_freqs(head_dim: int, seq_len: int, base: float = 10000.0,
                            rescale_factors: torch.Tensor | None = None,
                            max_position_embeddings: int = 4096,
                            original_max_position_embeddings: int = 4096,
                            dtype: torch.dtype = torch.float32,
                            device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical LongRoPE frequency computation with amplitude scaling.

    Reference: TVM ``rope_freq_longrope`` in position_embedding.py.

    Key formula:
    - divisor = ext_factors[k] * base^(2k/d)  (ext_factors multiply divisor)
    - scaling_factor = sqrt(1 + log(scale) / log(orig_max_pos)) if scale > 1
    - cos/sin are multiplied by scaling_factor (amplitude factor)
    """
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


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


class RopeTest(TestBase):
    """Generic test harness for RoPE ops.

    The op computes cos/sin internally; the test generates only x as input
    and computes the reference rotation using independently generated
    frequency tables.
    """

    def __init__(self, variant: str, layout: str, batch: int, seq_len: int,
                 num_heads: int, head_dim: int, dtype: torch.dtype,
                 extra_kwargs: dict | None = None):
        self.variant = variant
        self.layout = layout
        self.batch = batch
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.extra_kwargs = extra_kwargs or {}

    def gen_inputs(self) -> tuple[torch.Tensor]:
        """Generate only x; cos/sin are computed by the op internally."""
        if self.layout == "1d":
            x = torch.randn(self.seq_len, self.head_dim, device="cuda", dtype=self.dtype)
        else:
            x = torch.randn(self.batch, self.seq_len, self.num_heads, self.head_dim,
                             device="cuda", dtype=self.dtype)
        return (x,)

    def _compute_cos_sin(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Independently compute cos/sin for the reference implementation."""
        if self.variant in ("neox", "non_neox"):
            return _compute_freqs_cis_base(self.head_dim, self.seq_len, dtype=self.dtype)
        elif self.variant == "rope_llama31":
            return _compute_llama31_freqs(self.head_dim, self.seq_len, dtype=self.dtype,
                                          **self.extra_kwargs)
        elif self.variant == "yarn_rope":
            return _compute_yarn_freqs(self.head_dim, self.seq_len, dtype=self.dtype,
                                       **self.extra_kwargs)
        elif self.variant == "longrope":
            return _compute_longrope_freqs(self.head_dim, self.seq_len, dtype=self.dtype,
                                           **self.extra_kwargs)
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


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:
        return 1.6e-2, 1.6e-2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class RopeBasicFixture(FixtureBase):
    """Basic RoPE fixture: shapes x dtypes."""
    PARAMS = [
        ("batch, seq_len, num_heads, head_dim, dtype", [
            pytest.param(2, 128, 8, 64, torch.float16, marks=[pytest.mark.smoke, pytest.mark.packaging]),
            pytest.param(2, 128, 8, 64, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(2, 128, 8, 64, torch.float32, marks=pytest.mark.smoke),
            pytest.param(1, 256, 4, 128, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class RopeEdgeFixture(FixtureBase):
    """Edge case fixture: seq_len=1, small head_dim."""
    PARAMS = [
        ("batch, seq_len, num_heads, head_dim, dtype", [
            pytest.param(1, 1, 1, 16, torch.float32, marks=pytest.mark.smoke),
            pytest.param(1, 1, 1, 16, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 8, 64, torch.float16, marks=pytest.mark.full),
        ]),
    ]


# ---------------------------------------------------------------------------
# Neox RoPE tests
# ---------------------------------------------------------------------------


@RopeBasicFixture
def test_rope_neox_1d(batch: int, seq_len: int, num_heads: int,
                      head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeNeoxOp

    test = RopeTest("neox", "1d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeNeoxOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="1d")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeBasicFixture
def test_rope_neox_2d(batch: int, seq_len: int, num_heads: int,
                      head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeNeoxOp

    test = RopeTest("neox", "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeNeoxOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="2d",
                    batch=batch, num_heads=num_heads)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Non-neox (RoFormer) RoPE tests
# ---------------------------------------------------------------------------


@RopeBasicFixture
def test_rope_non_neox_1d(batch: int, seq_len: int, num_heads: int,
                          head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeNonNeoxOp

    test = RopeTest("non_neox", "1d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeNonNeoxOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="1d")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeBasicFixture
def test_rope_non_neox_2d(batch: int, seq_len: int, num_heads: int,
                          head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeNonNeoxOp

    test = RopeTest("non_neox", "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeNonNeoxOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="2d",
                       batch=batch, num_heads=num_heads)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Llama 3.1 RoPE tests
# ---------------------------------------------------------------------------


@RopeBasicFixture
def test_rope_llama31_1d(batch: int, seq_len: int, num_heads: int,
                         head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeLlama31Op

    extra = {"scale_factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
             "original_max_position": 8192}
    test = RopeTest("rope_llama31", "1d", batch, seq_len, num_heads, head_dim, dtype,
                    extra_kwargs=extra)
    op = RopeLlama31Op(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="1d",
                       **extra)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeBasicFixture
def test_rope_llama31_2d(batch: int, seq_len: int, num_heads: int,
                         head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeLlama31Op

    extra = {"scale_factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
             "original_max_position": 8192}
    test = RopeTest("rope_llama31", "2d", batch, seq_len, num_heads, head_dim, dtype,
                    extra_kwargs=extra)
    op = RopeLlama31Op(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="2d",
                       batch=batch, num_heads=num_heads, **extra)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# YaRN RoPE tests
# ---------------------------------------------------------------------------


@RopeBasicFixture
def test_rope_yarn_1d(batch: int, seq_len: int, num_heads: int,
                      head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeYarnOp

    extra = {"scale": 16.0, "original_max_position": 4096,
             "beta_fast": 32.0, "beta_slow": 1.0, "attn_factor": 1.0}
    test = RopeTest("yarn_rope", "1d", batch, seq_len, num_heads, head_dim, dtype,
                    extra_kwargs=extra)
    op = RopeYarnOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="1d", **extra)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeBasicFixture
def test_rope_yarn_2d(batch: int, seq_len: int, num_heads: int,
                      head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeYarnOp

    extra = {"scale": 16.0, "original_max_position": 4096,
             "beta_fast": 32.0, "beta_slow": 1.0, "attn_factor": 1.0}
    test = RopeTest("yarn_rope", "2d", batch, seq_len, num_heads, head_dim, dtype,
                    extra_kwargs=extra)
    op = RopeYarnOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="2d",
                    batch=batch, num_heads=num_heads, **extra)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# LongRoPE tests
# ---------------------------------------------------------------------------


@RopeBasicFixture
def test_rope_longrope_1d(batch: int, seq_len: int, num_heads: int,
                          head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeLongRopeOp

    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device="cuda")
    max_pos = 16384
    orig_max_pos = 4096
    extra = {"rescale_factors": rescale,
             "max_position_embeddings": max_pos,
             "original_max_position_embeddings": orig_max_pos}
    test = RopeTest("longrope", "1d", batch, seq_len, num_heads, head_dim, dtype,
                    extra_kwargs=extra)
    op = RopeLongRopeOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="1d",
                        rescale_factors=rescale, max_position_embeddings=max_pos,
                        original_max_position_embeddings=orig_max_pos)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeBasicFixture
def test_rope_longrope_2d(batch: int, seq_len: int, num_heads: int,
                          head_dim: int, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeLongRopeOp

    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device="cuda")
    max_pos = 16384
    orig_max_pos = 4096
    extra = {"rescale_factors": rescale,
             "max_position_embeddings": max_pos,
             "original_max_position_embeddings": orig_max_pos}
    test = RopeTest("longrope", "2d", batch, seq_len, num_heads, head_dim, dtype,
                    extra_kwargs=extra)
    op = RopeLongRopeOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="2d",
                        batch=batch, num_heads=num_heads, rescale_factors=rescale,
                        max_position_embeddings=max_pos,
                        original_max_position_embeddings=orig_max_pos)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


@RopeEdgeFixture
def test_rope_neox_edge(batch: int, seq_len: int, num_heads: int,
                        head_dim: int, dtype: torch.dtype) -> None:
    """Edge cases: seq_len=1 and longer sequences."""
    from tileops.ops.rope import RopeNeoxOp

    test = RopeTest("neox", "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeNeoxOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="2d",
                    batch=batch, num_heads=num_heads)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@RopeEdgeFixture
def test_rope_non_neox_edge(batch: int, seq_len: int, num_heads: int,
                            head_dim: int, dtype: torch.dtype) -> None:
    """Edge cases: seq_len=1 and longer sequences."""
    from tileops.ops.rope import RopeNonNeoxOp

    test = RopeTest("non_neox", "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeNonNeoxOp(seq_len=seq_len, head_dim=head_dim, dtype=dtype, layout="2d",
                       batch=batch, num_heads=num_heads)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Input validation regression tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_rope_rejects_wrong_shape_2d() -> None:
    """A same-numel but wrong-shape 2D tensor must be rejected."""
    from tileops.ops.rope import RopeNeoxOp

    # Op configured for batch=2, seq_len=2, num_heads=1, head_dim=4
    op = RopeNeoxOp(seq_len=2, head_dim=4, dtype=torch.float16, layout="2d",
                    batch=2, num_heads=1)
    # Wrong shape: (1, 4, 1, 4) has same numel (16) as (2, 2, 1, 4) but different layout
    x = torch.randn(1, 4, 1, 4, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="Expected input shape"):
        op(x)


@pytest.mark.smoke
def test_rope_noncontiguous_1d_works() -> None:
    """A non-contiguous 1D view must produce correct results after contiguity normalization."""
    from tileops.ops.rope import RopeNeoxOp

    seq_len, head_dim = 4, 8
    op = RopeNeoxOp(seq_len=seq_len, head_dim=head_dim, dtype=torch.float32, layout="1d")

    # Create a non-contiguous view: transpose makes it non-contiguous
    base = torch.randn(head_dim, seq_len, device="cuda", dtype=torch.float32)
    x_nc = base.t()  # shape (seq_len, head_dim), non-contiguous
    assert not x_nc.is_contiguous()

    # Reference with contiguous copy
    x_c = x_nc.contiguous()
    out_nc = op(x_nc)
    out_c = op(x_c)
    torch.testing.assert_close(out_nc, out_c, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_rope_rejects_non_float_dtype() -> None:
    from tileops.kernels.rope import RopeNeoxKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        RopeNeoxKernel(seq_len=16, head_dim=64, dtype=torch.int32)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
