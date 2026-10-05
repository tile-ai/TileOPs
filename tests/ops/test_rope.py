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

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, standard_tolerance
from workloads.device import run_device
from workloads.rope import (
    RopeWorkload,
    llama31_frequency_tables,
    longrope_frequency_tables,
    rope_frequency_tables,
    yarn_frequency_tables,
)


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
        return (x.float() * cos_full.float() + _rotate_half_neox(x).float() * sin_full.float()).to(
            x.dtype
        )
    elif x.ndim == 4:
        cos_full = cos_full.unsqueeze(0).unsqueeze(2)
        sin_full = sin_full.unsqueeze(0).unsqueeze(2)
        return (x.float() * cos_full.float() + _rotate_half_neox(x).float() * sin_full.float()).to(
            x.dtype
        )
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}")


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


# Test fixtures


class RopeTest(RopeWorkload, TestBase):
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


class RopeBasicFixture(FixtureBase):
    """Basic RoPE fixture: shapes x dtypes."""

    PARAMS = [
        (
            "batch, seq_len, num_heads, head_dim, dtype",
            [
                pytest.param(
                    2,
                    128,
                    8,
                    64,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="rope")],
                ),
                pytest.param(2, 128, 8, 64, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(2, 128, 8, 64, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1, 256, 4, 128, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class RopeEdgeFixture(FixtureBase):
    """Edge case fixture: seq_len=1, small head_dim."""

    PARAMS = [
        (
            "batch, seq_len, num_heads, head_dim, dtype",
            [
                pytest.param(1, 1, 1, 16, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1, 1, 1, 16, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 8, 64, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


# Base-frequency RoPE tests, both rotation conventions


@pytest.mark.parametrize("rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")])
@RopeBasicFixture
def test_rope_1d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    from tileops.ops.rope import RopeFwdOp

    test = RopeTest(variant, "1d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeFwdOp(rope_layout=rope_layout, input_layout="1d")
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


@pytest.mark.parametrize("rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")])
@RopeBasicFixture
def test_rope_2d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    from tileops.ops.rope import RopeFwdOp

    test = RopeTest(variant, "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeFwdOp(rope_layout=rope_layout, input_layout="2d")
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "rotary_dim, dtype",
    [
        (None, torch.float16),
        (32, torch.float16),
        (None, torch.bfloat16),
        (None, torch.float32),
    ],
)
def test_rope_neox_position_ids_thd(rotary_dim: int | None, dtype: torch.dtype) -> None:
    from tileops.ops.rope import RopeNeoxPositionIdsFwdOp

    num_tokens, num_heads, head_dim, max_position = 96, 8, 64, 512
    table_dim = head_dim if rotary_dim is None else rotary_dim
    x = torch.randn(num_tokens, num_heads, head_dim, device=run_device(), dtype=dtype)
    position_ids = (
        torch.arange(num_tokens, device=run_device(), dtype=torch.int32) * 3 + 17
    ) % max_position
    cos, sin = rope_frequency_tables(table_dim, max_position, dtype=dtype, device=run_device())
    ref = ref_rope_neox_position_ids(x, cos, sin, position_ids.long(), rotary_dim=rotary_dim)

    op = RopeNeoxPositionIdsFwdOp(
        max_position=max_position,
        rotary_dim=rotary_dim,
    )
    output = op(x, position_ids)
    torch.testing.assert_close(output, ref, **standard_tolerance(dtype))


@pytest.mark.smoke
def test_rope_neox_position_ids_validates_range() -> None:
    from tileops.ops.rope import RopeNeoxPositionIdsFwdOp

    op = RopeNeoxPositionIdsFwdOp(max_position=8)
    x = torch.randn(2, 1, 16, device=run_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="position_ids"):
        op(x, torch.tensor([0, 8], device=run_device(), dtype=torch.int32))


@pytest.mark.smoke
def test_rope_longrope_rejects_a_zero_rescale_factor() -> None:
    from tileops.ops.rope import RopeLongRopeFwdOp

    rescale = torch.tensor([1.0, 0.0, 2.0, 1.5], device=run_device())
    with pytest.raises(ValueError, match="rescale_factors"):
        RopeLongRopeFwdOp(rescale_factors=rescale)


@pytest.mark.smoke
def test_rope_neox_position_ids_none_rotary_dim_reinfers_head_dim() -> None:
    from tileops.ops.rope import RopeNeoxPositionIdsFwdOp

    max_position = 64
    position_ids = torch.arange(8, device=run_device(), dtype=torch.int32)
    op = RopeNeoxPositionIdsFwdOp(max_position=max_position, rotary_dim=None)

    x1 = torch.randn(8, 2, 16, device=run_device(), dtype=torch.float16)
    cos1, sin1 = rope_frequency_tables(16, max_position, dtype=x1.dtype, device=run_device())
    ref1 = ref_rope_neox_position_ids(x1, cos1, sin1, position_ids.long(), rotary_dim=None)
    torch.testing.assert_close(op(x1, position_ids), ref1, atol=5e-3, rtol=1e-5)

    x2 = torch.randn(8, 2, 32, device=run_device(), dtype=torch.float16)
    cos2, sin2 = rope_frequency_tables(32, max_position, dtype=x2.dtype, device=run_device())
    ref2 = ref_rope_neox_position_ids(x2, cos2, sin2, position_ids.long(), rotary_dim=None)
    torch.testing.assert_close(op(x2, position_ids), ref2, atol=5e-3, rtol=1e-5)


# Non-neox (RoFormer) RoPE tests


# Llama 3.1 RoPE tests


@RopeBasicFixture
def test_rope_llama31_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import RopeLlama31FwdOp

    extra = {
        "scale_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position": 8192,
    }
    test = RopeTest(
        "rope_llama31", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RopeLlama31FwdOp(input_layout="1d", **extra)
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


@RopeBasicFixture
def test_rope_llama31_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import RopeLlama31FwdOp

    extra = {
        "scale_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position": 8192,
    }
    test = RopeTest(
        "rope_llama31", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RopeLlama31FwdOp(input_layout="2d", **extra)
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


# YaRN RoPE tests


@RopeBasicFixture
def test_rope_yarn_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import RopeYarnFwdOp

    extra = {
        "scale": 16.0,
        "original_max_position": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attn_factor": 1.0,
    }
    test = RopeTest(
        "yarn_rope", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RopeYarnFwdOp(input_layout="1d", **extra)
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


@RopeBasicFixture
def test_rope_yarn_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import RopeYarnFwdOp

    extra = {
        "scale": 16.0,
        "original_max_position": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attn_factor": 1.0,
    }
    test = RopeTest(
        "yarn_rope", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RopeYarnFwdOp(input_layout="2d", **extra)
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


# LongRoPE tests


@RopeBasicFixture
def test_rope_longrope_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import RopeLongRopeFwdOp

    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=run_device())
    max_pos = 16384
    orig_max_pos = 4096
    extra = {
        "rescale_factors": rescale,
        "max_position_embeddings": max_pos,
        "original_max_position_embeddings": orig_max_pos,
    }
    test = RopeTest(
        "longrope", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RopeLongRopeFwdOp(
        input_layout="1d",
        rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


@RopeBasicFixture
def test_rope_longrope_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import RopeLongRopeFwdOp

    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=run_device())
    max_pos = 16384
    orig_max_pos = 4096
    extra = {
        "rescale_factors": rescale,
        "max_position_embeddings": max_pos,
        "original_max_position_embeddings": orig_max_pos,
    }
    test = RopeTest(
        "longrope", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RopeLongRopeFwdOp(
        input_layout="2d",
        rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


# Edge case tests


@pytest.mark.parametrize("rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")])
@RopeEdgeFixture
def test_rope_edge(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    """Edge cases: seq_len=1 and longer sequences."""
    from tileops.ops.rope import RopeFwdOp

    test = RopeTest(variant, "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RopeFwdOp(rope_layout=rope_layout, input_layout="2d")
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


# Layout and dtype regression tests


@pytest.mark.smoke
def test_rope_noncontiguous_1d_works() -> None:
    """A non-contiguous 1D view must produce correct results after contiguity normalization."""
    from tileops.ops.rope import RopeFwdOp

    seq_len, head_dim = 4, 8
    op = RopeFwdOp(input_layout="1d")

    # Create a non-contiguous view: transpose makes it non-contiguous
    base = torch.randn(head_dim, seq_len, device=run_device(), dtype=torch.float32)
    x_nc = base.t()  # shape (seq_len, head_dim), non-contiguous
    assert not x_nc.is_contiguous()

    # Reference with contiguous copy
    x_c = x_nc.contiguous()
    out_nc = op(x_nc)
    out_c = op(x_c)
    torch.testing.assert_close(out_nc, out_c, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_rope_rejects_non_float_dtype() -> None:
    from tileops.kernels.rope import RopeNeoxKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        RopeNeoxKernel(seq_len=16, head_dim=64, dtype=torch.int32)
