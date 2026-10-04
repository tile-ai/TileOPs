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

from tests.test_base import FixtureBase, TestBase
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.rope import (
    RoPECase,
    ref_rope_neox_position_ids,
    rope_frequency_tables,
    rope_verification,
)

# Test fixtures


class RoPETest(RoPECase, TestBase):
    pass


class RoPEBasicFixture(FixtureBase):
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


class RoPEEdgeFixture(FixtureBase):
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
@RoPEBasicFixture
def test_rope_1d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    from tileops.ops.rope import RoPEFwdOp

    test = RoPETest(variant, "1d", batch, seq_len, num_heads, head_dim, dtype)
    op = RoPEFwdOp(rope_layout=rope_layout, input_layout="1d")
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize("rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")])
@RoPEBasicFixture
def test_rope_2d(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    rope_layout: str,
    variant: str,
) -> None:
    from tileops.ops.rope import RoPEFwdOp

    test = RoPETest(variant, "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RoPEFwdOp(rope_layout=rope_layout, input_layout="2d")
    test.check(op, *test.gen_inputs())


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
    from tileops.ops.rope import RoPENeoxPositionIdsFwdOp

    num_tokens, num_heads, head_dim, max_position = 96, 8, 64, 512
    table_dim = head_dim if rotary_dim is None else rotary_dim
    x = torch.randn(num_tokens, num_heads, head_dim, device=run_device(), dtype=dtype)
    position_ids = (
        torch.arange(num_tokens, device=run_device(), dtype=torch.int32) * 3 + 17
    ) % max_position
    cos, sin = rope_frequency_tables(table_dim, max_position, dtype=dtype, device=run_device())
    ref = ref_rope_neox_position_ids(x, cos, sin, position_ids.long(), rotary_dim=rotary_dim)

    op = RoPENeoxPositionIdsFwdOp(
        max_position=max_position,
        rotary_dim=rotary_dim,
    )
    output = op(x, position_ids)
    compare_outputs(output, ref, rope_verification())


@pytest.mark.smoke
def test_rope_neox_position_ids_validates_range() -> None:
    from tileops.ops.rope import RoPENeoxPositionIdsFwdOp

    op = RoPENeoxPositionIdsFwdOp(max_position=8)
    x = torch.randn(2, 1, 16, device=run_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="position_ids"):
        op(x, torch.tensor([0, 8], device=run_device(), dtype=torch.int32))


@pytest.mark.smoke
def test_rope_longrope_rejects_a_zero_rescale_factor() -> None:
    from tileops.ops.rope import LongRoPEFwdOp

    rescale = torch.tensor([1.0, 0.0, 2.0, 1.5], device=run_device())
    with pytest.raises(ValueError, match="rescale_factors"):
        LongRoPEFwdOp(rescale_factors=rescale)


@pytest.mark.smoke
def test_rope_neox_position_ids_none_rotary_dim_reinfers_head_dim() -> None:
    from tileops.ops.rope import RoPENeoxPositionIdsFwdOp

    max_position = 64
    position_ids = torch.arange(8, device=run_device(), dtype=torch.int32)
    op = RoPENeoxPositionIdsFwdOp(max_position=max_position, rotary_dim=None)

    x1 = torch.randn(8, 2, 16, device=run_device(), dtype=torch.float16)
    cos1, sin1 = rope_frequency_tables(16, max_position, dtype=x1.dtype, device=run_device())
    ref1 = ref_rope_neox_position_ids(x1, cos1, sin1, position_ids.long(), rotary_dim=None)
    compare_outputs(op(x1, position_ids), ref1, rope_verification())

    x2 = torch.randn(8, 2, 32, device=run_device(), dtype=torch.float16)
    cos2, sin2 = rope_frequency_tables(32, max_position, dtype=x2.dtype, device=run_device())
    ref2 = ref_rope_neox_position_ids(x2, cos2, sin2, position_ids.long(), rotary_dim=None)
    compare_outputs(op(x2, position_ids), ref2, rope_verification())


# Non-neox (RoFormer) RoPE tests


# Llama 3.1 RoPE tests


@RoPEBasicFixture
def test_rope_llama31_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import RoPELlama31FwdOp

    extra = {
        "scale_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position": 8192,
    }
    test = RoPETest(
        "rope_llama31", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RoPELlama31FwdOp(input_layout="1d", **extra)
    test.check(op, *test.gen_inputs())


@RoPEBasicFixture
def test_rope_llama31_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import RoPELlama31FwdOp

    extra = {
        "scale_factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position": 8192,
    }
    test = RoPETest(
        "rope_llama31", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = RoPELlama31FwdOp(input_layout="2d", **extra)
    test.check(op, *test.gen_inputs())


# YaRN RoPE tests


@RoPEBasicFixture
def test_yarn_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import YaRNFwdOp

    extra = {
        "scale": 16.0,
        "original_max_position": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attn_factor": 1.0,
    }
    test = RoPETest(
        "yarn_rope", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = YaRNFwdOp(input_layout="1d", **extra)
    test.check(op, *test.gen_inputs())


@RoPEBasicFixture
def test_yarn_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import YaRNFwdOp

    extra = {
        "scale": 16.0,
        "original_max_position": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attn_factor": 1.0,
    }
    test = RoPETest(
        "yarn_rope", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = YaRNFwdOp(input_layout="2d", **extra)
    test.check(op, *test.gen_inputs())


# LongRoPE tests


@RoPEBasicFixture
def test_rope_longrope_1d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import LongRoPEFwdOp

    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=run_device())
    max_pos = 16384
    orig_max_pos = 4096
    extra = {
        "rescale_factors": rescale,
        "max_position_embeddings": max_pos,
        "original_max_position_embeddings": orig_max_pos,
    }
    test = RoPETest(
        "longrope", "1d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = LongRoPEFwdOp(
        input_layout="1d",
        rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    test.check(op, *test.gen_inputs())


@RoPEBasicFixture
def test_rope_longrope_2d(
    batch: int, seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    from tileops.ops.rope import LongRoPEFwdOp

    half = head_dim // 2
    rescale = torch.linspace(1.0, 2.0, half, device=run_device())
    max_pos = 16384
    orig_max_pos = 4096
    extra = {
        "rescale_factors": rescale,
        "max_position_embeddings": max_pos,
        "original_max_position_embeddings": orig_max_pos,
    }
    test = RoPETest(
        "longrope", "2d", batch, seq_len, num_heads, head_dim, dtype, extra_kwargs=extra
    )
    op = LongRoPEFwdOp(
        input_layout="2d",
        rescale_factors=rescale,
        max_position_embeddings=max_pos,
        original_max_position_embeddings=orig_max_pos,
    )
    test.check(op, *test.gen_inputs())


# Edge case tests


@pytest.mark.parametrize("rope_layout, variant", [("neox", "neox"), ("interleaved", "non_neox")])
@RoPEEdgeFixture
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
    from tileops.ops.rope import RoPEFwdOp

    test = RoPETest(variant, "2d", batch, seq_len, num_heads, head_dim, dtype)
    op = RoPEFwdOp(rope_layout=rope_layout, input_layout="2d")
    test.check(op, *test.gen_inputs())


# Layout and dtype regression tests


@pytest.mark.smoke
def test_rope_noncontiguous_1d_works() -> None:
    """A non-contiguous 1D view must produce correct results after contiguity normalization."""
    from tileops.ops.rope import RoPEFwdOp

    seq_len, head_dim = 4, 8
    op = RoPEFwdOp(input_layout="1d")

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
    from tileops.kernels.rope import RoPENeoxKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        RoPENeoxKernel(seq_len=16, head_dim=64, dtype=torch.int32)
