"""Tests for AdaLayerNorm and AdaLayerNormZero."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.backend import BUILTIN
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from workloads.device import run_device
from workloads.norm import (
    AdaLayerNormWorkload,
    AdaLayerNormZeroWorkload,
    LayerNormLargeOffsetWorkload,
)
from workloads.numerics import compare_outputs


class AdaLayerNormTest(AdaLayerNormWorkload, TestBase):
    pass


class AdaLayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                # Standard aligned shapes -- fp16
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                # Standard aligned shapes -- bf16
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@AdaLayerNormFixture
def test_ada_layer_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ada_layer_norm_kernel_handles_natural_unaligned_shape(
    dtype: torch.dtype,
) -> None:
    m, n = 16, 1152
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp(eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_ada_layer_norm_large_offset() -> None:
    """A padded row whose mean far outgrows its spread keeps the pad out of the variance."""
    large = LayerNormLargeOffsetWorkload(4, 1152, torch.float32)
    test = AdaLayerNormTest(4, 1152, torch.float32)
    x, _, _ = large.gen_inputs()
    _, scale, shift = test.gen_inputs()
    expected = test.ref_program(x, scale, shift)
    compare_outputs(AdaLayerNormFwdOp()(x, scale, shift), expected, large.verification(x))


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, dtype",
    [
        pytest.param(511, torch.float16, id="fp16-below"),
        pytest.param(514, torch.float16, id="fp16-lower-inside"),
    ],
)
def test_ada_layer_norm_async_policy_edge_correctness(
    n: int,
    dtype: torch.dtype,
) -> None:
    m = 4
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp(eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())


class AdaLayerNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@AdaLayerNorm3DFixture
def test_ada_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    scale = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    shift = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())

    op = AdaLayerNormFwdOp()

    test = AdaLayerNormTest(batch * seq, hidden, dtype)
    test.check(op, x, scale, shift)


class AdaLayerNormZeroTest(AdaLayerNormZeroWorkload, TestBase):
    pass


class AdaLayerNormZeroFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                # Standard aligned shapes -- fp16
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                # Standard aligned shapes -- bf16
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@AdaLayerNormZeroFixture
def test_ada_layer_norm_zero_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    op = AdaLayerNormZeroFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ada_layer_norm_zero_kernel_handles_natural_unaligned_shape(
    dtype: torch.dtype,
) -> None:
    m, n = 16, 1152
    test = AdaLayerNormZeroTest(m, n, dtype)
    op = AdaLayerNormZeroFwdOp(eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())


class AdaLayerNormZero3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@AdaLayerNormZero3DFixture
def test_ada_layer_norm_zero_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    scale = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    shift = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    gate = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())

    op = AdaLayerNormZeroFwdOp()

    test = AdaLayerNormZeroTest(batch * seq, hidden, dtype)
    test.check(op, x, scale, shift, gate)


_TUNE = [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)]


@pytest.mark.parametrize("tune", _TUNE)
def test_ada_layer_norm_under_tuning(tune: bool) -> None:
    test = AdaLayerNormTest(17, 514, torch.float16)
    op = AdaLayerNormFwdOp(eps=test.eps)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.parametrize("tune", _TUNE)
def test_ada_layer_norm_zero_under_tuning(tune: bool) -> None:
    test = AdaLayerNormZeroTest(17, 514, torch.float16)
    op = AdaLayerNormZeroFwdOp(eps=test.eps)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


def _misaligned(t: torch.Tensor) -> torch.Tensor:
    """*t* copied into a contiguous view that starts one element into its storage."""
    view = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:].view(t.shape)
    view.copy_(t)
    return view


@pytest.mark.smoke
@pytest.mark.parametrize("zero", [False, True], ids=["ada", "ada-zero"])
def test_ada_layer_norm_reads_inputs_off_the_vector_boundary(zero: bool) -> None:
    """A contiguous input and modulation tensors may start anywhere in their storage."""
    test = (AdaLayerNormZeroTest if zero else AdaLayerNormTest)(4, 1024, torch.float16)
    op = AdaLayerNormZeroFwdOp() if zero else AdaLayerNormFwdOp()
    test.check(op, *(_misaligned(t) for t in test.gen_inputs()))
