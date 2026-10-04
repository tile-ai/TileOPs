"""Tests for AdaLayerNorm and AdaLayerNormZero."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.norm.ada_layer_norm import (
    AdaLayerNormKernel,
    _should_use_cp_async,
)
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from workloads.device import run_device
from workloads.norm import AdaLayerNormWorkload, AdaLayerNormZeroWorkload
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
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(n, test.eps, dtype, has_gate=False)
    actual = kernel(*inputs)
    expected = test.ref_program(*inputs)
    assert actual.shape == (m, n)
    compare_outputs(actual, expected, test.verification(*inputs))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_ada_layer_norm_async_copy_handles_row_tail() -> None:
    """Regression: the async 2-D tile must support block_m > 1 and tail rows."""
    m, n, block_m = 17, 514, 4
    dtype = torch.float16
    test = AdaLayerNormTest(m, n, dtype)
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(
        n,
        test.eps,
        dtype,
        has_gate=False,
        config={"block_m": block_m, "threads": 128},
    )
    assert kernel.use_cp_async
    actual = kernel(*inputs)
    expected = test.ref_program(*inputs)
    compare_outputs(actual, expected, test.verification(*inputs))


@pytest.mark.smoke
def test_ada_layer_norm_async_policy_edges() -> None:
    cases = [
        (511, torch.float16, False),
        (512, torch.float16, False),
        (513, torch.float16, False),
        (514, torch.float16, True),
        (1918, torch.float16, True),
        (1919, torch.float16, False),
        (1920, torch.float16, True),
        (513, torch.float32, True),
        (1919, torch.float32, True),
    ]
    for n, dtype, expected_async in cases:
        assert _should_use_cp_async(n, dtype, has_gate=False) is expected_async


@pytest.mark.smoke
def test_ada_layer_norm_async_policy_shared_memory_limit() -> None:
    cases = [
        (8190, torch.float16, False, True),
        (8194, torch.float16, False, False),
        (6142, torch.float16, True, True),
        (6146, torch.float16, True, False),
        (4094, torch.float32, False, True),
        (4098, torch.float32, False, False),
    ]
    for n, dtype, has_gate, expected_async in cases:
        assert _should_use_cp_async(n, dtype, has_gate) is expected_async


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
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(n, test.eps, dtype, has_gate=False)
    if n == 514:
        assert kernel.use_cp_async
    actual = kernel(*inputs)
    expected = test.ref_program(*inputs)
    compare_outputs(actual, expected, test.verification(*inputs))


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
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(n, test.eps, dtype, has_gate=True)
    actual = kernel(*inputs)
    expected = test.ref_program(*inputs)
    assert actual.shape == (m, n)
    compare_outputs(actual, expected, test.verification(*inputs))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_ada_layer_norm_zero_async_copy_handles_row_tail() -> None:
    """Regression: the async 2-D tile must support block_m > 1 and tail rows."""
    m, n, block_m = 17, 514, 4
    dtype = torch.float16
    test = AdaLayerNormZeroTest(m, n, dtype)
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(
        n,
        test.eps,
        dtype,
        has_gate=True,
        config={"block_m": block_m, "threads": 128},
    )
    assert kernel.use_cp_async
    actual = kernel(*inputs)
    expected = test.ref_program(*inputs)
    compare_outputs(actual, expected, test.verification(*inputs))


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
