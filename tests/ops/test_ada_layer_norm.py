"""Tests for AdaLayerNorm and AdaLayerNormZero."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.backend import BUILTIN
from tileops.kernels.norm.ada_layer_norm import (
    AdaLayerNormKernel,
    AdaLayerNormZeroKernel,
    _should_use_cp_async,
)
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from workloads.device import run_device
from workloads.norm import AdaLayerNormWorkload, AdaLayerNormZeroWorkload


class AdaLayerNormTest(AdaLayerNormWorkload, TestBase):
    pass


# An autotune candidate for a 514-wide fp16 row: block_m > 1 puts a tail-row
# block in the async 2-D tile, which the block_m=1 default never does.
_ROW_TAIL_CONFIG = {"block_m": 4, "threads": 128}


class _RowTailAdaLayerNormKernel(AdaLayerNormKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "config": _ROW_TAIL_CONFIG})


class _RowTailAdaLayerNormZeroKernel(AdaLayerNormZeroKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "config": _ROW_TAIL_CONFIG})


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


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_ada_layer_norm_async_copy_handles_row_tail() -> None:
    """Regression: the async 2-D tile must support block_m > 1 and tail rows."""
    m, n = 17, 514
    dtype = torch.float16
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp(
        eps=test.eps, kernel_map={"ada_layer_norm": _RowTailAdaLayerNormKernel}, target=BUILTIN
    )
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("ada_layer_norm").values()
    assert type(kernel) is _RowTailAdaLayerNormKernel
    assert kernel.use_cp_async
    assert kernel.config == _ROW_TAIL_CONFIG
    assert kernel.config in kernel.autotune_configs


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
    op = AdaLayerNormFwdOp(eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("ada_layer_norm").values()
    assert kernel.use_cp_async is (n == 514)


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


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_ada_layer_norm_zero_async_copy_handles_row_tail() -> None:
    """Regression: the async 2-D tile must support block_m > 1 and tail rows."""
    m, n = 17, 514
    dtype = torch.float16
    test = AdaLayerNormZeroTest(m, n, dtype)
    op = AdaLayerNormZeroFwdOp(
        eps=test.eps, kernel_map={"ada_layer_norm": _RowTailAdaLayerNormZeroKernel}, target=BUILTIN
    )
    test.check(op, *test.gen_inputs())
    (kernel,) = op.built_kernels("ada_layer_norm").values()
    assert type(kernel) is _RowTailAdaLayerNormZeroKernel
    assert kernel.use_cp_async
    assert kernel.config == _ROW_TAIL_CONFIG
    assert kernel.config in kernel.autotune_configs


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
