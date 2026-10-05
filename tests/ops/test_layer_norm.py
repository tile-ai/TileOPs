"""Tests for LayerNorm and fused-add LayerNorm."""

import pytest
import torch
import torch.nn.functional as F

from tests.workload_test_base import FixtureBase, TestBase
from tileops.backend import BUILTIN
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from tileops.ops.norm.layer_norm import LayerNormFwdOp
from workloads.device import run_device
from workloads.norm import (
    FusedAddLayerNormWorkload,
    LayerNormLargeOffsetWorkload,
    LayerNormWorkload,
    layer_norm_verification,
    normalization_verification,
)
from workloads.numerics import compare_outputs


class LayerNormTest(LayerNormWorkload, TestBase):
    pass


class LayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.float32, False, marks=pytest.mark.full),
                # Standard aligned shapes -- fp16
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.float16, False, marks=pytest.mark.full),
                # Standard aligned shapes -- bf16
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.bfloat16, False, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.bfloat16, False, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@LayerNormFixture
def test_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LayerNormTest(m, n, dtype)
    op = LayerNormFwdOp(normalized_shape=(n,))
    test.check(op, *test.gen_inputs())


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_layer_norm_kernel_handles_unaligned_shape() -> None:
    """The kernel, not the Op layer, owns non-aligned boundary handling."""
    m, n = 16, 3000
    dtype = torch.float16
    test = LayerNormTest(m, n, dtype)
    op = LayerNormFwdOp(normalized_shape=(n,), eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())


class LayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@LayerNormNonContigFixture
def test_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device=run_device())
    bias = torch.randn(n, dtype=dtype, device=run_device())

    op = LayerNormFwdOp(normalized_shape=(n,))

    # Reference using torch.nn.functional.layer_norm
    x_ref = x.contiguous()
    y_ref = F.layer_norm(
        x_ref.float(),
        (n,),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    compare_outputs(y, y_ref, layer_norm_verification(dtype))


class LayerNorm3DFixture(FixtureBase):
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


@LayerNorm3DFixture
def test_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    weight = torch.randn(hidden, dtype=dtype, device=run_device())
    bias = torch.randn(hidden, dtype=dtype, device=run_device())

    op = LayerNormFwdOp(normalized_shape=(hidden,))

    # Reference using torch.nn.functional.layer_norm
    y_ref = F.layer_norm(
        x.float(),
        (hidden,),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    compare_outputs(y, y_ref, layer_norm_verification(dtype))


class LayerNormLargeOffsetFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(4, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(4, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(4, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


@LayerNormLargeOffsetFixture
def test_layer_norm_large_offset(m: int, n: int, dtype: torch.dtype) -> None:
    """Regression: large-mean, low-variance inputs stress the variance formula.

    E[x^2] - mean^2 would suffer catastrophic cancellation here (max_err > 1.0);
    the centered two-pass approach keeps error within a few percent.

    Note: fp32 reduction order differences between TileLang's T.reduce_sum and
    PyTorch's fused CUDA layer_norm cause inherent ~1-2% relative disagreement
    on adversarial large-offset inputs (var ~ 1e-4, mean ~ 10000).  We use
    a relative tolerance of 5% which is tight enough to catch the original
    catastrophic cancellation bug (which produced >100x error) while allowing
    the inherent fp32 parallel reduction precision limits.
    """
    workload = LayerNormLargeOffsetWorkload(m, n, dtype)
    inputs = workload.gen_inputs()
    op = LayerNormFwdOp(normalized_shape=(n,))
    TestBase.check(workload, op, *inputs)


@pytest.mark.smoke
def test_layer_norm_serves_a_changed_leading_dims_product_from_one_kernel() -> None:
    """The row count is a fact of the call, so it does not specialize the op's kernel."""
    n = 4096
    dtype = torch.float16

    op = LayerNormFwdOp(normalized_shape=(n,))
    weight = torch.randn(n, dtype=dtype, device=run_device())
    bias = torch.randn(n, dtype=dtype, device=run_device())

    x1 = torch.randn(512, n, dtype=dtype, device=run_device())
    y1 = op(x1, weight, bias)
    assert y1.shape == x1.shape

    x2 = torch.randn(1024, n, dtype=dtype, device=run_device())
    y2 = op(x2, weight, bias)
    assert y2.shape == x2.shape

    y_ref = F.layer_norm(
        x2.float(),
        (n,),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)
    compare_outputs(y2, y_ref, layer_norm_verification(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("give", ["weight", "bias", "neither"])
def test_either_affine_tensor_alone_matches_torch(give: str) -> None:
    n, dtype = 256, torch.float16
    x = torch.randn(8, n, dtype=dtype, device=run_device())
    kwargs = {} if give == "neither" else {give: torch.randn(n, dtype=dtype, device=run_device())}
    got = LayerNormFwdOp(normalized_shape=(n,))(x, **kwargs)
    compare_outputs(
        got, F.layer_norm(x, (n,), **kwargs), normalization_verification("LayerNormFwdOp", x.dtype)
    )


class FusedAddLayerNormTest(FusedAddLayerNormWorkload, TestBase):
    pass


class FusedAddLayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                # Standard aligned shapes -- fp16
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                # Standard aligned shapes -- bf16
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@FusedAddLayerNormFixture
def test_fused_add_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddLayerNormTest(m, n, dtype)
    op = FusedAddLayerNormFwdOp(tune=tune)
    test.check(op, *test.gen_inputs())


class FusedAddLayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FusedAddLayerNormNonContigFixture
def test_fused_add_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    r_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]  # non-contiguous slice
    residual = r_full[:, :n]
    weight = torch.randn(n, dtype=dtype, device=run_device())
    bias = torch.randn(n, dtype=dtype, device=run_device())

    op = FusedAddLayerNormFwdOp()

    # Reference on contiguous copies
    test = FusedAddLayerNormTest(m, n, dtype)
    y_ref, add_ref = test.ref_program(x.contiguous(), residual.contiguous(), weight, bias)

    y, residual_out = op(x, residual, weight, bias)

    compare_outputs(y, y_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype))
    compare_outputs(
        residual_out, add_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype)
    )


class FusedAddLayerNorm3DFixture(FixtureBase):
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


@FusedAddLayerNorm3DFixture
def test_fused_add_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    residual = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    weight = torch.randn(hidden, dtype=dtype, device=run_device())
    bias = torch.randn(hidden, dtype=dtype, device=run_device())

    M = batch * seq
    op = FusedAddLayerNormFwdOp()

    test = FusedAddLayerNormTest(M, hidden, dtype)
    y_ref, add_ref = test.ref_program(x, residual, weight, bias)

    y, residual_out = op(x, residual, weight, bias)

    compare_outputs(y, y_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype))
    compare_outputs(
        residual_out, add_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype)
    )
