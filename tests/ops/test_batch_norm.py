"""Tests for BatchNormFwdOp and BatchNormBwdOp.

Correctness is validated against torch.nn.functional.batch_norm and the
analytical gradient via torch.autograd.

Run:
    conda run -n tileops python -m pytest tests/ops/test_batch_norm.py -vvs
"""

import re

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.norm import BatchNormFwdTrainKernel
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.normalization import (
    BatchNormBwdWorkload,
    BatchNormFwdWorkload,
    batch_norm_fwd_ref,
)


class BatchNormBwdTest(BatchNormBwdWorkload, TestBase):
    pass


class BatchNormFwdTest(BatchNormFwdWorkload, TestBase):
    pass


class BatchNormFwdFixture(FixtureBase):
    """(N, C, *spatial, dtype, training)"""

    PARAMS = [
        (
            "N, C, spatial, dtype, training",
            [
                # BatchNorm1d – (N, C)
                pytest.param(32, 64, (), torch.float16, True, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.bfloat16, True, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.float16, False, marks=pytest.mark.full),
                pytest.param(32, 256, (), torch.bfloat16, True, marks=pytest.mark.full),
                # BatchNorm1d – (N, C, L)
                pytest.param(16, 64, (512,), torch.float16, True, marks=pytest.mark.full),
                # L > 8192, so the tiled kernel this shape falls back to reads global
                # memory twice rather than holding the tile in shared memory.
                pytest.param(4, 64, (64, 64), torch.float16, True, marks=pytest.mark.full),
                # BatchNorm2d – (N, C, H, W)
                pytest.param(8, 64, (1024, 1024), torch.float16, True, marks=pytest.mark.full),
                pytest.param(8, 64, (2048, 2048), torch.float16, False, marks=pytest.mark.full),
                pytest.param(4, 128, (32, 32), torch.bfloat16, True, marks=pytest.mark.full),
                # Non-aligned spatial: H*W=900, exercises partial-tile path
                pytest.param(8, 64, (30, 30), torch.float16, True, marks=pytest.mark.full),
                pytest.param(8, 64, (30, 30), torch.bfloat16, True, marks=pytest.mark.full),
                # High channel count oversubscribes the SMs, exposing the running-stat update race.
                pytest.param(16, 1024, (512,), torch.float16, True, marks=pytest.mark.full),
                # The streamed path, which no other case reaches: a channel too long to
                # hold in registers (L = 131072) and too many channels to split (C >= 1024).
                pytest.param(2048, 1024, (64,), torch.float16, True, marks=pytest.mark.full),
            ],
        ),
    ]


class BatchNormBwdFixture(FixtureBase):
    """(N, C, *spatial, dtype)"""

    PARAMS = [
        (
            "N, C, spatial, dtype",
            [
                pytest.param(32, 64, (), torch.float16, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(8, 64, (32, 32), torch.float16, marks=pytest.mark.full),
                pytest.param(4, 128, (32, 32), torch.bfloat16, marks=pytest.mark.full),
                # Non-persistent backward path (L=16384 > 8192).
                pytest.param(4, 64, (64, 64), torch.float16, marks=pytest.mark.full),
                # Non-aligned spatial: H*W=900, exercises partial-tile path
                pytest.param(8, 64, (30, 30), torch.float16, marks=pytest.mark.full),
                pytest.param(8, 64, (30, 30), torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@BatchNormFwdFixture
def test_batch_norm_fwd(N, C, spatial, dtype, training):
    test = BatchNormFwdTest(N, C, spatial, dtype, training)
    x, weight, bias, running_mean, running_var = test.gen_inputs()

    # Clone before op call so reference sees the same initial state.
    running_mean_ref = running_mean.clone()
    running_var_ref = running_var.clone()

    op = BatchNormFwdOp(training=training)
    # Manifest input order: (x, running_mean, running_var, weight, bias).
    y = op(x, running_mean, running_var, weight, bias)

    ref_y, ref_rm, ref_rv = batch_norm_fwd_ref(
        x, weight, bias, running_mean_ref, running_var_ref, training=training
    )

    # float16 accumulates more error; use loose tolerances.
    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (2e-2, 2e-2)
    max_err = (y.float() - ref_y.float()).abs().max()
    assert torch.allclose(y.float(), ref_y.float(), atol=atol, rtol=rtol), (
        f"fwd mismatch (training={training}): max_err={max_err:.4e}"
    )

    if training:
        # allclose is masked when running_mean starts near the batch mean; check determinism.
        rm2, rv2 = running_mean_ref.clone(), running_var_ref.clone()
        op(x, rm2, rv2, weight, bias)
        det_err = (running_mean.float() - rm2.float()).abs().max()
        assert torch.equal(running_mean, rm2) and torch.equal(running_var, rv2), (
            f"running stats non-deterministic across runs: max_err={det_err:.4e}"
        )

        rm_err = (running_mean.float() - ref_rm.float()).abs().max()
        assert torch.allclose(running_mean.float(), ref_rm.float(), atol=atol, rtol=rtol), (
            f"running_mean mismatch: max_err={rm_err:.4e}"
        )
        rv_err = (running_var.float() - ref_rv.float()).abs().max()
        assert torch.allclose(running_var.float(), ref_rv.float(), atol=atol, rtol=rtol), (
            f"running_var mismatch: max_err={rv_err:.4e}"
        )


@BatchNormBwdFixture
def test_batch_norm_bwd(N, C, spatial, dtype):
    test = BatchNormBwdTest(N, C, spatial, dtype)
    grad_out, x, weight, mean, rstd = test.gen_inputs()

    op = BatchNormBwdOp()
    grad_x, grad_weight, grad_bias = op(grad_out, x, weight, mean, rstd)

    ref_gx, ref_gw, ref_gb = test.ref_program(grad_out, x, weight, mean, rstd)

    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (2e-2, 2e-2)

    for name, got, ref in [
        ("grad_x", grad_x.float(), ref_gx.float()),
        ("grad_weight", grad_weight.float(), ref_gw.float()),
        ("grad_bias", grad_bias.float(), ref_gb.float()),
    ]:
        max_err = (got - ref).abs().max()
        assert torch.allclose(got, ref, atol=atol, rtol=rtol), (
            f"bwd {name} mismatch: max_err={max_err:.4e}"
        )


@pytest.mark.smoke
def test_batch_norm_fwd_returns_single_tensor() -> None:
    """BatchNormFwdOp forward must produce one tensor — manifest declares
    a single output. ``training`` is bound at ctor; the runtime kwarg is
    no longer accepted."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    N, C, H, W = 4, 8, 4, 4
    op = BatchNormFwdOp(training=False)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    rm = torch.zeros(C, device="cuda", dtype=torch.float32)
    rv = torch.ones(C, device="cuda", dtype=torch.float32)

    y = op(x, rm, rv, weight, bias)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_training_updates_a_non_contiguous_running_stat() -> None:
    """Contiguity normalization must not swallow the write a mutated input promises."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    N, C, H, W = 4, 8, 4, 4
    op = BatchNormFwdOp(training=True)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    weight = torch.ones(C, device="cuda", dtype=torch.float32)
    bias = torch.zeros(C, device="cuda", dtype=torch.float32)
    # Every other element of a wider buffer: a view the kernel cannot be handed as is.
    rm = torch.zeros(2 * C, device="cuda", dtype=torch.float32)[::2]
    rv = torch.ones(2 * C, device="cuda", dtype=torch.float32)[::2]
    assert not rm.is_contiguous()

    op(x, rm, rv, weight, bias)

    expected_mean = op.momentum * x.float().transpose(0, 1).reshape(C, -1).mean(dim=1)
    torch.testing.assert_close(rm, expected_mean, atol=1e-3, rtol=1e-3)
    assert not torch.equal(rv, torch.ones_like(rv)), "running_var was not written either"


@pytest.mark.smoke
def test_training_rejects_one_value_per_channel() -> None:
    """Bessel's correction divides by L - 1; torch refuses the same call."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    C = 4
    x = torch.randn(1, C, 1, 1, device="cuda", dtype=torch.float32)
    weight = torch.ones(C, device="cuda", dtype=torch.float32)
    bias = torch.zeros(C, device="cuda", dtype=torch.float32)
    rm = torch.zeros(C, device="cuda", dtype=torch.float32)
    rv = torch.ones(C, device="cuda", dtype=torch.float32)

    torch_message = ""
    try:
        torch.nn.functional.batch_norm(x, rm, rv, weight, bias, training=True)
    except ValueError as exc:
        torch_message = str(exc)
    assert torch_message, "torch accepted L=1 in training; the guard's premise is gone"

    with pytest.raises(ValueError, match=re.escape(torch_message)):
        BatchNormFwdOp(training=True)(x, rm, rv, weight, bias)

    # The kernel refuses on its own: it is exported, so a caller can reach it directly.
    with pytest.raises(ValueError, match="more than one value per channel"):
        BatchNormFwdTrainKernel(C, 1, torch.float32, S=1)

    # Inference applies no correction; torch normalizes the same shape.
    infer = BatchNormFwdOp(training=False)
    y = infer(x, rm, rv, weight, bias)
    expected = torch.nn.functional.batch_norm(
        x, rm, rv, weight, bias, training=False, eps=infer.eps
    )
    torch.testing.assert_close(y, expected, atol=1e-3, rtol=1e-3)
