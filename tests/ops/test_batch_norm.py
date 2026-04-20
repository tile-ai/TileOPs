"""Tests for BatchNormFwdOp and BatchNormBwdOp.

Correctness is validated against torch.nn.functional.batch_norm and the
analytical gradient via torch.autograd.

Run:
    conda run -n tileops python -m pytest tests/ops/test_batch_norm.py -vvs
"""


import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.batch_norm import (
    BatchNormBwdTest as _BatchNormBwdTestWorkload,
)
from workloads.batch_norm import (
    BatchNormFwdTest as _BatchNormFwdTestWorkload,
)


def _ref_fwd(x, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    """Reference: torch.nn.functional.batch_norm (float32 upcast)."""
    x32 = x.float()
    rm = running_mean.clone()
    rv = running_var.clone()
    y32 = torch.nn.functional.batch_norm(
        x32, rm, rv, weight.float(), bias.float(),
        training=training, momentum=momentum, eps=eps)
    return y32.to(x.dtype), rm, rv


class BatchNormBwdTest(_BatchNormBwdTestWorkload, TestBase):
    def ref_program(self, grad_out, x, weight, mean, rstd):
        """Reference via torch.autograd on a float32 graph."""
        x32 = x.float().requires_grad_(True)
        w32 = weight.float().requires_grad_(True)
        b32 = torch.zeros(self.C, device=x.device, dtype=torch.float32, requires_grad=True)
        rm = torch.zeros(self.C, device=x.device, dtype=torch.float32)
        rv = torch.ones(self.C, device=x.device, dtype=torch.float32)
        y32 = torch.nn.functional.batch_norm(
            x32, rm, rv, w32, b32, training=True, momentum=0.1, eps=1e-5)
        y32.backward(grad_out.float())
        return x32.grad.to(x.dtype), w32.grad, b32.grad

class BatchNormFwdTest(_BatchNormFwdTestWorkload, TestBase):
    def ref_program(self, x, weight, bias, running_mean, running_var):
        y, rm, rv = _ref_fwd(x, weight, bias, running_mean, running_var,
                             training=self.training)
        return (y,)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class BatchNormFwdFixture(FixtureBase):
    """(N, C, *spatial, dtype, training)"""
    PARAMS = [
        ("N, C, spatial, dtype, training", [
            # BatchNorm1d – (N, C)
            pytest.param(32, 64, (), torch.float16, True, marks=pytest.mark.smoke),
            pytest.param(32, 64, (), torch.bfloat16, True, marks=pytest.mark.smoke),
            pytest.param(32, 64, (), torch.float16, False, marks=pytest.mark.full),
            pytest.param(32, 256, (), torch.bfloat16, True, marks=pytest.mark.full),
            # BatchNorm1d – (N, C, L)
            pytest.param(16, 64, (512,), torch.float16, True, marks=pytest.mark.full),
            # Non-persistent path (L > 8192): smallest representative case L=16384.
            pytest.param(4, 64, (64, 64), torch.float16, True, marks=pytest.mark.full),
            # BatchNorm2d – (N, C, H, W)
            pytest.param(8, 64, (1024, 1024), torch.float16, True, marks=pytest.mark.full),
            pytest.param(8, 64, (2048, 2048), torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 128, (32, 32), torch.bfloat16, True, marks=pytest.mark.full),
            # Non-aligned spatial: H*W=900, exercises partial-tile path
            pytest.param(8, 64, (30, 30), torch.float16, True, marks=pytest.mark.full),
            pytest.param(8, 64, (30, 30), torch.bfloat16, True, marks=pytest.mark.full),
        ]),
    ]


class BatchNormBwdFixture(FixtureBase):
    """(N, C, *spatial, dtype)"""
    PARAMS = [
        ("N, C, spatial, dtype", [
            pytest.param(32, 64, (), torch.float16, marks=pytest.mark.smoke),
            pytest.param(32, 64, (), torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(8, 64, (32, 32), torch.float16, marks=pytest.mark.full),
            pytest.param(4, 128, (32, 32), torch.bfloat16, marks=pytest.mark.full),
            # Non-persistent backward path (L=16384 > 8192).
            pytest.param(4, 64, (64, 64), torch.float16, marks=pytest.mark.full),
            # Non-aligned spatial: H*W=900, exercises partial-tile path
            pytest.param(8, 64, (30, 30), torch.float16, marks=pytest.mark.full),
            pytest.param(8, 64, (30, 30), torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

@BatchNormFwdFixture
def test_batch_norm_fwd(N, C, spatial, dtype, training):
    test = BatchNormFwdTest(N, C, spatial, dtype, training)
    x, weight, bias, running_mean, running_var = test.gen_inputs()

    # Clone before op call so reference sees the same initial state.
    running_mean_ref = running_mean.clone()
    running_var_ref = running_var.clone()

    op = BatchNormFwdOp(C=C, dtype=dtype)
    y, mean, rstd = op(x, weight, bias, running_mean, running_var, training=training)

    ref_y, ref_rm, ref_rv = _ref_fwd(x, weight, bias, running_mean_ref, running_var_ref,
                                      training=training)

    # float16 accumulates more error; use loose tolerances.
    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (2e-2, 2e-2)
    max_err = (y.float() - ref_y.float()).abs().max()
    assert torch.allclose(y.float(), ref_y.float(), atol=atol, rtol=rtol), \
        f"fwd mismatch (training={training}): max_err={max_err:.4e}"

    if training:
        rm_err = (running_mean.float() - ref_rm.float()).abs().max()
        assert torch.allclose(running_mean.float(), ref_rm.float(), atol=atol, rtol=rtol), \
            f"running_mean mismatch: max_err={rm_err:.4e}"
        rv_err = (running_var.float() - ref_rv.float()).abs().max()
        assert torch.allclose(running_var.float(), ref_rv.float(), atol=atol, rtol=rtol), \
            f"running_var mismatch: max_err={rv_err:.4e}"

    print(f"test_batch_norm_fwd passed [training={training}]: max_err={max_err:.4e}")


@BatchNormBwdFixture
def test_batch_norm_bwd(N, C, spatial, dtype):
    test = BatchNormBwdTest(N, C, spatial, dtype)
    grad_out, x, weight, mean, rstd = test.gen_inputs()

    op = BatchNormBwdOp(C=C, dtype=dtype)
    grad_x, grad_weight, grad_bias = op(grad_out, x, weight, mean, rstd)

    ref_gx, ref_gw, ref_gb = test.ref_program(grad_out, x, weight, mean, rstd)

    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (2e-2, 2e-2)

    for name, got, ref in [
        ("grad_x",      grad_x.float(),    ref_gx.float()),
        ("grad_weight", grad_weight.float(), ref_gw.float()),
        ("grad_bias",   grad_bias.float(),   ref_gb.float()),
    ]:
        max_err = (got - ref).abs().max()
        assert torch.allclose(got, ref, atol=atol, rtol=rtol), \
            f"bwd {name} mismatch: max_err={max_err:.4e}"
    print("test_batch_norm_bwd passed: grad_x/weight/bias all match")


@pytest.mark.smoke
def test_batch_norm_bwd_input_validation():
    """BatchNormBwdOp rejects inconsistent backward inputs in both the
    user-facing ``forward`` path and the eager path used by the custom op
    (``_eager_forward``)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    C = 4
    dtype = torch.float16
    op = BatchNormBwdOp(C=C, dtype=dtype)

    def _good():
        grad_out = torch.randn(2, C, 3, device=device, dtype=dtype)
        x = torch.randn(2, C, 3, device=device, dtype=dtype)
        weight = torch.randn(C, device=device, dtype=torch.float32)
        mean = torch.randn(C, device=device, dtype=torch.float32)
        rstd = torch.ones(C, device=device, dtype=torch.float32)
        return grad_out, x, weight, mean, rstd

    # Baseline: valid inputs must pass _validate_inputs (no raise).
    op._validate_inputs(*_good())

    # Case 1: x.shape != grad_out.shape — the exact bug from the finding.
    grad_out, x, weight, mean, rstd = _good()
    x_bad = torch.randn(1, C, 3, device=device, dtype=dtype)
    with pytest.raises(ValueError, match="x.shape == grad_out.shape"):
        op._validate_inputs(grad_out, x_bad, weight, mean, rstd)
    with pytest.raises(ValueError, match="x.shape == grad_out.shape"):
        op.forward(grad_out, x_bad, weight, mean, rstd)
    # Eager path used by the custom op must validate consistently.
    with pytest.raises(ValueError, match="x.shape == grad_out.shape"):
        op._eager_forward(grad_out, x_bad, weight, mean, rstd)

    # Case 2: grad_out not CUDA.
    grad_out, x, weight, mean, rstd = _good()
    with pytest.raises(ValueError, match="grad_out must be a CUDA tensor"):
        op._validate_inputs(grad_out.cpu(), x, weight, mean, rstd)

    # Case 3: grad_out dtype mismatch.
    grad_out, x, weight, mean, rstd = _good()
    grad_out_bad = grad_out.to(torch.float32)
    with pytest.raises(ValueError, match="grad_out.dtype"):
        op._validate_inputs(grad_out_bad, x, weight, mean, rstd)

    # Case 4: grad_out.ndim < 2.
    grad_out_1d = torch.randn(C, device=device, dtype=dtype)
    x_1d = torch.randn(C, device=device, dtype=dtype)
    _, _, weight, mean, rstd = _good()
    with pytest.raises(ValueError, match="ndim >= 2"):
        op._validate_inputs(grad_out_1d, x_1d, weight, mean, rstd)

    # Case 5: channel dim mismatch.
    grad_out_badc = torch.randn(2, C + 1, 3, device=device, dtype=dtype)
    x_badc = torch.randn(2, C + 1, 3, device=device, dtype=dtype)
    _, _, weight, mean, rstd = _good()
    with pytest.raises(ValueError, match=f"Expected channel dim {C}"):
        op._validate_inputs(grad_out_badc, x_badc, weight, mean, rstd)

    # Case 6: x dtype mismatch.
    grad_out, x, weight, mean, rstd = _good()
    x_bad_dtype = x.to(torch.float32)
    with pytest.raises(ValueError, match="x.dtype"):
        op._validate_inputs(grad_out, x_bad_dtype, weight, mean, rstd)

    # Case 7: weight wrong shape.
    grad_out, x, weight, mean, rstd = _good()
    weight_bad = torch.randn(C + 1, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"weight\.shape"):
        op._validate_inputs(grad_out, x, weight_bad, mean, rstd)

    # Case 8: mean wrong dtype (must be float32).
    grad_out, x, weight, mean, rstd = _good()
    mean_bad = mean.to(torch.float16)
    with pytest.raises(ValueError, match=r"mean\.dtype"):
        op._validate_inputs(grad_out, x, weight, mean_bad, rstd)

    # Case 9: rstd wrong shape.
    grad_out, x, weight, mean, rstd = _good()
    rstd_bad = torch.ones(C + 2, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"rstd\.shape"):
        op._validate_inputs(grad_out, x, weight, mean, rstd_bad)

    print("test_batch_norm_bwd_input_validation passed")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
