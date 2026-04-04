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
from workloads.ops.batch_norm import (
    BatchNormBwdTest as _BatchNormBwdTestWorkload,
)
from workloads.ops.batch_norm import (
    BatchNormFwdTest as _BatchNormFwdTestWorkload,
)
from workloads.ops.batch_norm import (
    _ref_fwd,
)


class BatchNormBwdTest(_BatchNormBwdTestWorkload, TestBase):
    pass
class BatchNormFwdTest(_BatchNormFwdTestWorkload, TestBase):
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class BatchNormFwdFixture(FixtureBase):
    """(N, C, *spatial, dtype, training)"""
    PARAMS = [
        ("N, C, spatial, dtype, training", [
            # BatchNorm1d – (N, C)
            pytest.param(32, 64, (), torch.float16, True, marks=pytest.mark.smoke),
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

    op = BatchNormFwdOp(N, C, *spatial, dtype=dtype)
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

    op = BatchNormBwdOp(N, C, *spatial, dtype=dtype)
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


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
