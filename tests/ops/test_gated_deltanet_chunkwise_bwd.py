
import pytest
import torch

from tests.test_base import FixtureBase
from tileops.ops import GatedDeltaNetBwdOp
from workloads.ops.gated_deltanet_chunkwise_bwd import _autograd_bwd_ref

# =============================================================================
# Autograd-based reference: differentiable forward → torch.autograd.grad
# =============================================================================


# =============================================================================
# Backward correctness tests
# =============================================================================

def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-2, "rtol": 1e-2}
    elif dtype == torch.float16:
        return {"atol": 5e-2, "rtol": 5e-2}
    else:  # bfloat16 — wider tolerance due to compounding chunk-boundary
        # rounding in bf16 (7-bit mantissa); validated against FLA at 0.998+ cosine.
        return {"atol": 1e-1, "rtol": 1e-1}


class GatedDeltaNetBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@GatedDeltaNetBwdFixture
def test_gated_deltanet_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, H, S, device="cuda", dtype=dtype)
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5

    # Forward to get S for backward kernel
    from tileops.ops import GatedDeltaNetFwdOp
    fwd_op = GatedDeltaNetFwdOp(B, H, S, DK, DV, BC, dtype)
    _o, S_fwd, _Aw, _Au = fwd_op.forward(q, k, v, g, beta)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    # Reference via autograd
    ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta = _autograd_bwd_ref(do, q, k, v, g, beta, BC)
    ref_outputs = (ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta)

    # Kernel
    op = GatedDeltaNetBwdOp(B, H, S, DK, DV, BC, dtype, tune=tune)
    op_outputs = op.forward(do, q, k, v, g, beta, S_fwd)

    tols = _get_tolerances(dtype)
    names = ["dq", "dk", "dv", "dg", "dbeta"]
    for name, ref_out, op_out in zip(names, ref_outputs, op_outputs, strict=True):
        torch.testing.assert_close(
            op_out, ref_out.to(dtype), **tols,
            msg=lambda m, n=name: f"{n}: {m}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
