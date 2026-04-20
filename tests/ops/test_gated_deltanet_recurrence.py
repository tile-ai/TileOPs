

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GatedDeltaNetDecodeOp
from workloads.gated_deltanet import (
    GatedDeltaNetDecodeTest as _GatedDeltaNetDecodeTestWorkload,
)


def gated_deltanet_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step gated delta rule."""
    q, k, v = q.float(), k.float(), v.float()
    g, beta = g.float(), beta.float()
    state = state.float()

    alpha = torch.exp(g)
    old_val = torch.einsum("bhkv,bhk->bhv", state, k)

    beta_unsq = beta.unsqueeze(-1)
    alpha_unsq = alpha.unsqueeze(-1)
    v_new = beta_unsq * v - alpha_unsq * beta_unsq * old_val

    o_inter = alpha_unsq * torch.einsum("bhkv,bhk->bhv", state, q)
    qk_dot = torch.einsum("bhk,bhk->bh", q, k).unsqueeze(-1)
    o_intra = qk_dot * v_new
    o = o_inter + o_intra

    new_state = alpha_unsq.unsqueeze(-1) * state + k.unsqueeze(-1) * v_new.unsqueeze(-2)

    return o, new_state


class GatedDeltaNetDecodeTest(_GatedDeltaNetDecodeTestWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, new_state = gated_deltanet_decode_torch(q, k, v, g, beta, state)
        return o.to(self.dtype), new_state.to(self.dtype)


# =============================================================================
# Torch reference implementation (test-only)
# =============================================================================


# =============================================================================
# Correctness tests
# =============================================================================


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        # T.gemm uses tensor cores (TF32) which truncate fp32 mantissa to
        # 10 bits, giving ~1e-4 error per op; 5e-4 accommodates accumulation.
        return {"atol": 5e-4, "rtol": 5e-4}
    elif dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


class GatedDeltaNetDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype, tune", [
            pytest.param(1, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(
                1,
                4,
                64,
                64,
                torch.float16,
                False,
                marks=[
                    pytest.mark.smoke,
                    pytest.mark.skip(
                        reason="Temporarily skipped while isolating low-precision gated deltanet decode failures under TileLang 5f70374c (#999)."
                    ),
                ],
            ),
            pytest.param(
                1,
                4,
                64,
                64,
                torch.bfloat16,
                False,
                marks=[
                    pytest.mark.smoke,
                    pytest.mark.skip(
                        reason="Temporarily skipped while isolating low-precision gated deltanet decode failures under TileLang 5f70374c (#999)."
                    ),
                ],
            ),
            pytest.param(2, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
            pytest.param(
                2,
                8,
                64,
                64,
                torch.float16,
                False,
                marks=[
                    pytest.mark.full,
                    pytest.mark.skip(
                        reason="Temporarily skipped while isolating low-precision gated deltanet decode failures under TileLang 5f70374c (#999)."
                    ),
                ],
            ),
            pytest.param(
                2,
                8,
                64,
                64,
                torch.bfloat16,
                False,
                marks=[
                    pytest.mark.full,
                    pytest.mark.skip(
                        reason="Temporarily skipped while isolating low-precision gated deltanet decode failures under TileLang 5f70374c (#999)."
                    ),
                ],
            ),
        ]),
    ]


@GatedDeltaNetDecodeFixture
def test_gated_deltanet_decode(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetDecodeTest(batch, heads, dim_k, dim_v, dtype)
    op = GatedDeltaNetDecodeOp(batch, heads, dim_k, dim_v, dtype, tune=tune)
    tols = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), **tols)


@GatedDeltaNetDecodeFixture
def test_gated_deltanet_decode_multi_step(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Test multiple sequential decode steps to verify state propagation."""
    torch.manual_seed(42)
    num_steps = 8
    B, H, DK, DV = batch, heads, dim_k, dim_v

    op = GatedDeltaNetDecodeOp(B, H, DK, DV, dtype, tune=tune)
    tols = _get_tolerances(dtype)

    state_op = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)
    state_ref = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)

    for _ in range(num_steps):
        q = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=dtype) * 0.1
        g = -torch.rand(B, H, device="cuda", dtype=dtype)
        beta = torch.rand(B, H, device="cuda", dtype=dtype) * 0.5

        o_ref, state_ref = gated_deltanet_decode_torch(q, k, v, g, beta, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, g, beta, state_op)

        torch.testing.assert_close(o_op, o_ref, **tols)
        torch.testing.assert_close(state_op, state_ref, **tols)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
