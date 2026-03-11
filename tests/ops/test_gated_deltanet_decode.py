from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GatedDeltaNetDecodeOp

# =============================================================================
# Torch reference implementation (test-only)
# =============================================================================


def _gated_deltanet_decode_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step gated delta rule.

    Args:
        q: [B, H, DK]
        k: [B, H, DK]
        v: [B, H, DV]
        g: [B, H]       (log-space gate)
        beta: [B, H]    (writing strength)
        state: [B, H, DK, DV]

    Returns:
        o: [B, H, DV]
        new_state: [B, H, DK, DV]
    """
    q, k, v = q.float(), k.float(), v.float()
    g, beta = g.float(), beta.float()
    state = state.float()

    alpha = torch.exp(g)  # [B, H]

    # Step 1: old_val = state @ k  ->  [B, H, DV]
    # state: [B,H,DK,DV], k: [B,H,DK] -> einsum "bhkv,bhk->bhv"
    old_val = torch.einsum("bhkv,bhk->bhv", state, k)

    # Step 2: v_new = beta * v - alpha * beta * old_val  ->  [B, H, DV]
    beta_unsq = beta.unsqueeze(-1)   # [B, H, 1]
    alpha_unsq = alpha.unsqueeze(-1)  # [B, H, 1]
    v_new = beta_unsq * v - alpha_unsq * beta_unsq * old_val

    # Step 3: output
    # o_inter = alpha * (state @ q)  ->  [B, H, DV]
    o_inter = alpha_unsq * torch.einsum("bhkv,bhk->bhv", state, q)
    # o_intra = (q . k) * v_new  ->  [B, H, DV]
    qk_dot = torch.einsum("bhk,bhk->bh", q, k).unsqueeze(-1)  # [B, H, 1]
    o_intra = qk_dot * v_new
    o = o_inter + o_intra

    # Step 4: new_state = alpha * state + outer(k, v_new)
    # outer(k, v_new): [B,H,DK,1] * [B,H,1,DV] -> [B,H,DK,DV]
    new_state = alpha_unsq.unsqueeze(-1) * state + k.unsqueeze(-1) * v_new.unsqueeze(-2)

    return o, new_state


# =============================================================================
# Correctness tests
# =============================================================================


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    elif dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


class GatedDeltaNetDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype, tune", [
            pytest.param(1, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(2, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
            pytest.param(1, 4, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 8, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


class GatedDeltaNetDecodeTest(TestBase):

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
        B, H, DK, DV = self.batch, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, H, device="cuda", dtype=self.dtype)
        beta = torch.rand(B, H, device="cuda", dtype=self.dtype) * 0.5
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=self.dtype) * 0.1
        return q, k, v, g, beta, state

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, new_state = _gated_deltanet_decode_torch_ref(q, k, v, g, beta, state)
        return o.to(self.dtype), new_state.to(self.dtype)


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

        o_ref, state_ref = _gated_deltanet_decode_torch_ref(q, k, v, g, beta, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, g, beta, state_op)

        torch.testing.assert_close(o_op, o_ref, **tols)
        torch.testing.assert_close(state_op, state_ref, **tols)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
