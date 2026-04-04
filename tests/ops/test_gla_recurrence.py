
from typing import Tuple

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GLADecodeOp
from workloads.ops.gla_recurrence import GLADecodeTest as _GLADecodeTestWorkload
from workloads.ops.gla_recurrence import _gla_decode_torch_ref


class GLADecodeTest(_GLADecodeTestWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        o, new_state = _gla_decode_torch_ref(q, k, v, gk, state, self.scale)
        return o.to(self.dtype), new_state.to(self.dtype)


try:
    from fla.ops.gla import fused_recurrent_gla
except ImportError:
    fused_recurrent_gla = None

# =============================================================================
# Torch reference implementation (test-only)
# =============================================================================


# =============================================================================
# Correctness tests
# =============================================================================


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 5e-4, "rtol": 5e-4}
    elif dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


class GLADecodeFixture(FixtureBase):
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


@GLADecodeFixture
def test_gla_decode(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = GLADecodeTest(batch, heads, dim_k, dim_v, dtype)
    op = GLADecodeOp(batch, heads, dim_k, dim_v, dtype=dtype, tune=tune)
    tols = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), **tols)


@GLADecodeFixture
def test_gla_decode_multi_step(
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

    op = GLADecodeOp(B, H, DK, DV, dtype=dtype, tune=tune)
    tols = _get_tolerances(dtype)

    state_op = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)
    state_ref = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)

    for _ in range(num_steps):
        q = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=dtype) * 0.1
        gk = -torch.rand(B, H, DK, device="cuda", dtype=dtype)

        o_ref, state_ref = _gla_decode_torch_ref(q, k, v, gk, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, gk, state_op)

        torch.testing.assert_close(o_op, o_ref, **tols)
        torch.testing.assert_close(state_op, state_ref, **tols)


@GLADecodeFixture
def test_gla_decode_vs_fla(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Compare TileOPs GLA decode against FLA fused_recurrent_gla with T=1."""
    if fused_recurrent_gla is None:
        pytest.skip("FLA not installed")

    torch.manual_seed(42)
    B, H, DK, DV = batch, heads, dim_k, dim_v
    scale = DK ** -0.5

    q = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, DV, device="cuda", dtype=dtype) * 0.1
    gk = -torch.rand(B, H, DK, device="cuda", dtype=dtype)
    state = torch.randn(B, H, DK, DV, device="cuda", dtype=dtype) * 0.1

    # TileOPs
    op = GLADecodeOp(B, H, DK, DV, scale=scale, dtype=dtype, tune=tune)
    with torch.no_grad():
        o_tile, s_tile = op(q, k, v, gk, state)

    # FLA: needs BTHD layout with T=1
    # q [B,H,DK] -> [B,1,H,DK]
    q_fla = q.unsqueeze(1)
    k_fla = k.unsqueeze(1)
    v_fla = v.unsqueeze(1)
    gk_fla = gk.unsqueeze(1)

    o_fla, s_fla = fused_recurrent_gla(
        q_fla, k_fla, v_fla, gk=gk_fla,
        scale=scale, initial_state=state.contiguous(),
        output_final_state=True,
    )
    o_fla = o_fla.squeeze(1).to(dtype)

    tols = _get_tolerances(dtype)
    torch.testing.assert_close(o_tile, o_fla, **tols)
    torch.testing.assert_close(s_tile, s_fla.to(dtype), **tols)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
