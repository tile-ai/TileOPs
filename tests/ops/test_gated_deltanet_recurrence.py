

import pytest
import torch

import tileops.ops.gated_deltanet as gated_deltanet_ops
from tests.test_base import FixtureBase, TestBase
from tileops.kernels.gated_deltanet_recurrence import GatedDeltaNetDecodeRawCudaFlaStyleKernel
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
        # Keep fp32 tolerance aligned with the scalar decode reference path;
        # multi-step recurrence still accumulates small rounding differences.
        return {"atol": 5e-4, "rtol": 5e-4}
    elif dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


class GatedDeltaNetDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype, tune", [
            pytest.param(1, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 4, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 8, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 32, 128, 128, torch.bfloat16, False, marks=pytest.mark.full),
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


@pytest.mark.smoke
def test_gated_deltanet_decode_raw_cuda_config_requires_full_warp_mapping() -> None:
    with pytest.raises(ValueError, match="threads .* must equal raw_group_size \\* v_tile"):
        GatedDeltaNetDecodeRawCudaFlaStyleKernel(
            1,
            32,
            128,
            128,
            dtype="bfloat16",
            config={
                "threads": 16,
                "v_tile": 16,
                "raw_group_size": 2,
                "raw_maxrregcount": 146,
            },
        )


@pytest.mark.smoke
def test_gated_deltanet_decode_raw_cuda_config_requires_two_lane_group() -> None:
    with pytest.raises(ValueError, match="raw_group_size must equal 2"):
        GatedDeltaNetDecodeRawCudaFlaStyleKernel(
            1,
            32,
            128,
            128,
            dtype="bfloat16",
            config={
                "threads": 32,
                "v_tile": 8,
                "raw_group_size": 4,
                "raw_maxrregcount": 146,
            },
        )


@pytest.mark.smoke
def test_gated_deltanet_decode_raw_cuda_dispatch_handles_capability_errors(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _raise_capability_error():
        raise RuntimeError("mock capability failure")

    monkeypatch.setattr(torch.cuda, "get_device_capability", _raise_capability_error)

    assert not GatedDeltaNetDecodeOp._should_use_raw_cuda_decode(
        128,
        128,
        torch.bfloat16,
        tune=False,
    )


@pytest.mark.smoke
def test_gated_deltanet_decode_raw_cuda_map_only_registers_on_supported_arch(
    monkeypatch,
) -> None:
    op = object.__new__(GatedDeltaNetDecodeOp)

    monkeypatch.setattr(gated_deltanet_ops, "get_sm_version", lambda: 80)
    sm80_map = GatedDeltaNetDecodeOp.default_kernel_map.fget(op)
    assert "GatedDeltaNetDecodeRawCudaFlaStyleKernel" not in sm80_map

    monkeypatch.setattr(gated_deltanet_ops, "get_sm_version", lambda: 90)
    sm90_map = GatedDeltaNetDecodeOp.default_kernel_map.fget(op)
    assert (
        sm90_map["GatedDeltaNetDecodeRawCudaFlaStyleKernel"]
        is GatedDeltaNetDecodeRawCudaFlaStyleKernel
    )


@pytest.mark.smoke
def test_gated_deltanet_decode_raw_cuda_dispatch_rejects_unsupported_sm100(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gated_deltanet_ops, "get_sm_version", lambda: 100)

    assert not GatedDeltaNetDecodeOp._should_use_raw_cuda_decode(
        128,
        128,
        torch.bfloat16,
        tune=False,
    )


@pytest.mark.smoke
def test_gated_deltanet_decode_rejects_manifest_shape_mismatch() -> None:
    op = object.__new__(GatedDeltaNetDecodeOp)
    op.batch = 2
    op.heads = 3
    op.dim_k = 4
    op.dim_v = 5
    op.dtype = torch.float32

    q = torch.empty(2, 3, 4)
    k = torch.empty(2, 3, 4)
    v = torch.empty(2, 3, 5)
    g = torch.empty(2, 4)
    beta = torch.empty(2, 3)
    state = torch.empty(2, 3, 4, 5)

    with pytest.raises(ValueError, match="g must have shape"):
        op.forward(q, k, v, g, beta, state)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
