import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.linear_attention.deltanet_call import DeltaNetDecodeCall
from tileops.kernels.linear_attention.gated_deltanet_recurrence import (
    GatedDeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.ops import GatedDeltaNetDecodeFwdOp
from tileops.ops.linear_attention.gated_deltanet import GATED_DELTANET_DECODE_KEYS
from workloads.linear_attention import (
    GatedDeltaNetDecodeWorkload,
    gated_deltanet_decode_torch,
)


class GatedDeltaNetDecodeTest(GatedDeltaNetDecodeWorkload, TestBase):
    pass


# Correctness tests


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
        (
            "batch, heads, dim_k, dim_v, dtype, tune",
            [
                pytest.param(1, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(2, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 8, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1, 32, 128, 128, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
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
    op = GatedDeltaNetDecodeFwdOp(tune=tune)
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

    op = GatedDeltaNetDecodeFwdOp(tune=tune)
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
def test_gated_deltanet_decode_raw_cuda_dispatch_rejects_unsupported_sm100() -> None:
    """An architecture past the one it was written for is not assumed compatible.

    No implementation of this slot claims SM100, so the call is refused rather
    than run on any of them — and the refusal names the raw kernel as one that
    declined for its architecture, which is what this pins.
    """
    op = GatedDeltaNetDecodeFwdOp()
    call = DeltaNetDecodeCall(
        arch=100, batch=1, heads=4, dim_k=128, dim_v=128, dtype=torch.bfloat16
    )

    with pytest.raises(ValueError, match="no implementation serves this call") as excinfo:
        op.select_kernel_key(GATED_DELTANET_DECODE_KEYS, call)
    assert "GatedDeltaNetDecodeRawCudaFlaStyleKernel: built for architectures [90]" in (
        str(excinfo.value)
    )


@pytest.mark.smoke
def test_gated_deltanet_decode_rejects_manifest_shape_mismatch() -> None:
    op = object.__new__(GatedDeltaNetDecodeFwdOp)
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
