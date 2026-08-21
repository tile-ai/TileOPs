import pytest
import torch

import tileops.ops.deltanet_recurrence as deltanet_ops
from tests.test_base import FixtureBase, TestBase
from tileops.kernels.deltanet_call import DeltaNetDecodeCall
from tileops.kernels.deltanet_recurrence import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.ops import DeltaNetDecodeFwdOp
from tileops.ops.deltanet_recurrence import DELTANET_DECODE_KEYS
from workloads.linear_attention import DeltaNetDecodeWorkload, deltanet_decode_torch


class DeltaNetDecodeTest(DeltaNetDecodeWorkload, TestBase):
    pass


# Correctness tests


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 5e-4, "rtol": 5e-4}
    elif dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


class DeltaNetDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, dim_k, dim_v, dtype, tune",
            [
                pytest.param(1, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(2, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 8, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@DeltaNetDecodeFixture
def test_deltanet_decode(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = DeltaNetDecodeTest(batch, heads, dim_k, dim_v, dtype)
    op = DeltaNetDecodeFwdOp(tune=tune)
    tols = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), **tols)


@DeltaNetDecodeFixture
def test_deltanet_decode_multi_step(
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

    op = DeltaNetDecodeFwdOp(tune=tune)
    tols = _get_tolerances(dtype)

    state_op = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)
    state_ref = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)

    for _ in range(num_steps):
        q = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=dtype) * 0.1
        beta = torch.rand(B, H, device="cuda", dtype=dtype) * 0.5

        o_ref, state_ref = deltanet_decode_torch(q, k, v, beta, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, beta, state_op)

        torch.testing.assert_close(o_op, o_ref, **tols)
        torch.testing.assert_close(state_op, state_ref, **tols)


@pytest.mark.smoke
def test_deltanet_decode_rejects_manifest_shape_mismatch() -> None:
    op = object.__new__(DeltaNetDecodeFwdOp)
    op.batch = 2
    op.heads = 3
    op.dim_k = 4
    op.dim_v = 5
    op.dtype = torch.float32

    q = torch.empty(2, 3, 5)
    k = torch.empty(2, 3, 4)
    v = torch.empty(2, 3, 5)
    beta = torch.empty(2, 3)
    state = torch.empty(2, 3, 4, 5)

    with pytest.raises(ValueError, match="k must have shape"):
        op.forward(q, k, v, beta, state)


def _skip_unless_raw_cuda_decode_supported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for raw DeltaNet decode smoke coverage")
    try:
        sm_version = deltanet_ops.get_sm_version()
    except Exception as exc:
        pytest.skip(f"could not query CUDA architecture: {exc}")
    if sm_version not in DeltaNetDecodeRawCudaFlaStyleKernel.supported_archs:
        pytest.skip(f"raw DeltaNet decode requires SM90, got SM{sm_version}")


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deltanet_decode_raw_cuda_real_128x128_smoke(dtype: torch.dtype) -> None:
    """PR smoke must compile and execute the real raw CUDA 128x128 fast path."""
    _skip_unless_raw_cuda_decode_supported()

    torch.manual_seed(42)
    test = DeltaNetDecodeTest(2, 4, 128, 128, dtype)
    op = DeltaNetDecodeFwdOp(tune=False)
    inputs = test.gen_inputs()
    op(*inputs)
    assert isinstance(op.kernel, DeltaNetDecodeRawCudaFlaStyleKernel)
    test.check(op, *inputs, **_get_tolerances(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deltanet_decode_raw_cuda_real_128x128_multi_step_smoke(
    dtype: torch.dtype,
) -> None:
    """PR smoke must exercise raw CUDA state propagation across decode steps."""
    _skip_unless_raw_cuda_decode_supported()

    torch.manual_seed(42)
    num_steps = 8
    B, H, DK, DV = 2, 4, 128, 128
    op = DeltaNetDecodeFwdOp(tune=False)
    tols = _get_tolerances(dtype)

    state_op = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)
    state_ref = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)

    for _ in range(num_steps):
        q = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=dtype) * 0.1
        beta = torch.rand(B, H, device="cuda", dtype=dtype) * 0.5

        o_ref, state_ref = deltanet_decode_torch(q, k, v, beta, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, beta, state_op)

        assert isinstance(op.kernel, DeltaNetDecodeRawCudaFlaStyleKernel)

        torch.testing.assert_close(o_op, o_ref, **tols)
        torch.testing.assert_close(state_op, state_ref, **tols)


class _DispatchMarker:
    """Records its construction instead of compiling anything.

    Mixed in ahead of the class each marker stands in for, so the region that
    class states — and the architecture it declares — still decide selection.
    Overriding only construction is the point: a marker that answered
    ``applies`` differently would be testing itself.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _DefaultDispatchKernel(_DispatchMarker, DeltaNetDecodeKernel):
    pass


class _FP32DispatchKernel(_DispatchMarker, DeltaNetDecodeFP32Kernel):
    pass


class _RawDispatchKernel(_DispatchMarker, DeltaNetDecodeRawCudaFlaStyleKernel):
    pass


def _dispatch_kernel_map() -> dict:
    return {
        "DeltaNetDecodeKernel": _DefaultDispatchKernel,
        "DeltaNetDecodeFP32Kernel": _FP32DispatchKernel,
        "DeltaNetDecodeRawCudaFlaStyleKernel": _RawDispatchKernel,
    }


def _stated_call(
    sm_version: int, dtype: torch.dtype, dim_k: int = 128, dim_v: int = 128, tune: bool = False
) -> DeltaNetDecodeCall:
    """The record for a decode call, with the device stated rather than probed."""
    return DeltaNetDecodeCall(
        arch=sm_version, batch=1, heads=32, dim_k=dim_k, dim_v=dim_v, dtype=dtype, tune=tune
    )


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_deltanet_decode_raw_cuda_dispatch_selects_raw_on_supported_sm90(
    dtype: torch.dtype,
) -> None:
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map())

    key = op.select_kernel_key(DELTANET_DECODE_KEYS, _stated_call(90, dtype))

    assert op.kernel_map[key] is _RawDispatchKernel


@pytest.mark.parametrize(
    "tune",
    [
        pytest.param(False, marks=pytest.mark.smoke, id="untuned"),
        pytest.param(True, marks=pytest.mark.full, id="tuned"),
    ],
)
def test_deltanet_decode_build_carries_the_tune_flag(tune: bool) -> None:
    """Whatever selection picks is constructed with the op's autotune setting."""
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map(), tune=tune)

    kernel = op._get_kernel(1, 32, 128, 128, torch.bfloat16, device_index=None)

    assert kernel.kwargs["tune"] is tune


@pytest.mark.smoke
def test_deltanet_decode_raw_cuda_dispatch_falls_back_on_unsupported_sm() -> None:
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map())

    key = op.select_kernel_key(DELTANET_DECODE_KEYS, _stated_call(80, torch.bfloat16))

    assert op.kernel_map[key] is _DefaultDispatchKernel


@pytest.mark.smoke
@pytest.mark.parametrize(("dim_k", "dim_v"), [(64, 128), (128, 64)])
def test_deltanet_decode_raw_cuda_dispatch_falls_back_on_non_128_shapes(
    dim_k: int,
    dim_v: int,
) -> None:
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map())

    key = op.select_kernel_key(DELTANET_DECODE_KEYS, _stated_call(90, torch.bfloat16, dim_k, dim_v))

    assert op.kernel_map[key] is _DefaultDispatchKernel


@pytest.mark.smoke
def test_deltanet_decode_raw_cuda_dispatch_uses_fp32_kernel_for_fp32() -> None:
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map())

    key = op.select_kernel_key(DELTANET_DECODE_KEYS, _stated_call(90, torch.float32))

    assert op.kernel_map[key] is _FP32DispatchKernel


@pytest.mark.smoke
def test_deltanet_decode_raw_cuda_config_requires_full_warp_mapping() -> None:
    with pytest.raises(ValueError, match="threads .* must equal raw_group_size \\* v_tile"):
        DeltaNetDecodeRawCudaFlaStyleKernel(
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
def test_deltanet_decode_raw_cuda_config_requires_two_lane_group() -> None:
    with pytest.raises(ValueError, match="raw_group_size must equal 2"):
        DeltaNetDecodeRawCudaFlaStyleKernel(
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
