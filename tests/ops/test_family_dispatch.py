"""Which implementation each non-attention family lands on.

One row per region the family's own predicates used to draw, including the
boundaries they turned on: element type, dimensions, layout, and architecture.
Selection is asserted through ``select_kernel_key``, which resolves the key
without compiling anything.
"""

import itertools

import pytest
import torch

from tileops.kernels.deltanet_call import DeltaNetDecodeCall
from tileops.kernels.gemm_call import GemmCall
from tileops.ops.deltanet_recurrence import DELTANET_DECODE_KEYS, DeltaNetDecodeOp
from tileops.ops.gated_deltanet import GATED_DELTANET_DECODE_KEYS, GatedDeltaNetDecodeOp
from tileops.ops.gemm import GemmOp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="selection reads the device architecture")

_SM90 = 90
_SM80 = 80


# --- GEMM: a vector operand picks the GEMV kernel, but only in the two layouts
# it is written for. Off SM90 neither implementation can run.


@pytest.mark.smoke
@pytest.mark.parametrize(("m", "n", "trans_a", "trans_b", "expected"), [
    pytest.param(1, 8, False, True, "gemv_kernel", id="lhs-row"),
    pytest.param(8, 1, False, False, "gemv_kernel", id="rhs-col"),
    pytest.param(1, 8, False, False, "gemm_kernel", id="lhs-row-wrong-layout"),
    pytest.param(1, 8, True, True, "gemm_kernel", id="lhs-row-trans-a"),
    pytest.param(8, 1, False, True, "gemm_kernel", id="rhs-col-wrong-layout"),
    pytest.param(8, 1, True, False, "gemm_kernel", id="rhs-col-trans-a"),
    pytest.param(8, 8, False, False, "gemm_kernel", id="neither-is-a-vector"),
    pytest.param(1, 1, False, False, "gemv_kernel", id="both-are-vectors"),
])
def test_gemm_dispatch(m: int, n: int, trans_a: bool, trans_b: bool,
                       expected: str) -> None:
    op = GemmOp(trans_a=trans_a, trans_b=trans_b)
    call = GemmCall(arch=_SM90, m=m, n=n, k=64, dtype=torch.float16,
                    trans_a=trans_a, trans_b=trans_b)

    assert op.select_kernel_key(("gemv_kernel", "gemm_kernel"), call) == expected


@pytest.mark.smoke
def test_gemm_is_refused_where_neither_implementation_runs() -> None:
    """Both are SM90-only, so an older architecture has nothing to fall back to."""
    op = GemmOp()
    call = GemmCall(arch=_SM80, m=1, n=8, k=64, dtype=torch.float16, trans_b=True)

    with pytest.raises(ValueError, match="no implementation serves this call"):
        op.select_kernel_key(("gemv_kernel", "gemm_kernel"), call)


# --- DeltaNet decode: fp32 has its own kernel; the raw-CUDA one serves 16-bit
# at dim 128 on SM90; everything else is the general kernel.

_DELTANET_ROWS = [
    (torch.float32, 128, 128, _SM90, "DeltaNetDecodeFP32Kernel", "fp32"),
    (torch.float32, 64, 64, _SM80, "DeltaNetDecodeFP32Kernel", "fp32-any-dim-any-arch"),
    (torch.float16, 128, 128, _SM90, "DeltaNetDecodeRawCudaFlaStyleKernel", "fp16-raw"),
    (torch.bfloat16, 128, 128, _SM90, "DeltaNetDecodeRawCudaFlaStyleKernel", "bf16-raw"),
    (torch.float16, 64, 128, _SM90, "DeltaNetDecodeKernel", "dim-k-off"),
    (torch.float16, 128, 64, _SM90, "DeltaNetDecodeKernel", "dim-v-off"),
    (torch.float16, 128, 128, _SM80, "DeltaNetDecodeKernel", "arch-off"),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dtype", "dim_k", "dim_v", "arch", "expected"),
    [pytest.param(*row[:5], id=row[5]) for row in _DELTANET_ROWS],
)
def test_deltanet_decode_dispatch(dtype: torch.dtype, dim_k: int, dim_v: int,
                                  arch: int, expected: str) -> None:
    op = DeltaNetDecodeOp()
    call = DeltaNetDecodeCall(arch=arch, batch=1, heads=4, dim_k=dim_k, dim_v=dim_v,
                              dtype=dtype)

    assert op.select_kernel_key(DELTANET_DECODE_KEYS, call) == expected


# --- Gated DeltaNet decode: the same shape, and the raw-CUDA kernel additionally
# declines when the caller asked to autotune, having no knobs to tune.

_GATED_ROWS = [
    (torch.float32, 128, 128, _SM90, "GatedDeltaNetDecodeFP32Kernel", "fp32"),
    (torch.bfloat16, 128, 128, _SM90,
     "GatedDeltaNetDecodeRawCudaFlaStyleKernel", "bf16-raw"),
    (torch.float16, 128, 128, _SM90, "GatedDeltaNetDecodeKernel", "fp16-not-raw"),
    (torch.bfloat16, 64, 128, _SM90, "GatedDeltaNetDecodeKernel", "dim-k-off"),
    (torch.bfloat16, 128, 64, _SM90, "GatedDeltaNetDecodeKernel", "dim-v-off"),
    (torch.bfloat16, 128, 128, _SM80, "GatedDeltaNetDecodeKernel", "arch-off"),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dtype", "dim_k", "dim_v", "arch", "expected"),
    [pytest.param(*row[:5], id=row[5]) for row in _GATED_ROWS],
)
def test_gated_deltanet_decode_dispatch(dtype: torch.dtype, dim_k: int, dim_v: int,
                                        arch: int, expected: str) -> None:
    op = GatedDeltaNetDecodeOp()
    call = DeltaNetDecodeCall(arch=arch, batch=1, heads=4, dim_k=dim_k, dim_v=dim_v,
                              dtype=dtype)

    assert op.select_kernel_key(GATED_DELTANET_DECODE_KEYS, call) == expected


@pytest.mark.smoke
def test_gated_deltanet_raw_cuda_declines_when_asked_to_autotune() -> None:
    """It has no tunable knobs, so a caller asking to tune wants the other one.

    Nothing is autotuned here: only the key is resolved.
    """
    op = GatedDeltaNetDecodeOp(tune=True)
    call = DeltaNetDecodeCall(arch=_SM90, batch=1, heads=4, dim_k=128, dim_v=128,
                              dtype=torch.bfloat16, tune=True)

    assert op.select_kernel_key(GATED_DELTANET_DECODE_KEYS, call) == (
        "GatedDeltaNetDecodeKernel")


@pytest.mark.smoke
def test_every_family_call_record_reads_the_device_when_unstated() -> None:
    """A record built without an architecture resolves one; a stated one wins."""
    for record in (GemmCall(), DeltaNetDecodeCall()):
        assert record.arch > 0
    assert GemmCall(arch=_SM80).arch == _SM80
    assert DeltaNetDecodeCall(arch=_SM80).arch == _SM80


@pytest.mark.smoke
def test_gemv_region_matches_the_layouts_it_was_written_for() -> None:
    """The predicate the op used to carry, over every (m, n, layout) combination."""
    from tileops.kernels.gemm_call import gemv_region

    for m, n, trans_a, trans_b in itertools.product([1, 8], [1, 8], [False, True],
                                                    [False, True]):
        expected = (m == 1 and not trans_a and trans_b) or (
            n == 1 and not trans_a and not trans_b)
        call = GemmCall(arch=_SM90, m=m, n=n, k=64, dtype=torch.float16,
                        trans_a=trans_a, trans_b=trans_b)
        assert gemv_region(call) is expected, (m, n, trans_a, trans_b)
