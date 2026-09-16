"""Which implementation each non-attention family lands on.

One row per region the family's own predicates used to draw, including the
boundaries they turned on: element type, dimensions, layout, and architecture.
Selection is asserted through ``select_kernel`` / ``select_kernel_key``, which
resolve the implementation without compiling anything.
"""

import itertools

import pytest
import torch

from tileops.kernels.gemm import GemmBasicKernel
from tileops.kernels.gemm.call_spec import GemmCall
from tileops.kernels.linear_attention.deltanet_call import DeltaNetDecodeCall
from tileops.ops.gemm.gemm import GemmFwdOp
from tileops.ops.linear_attention.deltanet_recurrence import (
    DELTANET_DECODE_KEYS,
    DeltaNetDecodeFwdOp,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="selection reads the device architecture"
)

_SM90 = 90
_SM80 = 80


# --- GEMM: a vector operand picks the GEMV kernel, but only in the two layouts
# it is written for. Non-SM90 falls back to the pipelined mainloop.


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("m", "n", "trans_a", "trans_b", "expected"),
    [
        pytest.param(1, 8, False, True, "GemvKernel", id="lhs-row"),
        pytest.param(8, 1, False, False, "GemvKernel", id="rhs-col"),
        pytest.param(1, 8, False, False, "GemmKernel", id="lhs-row-wrong-layout"),
        pytest.param(8, 1, False, True, "GemmKernel", id="rhs-col-wrong-layout"),
        pytest.param(8, 8, False, False, "GemmKernel", id="neither-is-a-vector"),
        pytest.param(1, 1, False, False, "GemvKernel", id="both-are-vectors"),
    ],
)
def test_gemm_dispatch(m: int, n: int, trans_a: bool, trans_b: bool, expected: str) -> None:
    op = GemmFwdOp(trans_a=trans_a, trans_b=trans_b)
    call = GemmCall(
        arch=_SM90, m=m, n=n, k=64, dtype=torch.float16, trans_a=trans_a, trans_b=trans_b
    )

    assert op.select_kernel(call).__name__ == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("m", "n", "trans_a", "trans_b", "dim"),
    [
        pytest.param(1, 8, True, True, "m=1", id="lhs-row-trans-a"),
        pytest.param(8, 1, True, False, "n=1", id="rhs-col-trans-a"),
    ],
)
def test_gemm_vector_on_a_transposed_operand_is_refused(
    m: int, n: int, trans_a: bool, trans_b: bool, dim: str
) -> None:
    """A ``trans_a`` layout puts the vector on an operand's TMA-loaded innermost
    dimension, where the descriptor needs a multiple of 8 fp16 elements — and the
    GEMV kernel has no form for these layouts. Selection refuses, naming the
    dimension; it used to hand these to the general kernel, whose build then died
    inside TileLang (``T.tma_copy() ... TMA is not available``).
    """
    op = GemmFwdOp(trans_a=trans_a, trans_b=trans_b)
    call = GemmCall(
        arch=_SM90, m=m, n=n, k=64, dtype=torch.float16, trans_a=trans_a, trans_b=trans_b
    )

    with pytest.raises(ValueError, match=f"multiple of 8 .*and {dim}"):
        op.select_kernel(call)


@pytest.mark.smoke
def test_gemm_uses_basic_mainloop_off_sm90() -> None:
    op = GemmFwdOp()
    call = GemmCall(arch=_SM80, m=1, n=8, k=64, dtype=torch.float16, trans_b=True)

    assert op.select_kernel(call) is GemmBasicKernel


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
def test_deltanet_decode_dispatch(
    dtype: torch.dtype, dim_k: int, dim_v: int, arch: int, expected: str
) -> None:
    op = DeltaNetDecodeFwdOp()
    call = DeltaNetDecodeCall(arch=arch, batch=1, heads=4, dim_k=dim_k, dim_v=dim_v, dtype=dtype)

    assert op.select_kernel_key(DELTANET_DECODE_KEYS, call) == expected


@pytest.mark.smoke
def test_every_family_call_record_reads_the_device_when_unstated() -> None:
    """A record built without an architecture resolves one; a stated one wins."""
    for record in (GemmCall(), DeltaNetDecodeCall()):
        assert record.arch > 0
    assert GemmCall(arch=_SM80).arch == _SM80
    assert DeltaNetDecodeCall(arch=_SM80).arch == _SM80


@pytest.mark.smoke
def test_gemv_kernel_claims_the_layouts_it_was_written_for() -> None:
    """The predicate the op used to carry, over every (m, n, layout) combination."""
    from tileops.kernels.gemm import GemvKernel

    for m, n, trans_a, trans_b in itertools.product([1, 8], [1, 8], [False, True], [False, True]):
        expected = (m == 1 and not trans_a and trans_b) or (n == 1 and not trans_a and not trans_b)
        call = GemmCall(
            arch=_SM90, m=m, n=n, k=64, dtype=torch.float16, trans_a=trans_a, trans_b=trans_b
        )
        assert GemvKernel.applies(call) is expected, (m, n, trans_a, trans_b)
