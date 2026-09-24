"""Which implementation each non-attention family lands on.

One row per region the family's own predicates used to draw, including the
boundaries they turned on: element type, dimensions, layout, and architecture.
Selection is asserted through ``select_kernel`` / ``select_kernel_key``, which
resolve the implementation without compiling anything.
"""

import itertools

import pytest
import torch

from tileops.kernels.gemm import GemmCpAsyncKernel, GemmTmaKernel
from tileops.kernels.gemm.call_spec import GemmCall
from tileops.kernels.linear_attention.deltanet_call import DeltaNetDecodeCall
from tileops.ops.gemm.gemm import GemmFwdOp
from tileops.ops.linear_attention.deltanet_recurrence import DeltaNetDecodeFwdOp

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
        pytest.param(1, 8, False, False, "GemmTmaKernel", id="lhs-row-wrong-layout"),
        pytest.param(8, 1, False, True, "GemmTmaKernel", id="rhs-col-wrong-layout"),
        pytest.param(8, 8, False, False, "GemmTmaKernel", id="neither-is-a-vector"),
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
def test_gemm_vector_on_a_transposed_operand_takes_the_pipelined_mainloop(
    m: int, n: int, trans_a: bool, trans_b: bool, dim: str
) -> None:
    """A ``trans_a`` layout puts the vector on an operand's TMA-loaded innermost
    dimension, where the descriptor needs a multiple of 8 fp16 elements, and the
    GEMV kernel has no form for these layouts. ``GemmCpAsyncKernel`` takes them: it
    loads through ``cp.async``, so the dimension the TMA descriptor cannot address
    costs it nothing. ``GemmTmaKernel`` still refuses, naming that dimension.
    """
    op = GemmFwdOp(trans_a=trans_a, trans_b=trans_b)
    call = GemmCall(
        arch=_SM90, m=m, n=n, k=64, dtype=torch.float16, trans_a=trans_a, trans_b=trans_b
    )

    assert op.select_kernel(call) is GemmCpAsyncKernel
    assert f"and {dim}" in GemmTmaKernel.refusal(call)


@pytest.mark.smoke
def test_gemm_misaligned_k_on_sm90_takes_the_pipelined_mainloop() -> None:
    """A TMA-misaligned NT shape on SM90 reaches ``GemmCpAsyncKernel``.

    ``GemmTmaKernel`` refuses it because every structure it builds loads through TMA.
    ``GemmCpAsyncKernel`` excludes only the SM90 shapes TMA can address, so it takes
    this one — with ``GemmTmaKernel``'s whole architecture excluded instead, the call
    reached no implementation at all.
    """
    op = GemmFwdOp()
    call = GemmCall(
        arch=_SM90, sm_count=132, m=1024, n=4096, k=100, dtype=torch.float16, trans_b=True
    )

    assert op.select_kernel(call) is GemmCpAsyncKernel


@pytest.mark.smoke
def test_gemm_k_too_narrow_to_vectorize_is_refused_during_selection() -> None:
    """``k = 1`` fp16 spans 2 bytes, under the 4-byte load both mainloops issue.

    Neither implementation can serve it. The refusal states the reason during
    selection rather than letting a builder be entered and raise.
    """
    op = GemmFwdOp()
    call = GemmCall(arch=_SM90, sm_count=132, m=64, n=64, k=1, dtype=torch.float16, trans_b=True)

    with pytest.raises(ValueError, match="k must span at least one"):
        op.select_kernel(call)

    with pytest.raises(ValueError, match="cannot serve k=1"):
        GemmCpAsyncKernel(64, 64, 1, torch.float16, trans_b=True)


@pytest.mark.smoke
def test_gemm_uses_basic_mainloop_off_sm90() -> None:
    op = GemmFwdOp()
    call = GemmCall(arch=_SM80, m=1, n=8, k=64, dtype=torch.float16, trans_b=True)

    assert op.select_kernel(call) is GemmCpAsyncKernel


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

    assert op.select_kernel(call).__name__ == expected


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
        assert (GemvKernel.band_for(call) is not None) is expected


@pytest.mark.smoke
def test_gemv_kernel_takes_its_two_row_band_only_where_the_grid_underfills() -> None:
    """The ``lhs_rows`` band: m == 2 NT, and only while a 64-wide n-tiling underfills."""
    from tileops.kernels.gemm import GemvKernel

    def call(m: int, n: int, trans_b: bool = True) -> GemmCall:
        return GemmCall(
            arch=_SM90,
            sm_count=132,
            m=m,
            n=n,
            k=7168,
            dtype=torch.float16,
            trans_b=trans_b,
        )

    # The band is ceil(n / 64) * 8 < 132 * 3 = 396: n = 3136 gives 49 tiles (392), n = 3200 gives 50 (400).
    assert GemvKernel.band_for(call(2, 2112)) == "lhs_rows"
    assert GemvKernel.band_for(call(2, 3136)) == "lhs_rows"
    assert GemvKernel.band_for(call(2, 3200)) is None
    assert GemvKernel.band_for(call(3, 2112)) is None
    assert GemvKernel.band_for(call(2, 2112, trans_b=False)) is None


# --- Dense GQA: one row per region, plus each boundary between two of them.

_GQA_DENSE_ROWS = [
    # (dtype, batch, seq_len_q, heads, heads_kv, dim, seq_len_kv, window, rope, softcap)
    (
        ("fp8", 1, 1, 32, 4, 128, 2048, (-1, -1), False, 0.0),
        "GQADenseFP8DecodeKernel",
        "fp8-decode",
    ),
    (("fp8", 2, 1, 32, 4, 128, 2048, (-1, -1), False, 0.0), "GQADenseFP8Kernel", "fp8-batch-2"),
    (("fp8", 1, 1, 32, 4, 128, 512, (-1, -1), False, 0.0), "GQADenseFP8Kernel", "fp8-short-cache"),
    (("fp8", 1, 1, 32, 1, 128, 2048, (-1, -1), False, 0.0), "GQADenseFP8Kernel", "fp8-wide-group"),
    (("fp8", 1, 1, 32, 4, 128, 2048, (64, 0), False, 0.0), "GQADenseFP8Kernel", "fp8-window"),
    (("fp8", 1, 1, 32, 4, 128, 2048, (-1, -1), True, 0.0), "GQADenseFP8Kernel", "fp8-rope"),
    (
        ("fp16", 1, 1, 32, 4, 128, 2048, (-1, -1), False, 0.0),
        "GQADecodeLongContextKernel",
        "long-context",
    ),
    (("fp16", 1, 1, 32, 4, 128, 512, (-1, -1), False, 0.0), "GQADecodeBs1Kernel", "bs1-short"),
    (("fp16", 1, 1, 8, 4, 128, 2048, (-1, -1), False, 0.0), "GQADecodeBs1Kernel", "bs1-heads"),
    (("fp16", 1, 1, 32, 4, 128, 2048, (-1, -1), True, 0.0), "GQADecodeBs1Kernel", "bs1-rope"),
    (("bf16", 1, 1, 32, 4, 128, 2048, (-1, -1), False, 0.0), "GQADecodeKernel", "decode-bf16"),
    (
        ("fp16", 1, 1, 32, 4, 128, 2048, (-1, -1), False, 30.0),
        "GQADecodeKernel",
        "decode-softcap",
    ),
    (("fp16", 2, 1, 32, 4, 128, 2048, (-1, -1), False, 0.0), "GQADecodeKernel", "decode-batch-2"),
    (("fp16", 1, 1, 32, 4, 64, 2048, (-1, -1), False, 0.0), "GQADecodeKernel", "decode-dim-64"),
    (
        ("fp16", 1, 4, 32, 4, 128, 2048, (64, 0), False, 0.0),
        "GQADenseSlidingWindowKernel",
        "window",
    ),
    (
        ("fp16", 1, 1, 32, 4, 128, 2048, (64, 0), False, 0.0),
        "GQADenseSlidingWindowKernel",
        "window-beats-decode",
    ),
    (("fp16", 1, 4, 32, 4, 128, 2048, (-1, -1), False, 0.0), "GQADenseWsKernel", "prefill"),
    (
        ("bf16", 2, 8, 8, 8, 64, 512, (-1, -1), True, 30.0),
        "GQADenseWsKernel",
        "prefill-rope-softcap",
    ),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("row", "expected"),
    [pytest.param(row, expected, id=name) for row, expected, name in _GQA_DENSE_ROWS],
)
def test_gqa_dense_dispatch(row: tuple, expected: str) -> None:
    """Each region, and the boundary that separates it from the next."""
    from tileops.kernels.attention.call_spec import AttentionCall, fp8_dtype
    from tileops.ops.attention.gqa import GroupedQueryAttentionDenseFwdOp

    dtype_name, batch, seq_q, heads, heads_kv, dim, seq_kv, window, rope, softcap = row
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": fp8_dtype()}
    is_fp8 = dtype_name == "fp8"
    op = GroupedQueryAttentionDenseFwdOp(
        window_size_left=window[0],
        window_size_right=window[1],
        softcap=softcap,
        pos_encoding_mode="rope" if rope else "none",
        out_dtype=torch.float16 if is_fp8 else None,
    )
    call = AttentionCall(
        arch=_SM90,
        dtype=torch.float16 if is_fp8 else dtypes[dtype_name],
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        max_seqlen_q=seq_q,
        seqlen_kv=seq_kv,
        softcap=softcap,
        window_size_left=window[0],
        window_size_right=window[1],
        is_fp8=is_fp8,
        fuse_rope=rope,
    )

    assert op.select_kernel(call).__name__ == expected
