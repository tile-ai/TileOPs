"""Which implementation each non-attention family lands on.

One row per region the family's own predicates used to draw, including the
boundaries they turned on: element type, dimensions, layout, and architecture.
Selection is asserted through ``select_implementation``, which
resolve the implementation without compiling anything.
"""

import pytest
import torch

from tileops.kernels.gemm import GemmCpAsyncKernel, GemmTmaKernel
from tileops.kernels.gemm.call_spec import BmmFp8Call, GemmCall
from tileops.kernels.linear_attention import (
    DeltaNetDecodeCall,
    DeltaNetInferenceCall,
    GatedDeltaNetCall,
    GLAChunkCall,
    GLADecodeCall,
)
from tileops.kernels.linear_attention.gla.call_spec import GLAInferenceCallSpec
from tileops.ops.gemm.bmm import BmmFP8FwdOp
from tileops.ops.gemm.gemm import GemmFwdOp
from tileops.ops.linear_attention.deltanet.inference import DeltaNetInferenceFwdOp
from tileops.ops.linear_attention.deltanet.recurrent import DeltaNetRecurrentFwdOp
from tileops.ops.linear_attention.gated_deltanet import GatedDeltaNetFwdOp
from tileops.ops.linear_attention.gla.chunk import GLAChunkBwdOp, GLAChunkFwdOp
from tileops.ops.linear_attention.gla.inference import GLAInferenceFwdOp
from tileops.ops.linear_attention.gla.recurrent import GLARecurrentFwdOp
from workloads.device import run_device_available

pytestmark = pytest.mark.skipif(
    not run_device_available(), reason="selection reads the device architecture"
)

_SM90 = 90
_SM80 = 80


def _serves(op, call: GemmCall) -> type:
    """The implementation of *op*'s dense GEMM interface that serves *call*."""
    return op.kernel_map[op.select_implementation("gemm", call)]


# --- GEMM: a vector operand picks the GEMV kernel, but only in the two layouts
# it is written for. Non-SM90 falls back to the pipelined mainloop.


@pytest.mark.cuda_only
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

    assert op.kernel_map[op.select_implementation("gemm", call)].__name__ == expected


@pytest.mark.cuda_only
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

    assert _serves(op, call) is GemmCpAsyncKernel
    assert f"and {dim}" in GemmTmaKernel.refusal(call)


@pytest.mark.smoke
def test_gemm_misaligned_k_on_sm90_takes_the_pipelined_mainloop() -> None:
    """A TMA-misaligned NT shape on SM90 reaches ``GemmCpAsyncKernel``.

    ``GemmTmaKernel`` refuses it because every structure it builds loads through TMA.
    ``GemmCpAsyncKernel`` is the general implementation and takes what no other one
    claims, on SM90 as anywhere else.
    """
    op = GemmFwdOp()
    call = GemmCall(
        arch=_SM90, sm_count=132, m=1024, n=4096, k=100, dtype=torch.float16, trans_b=True
    )

    assert _serves(op, call) is GemmCpAsyncKernel


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_k_too_narrow_to_vectorize_is_refused_during_selection() -> None:
    """``k = 1`` fp16 spans 2 bytes, under the 4-byte load both mainloops issue.

    Neither implementation can serve it. The refusal states the reason during
    selection rather than letting a builder be entered and raise.
    """
    op = GemmFwdOp()
    call = GemmCall(arch=_SM90, sm_count=132, m=64, n=64, k=1, dtype=torch.float16, trans_b=True)

    with pytest.raises(ValueError, match="k must span at least one"):
        op.select_implementation("gemm", call)

    with pytest.raises(ValueError, match="cannot serve k=1"):
        GemmCpAsyncKernel(64, 64, 1, torch.float16, trans_b=True)


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_uses_basic_mainloop_off_sm90() -> None:
    op = GemmFwdOp()
    call = GemmCall(arch=_SM80, m=1, n=8, k=64, dtype=torch.float16, trans_b=True)

    assert _serves(op, call) is GemmCpAsyncKernel


# --- DeltaNet decode: fp32 has its own kernel; the raw-CUDA one serves 16-bit
# at dim 128 on SM90; everything else is the general kernel.

_DELTANET_ROWS = [
    (torch.float32, 128, 128, _SM90, "deltanet_decode_fp32", "fp32"),
    (torch.float32, 64, 64, _SM80, "deltanet_decode_fp32", "fp32-any-dim-any-arch"),
    (torch.float16, 128, 128, _SM90, "deltanet_decode_raw_cuda", "fp16-raw"),
    (torch.bfloat16, 128, 128, _SM90, "deltanet_decode_raw_cuda", "bf16-raw"),
    (torch.float16, 64, 128, _SM90, "deltanet_decode", "dim-k-off"),
    (torch.float16, 128, 64, _SM90, "deltanet_decode", "dim-v-off"),
    (torch.float16, 128, 128, _SM80, "deltanet_decode", "arch-off"),
]


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dtype", "dim_k", "dim_v", "arch", "expected"),
    [pytest.param(*row[:5], id=row[5]) for row in _DELTANET_ROWS],
)
def test_deltanet_decode_dispatch(
    dtype: torch.dtype, dim_k: int, dim_v: int, arch: int, expected: str
) -> None:
    op = DeltaNetRecurrentFwdOp()
    call = DeltaNetDecodeCall(arch=arch, batch=1, heads=4, dim_k=dim_k, dim_v=dim_v, dtype=dtype)

    assert op.select_implementation("deltanet_decode", call) == expected


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_deltanet_decode_refuses_a_key_dim_no_tile_divides() -> None:
    """The tile rule the three decode kernels share, which every served row satisfies."""
    call = DeltaNetDecodeCall(
        arch=_SM90, batch=1, heads=4, dim_k=72, dim_v=128, dtype=torch.bfloat16
    )

    with pytest.raises(ValueError, match="multiple of 16"):
        DeltaNetRecurrentFwdOp().select_implementation("deltanet_decode", call)


# --- Chunked GLA: the extents the three-pass forward and the two-pass backward tile, the
# backward also depending on the warp-group instruction SM90 offers 16-bit operands.


def _chunk_call(dim_k: int, dim_v: int, arch: int = _SM90, chunk_size: int = 64) -> GLAChunkCall:
    return GLAChunkCall(
        arch=arch,
        batch=2,
        seq_len=512,
        heads=8,
        dim_k=dim_k,
        dim_v=dim_v,
        chunk_size=chunk_size,
        scale=-1.0,
        dtype=torch.bfloat16,
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("call", "serves_fwd", "serves_bwd"),
    [
        pytest.param(_chunk_call(128, 128), True, True, id="square-128"),
        pytest.param(_chunk_call(32, 128), True, True, id="narrow-key"),
        pytest.param(_chunk_call(32, 64), True, False, id="narrow-key-half-value"),
        pytest.param(_chunk_call(192, 64), True, True, id="key-past-128-sm90"),
        pytest.param(_chunk_call(192, 64, arch=_SM80), True, False, id="key-past-128-sm80"),
        pytest.param(_chunk_call(128, 128, chunk_size=48), False, False, id="chunk-48"),
        pytest.param(_chunk_call(128, 16), False, False, id="value-below-a-tile"),
    ],
)
def test_gla_chunked_dispatch(call: GLAChunkCall, serves_fwd: bool, serves_bwd: bool) -> None:
    for op, interface, serves in (
        (GLAChunkFwdOp(chunk_size=call.chunk_size), "gla_fwd", serves_fwd),
        (GLAChunkBwdOp(chunk_size=call.chunk_size), "gla_bwd", serves_bwd),
    ):
        if serves:
            assert op.select_implementation(interface, call) == interface
        else:
            with pytest.raises(ValueError, match="no implementation serves this call"):
                op.select_implementation(interface, call)


# --- GLA decode: fp32 has its own kernel, every other element type the general one.


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        pytest.param(torch.float32, "gla_decode_fp32", id="fp32"),
        pytest.param(torch.float16, "gla_decode", id="fp16"),
        pytest.param(torch.bfloat16, "gla_decode", id="bf16"),
    ],
)
def test_gla_decode_dispatch(dtype: torch.dtype, expected: str) -> None:
    op = GLARecurrentFwdOp()
    call = GLADecodeCall(arch=_SM90, batch=1, heads=4, dim_k=128, dim_v=128, dtype=dtype)

    assert op.select_implementation("gla_decode", call) == expected


# --- GLA inference: one token is decode, whole 64-token rows are chunk-parallel prefill,
# and a packed call or a row that is not a whole chunk is the packed kernel, whose state
# walk is partitioned where a per-sequence walk serves the call badly. The dense partitioned
# kernel's region reads a device calibration, which a spec built without a device does not
# carry, so no row here selects it.


def _inference_call(
    seq_len: int, varlen: bool = False, dim: int = 64, heads: int = 4, sequences: int = 1
) -> GLAInferenceCallSpec:
    # Multiprocessors of the board the GLA inference regions are read against.
    sm_count = 132
    return GLAInferenceCallSpec(
        arch=_SM90,
        sm_count=sm_count,
        batch=1,
        seq_len=seq_len,
        heads=heads,
        dim_k=dim,
        dim_v=dim,
        dtype=torch.bfloat16,
        scale=dim**-0.5,
        varlen=varlen,
        num_sequences=sequences,
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("call", "expected"),
    [
        pytest.param(_inference_call(1), "gla_dense_decode", id="decode"),
        pytest.param(_inference_call(2048), "gla_dense_prefill_subchunk", id="whole-chunks"),
        pytest.param(_inference_call(100), "gla_varlen_prefill", id="part-chunk-row"),
        pytest.param(_inference_call(4096, varlen=True), "gla_varlen_prefill", id="packed"),
        pytest.param(
            _inference_call(4096, varlen=True, dim=128, heads=16, sequences=4),
            "gla_varlen_prefill_partitioned",
            id="packed-wide",
        ),
        pytest.param(
            _inference_call(3000),
            "gla_varlen_prefill_partitioned",
            id="part-chunk-long-row",
        ),
    ],
)
def test_gla_inference_dispatch(call: GLAInferenceCallSpec, expected: str) -> None:
    assert GLAInferenceFwdOp().select_implementation("gla_inference", call) == expected


# --- Gated DeltaNet: one token continuing a state is decode, whole chunks of 64 from
# zero are prefill, and nothing else is served.


def _gated_call(seq_len: int, has_initial_state: bool, **facts: object) -> GatedDeltaNetCall:
    return GatedDeltaNetCall(
        arch=_SM90,
        batch=1,
        seq_len=seq_len,
        heads=16,
        value_heads=facts.pop("value_heads", 16),
        dim_k=facts.pop("dim_k", 128),
        dim_v=facts.pop("dim_v", 128),
        dtype=torch.bfloat16,
        scale=0.088,
        has_initial_state=has_initial_state,
        **facts,
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("call", "expected"),
    [
        pytest.param(_gated_call(1, True), "gated_deltanet_dense_decode", id="decode"),
        pytest.param(
            _gated_call(1, False, dim_k=64, dim_v=64, value_heads=64, state_v_first=True),
            "gated_deltanet_dense_decode",
            id="decode-every-variant",
        ),
        pytest.param(_gated_call(64, False), "gated_deltanet_dense_prefill", id="prefill-64"),
        pytest.param(_gated_call(128, False), "gated_deltanet_dense_prefill", id="prefill-128"),
        pytest.param(
            _gated_call(64, True, dim_k=64, dim_v=64),
            "gated_deltanet_dense_prefill",
            id="prefill-narrow-state",
        ),
        pytest.param(
            _gated_call(4096, False, varlen=True, num_sequences=4),
            "gated_deltanet_dense_prefill",
            id="prefill-varlen",
        ),
        pytest.param(_gated_call(63, False), "gated_deltanet_dense_prefill", id="prefill-ragged"),
        pytest.param(
            _gated_call(64, False, value_heads=64),
            "gated_deltanet_dense_prefill",
            id="prefill-grouped-value-heads",
        ),
        pytest.param(
            _gated_call(64, False, l2norm=True, gate_in_kernel=True, beta_sigmoid=True),
            "gated_deltanet_dense_prefill",
            id="prefill-input-transforms",
        ),
        pytest.param(
            _gated_call(64, True, state_v_first=True),
            "gated_deltanet_dense_prefill",
            id="prefill-value-major-state",
        ),
    ],
)
def test_gated_deltanet_dispatch(call: GatedDeltaNetCall, expected: str) -> None:
    assert GatedDeltaNetFwdOp().select_implementation("gated_deltanet", call) == expected


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("call", "reason"),
    [
        pytest.param(
            _gated_call(1, True, dim_k=256, dim_v=256),
            "K and V other than matching 64 or 128",
            id="decode-wide-state",
        ),
    ],
)
def test_gated_deltanet_refuses_what_no_kernel_serves(call: GatedDeltaNetCall, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        GatedDeltaNetFwdOp().select_implementation("gated_deltanet", call)


# --- DeltaNet inference: whole chunks of 64 are prefill and one token is decode, and
# each kernel states what it does not serve.


def _inference_call(**facts: object) -> DeltaNetInferenceCall:
    return DeltaNetInferenceCall(
        arch=_SM90,
        batch=1,
        seq_len=facts.pop("seq_len", 128),
        heads=16,
        dim_k=facts.pop("dim_k", 128),
        dim_v=facts.pop("dim_v", 128),
        dtype=facts.pop("dtype", torch.bfloat16),
        scale=0.088,
        **facts,
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_deltanet_inference_dispatch() -> None:
    op = DeltaNetInferenceFwdOp()

    assert op.select_implementation("deltanet_inference", _inference_call()) == (
        "deltanet_dense_prefill"
    )
    assert op.select_implementation("deltanet_inference", _inference_call(seq_len=1)) == (
        "deltanet_dense_decode"
    )
    packed = _inference_call(seq_len=4096, varlen=True, num_sequences=4)
    assert op.select_implementation("deltanet_inference", packed) == "deltanet_dense_prefill"
    assert op.select_implementation("deltanet_inference", _inference_call(seq_len=63)) == (
        "deltanet_dense_prefill"
    )
    assert op.select_implementation("deltanet_inference", _inference_call(l2norm=True)) == (
        "deltanet_dense_prefill"
    )
    # Decode claims the state width and the in-kernel normalization it used to refuse.
    narrow = _inference_call(seq_len=1, dim_k=64, dim_v=64)
    assert op.select_implementation("deltanet_inference", narrow) == "deltanet_dense_decode"
    normalized = _inference_call(seq_len=1, l2norm=True)
    assert op.select_implementation("deltanet_inference", normalized) == "deltanet_dense_decode"


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("call", "reason"),
    [
        pytest.param(_inference_call(dim_v=64), "K/V dimensions", id="dim-k-not-dim-v"),
        pytest.param(_inference_call(dtype=torch.float32), "dtype other than", id="fp32"),
    ],
)
def test_deltanet_inference_refuses_what_the_kernel_does_not_serve(
    call: DeltaNetInferenceCall, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        DeltaNetInferenceFwdOp().select_implementation("deltanet_inference", call)


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_every_family_call_record_reads_the_device_when_unstated() -> None:
    """A record built without an architecture resolves one; a stated one wins."""
    for record in (GemmCall(), DeltaNetDecodeCall()):
        assert record.arch > 0
    assert GemmCall(arch=_SM80).arch == _SM80
    assert DeltaNetDecodeCall(arch=_SM80).arch == _SM80


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


# --- Batched FP8 GEMM: three programs, claimed by how much of a persistent wave the
# call's whole tiles fill.


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("batch", "m", "n", "k", "expected"),
    [
        pytest.param(8, 2048, 2048, 2048, "BmmFp8WsKernel", id="fills-a-warp-specialized-wave"),
        pytest.param(32, 128, 128, 2048, "BmmFp8PersistentKernel", id="whole-128-tiles-only"),
        pytest.param(1, 64, 64, 64, "BmmFp8Kernel", id="tails-on-both-axes"),
    ],
)
def test_bmm_fp8_dispatch(batch: int, m: int, n: int, k: int, expected: str) -> None:
    """The warp-specialized program wins wherever it applies; the classic one takes tails."""
    call = BmmFp8Call(
        arch=_SM90,
        sm_count=132,
        batch=batch,
        m=m,
        n=n,
        k=k,
        dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
    )

    op = BmmFP8FwdOp()
    assert op.kernel_map[op.select_implementation("bmm_fp8", call)].__name__ == expected


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
    (("fp8", 1, 1, 32, 4, 128, 2048, (64, 0), False, 0.0), None, "fp8-window"),
    (("fp8", 1, 1, 32, 4, 128, 2048, (-1, -1), True, 0.0), None, "fp8-rope"),
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
    (("fp16", 1, 1, 32, 4, 144, 2048, (-1, -1), False, 0.0), None, "decode-dim-144"),
    (
        ("fp16", 1, 4, 32, 4, 128, 4, (64, 0), False, 0.0),
        "GQADenseSlidingWindowKernel",
        "window",
    ),
    (("fp16", 1, 4, 32, 4, 128, 2048, (64, 0), False, 0.0), None, "window-unequal-lengths"),
    (
        ("fp16", 1, 1, 32, 4, 128, 1, (64, 0), False, 0.0),
        "GQADenseSlidingWindowKernel",
        "window-beats-decode",
    ),
    (("fp16", 1, 4, 32, 4, 128, 2048, (-1, -1), False, 0.0), "GQADenseWsKernel", "prefill"),
    (("fp16", 1, 4, 32, 4, 72, 2048, (-1, -1), False, 0.0), None, "prefill-dim-72"),
    (
        ("bf16", 2, 8, 8, 8, 64, 512, (-1, -1), True, 30.0),
        "GQADenseWsKernel",
        "prefill-rope-softcap",
    ),
]


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("row", "expected"),
    [pytest.param(row, expected, id=name) for row, expected, name in _GQA_DENSE_ROWS],
)
def test_gqa_dense_dispatch(row: tuple, expected: "str | None") -> None:
    """Each region, and the boundary that separates it from the next."""
    from tileops.kernels.attention.call_spec import AttentionCall
    from tileops.ops.attention.gqa.dense import GroupedQueryAttentionDenseFwdOp

    dtype_name, batch, seq_q, heads, heads_kv, dim, seq_kv, window, rope, softcap = row
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
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
        is_causal=True,
        softcap=softcap,
        window_size_left=window[0],
        window_size_right=window[1],
        is_fp8=is_fp8,
        fuse_rope=rope,
    )

    if expected is None:
        with pytest.raises(ValueError, match="no implementation serves"):
            op.select_implementation("gqa_dense", call)
    else:
        assert op.kernel_map[op.select_implementation("gqa_dense", call)].__name__ == expected
