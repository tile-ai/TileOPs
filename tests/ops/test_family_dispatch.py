"""Coverage gaps: inputs inside an op's signature domain that no implementation serves.

Each case hands the op's dispatch a call record for this device and asserts the refusal
and its reason, before anything is built.
"""

import pytest
import torch

from tileops.kernels.gemm.call_spec import GemmCall
from tileops.kernels.linear_attention import (
    DeltaNetChunkCall,
    DeltaNetDecodeCall,
    DeltaNetInferenceCall,
    GDNCall,
    GLAChunkCall,
    GLADecodeCall,
)
from tileops.kernels.linear_attention.gla.call_spec import GLAInferenceCallSpec
from tileops.ops.gemm.gemm import GemmFwdOp
from tileops.ops.linear_attention.deltanet.chunk import DeltaNetChunkBwdOp, DeltaNetChunkFwdOp
from tileops.ops.linear_attention.deltanet.inference import DeltaNetInferenceFwdOp
from tileops.ops.linear_attention.deltanet.recurrent import DeltaNetRecurrentFwdOp
from tileops.ops.linear_attention.gdn import GDNFwdOp
from tileops.ops.linear_attention.gla.chunk import GLAChunkBwdOp, GLAChunkFwdOp
from tileops.ops.linear_attention.gla.inference import GLAInferenceFwdOp
from tileops.ops.linear_attention.gla.recurrent import GLARecurrentFwdOp
from workloads.device import run_device_available

pytestmark = [
    pytest.mark.sm90,
    pytest.mark.in_tree_kernels,
    pytest.mark.skipif(
        not run_device_available(), reason="selection reads the device architecture"
    ),
]

_SM90 = 90


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


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_deltanet_decode_refuses_a_key_dim_no_tile_divides() -> None:
    """The tile rule the three decode kernels share, which every served row satisfies."""
    call = DeltaNetDecodeCall(
        arch=_SM90, batch=1, heads=4, dim_k=72, dim_v=128, dtype=torch.bfloat16
    )

    with pytest.raises(ValueError, match="multiple of 16"):
        DeltaNetRecurrentFwdOp().select_implementation("deltanet_decode", call)


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


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("call", "interfaces", "reason"),
    [
        pytest.param(
            _chunk_call(128, 128, chunk_size=48),
            ("gla_fwd", "gla_bwd"),
            "chunk_size=48 must be",
            id="chunk-48",
        ),
        pytest.param(
            _chunk_call(128, 16), ("gla_fwd", "gla_bwd"), "dim_v=16", id="value-below-a-tile"
        ),
        pytest.param(
            _chunk_call(32, 64), ("gla_bwd",), "dim_k=32, dim_v=64", id="bwd-narrow-key-half-value"
        ),
    ],
)
def test_gla_chunked_refuses_what_no_kernel_serves(
    call: GLAChunkCall, interfaces: tuple, reason: str
) -> None:
    ops = {"gla_fwd": GLAChunkFwdOp, "gla_bwd": GLAChunkBwdOp}
    for interface in interfaces:
        with pytest.raises(ValueError, match=reason):
            ops[interface](chunk_size=call.chunk_size).select_implementation(interface, call)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dim_k", "dim_v", "reason"),
    [
        pytest.param(128, 256, "shared memory", id="over-the-budget"),
        pytest.param(64, 160, "dim_v a multiple of 64", id="dims-no-tile-pairs"),
    ],
)
def test_gla_bwd_refuses_what_this_device_cannot_place(dim_k: int, dim_v: int, reason: str) -> None:
    call = GLAChunkCall(
        batch=1, seq_len=128, heads=2, dim_k=dim_k, dim_v=dim_v, chunk_size=64, dtype=torch.float32
    )
    with pytest.raises(ValueError, match=reason):
        GLAChunkBwdOp(chunk_size=64).select_implementation("gla_bwd", call)


def _gla_inference_call(
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


def _gated_call(seq_len: int, has_initial_state: bool, **facts: object) -> GDNCall:
    return GDNCall(
        arch=_SM90,
        batch=1,
        seq_len=seq_len,
        heads=facts.pop("heads", 16),
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
    ("call", "reason"),
    [
        pytest.param(
            _gated_call(1, True, dim_k=256, dim_v=256),
            "K and V other than matching 64 or 128",
            id="decode-wide-state",
        ),
    ],
)
def test_gdn_refuses_what_no_kernel_serves(call: GDNCall, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        GDNFwdOp().select_implementation("gdn", call)


def _inference_call(**facts: object) -> DeltaNetInferenceCall:
    return DeltaNetInferenceCall(
        arch=_SM90,
        batch=1,
        seq_len=facts.pop("seq_len", 128),
        heads=facts.pop("heads", 16),
        dim_k=facts.pop("dim_k", 128),
        dim_v=facts.pop("dim_v", 128),
        dtype=facts.pop("dtype", torch.bfloat16),
        scale=0.088,
        **facts,
    )


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


def _head_axis_cases(heads: int) -> list[tuple[object, str, object]]:
    """One ``(op, interface, call)`` per linear-attention implementation, at *heads* heads."""
    chunk = DeltaNetChunkCall(
        arch=_SM90,
        batch=1,
        heads=heads,
        seq_len=512,
        chunk_size=64,
        dim_k=128,
        dim_v=128,
        dtype=torch.bfloat16,
    )
    gla_chunk = GLAChunkCall(
        arch=_SM90,
        batch=1,
        seq_len=512,
        heads=heads,
        dim_k=128,
        dim_v=128,
        chunk_size=64,
        scale=-1.0,
        dtype=torch.bfloat16,
    )
    decode = {"arch": _SM90, "batch": 1, "heads": heads, "dim_k": 128, "dim_v": 128}
    return [
        (DeltaNetChunkFwdOp(), "deltanet_fwd", chunk),
        (DeltaNetChunkBwdOp(), "deltanet_bwd", chunk),
        (DeltaNetInferenceFwdOp(), "deltanet_inference", _inference_call(heads=heads)),
        (DeltaNetInferenceFwdOp(), "deltanet_inference", _inference_call(seq_len=1, heads=heads)),
        (
            DeltaNetRecurrentFwdOp(),
            "deltanet_decode",
            DeltaNetDecodeCall(dtype=torch.bfloat16, **decode),
        ),
        (
            DeltaNetRecurrentFwdOp(),
            "deltanet_decode",
            DeltaNetDecodeCall(dtype=torch.float32, **decode),
        ),
        (
            GDNFwdOp(),
            "gdn",
            _gated_call(2048, False, heads=heads, value_heads=2 * heads),
        ),
        (
            GDNFwdOp(),
            "gdn",
            _gated_call(1, True, heads=heads, value_heads=2 * heads),
        ),
        (GLAChunkFwdOp(), "gla_fwd", gla_chunk),
        (GLAChunkBwdOp(), "gla_bwd", gla_chunk),
        *(
            (GLAInferenceFwdOp(), "gla_inference", call)
            for call in (
                _gla_inference_call(1, heads=heads),
                _gla_inference_call(2048, heads=heads),
                _gla_inference_call(100, heads=heads),
                _gla_inference_call(4096, varlen=True, heads=heads),
                _gla_inference_call(4096, varlen=True, dim=128, heads=heads, sequences=4),
            )
        ),
        (
            GLARecurrentFwdOp(),
            "gla_decode",
            GLADecodeCall(scale=0.088, dtype=torch.bfloat16, **decode),
        ),
        (
            GLARecurrentFwdOp(),
            "gla_decode",
            GLADecodeCall(scale=0.088, dtype=torch.float32, **decode),
        ),
    ]


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_linear_attention_refuses_an_odd_head_count() -> None:
    for op, interface, call in _head_axis_cases(3):
        with pytest.raises(ValueError, match="even head count or a single head"):
            op.select_implementation(interface, call)


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("seq_len", [2048, 1])
def test_gdn_refuses_an_odd_value_head_count(seq_len: int) -> None:
    """The value heads are their own axis: one key head can group an odd number of them."""
    call = _gated_call(seq_len, seq_len == 1, heads=1, value_heads=3)

    with pytest.raises(ValueError, match="even head count or a single head"):
        GDNFwdOp().select_implementation("gdn", call)


_GQA_DENSE_GAPS = [
    # (dtype, batch, seq_len_q, heads, heads_kv, dim, seq_len_kv, window, rope, softcap)
    (
        ("fp8", 1, 1, 32, 4, 128, 2048, (64, 0), False, 0.0),
        "fp8-window",
        r"does not serve sliding windows",
    ),
    (
        ("fp8", 1, 1, 32, 4, 128, 2048, (-1, -1), True, 0.0),
        "fp8-rope",
        r"does not serve RoPE with one query position",
    ),
    (
        ("fp16", 1, 1, 32, 4, 144, 2048, (-1, -1), False, 0.0),
        "decode-dim-144",
        r"multiple of 16 in \[16, 128\]",
    ),
    (
        ("fp16", 1, 4, 32, 4, 128, 2048, (64, 0), False, 0.0),
        "window-unequal-lengths",
        r"requires equal Q and KV lengths",
    ),
    (
        ("fp16", 1, 4, 32, 4, 72, 2048, (-1, -1), False, 0.0),
        "prefill-dim-72",
        r"requires head dimension a multiple of 16",
    ),
]


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("row", "reason"), [pytest.param(row, reason, id=name) for row, name, reason in _GQA_DENSE_GAPS]
)
def test_gqa_dense_refuses_what_no_kernel_serves(row: tuple, reason: str) -> None:
    from tileops.kernels.attention.call_spec import AttentionCall
    from tileops.ops.attention.gqa.dense import GQADenseFwdOp

    dtype_name, batch, seq_q, heads, heads_kv, dim, seq_kv, window, rope, softcap = row
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
    is_fp8 = dtype_name == "fp8"
    op = GQADenseFwdOp(
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

    with pytest.raises(ValueError, match=reason):
        op.select_implementation("gqa_dense", call)
