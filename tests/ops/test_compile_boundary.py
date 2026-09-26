"""Compile-boundary contract for the ops whose operator is generated from the manifest.

One case per op: a cold ``torch.compile(op, fullgraph=True)`` must match eager and the
traced graph must hold nothing but that op's own operator. A case builds its inputs from
the op's own workload where there is one, so the shapes are the ones the family already
validates against rather than a second set written here.

One function per family, so a family's extents and dtype stay local to its cases. The
two tests are the contract and are written once.
"""

import pytest
import torch

from tests.compile_contract import (
    assert_fake_matches_eager,
    assert_op_owns_graph_nodes,
    assert_same_result,
    register_compile_contract,
)
from tileops.ops.attention.deepseek_dsa import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from tileops.ops.attention.deepseek_mla import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from tileops.ops.attention.deepseek_nsa import NSACmpVarlenFwdOp, NSATopkVarlenFwdOp, NSAVarlenFwdOp
from tileops.ops.attention.gqa import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionDecodePagedWithKVCacheFwdOp,
    GroupedQueryAttentionDenseFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
    GroupedQueryAttentionVarlenFwdOp,
)
from tileops.ops.attention.mha import (
    MultiHeadAttentionBwdOp,
    MultiHeadAttentionDecodePagedWithKVCacheFwdOp,
)
from tileops.ops.fp8_lightning_indexer import FP8LightningIndexerFwdOp
from tileops.ops.gemm.bmm import BmmFp8FwdOp, BmmFwdOp
from tileops.ops.gemm.gemm import GemmFp8FwdOp, GemmFwdOp, GemmW4A16FwdOp
from tileops.ops.gemm.grouped_gemm import GroupedGemmFwdOp
from tileops.ops.linear_attention.deltanet import DeltaNetBwdOp, DeltaNetFwdOp
from tileops.ops.linear_attention.deltanet_recurrence import DeltaNetDecodeFwdOp
from tileops.ops.linear_attention.gla import GLABwdOp, GLAFwdOp
from tileops.ops.linear_attention.gla_recurrence import GLADecodeFwdOp
from tileops.ops.mamba.cb_producer import CBProducerFwdOp
from tileops.ops.mamba.da_cumsum import DaCumsumFwdOp
from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops.ops.mamba.ssd_decode import SSDDecodeFwdOp
from tileops.ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from tileops.ops.rope import (
    RopeLlama31FwdOp,
    RopeLongRopeFwdOp,
    RopeNeoxFwdOp,
    RopeNeoxPositionIdsFwdOp,
    RopeNonNeoxFwdOp,
    RopeYarnFwdOp,
)
from tileops.ops.sequence_modeling.engram import EngramGateConvBwdOp, EngramGateConvFwdOp
from tileops.ops.sequence_modeling.engram_decode import EngramDecodeFwdOp
from tileops.ops.sequence_modeling.mhc import MHCPostFwdOp, MHCPreFwdOp
from tileops.ops.topk_selector import TopkSelectorFwdOp
from workloads.attention.deepseek import (
    DsaDecodeWorkload,
    MlaDecodeWorkload,
    NsaCmpFwdWorkload,
    NsaFwdWorkload,
    NsaTopkWorkload,
)
from workloads.attention.gqa import (
    GQAPrefillPagedWithKVCacheFwdWorkload,
    GQAPrefillVarlenFwdWorkload,
    GroupedQueryAttentionBwdWorkload,
    GroupedQueryAttentionDecodePagedWorkload,
    GroupedQueryAttentionDenseDecodeWorkload,
    GroupedQueryAttentionSlidingWindowVarlenFwdWorkload,
)
from workloads.attention.mha import MhaDecodePagedWorkload
from workloads.attention.paged import make_unit_cache_scales
from workloads.fp8_lightning_indexer import FP8LightningIndexerWorkload


def _attention_cases():
    """The attention ops, each with the inputs it is built for."""
    _DTYPE = torch.float16
    _HEADS, _HEADS_KV, _DIM = 8, 2, 128

    def gqa_dense():
        case = GroupedQueryAttentionDenseDecodeWorkload(2, _HEADS, _HEADS_KV, 256, _DIM, _DTYPE)
        return GroupedQueryAttentionDenseFwdOp(), case.gen_inputs()

    def gqa_bwd():
        case = GroupedQueryAttentionBwdWorkload(1, _HEADS, _HEADS_KV, 256, _DIM, True, _DTYPE)
        op = GroupedQueryAttentionBwdOp(is_causal=True)
        return op, case.gen_inputs()

    def gqa_varlen():
        lens = [128, 128]
        case = GQAPrefillVarlenFwdWorkload(2, _HEADS, _HEADS_KV, lens, lens, _DIM, True, _DTYPE)
        op = GroupedQueryAttentionVarlenFwdOp()
        return op, case.gen_inputs()

    def gqa_sliding_window_varlen():
        lens = [128, 128]
        case = GroupedQueryAttentionSlidingWindowVarlenFwdWorkload(
            2, lens, lens, _HEADS, _HEADS_KV, _DIM, True, 64, -1, _DTYPE
        )
        op = GroupedQueryAttentionVarlenFwdOp(is_causal=True, window_size_left=64)
        return op, case.gen_inputs()

    def gqa_prefill_varlen_compat():
        lens = [128, 128]
        case = GQAPrefillVarlenFwdWorkload(2, _HEADS, _HEADS_KV, lens, lens, _DIM, True, _DTYPE)
        op = GroupedQueryAttentionPrefillVarlenFwdOp(128, 128)
        return op, case.gen_inputs()

    def gqa_sliding_window_varlen_compat():
        lens = [128, 128]
        case = GroupedQueryAttentionSlidingWindowVarlenFwdWorkload(
            2, lens, lens, _HEADS, _HEADS_KV, _DIM, True, 64, -1, _DTYPE
        )
        op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(128, window_size_left=64)
        return op, case.gen_inputs()

    def gqa_prefill_paged():
        case = GQAPrefillPagedWithKVCacheFwdWorkload(
            2, _HEADS, _HEADS_KV, [64, 64], [128, 128], 64, _DIM, True, _DTYPE
        )
        op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
            page_size=64,
            max_seqlen_q=case.max_seqlen_q,
            is_causal=True,
        )
        q, k_new, v_new, k_pages, v_pages, cu_q, cache_seqlens, table = case.gen_inputs()
        k_scale, v_scale = make_unit_cache_scales()
        return op, (q, k_new, v_new, k_pages, v_pages, k_scale, v_scale, cu_q, cache_seqlens, table)

    def gqa_decode_paged():
        case = GroupedQueryAttentionDecodePagedWorkload(2, _HEADS, _HEADS_KV, 256, _DIM, 64, _DTYPE)
        op = GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(page_size=64)
        return op, case.gen_inputs()

    def mha_bwd():
        case = GroupedQueryAttentionBwdWorkload(1, _HEADS, _HEADS, 256, _DIM, True, _DTYPE)
        op = MultiHeadAttentionBwdOp(is_causal=True)
        return op, case.gen_inputs()

    def mha_decode_paged():
        case = MhaDecodePagedWorkload(1, _HEADS, 1, 256, _DIM, 128, False, _DTYPE)
        op = MultiHeadAttentionDecodePagedWithKVCacheFwdOp(page_size=128, is_causal=False)
        return op, case.gen_inputs()

    def mla_decode():
        case = MlaDecodeWorkload(1, 64, 1, 256, _DIM, 64, _DTYPE)
        op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp()
        return op, case.gen_inputs()

    def nsa_fwd():
        case = NsaFwdWorkload(1, 16, 512, 64, True, 0.1, 32, 16, 1, _DTYPE)
        op = NSAVarlenFwdOp(is_causal=True, scale=0.1, block_size=32)
        return op, case.gen_inputs()

    def nsa_cmp_fwd():
        case = NsaCmpFwdWorkload(1, 512, 32, _DIM, _DIM, 16, 0.088, 32, _DTYPE)
        op = NSACmpVarlenFwdOp(scale=0.088, bs=32)
        return op, case.gen_inputs()

    def nsa_topk():
        case = NsaTopkWorkload(1, 512, 32, _DIM, 16, 1.0, 16, 32, _DTYPE)
        op = NSATopkVarlenFwdOp(scale=1.0, selected_block_num=16, bs=32)
        return op, case.gen_inputs()

    def dsa_decode():
        batch, heads, seq_len, seq_len_kv, dim, tail, topk = 1, 64, 64, 128, 512, 64, 128
        case = DsaDecodeWorkload(batch, heads, seq_len, seq_len_kv, dim, tail, topk, 1, 1, seq_len)
        op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(tail, 1, q_start_index_s=seq_len)
        return op, case.gen_inputs()

    def fp8_lightning_indexer():
        case = FP8LightningIndexerWorkload(1, 2048, 8, 64, 4096, 1)
        return FP8LightningIndexerFwdOp(), case.gen_inputs()

    def topk_selector():
        score = torch.randn(1, 256, 512, 1, dtype=torch.float32, device="cuda")
        starts = torch.zeros(1, 256, dtype=torch.int32, device="cuda")
        ends = torch.full((1, 256), 512, dtype=torch.int32, device="cuda")
        return TopkSelectorFwdOp(topk=64), (score, starts, ends)

    return (
        ("gqa-dense", gqa_dense),
        ("gqa-bwd", gqa_bwd),
        ("gqa-varlen", gqa_varlen),
        ("gqa-sliding-window-varlen", gqa_sliding_window_varlen),
        ("gqa-prefill-varlen-compat", gqa_prefill_varlen_compat),
        ("gqa-sliding-window-varlen-compat", gqa_sliding_window_varlen_compat),
        ("gqa-prefill-paged", gqa_prefill_paged),
        ("gqa-decode-paged", gqa_decode_paged),
        ("mha-bwd", mha_bwd),
        ("mha-decode-paged", mha_decode_paged),
        ("mla-decode", mla_decode),
        ("nsa-fwd", nsa_fwd),
        ("nsa-cmp-fwd", nsa_cmp_fwd),
        ("nsa-topk", nsa_topk),
        ("dsa-decode", dsa_decode),
        ("fp8-lightning-indexer", fp8_lightning_indexer),
        ("topk-selector", topk_selector),
    )


def _gemm_cases():
    """The gemm ops, each with the inputs it is built for."""
    _DTYPE = torch.float16
    _M, _N, _K = 128, 256, 256

    def _x(*shape, dtype=_DTYPE):
        return torch.randn(*shape, dtype=dtype, device="cuda")

    def gemm():
        return GemmFwdOp(), (_x(_M, _K), _x(_N, _K))

    def gemm_fp8():
        fp8 = dict(dtype=torch.float8_e4m3fn)
        f32 = dict(dtype=torch.float32)
        return GemmFp8FwdOp(), (
            _x(_M, _K).to(**fp8),
            _x(_N, _K).to(**fp8),
            _x(1, 1, **f32).abs(),
            _x(1, 1, **f32).abs(),
            None,
        )

    def gemm_w4a16():
        group = 128
        return GemmW4A16FwdOp(), (
            _x(_M, _K),
            torch.randint(0, 255, (_N, _K // 2), dtype=torch.uint8, device="cuda"),
            _x(_N, _K // group).abs(),  # the scale follows the activation dtype
            torch.randint(0, 15, (_N, _K // group), dtype=torch.uint8, device="cuda"),
        )

    def grouped_gemm():
        groups, rows = 2, 128
        sizes = torch.full((groups,), rows, dtype=torch.int32, device="cuda")
        offsets = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
        return GroupedGemmFwdOp(), (
            _x(groups * rows, _K),
            _x(groups, _N, _K),
            sizes,
            offsets,
            offsets,
        )

    def bmm():
        batch = 2
        return BmmFwdOp(), (_x(batch, _M, _K), _x(batch, _K, _N))

    def bmm_fp8():
        batch = 2
        fp8 = dict(dtype=torch.float8_e4m3fn)
        scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        return BmmFp8FwdOp(out_dtype=torch.float16), (
            _x(batch, _M, _K).to(**fp8),
            _x(batch, _K, _N).to(**fp8),
            scale,
            scale.clone(),
        )

    return (
        ("gemm", gemm),
        ("gemm-fp8", gemm_fp8),
        ("gemm-w4a16", gemm_w4a16),
        ("grouped-gemm", grouped_gemm),
        ("bmm", bmm),
        ("bmm-fp8", bmm_fp8),
    )


def _mamba_cases():
    """The mamba ops, each with the inputs it is built for."""
    _DTYPE = torch.float16
    _B, _H, _P, _N, _G = 1, 8, 64, 128, 1
    _Q, _NC = 256, 2
    _S = _Q * _NC

    def _x(*shape, dtype=_DTYPE):
        return torch.randn(*shape, dtype=dtype, device="cuda")

    f32 = dict(dtype=torch.float32)

    def da_cumsum():
        op = DaCumsumFwdOp(chunk_len=_Q, out_dtype=_DTYPE, dt_softplus=True)
        return op, (_x(_B, _S, _H, **f32), -_x(_H, **f32).abs(), None)

    def cb_producer():
        op = CBProducerFwdOp(_Q)
        return op, (_x(_B, _S, _G, _N), _x(_B, _S, _G, _N))

    def ssd_chunk_state():
        return SSDChunkStateFwdOp(), (
            _x(_B, _S, _H, _P),
            _x(_B, _S, _G, _N),
            _x(_B, _H, _NC, _Q),
            _x(_B, _H, _NC, _Q, **f32),
            None,
        )

    def ssd_state_passing():
        return SSDStatePassingFwdOp(), (
            _x(_B, _NC, _H, _P * _N),
            _x(_B, _H, _NC, **f32),
            None,
        )

    def ssd_chunk_scan():
        return SSDChunkScanFwdOp(), (
            _x(_B, _S, _H, _P),
            _x(_B, _NC, _G, _Q, _Q),
            _x(_B, _H, _NC, _Q, **f32),
            _x(_B, _S, _G, _N),
            _x(_B, _NC, _H, _P, _N, **f32),
            _x(_B, _H, _NC, _Q),
        )

    def ssd_decode():
        return SSDDecodeFwdOp(), (
            -_x(_H, _P, _N, **f32).abs(),
            _x(_B, _H, _P, **f32),
            _x(_B, _H, _P),
            _x(_B, _G, _N),
            _x(_B, _G, _N),
            _x(_B, _H, _P, _N, **f32),
        )

    return (
        ("da-cumsum", da_cumsum),
        ("cb-producer", cb_producer),
        ("ssd-chunk-state", ssd_chunk_state),
        ("ssd-state-passing", ssd_state_passing),
        ("ssd-chunk-scan", ssd_chunk_scan),
        ("ssd-decode", ssd_decode),
    )


def _linear_attention_cases():
    """The linear attention ops, each with the inputs it is built for."""
    _DTYPE = torch.bfloat16
    _B, _H, _S, _D = 1, 4, 256, 64
    _CHUNK = 64
    _SCALE = _D**-0.5

    def _x(*shape, dtype=_DTYPE):
        return torch.randn(*shape, dtype=dtype, device="cuda")

    chunks = _S // _CHUNK + 1

    def gla_fwd():
        op = GLAFwdOp(chunk_size=_CHUNK, scale=_SCALE)
        # ``g`` is a log-space decay, so it must be non-positive.
        return op, (
            _x(_B, _S, _H, _D),
            _x(_B, _S, _H, _D),
            _x(_B, _S, _H, _D),
            -_x(_B, _S, _H, _D).abs(),
            None,
        )

    def gla_bwd():
        op = GLABwdOp(chunk_size=_CHUNK, scale=_SCALE)
        return op, (
            _x(_B, _S, _H, _D),
            _x(_B, _S, _H, _D),
            _x(_B, _S, _H, _D),
            -_x(_B, _S, _H, _D).abs(),
            _x(_B, chunks, _H, _D, _D, dtype=torch.float32),
            _x(_B, _S, _H, _D),
            _x(_B, _H, _D, _D, dtype=torch.float32),
        )

    def gla_decode():
        op = GLADecodeFwdOp(scale=_SCALE)
        return op, (
            _x(_B, _H, _D),
            _x(_B, _H, _D),
            _x(_B, _H, _D),
            -_x(_B, _H, _D).abs(),
            _x(_B, _H, _D, _D),
        )

    def deltanet_fwd():
        # The delta rule is a recurrence over S steps; unit-variance operands overflow it
        # into NaN, which compares unequal to itself. Scale as ``DeltaNetFwdWorkload`` does.
        op = DeltaNetFwdOp(chunk_size=_CHUNK)
        return op, (
            _x(_B, _H, _S, _D) * 0.1,
            _x(_B, _H, _S, _D) * 0.1,
            _x(_B, _H, _S, _D) * 0.1,
            _x(_B, _H, _S).sigmoid() * 0.5,
        )

    def deltanet_bwd():
        op = DeltaNetBwdOp(chunk_size=_CHUNK)
        return op, (
            _x(_B, _H, _S, _D),
            _x(_B, _H, _S, _D),
            _x(_B, _H, _S, _D),
            _x(_B, _H, _S, _D),
            _x(_B, _H, _S).sigmoid(),
            _x(_B, _H, chunks, _D, _D, dtype=torch.float32),
            _x(_B, _H, _S, _D),
            _x(_B, _H, _S, _D),
            _x(_B, _H, _S, _D),
            _x(_B, _H, _S, _D),
        )

    def deltanet_decode():
        return DeltaNetDecodeFwdOp(), (
            _x(_B, _H, _D),
            _x(_B, _H, _D),
            _x(_B, _H, _D),
            _x(_B, _H).sigmoid(),
            _x(_B, _H, _D, _D),
        )

    return (
        ("gla-fwd", gla_fwd),
        ("gla-bwd", gla_bwd),
        ("gla-decode", gla_decode),
        ("deltanet-fwd", deltanet_fwd),
        ("deltanet-bwd", deltanet_bwd),
        ("deltanet-decode", deltanet_decode),
    )


def _sequence_modeling_cases():
    """The sequence modeling ops, each with the inputs it is built for."""
    _DTYPE = torch.bfloat16
    _M = 1
    _SEQ_LEN = 32
    _D = 256
    _CONV_TAPS = 4

    def _x(*shape, dtype=_DTYPE):
        return torch.randn(*shape, dtype=dtype, device="cuda")

    def engram_gate_conv_fwd():
        op = EngramGateConvFwdOp(_M, _SEQ_LEN, _D)
        return op, (
            _x(_M, _SEQ_LEN, _D),
            _x(_M, _SEQ_LEN, _D),
            _x(_M, _SEQ_LEN, _D),
            _x(_D),
            _x(_D),
            _x(_CONV_TAPS, _D),
        )

    def engram_gate_conv_bwd():
        f32 = dict(dtype=torch.float32)
        op = EngramGateConvBwdOp(_M, _SEQ_LEN, _D)
        return op, (
            _x(_M, _SEQ_LEN, _D),
            _x(_M, _SEQ_LEN, _D),
            _x(_M, _SEQ_LEN, _D),
            _x(_M, _SEQ_LEN, _D),
            _x(_D),
            _x(_D),
            _x(_CONV_TAPS, _D),
            _x(_M, _SEQ_LEN, _D),
            _x(_M, _SEQ_LEN, **f32),
            _x(_M, _SEQ_LEN, **f32).abs(),
            _x(_M, _SEQ_LEN, **f32).abs(),
            _x(_M, _SEQ_LEN, **f32).abs(),
        )

    def engram_decode():
        batch, d_mem, max_conv_len, dilation = 2, 128, 8, 1
        op = EngramDecodeFwdOp(batch, d_mem, _D, max_conv_len, _CONV_TAPS, dilation)
        return op, (
            _x(batch, d_mem),
            _x(batch, _D),
            _x(batch, max_conv_len, _D),
            _x(d_mem, _D),
            _x(d_mem, _D),
            _x(_D),
            _x(_D),
            _x(_CONV_TAPS, _D),
        )

    def mhc_pre():
        n_expand, c_x, batch = 4, 1280, 1
        op = MHCPreFwdOp(0.5, 0.25, 0.125, sinkhorn_repeat=4)
        phi_dim = n_expand * n_expand + 2 * n_expand
        return op, (
            _x(n_expand * c_x, phi_dim, dtype=torch.float32),
            _x(batch, n_expand * c_x),
            _x(phi_dim, dtype=torch.float32),
        )

    def mhc_post():
        n_expand, c_x, batch = 4, 1280, 1
        return MHCPostFwdOp(), (
            _x(batch, c_x),
            _x(batch, n_expand, dtype=torch.float32),
            _x(batch, n_expand * c_x),
        )

    return (
        ("engram-gate-conv-fwd", engram_gate_conv_fwd),
        ("engram-gate-conv-bwd", engram_gate_conv_bwd),
        ("engram-decode", engram_decode),
        ("mhc-pre", mhc_pre),
        ("mhc-post", mhc_post),
    )


def _rope_cases():
    """The RoPE ops, one layout each, with the inputs they are built for."""
    _DTYPE = torch.float16
    _SEQ_LEN, _HEADS, _D = 64, 4, 64

    def _x(*shape):
        return torch.randn(*shape, dtype=_DTYPE, device="cuda")

    def one_d(op_cls):
        return lambda: (op_cls(layout="1d"), (_x(_SEQ_LEN, _D),))

    def two_d(op_cls):
        return lambda: (op_cls(layout="2d"), (_x(2, _SEQ_LEN, _HEADS, _D),))

    def longrope():
        rescale = torch.linspace(1.0, 2.0, _D // 2, device="cuda")
        return RopeLongRopeFwdOp(rescale_factors=rescale), (_x(_SEQ_LEN, _D),)

    def position_ids():
        op = RopeNeoxPositionIdsFwdOp(max_position=128)
        positions = torch.arange(_SEQ_LEN, device="cuda", dtype=torch.int32)
        return op, (_x(_SEQ_LEN, _HEADS, _D), positions)

    return (
        ("rope-neox", one_d(RopeNeoxFwdOp)),
        ("rope-non-neox", two_d(RopeNonNeoxFwdOp)),
        ("rope-llama31", one_d(RopeLlama31FwdOp)),
        ("rope-yarn", two_d(RopeYarnFwdOp)),
        ("rope-longrope", longrope),
        ("rope-neox-position-ids", position_ids),
    )


_FAMILIES = (
    _attention_cases,
    _gemm_cases,
    _mamba_cases,
    _linear_attention_cases,
    _sequence_modeling_cases,
    _rope_cases,
)


def _cases():
    """Every family's cases, as pytest params.

    The builders run inside the test: this module is imported on the CPU-only runner
    that enforces the compile-contract gate, where a CUDA tensor built at import time
    would fail before any test is selected.
    """
    return [pytest.param(builder, id=name) for family in _FAMILIES for name, builder in family()]


for _op_cls in (
    GroupedQueryAttentionDenseFwdOp,
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
    GroupedQueryAttentionSlidingWindowVarlenFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionDecodePagedWithKVCacheFwdOp,
    MultiHeadAttentionBwdOp,
    MultiHeadAttentionDecodePagedWithKVCacheFwdOp,
    MultiHeadLatentAttentionDecodeWithKVCacheFwdOp,
    NSAVarlenFwdOp,
    NSACmpVarlenFwdOp,
    NSATopkVarlenFwdOp,
    DeepSeekSparseAttentionDecodeWithKVCacheFwdOp,
    FP8LightningIndexerFwdOp,
    TopkSelectorFwdOp,
    GemmFwdOp,
    GemmFp8FwdOp,
    GemmW4A16FwdOp,
    GroupedGemmFwdOp,
    BmmFwdOp,
    BmmFp8FwdOp,
    DaCumsumFwdOp,
    CBProducerFwdOp,
    SSDChunkStateFwdOp,
    SSDStatePassingFwdOp,
    SSDChunkScanFwdOp,
    SSDDecodeFwdOp,
    GLAFwdOp,
    GLABwdOp,
    GLADecodeFwdOp,
    DeltaNetFwdOp,
    DeltaNetBwdOp,
    DeltaNetDecodeFwdOp,
    EngramGateConvFwdOp,
    EngramGateConvBwdOp,
    EngramDecodeFwdOp,
    MHCPreFwdOp,
    MHCPostFwdOp,
    RopeNeoxFwdOp,
    RopeNonNeoxFwdOp,
    RopeLlama31FwdOp,
    RopeYarnFwdOp,
    RopeLongRopeFwdOp,
    RopeNeoxPositionIdsFwdOp,
):
    register_compile_contract(_op_cls)


# Two kernels return a tensor a compiled call cannot be held equal to, so for these the
# contract is the graph the op traces to plus the shapes and dtypes it promises. The top-k
# selector picks the same set every time but lets the atomic increments that claim the
# slots decide which index lands where. Sparse MLA reads storage it was not handed, which
# a NaN-filled caching allocator turns its whole output into.
_NONDETERMINISTIC = frozenset({"topk-selector", "dsa-decode"})


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("build_case", _cases())
def test_a_cold_op_traces_fullgraph_and_matches_eager(build_case, request) -> None:
    """Cold is the whole contract: a warm op has nothing left for dynamo to trace into."""
    op, inputs = build_case()
    # Its own copies per call: a paged op writes its cache pages. ``detach`` because a
    # backward op's operator carries no autograd formula to answer a history-tracking input.
    compiled_inputs = tuple(None if t is None else t.detach().clone() for t in inputs)
    eager_inputs = tuple(None if t is None else t.detach().clone() for t in inputs)

    compiled = torch.compile(op, fullgraph=True)(*compiled_inputs)

    assert_same_result(
        compiled, op(*eager_inputs), exact=request.node.callspec.id not in _NONDETERMINISTIC
    )


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("build_case", _cases())
def test_the_fake_reports_what_the_op_returns(build_case) -> None:
    """The fake is the op's whole promise to the compiler, so it must be the truth."""
    op, inputs = build_case()
    inputs = tuple(None if t is None else t.detach() for t in inputs)

    assert_fake_matches_eager(op, *inputs)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("build_case", _cases())
def test_the_traced_graph_holds_only_this_ops_operator(build_case) -> None:
    """The node is the op's, so replacing the kernel cannot change the graph."""
    op, inputs = build_case()
    inputs = tuple(None if t is None else t.detach() for t in inputs)

    assert_op_owns_graph_nodes(op, *inputs)
