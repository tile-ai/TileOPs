"""Compile-boundary contract for the ops whose operator is generated from the manifest.

One case per op: a cold ``torch.compile(op, fullgraph=True)`` must match eager and the
traced graph must hold nothing but that op's own operator. A case builds its inputs from
the op's own workload where there is one, so the shapes are the ones the family already
validates against rather than a second set written here.

One function per family, so a family's extents and dtype stay local to its cases. The
two tests are the contract and are written once.
"""

from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch

from tests.compile_contract import (
    assert_fake_matches_eager,
    assert_op_owns_graph_nodes,
    assert_same_result,
    register_compile_contract,
)
from tileops._ops.attention.dsa import DSADenseFwdOp, DSAPagedFwdOp
from tileops._ops.attention.fp8_lightning_indexer import FP8LightningIndexerFwdOp
from tileops._ops.attention.gqa.bwd import GQADenseBwdOp
from tileops._ops.attention.gqa.dense import GQADenseFwdOp, GQADenseWithLSEFwdOp
from tileops._ops.attention.gqa.paged import GQAPagedFwdOp
from tileops._ops.attention.gqa.varlen import GQAVarlenFwdOp
from tileops._ops.attention.mha import (
    MHAPagedFwdOp,
)
from tileops._ops.attention.mla import (
    MLADenseBwdOp,
    MLADenseDecodeFwdOp,
    MLAVarlenBwdOp,
    MLAVarlenFwdOp,
)
from tileops._ops.attention.nsa import (
    NSACompressedVarlenFwdOp,
    NSATopKVarlenFwdOp,
    NSAVarlenFwdOp,
)
from tileops._ops.attention.topk_select import TopKSelectFwdOp
from tileops._ops.fft import FFTC2CFwdOp
from tileops._ops.gemm.bmm import BMMFP8FwdOp, BMMFwdOp
from tileops._ops.gemm.gemm import (
    GEMMFP8FwdOp,
    GEMMFwdOp,
    GEMMINT8W8A8FwdOp,
    GEMMW4A8FwdOp,
    GEMMW4A16FwdOp,
)
from tileops._ops.gemm.grouped_gemm import GroupedGEMMFwdOp
from tileops._ops.linear_attention.deltanet.chunk import DeltaNetChunkBwdOp, DeltaNetChunkFwdOp
from tileops._ops.linear_attention.deltanet.fwd import DeltaNetFwdOp
from tileops._ops.linear_attention.deltanet.recurrent import DeltaNetRecurrentFwdOp
from tileops._ops.linear_attention.gdn import GDNFwdOp
from tileops._ops.linear_attention.gla.chunk import GLAChunkBwdOp, GLAChunkFwdOp
from tileops._ops.linear_attention.gla.fwd import GLAFwdOp
from tileops._ops.linear_attention.gla.recurrent import GLARecurrentFwdOp
from tileops._ops.linear_attention.kda import KDAFwdOp
from tileops._ops.mamba.causal_conv1d import CausalConv1dFwdOp, CausalConv1dUpdateFwdOp
from tileops._ops.mamba.selective_scan import SelectiveScanFwdOp
from tileops._ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
from tileops._ops.mamba.ssd_chunk_cumsum import SSDChunkCumsumFwdOp
from tileops._ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops._ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops._ops.mamba.ssd_recurrent import SSDRecurrentFwdOp
from tileops._ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from tileops._ops.pool.mean_pooling import MeanPoolingFwdOp
from tileops._ops.quantization import (
    FP8QuantPerBlockFwdOp,
    INT4QuantPerGroupFwdOp,
    INT8DequantPerBlockFwdOp,
    INT8DequantPerChannelFwdOp,
    INT8DequantPerTensorFwdOp,
    INT8QuantPerChannelFwdOp,
    INT8QuantPerTensorFwdOp,
    INT8QuantPerTokenGroupFwdOp,
    SmoothQuantFwdOp,
)
from tileops._ops.quantization.fp8_quant import FP8QuantPerTokenGroupFwdOp
from tileops._ops.rope import (
    LongRoPEFwdOp,
    RoPEFwdOp,
    RoPELlama31FwdOp,
    RoPENeoxPositionIdsFwdOp,
    YaRNFwdOp,
)
from tileops._ops.sampling.chain_speculative_sampling import ChainSpeculativeSamplingFwdOp
from tileops._ops.sampling.min_p_mask import MinPMaskFwdOp
from tileops._ops.sampling.sampling_from_probs import SamplingFromProbsFwdOp
from tileops._ops.sampling.top_k_mask import TopKMaskFwdOp
from tileops._ops.sampling.top_k_top_p_mask import TopKTopPMaskFwdOp
from tileops._ops.sampling.top_p_mask import TopPMaskFwdOp
from tileops._ops.sequence_modeling.engram import EngramGateConvBwdOp, EngramGateConvFwdOp
from tileops._ops.sequence_modeling.engram_decode import EngramDecodeFwdOp
from tileops._ops.sequence_modeling.mhc import MHCPostFwdOp, MHCPreFwdOp
from tileops._ops.transform import HadamardTransformFwdOp
from workloads.attention.dsa import DSADecodeWorkload, DSAPagedCall
from workloads.attention.fp8_lightning_indexer import FP8LightningIndexerWorkload
from workloads.attention.gqa.bwd import GQABwdWorkload
from workloads.attention.gqa.dense import GQADenseDecodeWorkload, GQADenseWithLSECall
from workloads.attention.gqa.paged import GQAPagedFwdWorkload
from workloads.attention.gqa.varlen import (
    GQAPrefillVarlenFwdWorkload,
    GQASlidingWindowVarlenFwdWorkload,
)
from workloads.attention.mha import MHADecodePagedWorkload
from workloads.attention.mla import (
    MLADecodeWorkload,
    MLADenseBwdWorkload,
    MLAVarlenBwdWorkload,
    MLAVarlenWorkload,
)
from workloads.attention.nsa import NSACompressedFwdWorkload, NSAFwdWorkload, NSATopKWorkload
from workloads.device import run_device
from workloads.quantization.int8_dequant import (
    INT8DequantPerBlockWorkload,
    INT8DequantPerChannelWorkload,
    INT8DequantPerTensorWorkload,
)
from workloads.quantization.quantize import (
    FP8QuantPerBlockWorkload,
    INT4QuantPerGroupWorkload,
    INT8QuantPerBlockWorkload,
    INT8QuantPerChannelWorkload,
    INT8QuantPerTensorWorkload,
    SmoothQuantWorkload,
)
from workloads.workload_base import manifest_call


def _attention_cases():
    """The attention ops, each with the inputs it is built for."""
    _DTYPE = torch.float16
    _HEADS, _HEADS_KV, _DIM = 8, 2, 128

    def gqa_dense():
        case = GQADenseDecodeWorkload(2, _HEADS, _HEADS_KV, 256, _DIM, _DTYPE)
        return GQADenseFwdOp(), case.gen_inputs()

    def gqa_dense_with_lse():
        row = dict(B=1, S_q=256, S_kv=256, H=_HEADS, H_kv=_HEADS_KV, D=_DIM, is_causal=True)
        call = manifest_call("GQADenseWithLSEFwdOp", {"T": "float16"}, **row)
        return GQADenseWithLSEFwdOp(is_causal=True), GQADenseWithLSECall(call).gen_inputs()

    def gqa_bwd():
        case = GQABwdWorkload(1, _HEADS, _HEADS_KV, 256, _DIM, True, _DTYPE)
        op = GQADenseBwdOp(is_causal=True)
        return op, case.gen_inputs()

    def gqa_varlen():
        lens = [128, 128]
        case = GQAPrefillVarlenFwdWorkload(2, _HEADS, _HEADS_KV, lens, lens, _DIM, True, _DTYPE)
        op = GQAVarlenFwdOp()
        return op, case.gen_inputs()

    def gqa_sliding_window_varlen():
        lens = [128, 128]
        case = GQASlidingWindowVarlenFwdWorkload(
            2, lens, lens, _HEADS, _HEADS_KV, _DIM, True, 64, -1, _DTYPE
        )
        op = GQAVarlenFwdOp(is_causal=True, window_size_left=64)
        return op, case.gen_inputs()

    def gqa_paged_decode():
        case = GQAPagedFwdWorkload(_HEADS, _HEADS_KV, _DIM, [1, 1], [256, 200], 64, 4, 8, _DTYPE)
        return GQAPagedFwdOp(), case.gen_inputs()

    def gqa_bwd_mha_heads():
        # One KV head per query head: the call the warp-specialized kernel serves.
        case = GQABwdWorkload(1, _HEADS, _HEADS, 256, _DIM, True, _DTYPE)
        op = GQADenseBwdOp(is_causal=True)
        return op, case.gen_inputs()

    def mha_decode_paged():
        case = MHADecodePagedWorkload(1, _HEADS, 1, 256, _DIM, 128, False, _DTYPE)
        op = MHAPagedFwdOp(page_size=128, is_causal=False)
        return op, case.gen_inputs()

    def mla_decode():
        case = MLADecodeWorkload(1, 64, 1, 256, _DIM, 64, _DTYPE)
        op = MLADenseDecodeFwdOp()
        return op, case.gen_inputs()

    def mla_varlen():
        case = MLAVarlenWorkload([128, 128], _HEADS, _DIM, 64, _DIM, _DTYPE)
        return MLAVarlenFwdOp(), case.gen_inputs()

    def mla_dense_bwd():
        case = MLADenseBwdWorkload(1, 256, 256, _HEADS, 192, _DIM, _DTYPE)
        return MLADenseBwdOp(), case.gen_inputs()

    def mla_varlen_bwd():
        case = MLAVarlenBwdWorkload([128, 128], [128, 128], _HEADS, 192, _DIM, _DTYPE)
        return MLAVarlenBwdOp(), case.gen_inputs()

    def nsa_fwd():
        case = NSAFwdWorkload(1, 16, 512, 64, True, 0.1, 32, 16, 1, _DTYPE)
        op = NSAVarlenFwdOp(is_causal=True, sm_scale=0.1, block_size=32)
        return op, case.gen_inputs()

    def nsa_compressed_fwd():
        case = NSACompressedFwdWorkload(1, 512, 32, _DIM, _DIM, 16, 0.088, 32, _DTYPE)
        op = NSACompressedVarlenFwdOp(sm_scale=0.088, block_size=32)
        return op, case.gen_inputs()

    def nsa_topk():
        case = NSATopKWorkload(1, 512, 32, _DIM, 16, 1.0, 16, 32, _DTYPE)
        op = NSATopKVarlenFwdOp(sm_scale=1.0, selected_block_num=16, block_size=32)
        return op, case.gen_inputs()

    def dsa_decode():
        batch, heads, seq_len, seq_len_kv, dim, tail, topk = 1, 64, 64, 128, 512, 64, 128
        case = DSADecodeWorkload(batch, heads, seq_len, seq_len_kv, dim, tail, topk, 1, 1, seq_len)
        op = DSADenseFwdOp(tail, 1, q_start_index_s=seq_len)
        return op, case.gen_inputs()

    def dsa_paged():
        call = manifest_call(
            "DSAPagedFwdOp", S_q=1, H=64, K=128, NP=8, PS=64, W=2, cache_lens="repeat(128, 2)"
        )
        return DSAPagedFwdOp(), DSAPagedCall(call).gen_inputs()

    def fp8_lightning_indexer():
        case = FP8LightningIndexerWorkload(1, 2048, 8, 64, 4096, 1)
        return FP8LightningIndexerFwdOp(), case.gen_inputs()

    def topk_select():
        score = torch.randn(1, 256, 512, 1, dtype=torch.float32, device=run_device())
        starts = torch.zeros(1, 256, dtype=torch.int32, device=run_device())
        ends = torch.full((1, 256), 512, dtype=torch.int32, device=run_device())
        return TopKSelectFwdOp(top_k=64), (score, starts, ends)

    return (
        ("gqa-dense", gqa_dense),
        # Only the SM90 kernel serves it.
        ("gqa-dense-with-lse", gqa_dense_with_lse, True, (pytest.mark.sm90,)),
        ("gqa-bwd", gqa_bwd),
        ("gqa-varlen", gqa_varlen),
        ("gqa-sliding-window-varlen", gqa_sliding_window_varlen),
        ("gqa-paged-decode", gqa_paged_decode),
        ("gqa-bwd-mha-heads", gqa_bwd_mha_heads),
        ("mha-decode-paged", mha_decode_paged),
        ("mla-decode", mla_decode),
        ("mla-varlen", mla_varlen),
        # dQ lands through float32 atomic adds, whose order differs from run to run.
        ("mla-dense-bwd", mla_dense_bwd, False),
        ("mla-varlen-bwd", mla_varlen_bwd, False),
        ("nsa-fwd", nsa_fwd),
        ("nsa-cmp-fwd", nsa_compressed_fwd),
        ("nsa-topk", nsa_topk),
        ("dsa-decode", dsa_decode),
        ("dsa-paged", dsa_paged),
        ("fp8-lightning-indexer", fp8_lightning_indexer),
        # Picks the same set every time, but the atomic increments that claim the slots
        # decide which index lands where.
        ("topk-selector", topk_select, False),
    )


def _gemm_cases():
    """The gemm ops, each with the inputs it is built for."""
    _DTYPE = torch.float16
    _M, _N, _K = 128, 256, 256

    def _x(*shape, dtype=_DTYPE):
        return torch.randn(*shape, dtype=dtype, device=run_device())

    def gemm():
        return GEMMFwdOp(), (_x(_M, _K), _x(_N, _K))

    def gemm_fp8():
        fp8 = dict(dtype=torch.float8_e4m3fn)
        f32 = dict(dtype=torch.float32)
        return GEMMFP8FwdOp(), (
            _x(_M, _K).to(**fp8),
            _x(_N, _K).to(**fp8),
            _x(1, 1, **f32).abs(),
            _x(1, 1, **f32).abs(),
            None,
        )

    def gemm_int8_w8a8():
        i8 = dict(dtype=torch.int8, device=run_device())
        f32 = dict(dtype=torch.float32)
        return GEMMINT8W8A8FwdOp(), (
            torch.randint(-127, 128, (_M, _K), **i8),
            torch.randint(-127, 128, (_N, _K), **i8),
            _x(_M, 1, **f32).abs(),
            _x(1, _N, **f32).abs(),
            None,
        )

    def gemm_w4a16():
        group = 128
        return GEMMW4A16FwdOp(), (
            _x(_M, _K),
            torch.randint(0, 255, (_N, _K // 2), dtype=torch.uint8, device=run_device()),
            _x(_N, _K // group).abs(),  # the scale follows the activation dtype
            torch.randint(0, 15, (_N, _K // group), dtype=torch.uint8, device=run_device()),
        )

    def gemm_w4a8():
        group = 128
        fp8 = dict(dtype=torch.float8_e4m3fn)
        f32 = dict(dtype=torch.float32)
        return GEMMW4A8FwdOp(), (
            _x(_M, _K).to(**fp8),
            # Any byte pattern is a valid prepacked weight.
            torch.randint(0, 255, (_N, _K // 2), dtype=torch.uint8, device=run_device()),
            _x(_N, _K // group).abs().to(**fp8),
            _x(_N, **f32).abs(),
            _x(_M, **f32).abs(),
        )

    def grouped_gemm():
        groups, rows = 2, 128
        sizes = torch.full((groups,), rows, dtype=torch.int32, device=run_device())
        offsets = torch.tensor([0, rows], dtype=torch.int32, device=run_device())
        return GroupedGEMMFwdOp(), (
            _x(groups * rows, _K),
            _x(groups, _N, _K),
            sizes,
            offsets,
        )

    def bmm():
        batch = 2
        return BMMFwdOp(), (_x(batch, _M, _K), _x(batch, _K, _N))

    def bmm_fp8():
        batch = 2
        fp8 = dict(dtype=torch.float8_e4m3fn)
        scale = torch.tensor(1.0, dtype=torch.float32, device=run_device())
        return BMMFP8FwdOp(out_dtype=torch.float16, trans_b=False), (
            _x(batch, _M, _K).to(**fp8),
            _x(batch, _K, _N).to(**fp8),
            scale,
            scale.clone(),
        )

    return (
        ("gemm", gemm),
        ("gemm-fp8", gemm_fp8),
        ("gemm-int8-w8a8", gemm_int8_w8a8),
        ("gemm-w4a16", gemm_w4a16),
        ("gemm-w4a8", gemm_w4a8),
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
        return torch.randn(*shape, dtype=dtype, device=run_device())

    f32 = dict(dtype=torch.float32)

    def ssd_chunk_cumsum():
        op = SSDChunkCumsumFwdOp(chunk_size=_Q, out_dtype=_DTYPE, dt_softplus=True)
        return op, (_x(_B, _S, _H, **f32), -_x(_H, **f32).abs(), None)

    def ssd_chunk_coupling():
        op = SSDChunkCouplingFwdOp(_Q)
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
        return SSDRecurrentFwdOp(), (
            -_x(_H, _P, _N, **f32).abs(),
            _x(_B, _H, _P, **f32),
            _x(_B, _H, _P),
            _x(_B, _G, _N),
            _x(_B, _G, _N),
            _x(_B, _H, _P, _N, **f32),
        )

    def causal_conv1d():
        cu_seqlens = torch.tensor([0, 5, 64], dtype=torch.int32, device=run_device())
        flags = torch.tensor([True, False], device=run_device())
        return CausalConv1dFwdOp(), (
            _x(64, 256),
            _x(256, 4),
            _x(2, 256, 3),
            cu_seqlens,
            _x(256),
            flags,
        )

    def causal_conv1d_update():
        return CausalConv1dUpdateFwdOp(), (_x(3, 256), _x(256, 4), _x(3, 256, 3), _x(256))

    def selective_scan():
        return SelectiveScanFwdOp(), (
            _x(1, 16, 64),
            0.1 * _x(1, 16, 64),
            -_x(16, 16, **f32).abs(),
            _x(1, 1, 16, 64),
            _x(1, 1, 16, 64),
            _x(16, **f32),
            _x(1, 16, 64),
            _x(16, **f32),
            None,
            None,
        )

    return (
        ("causal-conv1d", causal_conv1d),
        ("causal-conv1d-update", causal_conv1d_update),
        ("selective-scan", selective_scan),
        ("da-cumsum", ssd_chunk_cumsum),
        ("cb-producer", ssd_chunk_coupling),
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
        return torch.randn(*shape, dtype=dtype, device=run_device())

    chunks = _S // _CHUNK + 1

    def gla_chunk_fwd():
        op = GLAChunkFwdOp(chunk_size=_CHUNK, scale=_SCALE)
        # ``g`` is a log-space decay, so it must be non-positive.
        return op, (
            _x(_B, _S, _H, _D),
            _x(_B, _S, _H, _D),
            _x(_B, _S, _H, _D),
            -_x(_B, _S, _H, _D).abs(),
            None,
        )

    def gla_chunk_bwd():
        op = GLAChunkBwdOp(chunk_size=_CHUNK, scale=_SCALE)
        return op, (
            _x(_B, _S, _H, _D),
            _x(_B, _H, _D, _D, dtype=torch.float32),
            _x(_B, _S, _H, _D),
            _x(_B, _S, _H, _D),
            _x(_B, _S, _H, _D),
            -_x(_B, _S, _H, _D).abs(),
            _x(_B, chunks, _H, _D, _D, dtype=torch.float32),
        )

    def gla_decode():
        op = GLARecurrentFwdOp(scale=_SCALE)
        return op, (
            _x(_B, _H, _D),
            _x(_B, _H, _D),
            _x(_B, _H, _D),
            -_x(_B, _H, _D).abs(),
            _x(_B, _H, _D, _D),
        )

    def deltanet_chunk_fwd():
        # The delta rule is a recurrence over S steps; unit-variance operands overflow it
        # into NaN, which compares unequal to itself. Scale as ``DeltaNetChunkwiseWorkload`` does.
        op = DeltaNetChunkFwdOp(chunk_size=_CHUNK)
        return op, (
            _x(_B, _H, _S, _D) * 0.1,
            _x(_B, _H, _S, _D) * 0.1,
            _x(_B, _H, _S, _D) * 0.1,
            _x(_B, _H, _S).sigmoid() * 0.5,
        )

    def deltanet_chunk_bwd():
        op = DeltaNetChunkBwdOp(chunk_size=_CHUNK)
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
        return DeltaNetRecurrentFwdOp(), (
            _x(_B, _H, _D),
            _x(_B, _H, _D),
            _x(_B, _H, _D),
            _x(_B, _H).sigmoid(),
            _x(_B, _H, _D, _D),
        )

    def gla():
        return GLAFwdOp(scale=_SCALE), (
            _x(_B, _S, _H, _D) * 0.1,
            _x(_B, _S, _H, _D) * 0.1,
            _x(_B, _S, _H, _D) * 0.1,
            -_x(_B, _S, _H, _D).abs() * 0.1,
        )

    def deltanet():
        k = torch.nn.functional.normalize(_x(_B, _S, _H, _D, dtype=torch.float32), dim=-1)
        return DeltaNetFwdOp(), (
            _x(_B, _S, _H, _D) * 0.1,
            k.to(_DTYPE),
            _x(_B, _S, _H, _D) * 0.1,
            _x(_B, _S, _H).sigmoid() * 0.5,
        )

    def gdn():
        # A 128-wide head, which both the in-tree prefill and decode kernels serve.
        k = torch.nn.functional.normalize(_x(_B, _S, _H, 128, dtype=torch.float32), dim=-1)
        return GDNFwdOp(), (
            _x(_B, _S, _H, 128) * 0.1,
            k.to(_DTYPE),
            _x(_B, _S, _H, 128) * 0.1,
            -_x(_B, _S, _H).abs() * 0.1,
            _x(_B, _S, _H).sigmoid(),
        )

    def kda():
        # A 128-wide square state with a precomputed gate, which the in-tree prefill serves.
        k = torch.nn.functional.normalize(_x(_B, _S, _H, 128, dtype=torch.float32), dim=-1)
        return KDAFwdOp(), (
            _x(_B, _S, _H, 128) * 0.1,
            k.to(_DTYPE),
            _x(_B, _S, _H, 128) * 0.1,
            -_x(_B, _S, _H, 128).abs() * 0.1,
            _x(_B, _S, _H).sigmoid(),
        )

    return (
        ("gla-chunk-fwd", gla_chunk_fwd),
        ("gla-chunk-bwd", gla_chunk_bwd),
        ("gla-decode", gla_decode),
        ("deltanet-chunk-fwd", deltanet_chunk_fwd),
        ("deltanet-chunk-bwd", deltanet_chunk_bwd),
        ("deltanet-decode", deltanet_decode),
        ("gla", gla),
        ("deltanet", deltanet),
        ("gdn", gdn),
        ("kda", kda),
    )


def _other_cases():
    """The FFT, Hadamard, quantization and mean pooling ops, with the inputs they are built for."""

    def fft_c2c():
        x = torch.randn(2, 64, dtype=torch.complex64, device=run_device())
        return FFTC2CFwdOp(), (x,)

    def hadamard_transform():
        x = torch.randn(2, 128, dtype=torch.bfloat16, device=run_device())
        return HadamardTransformFwdOp(), (x,)

    def fp8_quant():
        x = torch.randn(1, 64, 1, 64, dtype=torch.float16, device=run_device())
        return FP8QuantPerTokenGroupFwdOp(), (x,)

    def fp8_quant_per_block():
        return FP8QuantPerBlockFwdOp(), FP8QuantPerBlockWorkload(
            200, 392, torch.bfloat16
        ).gen_inputs()

    def int8_dequant_per_channel():
        case = INT8DequantPerChannelWorkload(64, 64, torch.bfloat16)
        return INT8DequantPerChannelFwdOp(torch.bfloat16), case.gen_inputs()

    def int8_dequant_per_block():
        case = INT8DequantPerBlockWorkload(64, 64, torch.bfloat16)
        return INT8DequantPerBlockFwdOp(torch.bfloat16), case.gen_inputs()

    def int8_dequant_per_tensor():
        case = INT8DequantPerTensorWorkload(64, 64, torch.bfloat16)
        return INT8DequantPerTensorFwdOp(torch.bfloat16), case.gen_inputs()

    def int8_quant_per_block():
        return INT8QuantPerTokenGroupFwdOp(), INT8QuantPerBlockWorkload(
            64, 256, torch.bfloat16
        ).gen_inputs()

    def int8_quant_per_channel():
        return INT8QuantPerChannelFwdOp(), INT8QuantPerChannelWorkload(
            64, 64, torch.bfloat16
        ).gen_inputs()

    def int8_quant_per_tensor():
        return INT8QuantPerTensorFwdOp(), INT8QuantPerTensorWorkload(
            64, 64, torch.bfloat16
        ).gen_inputs()

    def int4_quant_per_group():
        return INT4QuantPerGroupFwdOp(), INT4QuantPerGroupWorkload(
            64, 256, torch.float16
        ).gen_inputs()

    def smooth_quant():
        return SmoothQuantFwdOp(), SmoothQuantWorkload(64, 64, torch.bfloat16).gen_inputs()

    def mean_pooling():
        x = torch.randn(1, 64, 2, 64, dtype=torch.float16, device=run_device())
        return MeanPoolingFwdOp(32, torch.float32), (x,)

    def top_k_top_p_mask():
        logits = torch.randn(2, 256, dtype=torch.bfloat16, device=run_device())
        k = torch.tensor([1, 40], dtype=torch.int32, device=run_device())
        p = torch.tensor([0.9, 0.7], dtype=torch.float32, device=run_device())
        return TopKTopPMaskFwdOp(), (logits, k, p)

    return (
        ("fft-c2c", fft_c2c),
        ("hadamard-transform", hadamard_transform),
        ("fp8-quant", fp8_quant),
        ("fp8-quant-per-block", fp8_quant_per_block),
        ("int4-quant-per-group", int4_quant_per_group),
        ("int8-dequant-per-block", int8_dequant_per_block),
        ("int8-dequant-per-channel", int8_dequant_per_channel),
        ("int8-dequant-per-tensor", int8_dequant_per_tensor),
        ("int8-quant-per-block", int8_quant_per_block),
        ("int8-quant-per-channel", int8_quant_per_channel),
        ("int8-quant-per-tensor", int8_quant_per_tensor),
        ("mean-pooling", mean_pooling),
        ("smooth-quant", smooth_quant),
        ("top-k-top-p-mask", top_k_top_p_mask),
    )


def _sampling_cases():
    """The logit filters, the token draw and the chain verification, each with the inputs it is built for."""

    def min_p_mask():
        logits = torch.randn(4, 256, dtype=torch.bfloat16, device=run_device())
        min_p = torch.full((4,), 0.1, device=run_device())
        return MinPMaskFwdOp(), (logits, min_p)

    def top_k_mask():
        logits = torch.randn(2, 256, dtype=torch.bfloat16, device=run_device())
        k = torch.tensor([1, 40], dtype=torch.int32, device=run_device())
        return TopKMaskFwdOp(), (logits, k)

    def top_p_mask():
        logits = torch.randn(4, 256, dtype=torch.bfloat16, device=run_device())
        p = torch.full((4,), 0.9, device=run_device())
        return TopPMaskFwdOp(), (logits, p)

    def sampling_from_probs():
        probs = torch.rand(4, 256, device=run_device()).softmax(-1)
        state = torch.tensor([7], dtype=torch.int64, device=run_device())
        return SamplingFromProbsFwdOp(), (probs, state, state)

    def chain_speculative_sampling():
        draft = torch.rand(4, 2, 256, device=run_device()).softmax(-1)
        target = torch.rand(4, 3, 256, device=run_device()).softmax(-1)
        ids = torch.randint(0, 256, (4, 2), dtype=torch.int32, device=run_device())
        state = torch.tensor([7], dtype=torch.int64, device=run_device())
        return ChainSpeculativeSamplingFwdOp(), (draft, ids, target, state, state)

    return (
        ("min-p-mask", min_p_mask),
        ("top-k-mask", top_k_mask),
        ("top-p-mask", top_p_mask),
        ("sampling-from-probs", sampling_from_probs),
        ("chain-speculative-sampling", chain_speculative_sampling),
    )


def _sequence_modeling_cases():
    """The sequence modeling ops, each with the inputs it is built for."""
    _DTYPE = torch.bfloat16
    _M = 1
    _SEQ_LEN = 32
    _D = 256
    _CONV_TAPS = 4

    def _x(*shape, dtype=_DTYPE):
        return torch.randn(*shape, dtype=dtype, device=run_device())

    def engram_gate_conv_fwd():
        op = EngramGateConvFwdOp()
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
        op = EngramGateConvBwdOp()
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
        op = EngramDecodeFwdOp(max_conv_len, _CONV_TAPS, dilation)
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
        return torch.randn(*shape, dtype=_DTYPE, device=run_device())

    def one_d(op_cls):
        return lambda: (op_cls(input_layout="1d"), (_x(_SEQ_LEN, _D),))

    def two_d(op_cls):
        return lambda: (op_cls(input_layout="2d"), (_x(2, _SEQ_LEN, _HEADS, _D),))

    def longrope():
        rescale = torch.linspace(1.0, 2.0, _D // 2, device=run_device())
        return LongRoPEFwdOp(rescale_factors=rescale), (_x(_SEQ_LEN, _D),)

    def position_ids():
        op = RoPENeoxPositionIdsFwdOp(max_position=128)
        positions = torch.arange(_SEQ_LEN, device=run_device(), dtype=torch.int32)
        return op, (_x(_SEQ_LEN, _HEADS, _D), positions)

    return (
        ("rope-neox", one_d(RoPEFwdOp)),
        (
            "rope-interleaved",
            lambda: (
                RoPEFwdOp(rope_layout="interleaved", input_layout="2d"),
                (_x(2, _SEQ_LEN, _HEADS, _D),),
            ),
        ),
        ("rope-llama31", one_d(RoPELlama31FwdOp)),
        ("rope-yarn", two_d(YaRNFwdOp)),
        ("rope-longrope", longrope),
        ("rope-neox-position-ids", position_ids),
    )


_FAMILIES = (
    _attention_cases,
    _gemm_cases,
    _mamba_cases,
    _linear_attention_cases,
    _sampling_cases,
    _sequence_modeling_cases,
    _rope_cases,
    _other_cases,
)


class _Case(NamedTuple):
    """One op's case. ``exact`` is False where the kernel returns a tensor a compiled call
    cannot be held equal to; the contract there is the traced graph plus the shapes and
    dtypes the op promises. ``marks`` are its pytest marks, such as the architecture its
    kernel needs."""

    name: str
    build: Callable[[], tuple]
    exact: bool = True
    marks: tuple = ()


def _cases():
    """Every family's cases, as pytest params.

    The builders run inside the test: this module is imported on the CPU-only runner
    that enforces the compile-contract gate, where a CUDA tensor built at import time
    would fail before any test is selected.
    """
    cases = [_Case(*entry) for family in _FAMILIES for entry in family()]
    return [pytest.param(case, id=case.name, marks=case.marks) for case in cases]


# A spec-only entry's case runs but is not registered: only an implemented entry's
# declaration is contract evidence.
for _op_cls in (
    GQADenseFwdOp,
    GQADenseWithLSEFwdOp,
    GQADenseBwdOp,
    GQAVarlenFwdOp,
    GQAPagedFwdOp,
    MHAPagedFwdOp,
    MLADenseDecodeFwdOp,
    MLADenseBwdOp,
    MLAVarlenBwdOp,
    MLAVarlenFwdOp,
    NSAVarlenFwdOp,
    NSACompressedVarlenFwdOp,
    NSATopKVarlenFwdOp,
    DSADenseFwdOp,
    DSAPagedFwdOp,
    FP8LightningIndexerFwdOp,
    TopKSelectFwdOp,
    GEMMFwdOp,
    GEMMFP8FwdOp,
    GEMMINT8W8A8FwdOp,
    GEMMW4A16FwdOp,
    GEMMW4A8FwdOp,
    GroupedGEMMFwdOp,
    BMMFwdOp,
    BMMFP8FwdOp,
    CausalConv1dFwdOp,
    CausalConv1dUpdateFwdOp,
    SelectiveScanFwdOp,
    SSDChunkCumsumFwdOp,
    SSDChunkCouplingFwdOp,
    SSDChunkStateFwdOp,
    SSDStatePassingFwdOp,
    SSDChunkScanFwdOp,
    SSDRecurrentFwdOp,
    GLAChunkFwdOp,
    GLAChunkBwdOp,
    GLARecurrentFwdOp,
    GLAFwdOp,
    DeltaNetChunkFwdOp,
    DeltaNetChunkBwdOp,
    DeltaNetRecurrentFwdOp,
    DeltaNetFwdOp,
    GDNFwdOp,
    KDAFwdOp,
    FFTC2CFwdOp,
    HadamardTransformFwdOp,
    FP8QuantPerTokenGroupFwdOp,
    ChainSpeculativeSamplingFwdOp,
    FP8QuantPerBlockFwdOp,
    INT4QuantPerGroupFwdOp,
    INT8DequantPerBlockFwdOp,
    INT8DequantPerChannelFwdOp,
    INT8DequantPerTensorFwdOp,
    INT8QuantPerTokenGroupFwdOp,
    INT8QuantPerChannelFwdOp,
    INT8QuantPerTensorFwdOp,
    SmoothQuantFwdOp,
    MeanPoolingFwdOp,
    MinPMaskFwdOp,
    SamplingFromProbsFwdOp,
    TopPMaskFwdOp,
    EngramGateConvFwdOp,
    EngramGateConvBwdOp,
    EngramDecodeFwdOp,
    MHCPreFwdOp,
    MHCPostFwdOp,
    RoPEFwdOp,
    RoPELlama31FwdOp,
    YaRNFwdOp,
    LongRoPEFwdOp,
    RoPENeoxPositionIdsFwdOp,
    TopKMaskFwdOp,
    TopKTopPMaskFwdOp,
):
    register_compile_contract(_op_cls)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("case", _cases())
def test_a_cold_op_traces_fullgraph_and_matches_eager(case) -> None:
    """Cold is the whole contract: a warm op has nothing left for dynamo to trace into."""
    op, inputs = case.build()
    # Its own copies per call: a paged op writes its cache pages. ``detach`` because a
    # backward op's operator carries no autograd formula to answer a history-tracking input.
    compiled_inputs = tuple(None if t is None else t.detach().clone() for t in inputs)
    eager_inputs = tuple(None if t is None else t.detach().clone() for t in inputs)

    compiled = torch.compile(op, fullgraph=True)(*compiled_inputs)

    assert_same_result(compiled, op(*eager_inputs), exact=case.exact)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("case", _cases())
def test_the_fake_reports_what_the_op_returns(case) -> None:
    """The fake is the op's whole promise to the compiler, so it must be the truth."""
    op, inputs = case.build()
    inputs = tuple(None if t is None else t.detach() for t in inputs)

    assert_fake_matches_eager(op, *inputs)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.parametrize("case", _cases())
def test_the_traced_graph_holds_only_this_ops_operator(case) -> None:
    """The node is the op's, so replacing the kernel cannot change the graph."""
    op, inputs = case.build()
    inputs = tuple(None if t is None else t.detach() for t in inputs)

    assert_op_owns_graph_nodes(op, *inputs)
