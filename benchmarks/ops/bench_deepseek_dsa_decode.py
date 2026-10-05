"""Benchmark TileOPs DeepSeek sparse-attention (DSA) decode, against FlashMLA and PyTorch."""

import pytest
import torch

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    reference_tolerance,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from workloads.attention.dsa import DsaDecodeCall


def _torch_sdpa_dsa(workload: DsaDecodeCall):
    """SDPA over the selection ``ref_program`` masks, or None for a row it cannot serve.

    Same computation, without the reference's float32 upcast and materialized score
    tensor. A single kv head lets the mask and the cache broadcast over the query heads.
    """
    if workload.heads_kv != 1:
        return None

    def fn(q, kv, indices):
        b, sq, h, dim_q = q.shape
        sk = kv.shape[1]
        dim = workload.dim
        mask = workload.selection_mask(indices).expand(b, h, sq, sk)
        k = kv.permute(0, 2, 1, 3).expand(b, h, sk, dim_q)
        v = kv[..., :dim].permute(0, 2, 1, 3).expand(b, h, sk, dim)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k,
            v,
            attn_mask=mask,
            scale=workload.sm_scale if workload.sm_scale is not None else dim_q**-0.5,
        )
        return out.permute(0, 2, 1, 3).reshape(b, sq, h, dim).to(q.dtype)

    return fn


def _torch_gather_dsa(workload: DsaDecodeCall):
    """Dense attention over only the gathered selection, or None when it buys nothing.

    Gathering beats masking only where the selection is smaller than the cache.
    """
    if workload.heads_kv != 1 or workload.topk >= workload.seq_len_kv:
        return None

    def fn(q, kv, indices):
        b, sq, h, dim_q = q.shape
        sk = kv.shape[1]
        dim, topk = workload.dim, indices.shape[-1]
        idx = indices.transpose(1, 2).clamp(max=sk - 1).long()
        valid = torch.gather(workload.selection_mask(indices), 3, idx)
        gathered = torch.gather(
            kv.squeeze(2), 1, idx.reshape(b, sq * topk, 1).expand(-1, -1, dim_q)
        ).view(b, sq, topk, dim_q)
        scale = workload.sm_scale if workload.sm_scale is not None else dim_q**-0.5
        scores = torch.einsum("bhqd,bqkd->bhqk", q.permute(0, 2, 1, 3), gathered)
        probs = scores.float().mul(scale).masked_fill(~valid, float("-inf")).softmax(-1)
        out = torch.einsum("bhqk,bqkd->bhqd", probs.to(kv.dtype), gathered[..., :dim])
        return out.permute(0, 2, 1, 3).reshape(b, sq, h, dim).to(q.dtype)

    return fn


def _flashmla_sparse(workload: DsaDecodeCall):
    run = vllm_op("flash_mla_sparse_fwd", "v1.attention.ops.flashmla")
    scale = workload.sm_scale
    if scale is None:
        scale = (workload.dim + workload.dim_tail) ** -0.5

    def fn(q, kv, indices):
        query_pos = torch.arange(q.shape[1], device=q.device) + workload.q_start_index_s
        valid = (indices >= 0) & (indices < kv.shape[1])
        valid &= (
            indices * workload.stride_kv + workload.stride_kv - 1 <= query_pos[None, :, None, None]
        )
        selected = indices.masked_fill(~valid, -1)
        outputs = [
            run(q[b], kv[b], selected[b], scale, d_v=workload.dim)[0] for b in range(q.shape[0])
        ]
        return outputs[0].unsqueeze(0) if len(outputs) == 1 else torch.stack(outputs)

    return fn


@pytest.mark.parametrize("call", manifest_calls(DeepSeekSparseAttentionDecodeWithKVCacheFwdOp))
def test_dsa_decode_bench(call) -> None:
    workload = DsaDecodeCall(call)
    inputs = workload.gen_inputs()
    dtype = workload.dtype

    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(**workload.arguments())
    bm = ManifestBenchmark(op, workload)

    if dtype == torch.bfloat16:
        bm.compare({"tileops": op, "flashmla": _flashmla_sparse(workload)}, *inputs)
        return

    baselines = {}
    sdpa_fn = _torch_sdpa_dsa(workload)
    if sdpa_fn is not None:
        baselines["torch-sdpa"] = sdpa_fn
    gather_fn = _torch_gather_dsa(workload)
    if gather_fn is not None:
        baselines["torch-gather"] = gather_fn
    for fn in baselines.values():
        assert_matches_reference(fn, workload.ref_program, *inputs, **reference_tolerance(dtype))

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
            **baselines,
        },
        *inputs,
    )
