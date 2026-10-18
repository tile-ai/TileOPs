"""MLA decode against FlashInfer and vLLM's FlashMLA; packed MLA prefill against FA3 and
FlashInfer."""

import math

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import FLASHINFER_TAG, baseline_module, flashinfer_op, vllm_op
from tileops.ops import MLADecodeWithKVCacheFwdOp, MLAVarlenFwdOp


@pytest.mark.parametrize("case", bench.cases(MLADecodeWithKVCacheFwdOp), ids=lambda case: case.id)
def test_mla_decode_bench(case) -> None:
    q, q_pe, k, _ = case.inputs
    batch, heads, dim = q.shape
    length, dim_pe = (k.shape[1], q_pe.shape[-1])
    page = 64
    pages = length // page
    indices = torch.arange(batch * pages, dtype=torch.int32, device=q.device)
    lengths = torch.full((batch,), length, dtype=torch.int32, device=q.device)
    wrapper_cls = flashinfer_op("mla.BatchMLAPagedAttentionWrapper")
    wrapper = wrapper_cls(torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device))
    wrapper.plan(
        torch.arange(batch + 1, dtype=torch.int32, device=q.device),
        torch.arange(batch + 1, dtype=torch.int32, device=q.device) * pages,
        indices,
        lengths,
        heads,
        dim,
        dim_pe,
        page,
        False,
        (dim + dim_pe) ** (-0.5),
        q.dtype,
        k.dtype,
    )

    def flashinfer_fn(q, q_pe, k, k_pe):
        return wrapper.run(q, q_pe, k.reshape(-1, page, dim), k_pe.reshape(-1, page, dim_pe))

    flashmla = vllm_op("flash_mla_with_kvcache", "v1.attention.ops.flashmla")
    metadata, _ = vllm_op("get_mla_metadata", "v1.attention.ops.flashmla")()
    table = indices.reshape(batch, pages)

    def flashmla_fn(q, q_pe, k, k_pe):
        query = torch.cat((q, q_pe), -1).unsqueeze(1)
        cache = torch.cat((k, k_pe), -1).reshape(-1, page, 1, dim + dim_pe)
        out, _ = flashmla(query, cache, table, lengths, dim, metadata)
        return out.squeeze(1)

    op = MLADecodeWithKVCacheFwdOp(**case.arguments)
    op.request_tune()
    bench.Runner(op, case).compare({FLASHINFER_TAG: flashinfer_fn, "flashmla": flashmla_fn})


def _expanded_key(k_nope, k_pe):
    """The per-head key these libraries take: ``k_pe`` repeated beside each head's nope."""
    return torch.cat([k_nope, k_pe[:, None].expand(-1, k_nope.shape[1], -1)], dim=-1)


@pytest.mark.parametrize("case", bench.cases(MLAVarlenFwdOp), ids=lambda case: case.id)
def test_mla_varlen_fwd_bench(case) -> None:
    """Neither library takes ``k_pe`` once per token, so each call builds the per-head key
    first, as vLLM's MLA prefill does before calling FA3."""
    workload = case.workload
    q, _, _, _, cu_seqlens = case.inputs
    max_seqlen = max(workload.seq_lens)
    flash_attn_varlen_func = baseline_module("flash_attn_interface", "fa3").flash_attn_varlen_func

    def fa3_fn(q, k_nope, k_pe, v, cu_seqlens):
        out, lse = flash_attn_varlen_func(
            q,
            _expanded_key(k_nope, k_pe),
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale=workload.sm_scale,
            causal=workload.is_causal,
            return_attn_probs=True,
        )
        # FA3 returns the log-sum-exp head-major.
        return out, lse.T

    wrapper = flashinfer_op("prefill.BatchPrefillWithRaggedKVCacheWrapper")(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device), kv_layout="NHD"
    )
    wrapper.plan(
        qo_indptr=cu_seqlens,
        kv_indptr=cu_seqlens,
        num_qo_heads=workload.heads,
        num_kv_heads=workload.heads,
        head_dim_qk=workload.dim_nope + workload.dim_pe,
        head_dim_vo=workload.dim_v,
        causal=workload.is_causal,
        sm_scale=workload.sm_scale,
        q_data_type=q.dtype,
    )

    def flashinfer_fn(q, k_nope, k_pe, v, _cu_seqlens):
        out, lse = wrapper.run(q, _expanded_key(k_nope, k_pe), v, return_lse=True)
        # FlashInfer returns the log-sum-exp in base 2.
        return out, lse * math.log(2)

    op = MLAVarlenFwdOp(**case.arguments)
    bench.Runner(op, case).compare({"fa3": fa3_fn, FLASHINFER_TAG: flashinfer_fn})
