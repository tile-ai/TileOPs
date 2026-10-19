"""MLA decode (dense and paged) against FlashInfer and vLLM's FlashMLA; packed MLA prefill
against FA3 and FlashInfer."""

import inspect
import math

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import FLASHINFER_TAG, baseline_module, flashinfer_op, vllm_op
from tileops.ops import MLADecodeWithKVCacheFwdOp, MLAPagedFwdOp, MLAVarlenFwdOp


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


def _flashmla_paged(inputs, op):
    """vLLM's FlashMLA on the same paged cache: it reads ``[pages, page, 1, dim]`` rows
    through the same block table and returns the log-sum-exp as ``(B, H, S_q)``, which the
    adapter hands back transposed, as a view. It takes ``None`` for the split count and
    fills its scheduling metadata on its first call, before timing."""
    q, kv_cache, table, lens, _ = inputs
    batch, s_q, heads, dim = q.shape
    flash_mla = vllm_op("flash_mla_with_kvcache", "v1.attention.ops.flashmla")
    metadata, _ = vllm_op("get_mla_metadata", "v1.attention.ops.flashmla")(lens, s_q * heads, 1)
    cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], 1, dim)

    def run(q, kv_cache, block_table, cache_seqlens, kv_scale):
        out, lse = flash_mla(
            q, cache, block_table, cache_seqlens, op.kv_lora_rank, metadata, None,
            softmax_scale=op.sm_scale, causal=op.is_causal,
        )  # fmt: skip
        return out, lse.transpose(1, 2)

    return run


def _flashinfer_paged(inputs, op):
    """FlashInfer's ``BatchMLAPagedAttentionWrapper`` over the pages each request reaches.

    It takes the latent and rope columns as two caches, the column ranges of the one cache,
    and the query as two halves, split once outside timing; the plan (the page lists each
    request reaches) is also built outside timing. An FP8 cache passes its scale for both.
    """
    q, kv_cache, table, lens, kv_scale = inputs
    batch, s_q, heads, dim = q.shape
    dim_v, page = op.kv_lora_rank, kv_cache.shape[1]
    reach = [-(-int(n) // page) for n in lens.tolist()]
    indices = torch.cat([table[b, : reach[b]] for b in range(batch)])
    indptr = torch.tensor(
        [0, *torch.tensor(reach).cumsum(0).tolist()], dtype=torch.int32, device=q.device
    )
    wrapper = flashinfer_op("mla.BatchMLAPagedAttentionWrapper")(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    )
    scale = dim**-0.5 if op.sm_scale is None else op.sm_scale
    wrapper.plan(
        torch.arange(batch + 1, dtype=torch.int32, device=q.device) * s_q, indptr, indices,
        lens, heads, dim_v, dim - dim_v, page, op.is_causal, scale, q.dtype, kv_cache.dtype,
    )  # fmt: skip
    q_nope = q[..., :dim_v].reshape(batch * s_q, heads, dim_v).contiguous()
    q_pe = q[..., dim_v:].reshape(batch * s_q, heads, dim - dim_v).contiguous()
    extra = {}
    if kv_scale is not None:
        extra = {"ckv_scale": float(kv_scale), "kpe_scale": float(kv_scale)}

    def run(q, kv_cache, block_table, cache_seqlens, kv_scale):
        out, lse = wrapper.run(
            q_nope, q_pe, kv_cache[..., :dim_v], kv_cache[..., dim_v:],
            return_lse=True, return_lse_base_on_e=True, **extra,
        )  # fmt: skip
        return out.view(batch, s_q, heads, dim_v), lse.view(batch, s_q, heads)

    return run


@pytest.mark.parametrize("case", bench.cases(MLAPagedFwdOp), ids=lambda case: case.id)
def test_mla_paged_bench(case) -> None:
    op = MLAPagedFwdOp(**case.arguments)
    q, kv_cache = case.inputs[0], case.inputs[1]
    implementations = {}
    fp8 = kv_cache.dtype == torch.float8_e4m3fn
    # FlashMLA's dense decode runs on SM90 and SM100 only, and its FP8 cache is its own
    # packed layout with per-tile scales, not a float8_e4m3fn cache with one scale.
    dense, _ = vllm_op("is_flashmla_dense_supported", "v1.attention.ops.flashmla")()
    if dense and not fp8:
        implementations["flashmla"] = _flashmla_paged(case.inputs, op)
    # FlashInfer's FP8 MLA takes bfloat16 queries only, and an older FlashInfer's ``run``
    # takes no cache scale.
    run = flashinfer_op("mla.BatchMLAPagedAttentionWrapper").run
    if not fp8 or (q.dtype == torch.bfloat16 and "ckv_scale" in inspect.signature(run).parameters):
        implementations[FLASHINFER_TAG] = _flashinfer_paged(case.inputs, op)
    bench.Runner(op, case).compare(implementations)


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
