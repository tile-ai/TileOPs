"""Benchmark for GroupedQueryAttentionSlidingWindowVarlenFwdOp vs FA3 baseline."""

import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import (
    gqa_sliding_window_varlen_args,
    manifest_params,
)
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionSlidingWindowVarlenFwdOp
from workloads.attention.gqa_sliding_window_varlen import (
    GroupedQueryAttentionSlidingWindowVarlenFwdTest,
)

_OP_NAME = "GroupedQueryAttentionSlidingWindowVarlenFwdOp"

_GQA_SLIDING_WINDOW_VARLEN_FWD_BENCH_PARAMS = manifest_params(
    load_workloads(_OP_NAME),
    gqa_sliding_window_varlen_args,
    tune=False,
)


def _torch_sliding_window_varlen_fwd(test):
    """Torch SDPA forward baseline: unpack varlen to padded batch, single SDPA call."""

    def fn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q):
        B = test.batch
        seqlens_q = test.seqlens_q
        seqlens_k = test.seqlens_k
        max_sq = max(seqlens_q)
        max_sk = max(seqlens_k)
        H, Hkv, D = test.heads, test.heads_kv, test.dim

        # Unpack packed varlen tensors to padded [B, max_s, H, D]
        q_pad = q.new_zeros(B, max_sq, H, D)
        k_pad = k.new_zeros(B, max_sk, Hkv, D)
        v_pad = v.new_zeros(B, max_sk, Hkv, D)
        for i in range(B):
            qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            ks, ke = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
            q_pad[i, : qe - qs] = q[qs:qe]
            k_pad[i, : ke - ks] = k[ks:ke]
            v_pad[i, : ke - ks] = v[ks:ke]

        # Build combined mask [B, max_sq, max_sk]
        q_idx = torch.arange(max_sq, device=q.device).view(1, -1, 1)
        k_idx = torch.arange(max_sk, device=q.device).view(1, 1, -1)
        sq_t = torch.tensor(seqlens_q, device=q.device, dtype=torch.long).view(B, 1, 1)
        sk_t = torch.tensor(seqlens_k, device=q.device, dtype=torch.long).view(B, 1, 1)
        offsets = sk_t - sq_t  # per-sample offset [B, 1, 1]

        # Padding positions
        mask = (q_idx >= sq_t) | (k_idx >= sk_t)
        # Sliding window + causal constraints
        if test.is_causal:
            mask = mask | (k_idx > q_idx + offsets)
        if test.wl >= 0:
            mask = mask | (k_idx < q_idx + offsets - test.wl)
        if test.wr >= 0:
            mask = mask | (k_idx > q_idx + offsets + test.wr)

        attn_mask = torch.zeros(B, max_sq, max_sk, dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(mask, float("-inf"))

        # SDPA call: transpose to [B, H, S, D], mask broadcasts as [B, 1, max_sq, max_sk]
        out = F.scaled_dot_product_attention(
            q_pad.transpose(1, 2),
            k_pad.transpose(1, 2),
            v_pad.transpose(1, 2),
            attn_mask=attn_mask.unsqueeze(1),
            enable_gqa=True,
        )
        out = out.transpose(1, 2)  # [B, max_sq, H, D]

        # Repack valid positions to [total_q, H, D]
        parts = [out[i, : seqlens_q[i]] for i in range(B)]
        return torch.cat(parts, dim=0)

    return fn


def _fa3_varlen_baseline(max_seqlen_k, is_causal, wl, wr):
    """Return FA3 varlen baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_varlen_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q):
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=is_causal,
            window_size=(wl, wr),
        )
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_varlen_sliding_window_fwd(test, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q):
    """Set up FlashInfer ragged prefill wrapper. Returns callable or None.

    FlashInfer only supports window_left; skip when window_right >= 0.
    """
    if test.wr >= 0:
        return None
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=cu_seqlens_q,
        kv_indptr=cu_seqlens_k,
        num_qo_heads=test.heads,
        num_kv_heads=test.heads_kv,
        head_dim_qk=test.dim,
        causal=test.is_causal,
        window_left=test.wl,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q):
        return wrapper.run(q, k, v)

    return run_fn


@pytest.mark.parametrize(
    "batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype, tune",
    _GQA_SLIDING_WINDOW_VARLEN_FWD_BENCH_PARAMS,
)
def test_gqa_sliding_window_varlen_fwd_bench(
    batch: int,
    seqlens_q,
    seqlens_k,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    wl: int,
    wr: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionSlidingWindowVarlenFwdTest(
        batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype
    )
    inputs = test.gen_inputs()
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q = inputs

    op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
        dtype=dtype,
        tune=tune,
    )
    op.total_q = sum(seqlens_q)
    op.total_k = sum(seqlens_k)
    op.q_lens = seqlens_q
    op.k_lens = seqlens_k
    op.max_seqlen_q = max(seqlens_q)
    op.max_seqlen_k = max(seqlens_k)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    # Warmup: trigger JIT compilation before timed profiling
    op(*inputs)
    torch.cuda.synchronize()

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # FA3 baseline
    max_seqlen_k = max(seqlens_k)
    fa3_fn = _fa3_varlen_baseline(max_seqlen_k, is_causal, wl, wr)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")

    # FlashInfer baseline
    fi_fn = _flashinfer_varlen_sliding_window_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_fn is None and fi_fn is None:
        result_bl = bm.profile(_torch_sliding_window_varlen_fwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
