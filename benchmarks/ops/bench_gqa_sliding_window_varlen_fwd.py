"""Benchmark for GqaSlidingWindowVarlenFwdOp vs FA3 baseline."""
from typing import Optional

import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gqa_sliding_window_varlen_fwd import GqaSlidingWindowVarlenFwdTest
from tileops.ops import GqaSlidingWindowVarlenFwdOp

_GQA_SLIDING_WINDOW_VARLEN_FWD_BENCH_PARAMS = [
    pytest.param(1, [3000], [3000], 32, 8, 128, True, -1, -1, torch.float16, False, id="single-seq-causal"),
    pytest.param(2, [1537, 3073], [1537, 3073], 32, 8, 128, True, 2000, -1, torch.float16, False, id="mixed-length-left-window"),
    pytest.param(4, [1500, 2000, 3000, 3500], [1500, 2000, 3000, 3500], 32, 8, 128, False, 1500, 1500, torch.float16, False, id="moderate-batch-window"),
    pytest.param(2, [512, 1024], [4096, 8192], 32, 8, 128, True, -1, -1, torch.float16, False, id="prefill-cache"),
    pytest.param(2, [5001, 8193], [5001, 8193], 32, 8, 128, True, -1, -1, torch.float16, False, id="long-sequence-stress"),
]


class GqaSlidingWindowVarlenFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        """Approximate FLOPs for QK^T and PV GEMMs, summed over all samples."""
        t = self.test
        total = 0.0
        for sq, sk in zip(t.seqlens_q, t.seqlens_k, strict=True):
            offset = sk - sq
            seq_attended = 0
            for q_pos in range(sq):
                hi = min(q_pos + offset, sk - 1) if t.is_causal else (
                    min(q_pos + offset + t.wr, sk - 1) if t.wr >= 0 else sk - 1)
                lo = max(0, q_pos + offset - t.wl) if t.wl >= 0 else 0
                seq_attended += max(0, hi - lo + 1)
            total += 4 * t.heads * seq_attended * t.dim
        return total

    def calculate_memory(self) -> Optional[float]:
        """Approximate bytes accessed: read Q/K/V, write O."""
        t = self.test
        elem = torch.tensor([], dtype=t.dtype).element_size()
        total_q = sum(t.seqlens_q)
        total_k = sum(t.seqlens_k)
        return (total_q * t.heads * t.dim +
                total_k * t.heads_kv * t.dim * 2 +
                total_q * t.heads * t.dim) * elem


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
            q_pad[i, :qe - qs] = q[qs:qe]
            k_pad[i, :ke - ks] = k[ks:ke]
            v_pad[i, :ke - ks] = v[ks:ke]

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
        attn_mask.masked_fill_(mask, float('-inf'))

        # SDPA call: transpose to [B, H, S, D], mask broadcasts as [B, 1, max_sq, max_sk]
        out = F.scaled_dot_product_attention(
            q_pad.transpose(1, 2), k_pad.transpose(1, 2), v_pad.transpose(1, 2),
            attn_mask=attn_mask.unsqueeze(1), enable_gqa=True)
        out = out.transpose(1, 2)  # [B, max_sq, H, D]

        # Repack valid positions to [total_q, H, D]
        parts = [out[i, :seqlens_q[i]] for i in range(B)]
        return torch.cat(parts, dim=0)
    return fn


def _fa3_varlen_baseline(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                         max_seqlen_k, is_causal, wl, wr):
    """FA3 varlen reference baseline."""
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: PLC0415
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=is_causal,
            window_size=(wl, wr),
        )
    except ImportError:
        return None


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
    test = GqaSlidingWindowVarlenFwdTest(
        batch, seqlens_q, seqlens_k, heads, heads_kv, dim, is_causal, wl, wr, dtype)
    bm = GqaSlidingWindowVarlenFwdBenchmark(test)
    inputs = test.gen_inputs()
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q = inputs

    op = GqaSlidingWindowVarlenFwdOp(
        batch=batch, heads=heads, heads_kv=heads_kv, dim=dim,
        is_causal=is_causal, window_size_left=wl, window_size_right=wr,
        dtype=dtype, tune=tune)

    # Warmup: trigger JIT compilation before timed profiling
    op(*inputs)
    torch.cuda.synchronize()

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # FA3 baseline
    max_seqlen_k = max(seqlens_k)
    fa3_out = _fa3_varlen_baseline(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        is_causal, wl, wr)
    if fa3_out is not None:
        result_bl = bm.profile(
            lambda q, k, v, csq, csk, msq: _fa3_varlen_baseline(
                q, k, v, csq, csk, msq, max_seqlen_k, is_causal, wl, wr),
            *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")

    # FlashInfer baseline
    fi_fn = _flashinfer_varlen_sliding_window_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_out is None and fi_fn is None:
        result_bl = bm.profile(_torch_sliding_window_varlen_fwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
