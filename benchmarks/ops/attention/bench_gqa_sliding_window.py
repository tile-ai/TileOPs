"""Benchmark for GroupedQueryAttentionSlidingWindowFwdOp vs FA3 baseline."""

import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import (
    gqa_sliding_window_args,
    manifest_params,
)
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionSlidingWindowFwdOp
from workloads.attention.gqa_sliding_window import GroupedQueryAttentionSlidingWindowFwdTest

_OP_NAME = "GroupedQueryAttentionSlidingWindowFwdOp"


def _torch_sliding_window_fwd(test):
    """Torch SDPA forward baseline with explicit sliding window mask."""

    def fn(q, k, v):
        S = test.seq
        q_idx = torch.arange(S, device=q.device).unsqueeze(1)
        k_idx = torch.arange(S, device=q.device).unsqueeze(0)
        mask = torch.zeros(S, S, dtype=torch.bool, device=q.device)
        if test.is_causal:
            mask |= k_idx > q_idx
        if test.wl >= 0:
            mask |= k_idx < q_idx - test.wl
        if test.wr >= 0:
            mask |= k_idx > q_idx + test.wr
        attn_mask = torch.zeros(S, S, dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(mask, float("-inf"))
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            enable_gqa=True,
        )
        return out.transpose(1, 2)

    return fn


def _fa3_baseline(is_causal, wl, wr):
    """Return FA3 sliding-window baseline callable, or None if not installed."""
    try:
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v):
        out = flash_attn_func(q, k, v, causal=is_causal, window_size=(wl, wr))
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _flashinfer_sliding_window_fwd(test, q, k, v):
    """Set up FlashInfer batched prefill with sliding window. Returns callable or None.

    FlashInfer only supports window_left; skip when window_right >= 0.
    """
    if test.wr >= 0:
        return None
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    B, S, H, D = q.shape
    Hkv = k.shape[2]
    cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * S

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=cu_seqlens,
        kv_indptr=cu_seqlens,
        num_qo_heads=H,
        num_kv_heads=Hkv,
        head_dim_qk=D,
        causal=test.is_causal,
        window_left=test.wl,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        return wrapper.run(
            q.reshape(-1, H, D),
            k.reshape(-1, Hkv, D),
            v.reshape(-1, Hkv, D),
        ).reshape(B, S, H, D)

    return run_fn


_GQA_SLIDING_WINDOW_FWD_BENCH_PARAMS = manifest_params(
    load_workloads(_OP_NAME),
    gqa_sliding_window_args,
)


@pytest.mark.parametrize(
    "batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype, tune",
    _GQA_SLIDING_WINDOW_FWD_BENCH_PARAMS,
)
def test_gqa_sliding_window_fwd_bench(
    batch: int,
    seq: int,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    wl: int,
    wr: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionSlidingWindowFwdTest(
        batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype
    )
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionSlidingWindowFwdOp(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len=seq,
        dim=dim,
        is_causal=is_causal,
        window_size_left=wl,
        window_size_right=wr,
        dtype=dtype,
        tune=tune,
    )
    bm = ManifestBenchmark(_OP_NAME, op, test)

    # Warmup: trigger JIT compilation before timed profiling
    op(*inputs)
    torch.cuda.synchronize()

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # FA3 baseline
    fa3_fn = _fa3_baseline(is_causal, wl, wr)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")

    # FlashInfer baseline
    fi_fn = _flashinfer_sliding_window_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_fn is None and fi_fn is None:
        result_bl = bm.profile(_torch_sliding_window_fwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
