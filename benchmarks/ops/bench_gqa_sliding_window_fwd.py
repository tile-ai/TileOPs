"""Benchmark for GqaSlidingWindowFwdOp vs FA3 baseline."""
from typing import Optional

import pytest
import torch
from torch.nn import functional as F

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gqa_sliding_window_fwd import GqaSlidingWindowFwdTest
from tileops.ops import GqaSlidingWindowFwdOp


class GqaSlidingWindowFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        """Approximate FLOPs for QK^T and PV GEMMs."""
        t = self.test
        S = t.seq
        wl, wr = t.wl, t.wr
        total_attended = 0
        for q in range(S):
            hi = q if t.is_causal else (min(S - 1, q + wr) if wr >= 0 else S - 1)
            lo = max(0, q - wl) if wl >= 0 else 0
            total_attended += hi - lo + 1
        return 4 * t.batch * t.heads * total_attended * t.dim

    def calculate_memory(self) -> Optional[float]:
        """Approximate bytes accessed: read Q/K/V, write O."""
        t = self.test
        elem = torch.tensor([], dtype=t.dtype).element_size()
        return 2 * t.batch * t.seq * (t.heads + t.heads_kv) * t.dim * elem


def _torch_sliding_window_fwd(test):
    """Torch SDPA forward baseline with explicit sliding window mask."""
    def fn(q, k, v):
        S = test.seq
        q_idx = torch.arange(S, device=q.device).unsqueeze(1)
        k_idx = torch.arange(S, device=q.device).unsqueeze(0)
        mask = torch.zeros(S, S, dtype=torch.bool, device=q.device)
        if test.is_causal:
            mask |= (k_idx > q_idx)
        if test.wl >= 0:
            mask |= (k_idx < q_idx - test.wl)
        if test.wr >= 0:
            mask |= (k_idx > q_idx + test.wr)
        attn_mask = torch.zeros(S, S, dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(mask, float('-inf'))
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask, enable_gqa=True)
        return out.transpose(1, 2)
    return fn


def _fa3_baseline(q, k, v, is_causal, wl, wr):
    """FA3 reference baseline."""
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415
        return flash_attn_func(q, k, v, causal=is_causal, window_size=(wl, wr))
    except ImportError:
        return None


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
        qo_indptr=cu_seqlens, kv_indptr=cu_seqlens,
        num_qo_heads=H, num_kv_heads=Hkv, head_dim_qk=D,
        causal=test.is_causal,
        window_left=test.wl,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        return wrapper.run(
            q.reshape(-1, H, D), k.reshape(-1, Hkv, D), v.reshape(-1, Hkv, D),
        ).reshape(B, S, H, D)

    return run_fn


_GQA_SLIDING_WINDOW_FWD_BENCH_PARAMS = [
    pytest.param(2, 512, 8, 2, 64, True, -1, -1, torch.float16, True, id="causal-mainstream"),
    pytest.param(2, 512, 8, 2, 64, True, 128, -1, torch.float16, True, id="causal-left-window"),
    pytest.param(2, 768, 8, 2, 64, False, 256, -1, torch.float16, True, id="bidirectional-long"),
    pytest.param(2, 512, 8, 2, 64, False, 64, 64, torch.bfloat16, True, id="window-bf16"),
    pytest.param(1, 2048, 8, 2, 64, True, 512, -1, torch.float16, True, id="long-sequence"),
]


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
    test = GqaSlidingWindowFwdTest(batch, seq, heads, heads_kv, dim, is_causal, wl, wr, dtype)
    bm = GqaSlidingWindowFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = GqaSlidingWindowFwdOp(
        batch=batch, heads=heads, heads_kv=heads_kv, seq_len=seq, dim=dim,
        is_causal=is_causal, window_size_left=wl, window_size_right=wr,
        dtype=dtype, tune=tune)

    # Warmup: trigger JIT compilation before timed profiling
    op(*inputs)
    torch.cuda.synchronize()

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # FA3 baseline
    q, k, v = inputs
    fa3_out = _fa3_baseline(q, k, v, is_causal, wl, wr)
    if fa3_out is not None:
        result_bl = bm.profile(
            lambda q, k, v: _fa3_baseline(q, k, v, is_causal, wl, wr), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")

    # FlashInfer baseline
    fi_fn = _flashinfer_sliding_window_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_out is None and fi_fn is None:
        result_bl = bm.profile(_torch_sliding_window_fwd(test), *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
