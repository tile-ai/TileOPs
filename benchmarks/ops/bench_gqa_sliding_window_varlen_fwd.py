"""Benchmark for GqaSlidingWindowVarlenFwdOp vs FA3 baseline."""
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gqa_sliding_window_varlen_fwd import GqaSlidingWindowVarlenFwdTest
from tests.test_base import FixtureBase
from tileops.ops import GqaSlidingWindowVarlenFwdOp


@pytest.fixture(scope="session", autouse=True)
def warmup_cupti():
    """Pre-initialize the CUPTI profiler once per session.

    The first torch.profiler.profile() call with CUDA activity tracking
    incurs a one-time initialization cost.  If this happens inside do_bench's
    estimation phase, estimate_ms is inflated and n_repeat is computed as 1,
    causing the measured latency to include initialization overhead.
    """
    if not torch.cuda.is_available():
        return
    dummy = torch.empty(1, device="cuda")
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
    ) as prof:
        for _ in range(2):
            dummy.zero_()
            prof.step()
    torch.cuda.synchronize()


class GqaSlidingWindowVarlenFwdBenchFixture(FixtureBase):
    """Long-sequence configs for throughput benchmarking."""
    PARAMS = [
        ("batch, seqlens_q, seqlens_k, heads, heads_kv, dim,"
         " is_causal, wl, wr, dtype, tune", [
             # ── batch=1: single sequence ──────────────────────────────────────
             (1, [3000],                [3000],                32, 8, 128, True,  -1,   -1, torch.float16, False),
             (1, [6001],                [6001],                32, 8, 128, True,  3000, -1, torch.float16, False),
             (1, [5999],                [5999],                32, 8, 128, False, 1500, 1500, torch.float16, False),
             # ── batch=2: mixed non-power-of-2 lengths ────────────────────────
             (2, [1537, 3073],          [1537, 3073],          32, 8, 128, True,  -1,   -1, torch.float16, False),
             (2, [1537, 3073],          [1537, 3073],          32, 8, 128, True,  2000, -1, torch.float16, False),
             (2, [1537, 3073],          [1537, 3073],          32, 8, 128, False, 1000, 1000, torch.float16, False),
             # ── batch=4: moderate parallelism ─────────────────────────────────
             (4, [1500, 2000, 3000, 3500], [1500, 2000, 3000, 3500], 32, 8, 128, True,  -1,   -1, torch.float16, False),
             (4, [1500, 2000, 3000, 3500], [1500, 2000, 3000, 3500], 32, 8, 128, True,  2000, -1, torch.float16, False),
             (4, [1500, 2000, 3000, 3500], [1500, 2000, 3000, 3500], 32, 8, 128, False, 1500, 1500, torch.float16, False),
             # ── batch=8: high parallelism ─────────────────────────────────────
             (8, [999, 1200, 1537, 1800, 2100, 2500, 3001, 3500],
                 [999, 1200, 1537, 1800, 2100, 2500, 3001, 3500],
                 32, 8, 128, True,  -1,   -1, torch.float16, False),
             (8, [999, 1200, 1537, 1800, 2100, 2500, 3001, 3500],
                 [999, 1200, 1537, 1800, 2100, 2500, 3001, 3500],
                 32, 8, 128, False, 1500, 1500, torch.float16, False),
             # ── KV-cache decode: seqlen_q=1 (memory-bound) ───────────────────
             (2, [1, 1],       [4096, 8192], 32, 8, 128, True,  -1,   -1, torch.float16, False),
             # ── KV-cache prefill+cache (memory-bound) ─────────────────────────
             (2, [512, 1024],  [4096, 8192], 32, 8, 128, True,  -1,   -1, torch.float16, False),
             (2, [512, 1024],  [4096, 8192], 32, 8, 128, True,  2000, -1, torch.float16, False),
             (4, [513, 700, 1025, 1200], [3001, 4000, 6001, 7000],
                 32, 8, 128, True, 2000, -1, torch.float16, False),
             # ── Long sequence stress: non-power-of-2 ─────────────────────────
             (2, [5001, 8193],          [5001, 8193],          32, 8, 128, True,  -1,   -1, torch.float16, False),
             (2, [5001, 8193],          [5001, 8193],          32, 8, 128, False, 3000, 3000, torch.float16, False),
         ]),
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


@GqaSlidingWindowVarlenFwdBenchFixture
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
    BenchmarkReport.record("gqa_sliding_window_varlen_fwd", locals(), result, tag="tileops")

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
        BenchmarkReport.record(
            "gqa_sliding_window_varlen_fwd", locals(), result_bl, tag="fa3")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
