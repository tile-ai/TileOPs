"""Benchmark for GqaSlidingWindowFwdOp vs FA3 baseline."""
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_gqa_sliding_window_fwd import (
    GqaSlidingWindowFwdFixture,
    GqaSlidingWindowFwdTest,
)
from tileops.ops import GqaSlidingWindowFwdOp


class GqaSlidingWindowFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        """Approximate FLOPs for QK^T and PV GEMMs."""
        t = self.test
        if t.is_causal:
            eff_kv = t.seq // 2
        elif t.wl >= 0 or t.wr >= 0:
            wl = t.wl if t.wl >= 0 else t.seq
            wr = t.wr if t.wr >= 0 else t.seq
            eff_kv = min(t.seq, wl + wr + 1)
        else:
            eff_kv = t.seq
        return 4 * t.batch * t.heads * t.seq * eff_kv * t.dim

    def calculate_memory(self) -> Optional[float]:
        """Approximate bytes accessed: read Q/K/V, write O."""
        t = self.test
        elem = torch.tensor([], dtype=t.dtype).element_size()
        return (3 * t.batch * t.seq * t.heads_kv * t.dim +
                t.batch * t.seq * t.heads * t.dim) * elem


def _fa3_baseline(q, k, v, is_causal, wl, wr):
    """FA3 reference baseline."""
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415
        return flash_attn_func(q, k, v, causal=is_causal, window_size=(wl, wr))
    except ImportError:
        return None


@GqaSlidingWindowFwdFixture
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
    BenchmarkReport.record("gqa_sliding_window_fwd", locals(), result, tag="tileops")

    # FA3 baseline
    q, k, v = inputs
    fa3_out = _fa3_baseline(q, k, v, is_causal, wl, wr)
    if fa3_out is not None:
        result_bl = bm.profile(
            lambda q, k, v: _fa3_baseline(q, k, v, is_causal, wl, wr), *inputs)
        BenchmarkReport.record("gqa_sliding_window_fwd", locals(), result_bl, tag="fa3")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
