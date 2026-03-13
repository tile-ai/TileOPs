"""Benchmark for GqaSlidingWindowFwdOp vs FA3 baseline."""
from typing import Optional

import pytest
import torch

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


def _fa3_baseline(q, k, v, is_causal, wl, wr):
    """FA3 reference baseline."""
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415
        return flash_attn_func(q, k, v, causal=is_causal, window_size=(wl, wr))
    except ImportError:
        return None


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
