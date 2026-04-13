from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import NSAFwdVarlenOp
from workloads.attention.deepseek_nsa import NsaFwdTest


class _NsaFwdTestBaseline(NsaFwdTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    block_indices: torch.Tensor, block_counts: torch.Tensor,
                    offsets: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        _ = token_indices
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        block_indices = block_indices.unsqueeze(0)
        block_counts = block_counts.unsqueeze(0)
        return self.naive_nsa(
            q=q,
            k=k,
            v=v,
            g_slc=self.g_slc,
            g_swa=self.g_swa,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=self.block_size,
            window_size=0,
            scale=self.scale,
            cu_seqlens=offsets,
            head_first=False,
        )


class NsaFwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        flops_per_token = 4 * t.dim * t.selected_blocks * t.block_size
        return flops_per_token * t.c_seq_len * t.heads

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        # q, k, v, output, block_indices, block_counts, offsets, token_indices
        # ignore block counts, offsets and token_indices memory
        q_memory = t.heads * t.c_seq_len * t.dim * t.dtype.itemsize
        k_memory = t.head_kv * t.c_seq_len * t.dim * t.dtype.itemsize
        v_memory = t.head_kv * t.c_seq_len * t.dim * t.dtype.itemsize
        output_memory = t.heads * t.c_seq_len * t.dim * t.dtype.itemsize
        block_indices_memory = t.head_kv * t.c_seq_len * t.selected_blocks * 4
        return (q_memory + k_memory + v_memory + output_memory + block_indices_memory)


_NSA_FWD_BENCH_PARAMS = [
    pytest.param(
        1, 16, 1024, 64, True, 0.1, 32, 16, 1, torch.float16, torch.float32, False, id="single-block",
    ),
    pytest.param(
        4, 16, 8192, 64, True, 0.1, 32, 16, 1, torch.float16, torch.float32, False, id="long-context",
    ),
    pytest.param(
        2, 16, 8192, 64, True, 0.1, 32, 16, 4, torch.float16, torch.float32, False, id="multi-selected-blocks",
    ),
]


@pytest.mark.parametrize(
    "batch, heads, c_seq_len, dim, is_causal, scale, block_size, groups, selected_blocks, dtype, accum_dtype, tune",
    _NSA_FWD_BENCH_PARAMS,
)
def test_nsa_fwd_bench(batch: int, heads: int, c_seq_len: int, dim: int, is_causal: bool,
                       scale: float, block_size: int, groups: int, selected_blocks: int,
                       dtype: torch.dtype, accum_dtype: torch.dtype, tune: bool) -> None:
    test = _NsaFwdTestBaseline(batch, heads, c_seq_len, dim, is_causal, scale, block_size, groups,
                      selected_blocks, dtype, accum_dtype)
    bm = NsaFwdBenchmark(test)
    inputs = test.gen_inputs()

    op = NSAFwdVarlenOp(
        batch=batch, heads=heads, c_seq_len=c_seq_len, dim=dim,
        is_causal=is_causal, scale=scale, block_size=block_size,
        groups=groups, selected_blocks=selected_blocks, dtype=dtype,
        accum_dtype=accum_dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # Use reduced warmup/rep for the slow Python-loop baseline to avoid timeouts.
    result_bl = bm.profile(test.ref_program, *inputs, warmup=5, rep=10)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
