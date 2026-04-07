from typing import Optional, Tuple

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import Fp8LightingIndexerOp
from workloads.ops.fp8_lighting_indexer import Fp8LightingIndexerTest


class _Fp8LightingIndexerTestBaseline(Fp8LightingIndexerTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                    cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> Tuple[torch.Tensor]:
        k = kv
        q = q.float()
        k = k.float()
        batch, seq_len, heads, index_dim = q.shape
        seq_len_kv = self.seq_len_kv
        kv_group = self.kv_group
        heads_per_group = heads // kv_group

        k = k.view(batch, seq_len_kv, kv_group, index_dim)
        q = q.view(batch, seq_len, kv_group, heads_per_group, index_dim)

        mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum("bsghd,bngd->bghsn", q, k)
        weights = weights.view(seq_len, kv_group, heads_per_group)
        weights = weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(-1)
        score = score.relu() * weights
        logits = score.sum(dim=2)
        logits = logits.permute(0, 2, 3, 1)
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1)
        logits = logits.masked_fill(~mask_expanded, float("-inf"))
        return (logits,)


class Fp8LightingIndexerBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        # Flops depend on the actual mask cost which varies per input
        return None

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        dtype = torch.float8_e4m3fn
        accum_dtype = torch.float32
        index_dtype = torch.int32

        index_q_memory = t.batch * t.seq_len * t.heads * t.index_dim * dtype.itemsize
        index_k_memory = t.batch * t.seq_len_kv * t.index_dim * t.kv_group * dtype.itemsize
        index_k_scale_memory = t.batch * t.seq_len_kv * t.kv_group * accum_dtype.itemsize
        logits_memory = t.batch * t.seq_len * t.seq_len_kv * t.kv_group * accum_dtype.itemsize
        weights_memory = t.seq_len * t.heads * accum_dtype.itemsize
        cu_seqlens_ks_memory = t.seq_len * index_dtype.itemsize
        cu_seqlens_ke_memory = t.seq_len * index_dtype.itemsize

        return (index_q_memory + index_k_memory + index_k_scale_memory + logits_memory +
                weights_memory + cu_seqlens_ks_memory + cu_seqlens_ke_memory)


_FP8_LIGHTING_INDEXER_BENCH_PARAMS = [
    pytest.param(1, 4096, 32, 64, 8192, 1, True, None, False, id="default-config"),
    pytest.param(1, 2048, 16, 64, 4096, 1, True, None, False, id="mid-shape"),
]


@pytest.mark.parametrize(
    "batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, config, tune",
    _FP8_LIGHTING_INDEXER_BENCH_PARAMS,
)
def test_fp8_lighting_indexer_bench(batch: int, seq_len: int, heads: int, index_dim: int,
                                    seq_len_kv: int, kv_group: int, clean_logits: bool,
                                    config: Optional[dict], tune: bool) -> None:
    test = _Fp8LightingIndexerTestBaseline(batch, seq_len, heads, index_dim, seq_len_kv, kv_group,
                                  clean_logits, config)
    bm = Fp8LightingIndexerBenchmark(test)
    inputs = test.gen_inputs()

    op = Fp8LightingIndexerOp(batch=batch,
                              seq_len=seq_len,
                              heads=heads,
                              index_dim=index_dim,
                              seq_len_kv=seq_len_kv,
                              kv_group=kv_group,
                              clean_logits=clean_logits,
                              config=config,
                              tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
