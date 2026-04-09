from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from einops import einsum, rearrange

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import DeepSeekMlaDecodeFwdOp
from workloads.ops.deepseek_mla_decode import MlaDecodeTest


class _MlaDecodeTestBaseline(MlaDecodeTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, q: torch.Tensor, q_pe: torch.Tensor, kv: torch.Tensor,
                    k_pe: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
        - q (Tensor): [batch, heads, dim]
        - q_pe (Tensor): [batch, heads, dim_pe]
        - kv (Tensor): [batch, seqlen_kv, heads_kv, dim]
        - k_pe (Tensor): [batch, seqlen_kv, heads_kv, dim_pe]
        Outputs:
        - output (Tensor): [batch, heads, dim]
        """
        dim = q.shape[-1]
        dim_pe = q_pe.shape[-1]
        num_head_groups = q.shape[1] // kv.shape[2]
        scale = (dim + dim_pe)**0.5
        Q = rearrange(
            q, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

        Q_pe = rearrange(
            q_pe, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim_pe]

        KV = rearrange(kv, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

        K_pe = rearrange(k_pe,
                         'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, dim_pe]

        query = torch.concat([Q, Q_pe], dim=-1)
        key = torch.concat([KV, K_pe], dim=-1)

        scores = einsum(
            query, key,
            'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

        attention = F.softmax(
            scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

        out = einsum(attention, KV,
                     'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
        out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
        return out


class MlaDecodeBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        # qk_flops = 2 * batch * heads * seq_len_kv * (dim + dim_pe)
        # pv_flops = 2 * batch * heads * seq_len_kv * dim
        qk_flops = 2 * t.batch * t.heads * t.seq_len_kv * (t.dim + t.dim_pe)
        pv_flops = 2 * t.batch * t.heads * t.seq_len_kv * t.dim
        return qk_flops + pv_flops

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        # Q: batch * heads * dim
        # Q_pe: batch * heads * dim_pe
        # K: batch * seq_len_kv * heads_kv * dim
        # K_pe: batch * seq_len_kv * heads_kv * dim_pe
        # Output: batch * heads * dim
        return t.batch * (t.heads + t.seq_len_kv * t.heads_kv) * (
            t.dim + t.dim_pe) * t.dtype.itemsize


_MLA_DECODE_BENCH_PARAMS = [
    pytest.param(32, 128, 1, 8192, 512, 64, torch.float16, True, id="mainstream-fp16"),
    pytest.param(16, 128, 1, 4096, 512, 64, torch.float16, True, id="mid-cache-fp16"),
]


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune",
    _MLA_DECODE_BENCH_PARAMS,
)
def test_mla_decode_bench(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                          dim_pe: int, dtype: torch.dtype, tune: bool) -> None:
    test = _MlaDecodeTestBaseline(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    bm = MlaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = DeepSeekMlaDecodeFwdOp(
        batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
