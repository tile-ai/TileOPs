import pytest
import torch
import torch.nn.functional as F
from einops import einsum, rearrange

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params, mla_decode_args
from tileops.manifest import load_workloads
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek_mla_decode import MlaDecodeTest

_OP_NAME = "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"


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


_MLA_DECODE_BENCH_PARAMS = manifest_params(load_workloads(_OP_NAME), mla_decode_args)


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune",
    _MLA_DECODE_BENCH_PARAMS,
)
def test_mla_decode_bench(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                          dim_pe: int, dtype: torch.dtype, tune: bool) -> None:
    test = _MlaDecodeTestBaseline(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    inputs = test.gen_inputs()

    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
