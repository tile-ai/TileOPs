from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek_dsa_decode import DsaDecodeTest


class _DsaDecodeTestBaseline(DsaDecodeTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, q: torch.Tensor, kv: torch.Tensor,
                    indices: torch.Tensor) -> torch.Tensor:
        q = q.float()
        kv = kv.float()
        indices = indices.transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape
        q_start_index_s = self.q_start_index_s
        if self.q_start_index_s is None:
            q_start_index_s = sk * self.stride_kv - sq

        assert kv.shape[-1] == self.dim + self.dim_tail, 'you should assign dim otherwise'
        dim = self.dim
        k = kv
        v = kv[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        compressed_causal_mask = torch.arange(
            q_start_index_s, sq + q_start_index_s, dtype=torch.int32,
            device="cuda").view(-1, 1) >= torch.arange(
                self.stride_kv - 1,
                sk * self.stride_kv,
                self.stride_kv,
                dtype=torch.int32,
                device="cuda").view(1, -1)

        mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
        mask[:, :, :self.stride_kv - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        q = q.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        sm_scale = dim_q**-0.5 if self.sm_scale is None else self.sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        return o.to(torch.float16)


class DsaDecodeBenchmark(BenchmarkBase[DsaDecodeTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        return t.batch * t.seq_len * (2 * t.dim + t.dim_tail) * t.topk * 2 * t.heads

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        # q: batch, seq_len, heads, dim + dim_tail
        # kv: batch, seq_len_kv, heads_kv, dim + dim_tail
        # indices: batch, seq_len, heads_kv, topk
        # Output: batch, seq_len, heads, dim
        q_memory = (
            t.batch * t.seq_len * t.heads * (t.dim + t.dim_tail) * t.dtype.itemsize)
        kv_memory = t.batch * t.seq_len_kv * t.heads_kv * (
            t.dim + t.dim_tail) * t.dtype.itemsize
        indices_memory = t.batch * t.seq_len * t.heads_kv * t.topk * 4  # int32
        output_memory = t.batch * t.seq_len * t.heads * t.dim * t.dtype.itemsize
        return q_memory + kv_memory + indices_memory + output_memory


_DSA_DECODE_BENCH_PARAMS = [
    pytest.param(
        1, 128, 1024, 2048, 512, 64, 2048, 1, 1, 1024, None, torch.float16, False,
        id="single-batch-mainstream",
    ),
    pytest.param(
        1, 128, 512, 4096, 512, 64, 1024, 1, 1, 512, None, torch.float16, False,
        id="longer-kv-lower-topk",
    ),
]


@pytest.mark.parametrize(
    "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start_index_s, sm_scale, dtype, tune",
    _DSA_DECODE_BENCH_PARAMS,
)
def test_dsa_decode_bench(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                          dim_tail: int, topk: int, stride_kv: int, heads_kv: int,
                          q_start_index_s: int, sm_scale: float, dtype: torch.dtype,
                          tune: bool) -> None:
    test = _DsaDecodeTestBaseline(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    bm = DsaDecodeBenchmark(test)
    inputs = test.gen_inputs()

    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype, tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
