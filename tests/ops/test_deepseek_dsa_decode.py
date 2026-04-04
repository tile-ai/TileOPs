
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheOp
from workloads.ops.deepseek_dsa_decode import DsaDecodeTest as _DsaDecodeTestWorkload


class DsaDecodeTest(_DsaDecodeTestWorkload, TestBase):
    pass


class DsaDecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, "
         "q_start_index_s, sm_scale, dtype, tune", [
             pytest.param(
                 1, 128, 1024, 2048, 512, 64, 2048, 1, 1, 1024, None, torch.float16, False,
                 marks=pytest.mark.smoke,
             ),
         ]),
    ]


@DsaDecodeFixture
def test_sparse_mla_decode(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                           dim_tail: int, topk: int, stride_kv: int, heads_kv: int,
                           q_start_index_s: int, sm_scale: float, dtype: torch.dtype,
                           tune: bool) -> None:
    test = DsaDecodeTest(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    op = DeepSeekSparseAttentionDecodeWithKVCacheOp(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype, tune=tune)
    test.check(op, *test.gen_inputs(), atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
