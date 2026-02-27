import pytest
import torch

from tests.test_base import FixtureBase
from tests.ops.test_deepseek_dsa_decode import DsaDecodeTest
from tileops.functions import DeepSeekSparseAttentionDecodeWithKVCacheFunc
from tileops.layers import DeepSeekSparseAttentionDecodeLayer


class DsaDecodeFuncFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, group_kv, "
         "q_start_index_s, sm_scale, dtype, tune", [
             (1, 128, 1024, 2048, 512, 64, 2048, 1, 1, 1024, None, torch.float16, False),
         ]),
    ]


@DsaDecodeFuncFixture
def test_sparse_mla_decode(batch: int, heads: int, seq_len_q: int, seq_len_kv: int, dim: int,
                           dim_tail: int, topk: int, stride_kv: int, group_kv: int,
                           q_start_index_s: int, sm_scale: float, dtype: torch.dtype, tune: bool):
    test = DsaDecodeTest(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, group_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype)
    inputs = test.gen_inputs()

    print("=========Testing dsa_fn=========")
    fn = DeepSeekSparseAttentionDecodeWithKVCacheFunc(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, group_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype, tune=tune)
    test.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)

    print("=========Testing dsa_layer=========")
    layer = DeepSeekSparseAttentionDecodeLayer(
        batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, group_kv,
        q_start_index_s, sm_scale=sm_scale, dtype=dtype, tune=tune)
    test.check_fn(layer, *inputs, grad=False, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
