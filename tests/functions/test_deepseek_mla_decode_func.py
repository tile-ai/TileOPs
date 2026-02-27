import pytest
import torch

from tests.test_base import FixtureBase
from tests.ops.test_deepseek_mla_decode import MlaDecodeTest
from tileops.functions import MultiHeadLatentAttentionDecodeWithKVCacheFunc, mla_decode_with_kvcache
from tileops.layers import MultiHeadLatentAttentionDecodeLayer


class MlaDecodeFuncFixture(FixtureBase):
    PARAMS = [
        ("batch, kv_head_num, seq_len_kv, heads, dim, pe_dim, dtype", [
            (32, 1, 8192, 128, 512, 64, torch.float16),
        ]),
    ]


@MlaDecodeFuncFixture
def test_mla_decode_fn(batch: int, kv_head_num: int, seq_len_kv: int, heads: int, dim: int,
                       pe_dim: int, dtype: torch.dtype):
    test = MlaDecodeTest(batch, heads, kv_head_num, seq_len_kv, dim, pe_dim, dtype)
    inputs = test.gen_inputs()

    print("=========Testing mla_fn interface=========")
    test.check_fn(mla_decode_with_kvcache, *inputs, grad=False, atol=3e-4, rtol=1e-5)

    print("=========Testing mla_fn class=========")
    fn = MultiHeadLatentAttentionDecodeWithKVCacheFunc(batch, heads, kv_head_num, seq_len_kv, dim,
                                                       pe_dim, dtype)
    test.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)

    print("=========Testing mla_layer=========")
    mla_layer = MultiHeadLatentAttentionDecodeLayer(batch, heads, kv_head_num, seq_len_kv, dim,
                                                    pe_dim, dtype)
    test.check_fn(mla_layer, *inputs, grad=False, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
