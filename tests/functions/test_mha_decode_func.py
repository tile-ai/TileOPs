import pytest
import torch

from tests.test_base import FixtureBase
from tests.ops.test_mha_decode import MhaDecodeTest
from tileops.functions import MultiHeadAttentionDecodeWithKVCacheFunc, mha_decode_with_kvcache


class MhaDecodeFuncFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len_q, seq_len_kv, heads, dim, dtype", [
            (1, 128, 8192, 32, 128, torch.float16),
        ]),
    ]


@MhaDecodeFuncFixture
def test_mha_decode_fn(batch: int, seq_len_q: int, seq_len_kv: int, heads: int, dim: int,
                       dtype: torch.dtype):
    test = MhaDecodeTest(batch, heads, seq_len_q, seq_len_kv, dim, dtype)
    inputs = test.gen_inputs()

    print("=========Testing mha decode function inference=========")
    test.check_fn(mha_decode_with_kvcache, *inputs, grad=False, atol=3e-4, rtol=1e-5)

    print("=========Testing mha decode function class=========")
    fn = MultiHeadAttentionDecodeWithKVCacheFunc(batch, heads, seq_len_q, seq_len_kv, dim, dtype)
    test.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
