import pytest
import torch

from tests.test_base import FixtureBase
from tests.ops.test_gqa_decode import GqaDecodeTest
from tileops.functions import GroupQueryAttentionDecodeWithKVCacheFunc, gqa_decode_with_kvcache


class GqaDecodeFuncFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, seq_len_kv, dim, groups, dtype", [
            (1, 32, 8192, 128, 1, torch.float16),
        ]),
    ]


@GqaDecodeFuncFixture
def test_gqa_decode_fn(batch: int, heads: int, seq_len_kv: int, dim: int, groups: int,
                       dtype: torch.dtype):
    test = GqaDecodeTest(batch, heads, groups, seq_len_kv, dim, dtype)
    inputs = test.gen_inputs()

    print("=========Testing gqa decode function inference=========")
    test.check_fn(gqa_decode_with_kvcache, *inputs, grad=False, atol=3e-4, rtol=1e-5)

    print("=========Testing gqa decode function class=========")
    fn = GroupQueryAttentionDecodeWithKVCacheFunc(batch, heads, groups, seq_len_kv, dim, dtype)
    test.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
