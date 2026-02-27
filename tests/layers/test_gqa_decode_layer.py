import pytest
import torch

from tests.ops.test_gqa_decode import GqaDecodeTest
from tests.test_base import FixtureBase
from tileops.layers import GroupQueryAttentionDecodeLayer


class GqaDecodeLayerFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, seq_len_kv, dim, groups, dtype", [
            (1, 32, 8192, 128, 1, torch.float16),
        ]),
    ]


@GqaDecodeLayerFixture
def test_gqa_decode_layer(batch: int, heads: int, seq_len_kv: int, dim: int, groups: int,
                          dtype: torch.dtype):
    fn = GroupQueryAttentionDecodeLayer(batch, heads, groups, seq_len_kv, dim, dtype)
    test = GqaDecodeTest(batch, heads, groups, seq_len_kv, dim, dtype)
    inputs = test.gen_inputs()
    test.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
