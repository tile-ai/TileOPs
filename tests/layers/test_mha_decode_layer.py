import pytest
import torch

from tests.ops.test_mha_decode import MhaDecodeTest
from tests.test_base import FixtureBase
from tileops.layers import MultiHeadAttentionDecodeLayer


class MhaDecodeLayerFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len_q, seq_len_kv, heads, dim, dtype", [
            (1, 128, 8192, 32, 128, torch.float16),
        ]),
    ]


@MhaDecodeLayerFixture
def test_mha_decode_layer(batch: int, seq_len_q: int, seq_len_kv: int, heads: int, dim: int,
                          dtype: torch.dtype):
    fn = MultiHeadAttentionDecodeLayer(batch, heads, seq_len_q, seq_len_kv, dim, dtype)
    test = MhaDecodeTest(batch, heads, seq_len_q, seq_len_kv, dim, dtype)
    inputs = test.gen_inputs()
    test.check_fn(fn, *inputs, grad=False, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
