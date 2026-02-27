import pytest
import torch

from tests.functions.test_gqa_func import GqaFuncTest
from tests.test_base import FixtureBase
from tileops.layers import GroupQueryAttentionLayer


class GqaLayerFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, heads_kv, dim, causal, dtype", [
            (8, 1024, 32, 32, 128, False, torch.float16),
        ]),
    ]


@GqaLayerFixture
def test_gqa_layer(batch: int, seq_len: int, heads: int, heads_kv: int, dim: int, causal: bool,
                   dtype: torch.dtype) -> None:
    gqa = GroupQueryAttentionLayer(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    test = GqaFuncTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()
    test.check_fn(gqa, *inputs, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
