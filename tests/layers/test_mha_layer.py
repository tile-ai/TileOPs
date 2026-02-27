import pytest
import torch

from tests.functions.test_mha_func import MhaFuncTest
from tests.test_base import FixtureBase
from tileops.layers import MultiHeadAttentionLayer


class MhaLayerFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim, causal, dtype", [
            (8, 1024, 32, 128, False, torch.float16),
        ]),
    ]


@MhaLayerFixture
def test_mha_layer(batch: int, seq_len: int, heads: int, dim: int, causal: bool,
                   dtype: torch.dtype) -> None:
    mha = MultiHeadAttentionLayer(batch, heads, seq_len, dim, causal, dtype)
    test = MhaFuncTest(batch, heads, seq_len, dim, causal, dtype)
    inputs = test.gen_inputs()
    test.check_fn(mha, *inputs, atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
