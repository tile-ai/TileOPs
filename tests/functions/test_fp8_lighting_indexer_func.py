from typing import Optional

import pytest

from tests.test_base import FixtureBase
from tests.ops.test_fp8_lighting_indexer import Fp8LightingIndexerTest
from tileops.functions import Fp8LightingIndexerFunc
from tileops.layers import Fp8LightingIndexerDecodeLayer


class Fp8LightingIndexerFuncFixture(FixtureBase):
    PARAMS = [
        ("seq_len, heads, index_dim, seq_len_kv, clean_logits, config", [
            (4096, 32, 64, 8192, True, None),
        ]),
    ]


@Fp8LightingIndexerFuncFixture
def test_fp8_lighting_indexer(seq_len: int, heads: int, index_dim: int, seq_len_kv: int,
                              clean_logits: bool, config: Optional[dict]) -> None:
    test = Fp8LightingIndexerTest(seq_len, heads, index_dim, seq_len_kv, clean_logits, config)
    inputs = test.gen_inputs()

    print("Testing indexer_fn...")
    fn = Fp8LightingIndexerFunc(seq_len, heads, index_dim, seq_len_kv, clean_logits, config)
    test.check_fn(fn, *inputs, grad=False)

    print("Testing indexer_layer...")
    layer = Fp8LightingIndexerDecodeLayer(seq_len, heads, index_dim, seq_len_kv, clean_logits,
                                          config)
    test.check_fn(layer, *inputs, grad=False)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
