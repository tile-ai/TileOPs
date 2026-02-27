import torch
import pytest

from tests.test_base import FixtureBase
from tests.ops.test_topk_selector import TopkSelectorTest
from tileops.functions import TopkSelectorFunc
from tileops.layers import TopkSelectorLayer


class TopkSelectorFuncFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, topk, in_dtype, out_dtype, tune", [
            (64, 32 * 1024, 2048, torch.float32, torch.int32, False),
        ]),
    ]


@TopkSelectorFuncFixture
def test_topk_selector(batch: int, seq_len: int, topk: int, in_dtype: torch.dtype,
                       out_dtype: torch.dtype, tune: bool) -> None:
    test = TopkSelectorTest(batch, seq_len, topk, in_dtype, out_dtype)
    inputs = test.gen_inputs()

    print("Testing topk_selector_fn...")
    fn = TopkSelectorFunc(batch, seq_len, topk, in_dtype, out_dtype, tune=tune)
    test.check_fn(fn, *inputs, grad=False)

    print("Testing topk_selector_layer...")
    layer = TopkSelectorLayer(batch, seq_len, topk, in_dtype, out_dtype, tune=tune)
    test.check_fn(layer, *inputs, grad=False)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
