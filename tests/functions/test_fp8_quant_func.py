import torch
import pytest

from tests.test_base import FixtureBase
from tests.ops.test_fp8_quant import Fp8QuantTest
from tileops.functions import Fp8QuantFunc
from tileops.layers import Fp8QuantLayer


class Fp8QuantFuncFixture(FixtureBase):
    PARAMS = [
        ("seq_len_kv, index_dim, in_dtype, tune", [
            (8192, 64, torch.float16, False),
            (8192, 64, torch.bfloat16, False),
            (4096, 128, torch.float32, False),
            (16384, 32, torch.float32, False),
        ]),
    ]


@Fp8QuantFuncFixture
def test_fp8_quant(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype,
                   tune: bool) -> None:
    test = Fp8QuantTest(seq_len_kv, index_dim, in_dtype)
    inputs = test.gen_inputs()

    print("Testing fp8_quant_fn...")
    fn = Fp8QuantFunc(seq_len_kv, index_dim, in_dtype, tune=tune)
    test.check_fn(fn, *inputs, grad=False)

    print("Testing fp8_quant_layer...")
    layer = Fp8QuantLayer(seq_len_kv, index_dim, in_dtype, tune=tune)
    test.check_fn(layer, *inputs, grad=False)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
