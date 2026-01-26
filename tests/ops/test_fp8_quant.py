import torch
import pytest

from benchmarks import Fp8QuantBenchmark
from top.ops import Fp8QuantOp


@pytest.mark.parametrize(
    ("seq_len_kv, index_dim, in_dtype, tune"),
    [
        (8192, 64, torch.float16, False),
        (8192, 64, torch.bfloat16, False),
        (4096, 128, torch.float32, False),
        (8192, 64, torch.float16, True),
        (16384, 32, torch.float32, False),
    ],
)
def test_fp8_quant_op(seq_len_kv: int, index_dim: int, in_dtype: torch.dtype, tune: bool) -> None:

    params = {
        "seq_len_kv": seq_len_kv,
        "index_dim": index_dim,
        "in_dtype": in_dtype,
        "tune": tune,
    }
    op = Fp8QuantOp(**params)
    benchmark = Fp8QuantBenchmark(**params)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)


if __name__ == "main":
    test_fp8_quant_op(8192, 64, torch.float16, False)
    test_fp8_quant_op(8192, 64, torch.bfloat16, False)
    test_fp8_quant_op(4096, 128, torch.float32, False)
    test_fp8_quant_op(8192, 64, torch.float16, True)
    test_fp8_quant_op(16384, 32, torch.float32, False)
