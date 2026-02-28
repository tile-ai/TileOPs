import torch
import pytest

from benchmarks import Fp8QuantBenchmark
from top.ops import Fp8QuantOp


@pytest.mark.parametrize(
    ("batch, seq_len_kv, kv_group, index_dim, in_dtype, tune"),
    [
        (1, 8192, 1, 64, torch.float16, False),
        (1, 8192, 1, 64, torch.bfloat16, False),
        (1, 4096, 1, 128, torch.float32, False),
        (1, 16384, 1, 32, torch.float32, False),
        (1, 1024, 4, 64, torch.float16, False),
    ],
)
def test_fp8_quant_op(batch: int, seq_len_kv: int, kv_group: int, index_dim: int,
                      in_dtype: torch.dtype, tune: bool) -> None:

    params = {
        "batch": batch,
        "seq_len_kv": seq_len_kv,
        "kv_group": kv_group,
        "index_dim": index_dim,
        "in_dtype": in_dtype,
        "tune": tune,
    }
    op = Fp8QuantOp(**params)
    benchmark = Fp8QuantBenchmark(**params)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, inputs)
    benchmark.profile(op, inputs)


if __name__ == "__main__":
    test_fp8_quant_op(1, 8192, 1, 64, torch.float16, False)
    test_fp8_quant_op(1, 8192, 1, 64, torch.bfloat16, False)
    test_fp8_quant_op(1, 4096, 1, 128, torch.float32, False)
    test_fp8_quant_op(1, 16384, 1, 32, torch.float32, False)
    test_fp8_quant_op(1, 1024, 4, 64, torch.float16, False)
