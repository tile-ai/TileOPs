import torch
import pytest

from benchmarks import GemmBenchmark
from top.ops import GemmOp


@pytest.mark.parametrize(
    "m, n, k, dtype, trans_a, trans_b, tune",
    [
        (1024, 1024, 1024, torch.float16, False, False, False),
    ],
)
def test_gemm(m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool,
              tune: bool) -> None:
    op = GemmOp(m, n, k, trans_a=trans_a, trans_b=trans_b, dtype=dtype, tune=tune)
    benchmark = GemmBenchmark(m, n, k, dtype, trans_a=trans_a, trans_b=trans_b)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-vvs"])
