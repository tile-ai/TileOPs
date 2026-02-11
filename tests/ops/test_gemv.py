import torch
import pytest

from benchmarks import GemvBenchmark
from top.ops import GemvOp


@pytest.mark.parametrize(
    "n, k, dtype, tune",
    [
        (1024, 1024, torch.float16, False),
        (7168, 16384, torch.float16, True),
        (18432, 7168, torch.float16, True),
        (1024, 1024, torch.bfloat16, False),
        (7168, 16384, torch.bfloat16, True),
        (18432, 7168, torch.bfloat16, True),
    ],
)
def test_gemv(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    op = GemvOp(n, k, dtype=dtype, tune=tune)
    benchmark = GemvBenchmark(n, k, dtype)

    inputs = benchmark.gen_inputs()

    if dtype == torch.float16:
        benchmark.check(op, *inputs, atol=1e-3, rtol=1e-3)
    else:
        benchmark.check(op, *inputs, atol=1.6e-2, rtol=1.6e-2)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-vvs"])
