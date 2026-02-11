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
    ],
)
def test_gemv(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    op = GemvOp(n, k, dtype=dtype, tune=tune)
    benchmark = GemvBenchmark(n, k, dtype)

    inputs = benchmark.gen_inputs()

    benchmark.check(op, *inputs, atol=1e-3, rtol=1e-3)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-vvs"])
