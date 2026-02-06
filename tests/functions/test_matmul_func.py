import pytest
import torch

from benchmarks import MatMulBenchmark
from top.functions import MatMulFunc, matmul


@pytest.fixture(autouse=True)
def setup() -> None:
    """Set up the test environment."""
    torch.manual_seed(1234)


@pytest.mark.parametrize(
    "m, n, k, dtype, tune",
    [
        (1024, 1024, 1024, torch.float16, False),
    ],
)
def test_matmul(m: int, n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    benchmark = MatMulBenchmark(m, n, k, dtype)

    inputs = benchmark.gen_inputs()

    print("=========Testing matmul function inference=========")
    benchmark.check_fn(matmul, *inputs)

    print("=========Testing matmul function class=========")
    fn = MatMulFunc(m, n, k, dtype, tune)
    benchmark.check_fn(fn, *inputs)


if __name__ == "__main__":
    import sys

    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
