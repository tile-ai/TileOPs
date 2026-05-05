"""Strategy benchmark for BinaryKernel: direct vs explicit_parallel.

Uses add as the representative op. Sweeps DNN-realistic 2D shapes
(tokens x hidden_dim), all SUPPORTED_DTYPES, and both strategies to
validate DEFAULT_STRATEGY = "explicit_parallel".

Acceptance criteria:
  >= 3 shapes x 3 dtypes x 2 strategies = 18 benchmark points.
"""

from math import prod
from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops.elementwise import AddFwdOp
from workloads.workload_base import FixtureBase

# DNN-realistic 2D shapes — same-shape (no broadcast) for clean strategy
# comparison. The third entry is non-pow2 in the hidden dim to exercise
# tail-handling code paths.
_SHAPES_2D = [
    (1024, 4096),   # 4M  — small transformer hidden dim
    (1024, 10240),  # 10M — medium (e.g. Llama-2 intermediate)
    (1024, 11008),  # 11M — non-pow2 LLaMA-7B intermediate
]

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_BINARY_STRATEGIES = ("direct", "explicit_parallel")


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class BinaryStrategyBenchCase:
    """Minimal test harness for binary strategy benchmarks."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(*self.shape, device="cuda", dtype=self.dtype)
        b = torch.randn(*self.shape, device="cuda", dtype=self.dtype)
        return a, b


class BinaryStrategyBenchmark(BenchmarkBase[BinaryStrategyBenchCase]):
    """Bandwidth-oriented benchmark for binary elementwise strategy comparison."""

    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        """Read a + read b + write y."""
        t = self.workload
        elem_bytes = t.dtype.itemsize
        return 3 * t.n_total * elem_bytes


# ---------------------------------------------------------------------------
# Parametrize: 3 shapes x 3 dtypes x 2 strategies = 18 cases
# ---------------------------------------------------------------------------


def _shape_id(shape: tuple[int, ...]) -> str:
    return "x".join(str(s) for s in shape)


def _binary_strategy_params() -> list:
    params = []
    smoke_shape = _SHAPES_2D[0]
    for shape in _SHAPES_2D:
        for dtype in _DTYPES:
            for strategy in _BINARY_STRATEGIES:
                is_smoke = shape == smoke_shape and dtype == torch.float16
                mark = pytest.mark.smoke if is_smoke else pytest.mark.full
                params.append(pytest.param(
                    shape, dtype, strategy,
                    id=f"{_shape_id(shape)}-{dtype}-{strategy}",
                    marks=mark,
                ))
    return params


class BinaryStrategyFixture(FixtureBase):
    PARAMS = [("shape, dtype, strategy", _binary_strategy_params())]


@BinaryStrategyFixture
def test_binary_strategy_bench(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    strategy: str,
) -> None:
    """Benchmark BinaryKernel (add) per strategy to validate DEFAULT_STRATEGY."""
    test = BinaryStrategyBenchCase(shape, dtype)
    bm = BinaryStrategyBenchmark(test)
    inputs = test.gen_inputs()

    op = AddFwdOp(a_shape=shape, b_shape=shape, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "binary_strategy",
        {"shape": shape, "dtype": dtype, "strategy": strategy},
        result,
        tag=f"add_{strategy}",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
