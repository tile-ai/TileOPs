"""Strategy benchmark for UnaryKernel: direct vs explicit_parallel vs register_copy.

Uses relu as the representative op. Sweeps DNN-realistic 2D shapes
(tokens x hidden_dim), all SUPPORTED_DTYPES, and all 3 strategies to
evaluate DEFAULT_STRATEGY.

H200 observations:
  - register_copy wins clearly for fp16/bf16 across all shapes.
  - fp32 small shapes (1024x4096) show run-to-run variance between
    register_copy and explicit_parallel; neither dominates reliably.
  - Current DEFAULT_STRATEGY = "register_copy" is a reasonable choice
    but not proven dominant for every dtype/shape combination.

Acceptance criteria:
  >= 3 shapes x 3 dtypes x 3 strategies = 27 benchmark points.
"""

from math import prod
from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops.elementwise import ReluFwdOp
from workloads.workload_base import FixtureBase

# DNN-realistic 2D shapes (tokens x hidden_dim). The third entry is
# non-pow2 in the hidden dim to exercise tail-handling code paths.
_SHAPES_2D = [
    (1024, 4096),   # 4M  — small transformer hidden dim
    (1024, 10240),  # 10M — medium (e.g. Llama-2 intermediate)
    (1024, 11008),  # 11M — non-pow2 LLaMA-7B intermediate
]

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_UNARY_STRATEGIES = ("direct", "explicit_parallel", "register_copy")


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class UnaryStrategyBenchCase:
    """Minimal test harness for unary strategy benchmarks."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(*self.shape, device="cuda", dtype=self.dtype),)


class UnaryStrategyBenchmark(BenchmarkBase[UnaryStrategyBenchCase]):
    """Bandwidth-oriented benchmark for unary elementwise strategy comparison."""

    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return t.n_total * (t.dtype.itemsize + t.output_dtype.itemsize)


# ---------------------------------------------------------------------------
# Parametrize: 3 shapes x 3 dtypes x 3 strategies = 27 cases
# ---------------------------------------------------------------------------


def _shape_id(shape: tuple[int, ...]) -> str:
    return "x".join(str(s) for s in shape)


def _unary_strategy_params() -> list:
    params = []
    smoke_shape = _SHAPES_2D[0]
    for shape in _SHAPES_2D:
        for dtype in _DTYPES:
            for strategy in _UNARY_STRATEGIES:
                is_smoke = shape == smoke_shape and dtype == torch.float16
                mark = pytest.mark.smoke if is_smoke else pytest.mark.full
                params.append(pytest.param(
                    shape, dtype, strategy,
                    id=f"{_shape_id(shape)}-{dtype}-{strategy}",
                    marks=mark,
                ))
    return params


class UnaryStrategyFixture(FixtureBase):
    PARAMS = [("shape, dtype, strategy", _unary_strategy_params())]


@UnaryStrategyFixture
def test_unary_strategy_bench(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    strategy: str,
) -> None:
    """Benchmark UnaryKernel (relu) per strategy to validate DEFAULT_STRATEGY."""
    test = UnaryStrategyBenchCase(shape, dtype)
    bm = UnaryStrategyBenchmark(test)
    inputs = test.gen_inputs()

    n_total = prod(shape)
    op = ReluFwdOp(N_total=n_total, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "unary_strategy",
        {"shape": shape, "dtype": dtype, "strategy": strategy},
        result,
        tag=f"relu_{strategy}",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
