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

Acceptance criteria (issue #498):
  >= 3 shapes x 3 dtypes x 3 strategies = 27 benchmark points.
"""

from math import prod
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.test_base import FixtureBase
from tileops.ops.elementwise import ReluOp

# DNN-realistic 2D shapes flattened to 1D total element counts
_SHAPES_2D = [
    (1024, 4096),   # 4M  — small transformer hidden dim
    (1024, 10240),  # 10M — medium (e.g. Llama-2 intermediate)
    (1024, 20480),  # 20M — large (e.g. Llama-2 70B intermediate)
]
_SHAPES = [prod(s) for s in _SHAPES_2D]

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_UNARY_STRATEGIES = ("direct", "explicit_parallel", "register_copy")


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class UnaryStrategyBenchCase:
    """Minimal test harness for unary strategy benchmarks."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.n_total, device="cuda", dtype=self.dtype),)


class UnaryStrategyBenchmark(BenchmarkBase):
    """Bandwidth-oriented benchmark for unary elementwise strategy comparison."""

    def calculate_flops(self) -> Optional[float]:
        return self.test.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        return t.n_total * (t.dtype.itemsize + t.output_dtype.itemsize)


# ---------------------------------------------------------------------------
# Parametrize: 3 shapes x 3 dtypes x 3 strategies = 27 cases
# ---------------------------------------------------------------------------


class UnaryStrategyFixture(FixtureBase):
    PARAMS = [
        ("n_total, shape_label, dtype, strategy", [
            # --- (1024, 4096) = 4_194_304 ---
            pytest.param(
                _SHAPES[0], "1024x4096", torch.float16, "direct",
                marks=pytest.mark.smoke,
            ),
            pytest.param(
                _SHAPES[0], "1024x4096", torch.float16, "explicit_parallel",
                marks=pytest.mark.smoke,
            ),
            pytest.param(
                _SHAPES[0], "1024x4096", torch.float16, "register_copy",
                marks=pytest.mark.smoke,
            ),
            pytest.param(
                _SHAPES[0], "1024x4096", torch.bfloat16, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[0], "1024x4096", torch.bfloat16, "explicit_parallel",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[0], "1024x4096", torch.bfloat16, "register_copy",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[0], "1024x4096", torch.float32, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[0], "1024x4096", torch.float32, "explicit_parallel",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[0], "1024x4096", torch.float32, "register_copy",
                marks=pytest.mark.full,
            ),
            # --- (1024, 10240) = 10_485_760 ---
            pytest.param(
                _SHAPES[1], "1024x10240", torch.float16, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[1], "1024x10240", torch.float16, "explicit_parallel",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[1], "1024x10240", torch.float16, "register_copy",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[1], "1024x10240", torch.bfloat16, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[1], "1024x10240", torch.bfloat16, "explicit_parallel",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[1], "1024x10240", torch.bfloat16, "register_copy",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[1], "1024x10240", torch.float32, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[1], "1024x10240", torch.float32, "explicit_parallel",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[1], "1024x10240", torch.float32, "register_copy",
                marks=pytest.mark.full,
            ),
            # --- (1024, 20480) = 20_971_520 ---
            pytest.param(
                _SHAPES[2], "1024x20480", torch.float16, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[2], "1024x20480", torch.float16, "explicit_parallel",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[2], "1024x20480", torch.float16, "register_copy",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[2], "1024x20480", torch.bfloat16, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[2], "1024x20480", torch.bfloat16, "explicit_parallel",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[2], "1024x20480", torch.bfloat16, "register_copy",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[2], "1024x20480", torch.float32, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[2], "1024x20480", torch.float32, "explicit_parallel",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[2], "1024x20480", torch.float32, "register_copy",
                marks=pytest.mark.full,
            ),
        ]),
    ]


@UnaryStrategyFixture
def test_unary_strategy_bench(
    n_total: int,
    shape_label: str,
    dtype: torch.dtype,
    strategy: str,
) -> None:
    """Benchmark UnaryKernel (relu) per strategy to validate DEFAULT_STRATEGY."""
    test = UnaryStrategyBenchCase(n_total, dtype)
    bm = UnaryStrategyBenchmark(test)
    inputs = test.gen_inputs()

    op = ReluOp(N_total=n_total, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "unary_strategy",
        locals(),
        result,
        tag=f"relu_{strategy}",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
