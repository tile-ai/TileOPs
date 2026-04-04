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

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops.elementwise import AddOp
from workloads.base import FixtureBase

# DNN-realistic 2D shapes — same-shape (no broadcast) for clean strategy comparison
_SHAPES_2D = [
    (1024, 4096),   # 4M  — small transformer hidden dim
    (1024, 10240),  # 10M — medium (e.g. Llama-2 intermediate)
    (1024, 20480),  # 20M — large (e.g. Llama-2 70B intermediate)
]
_SHAPES = [prod(s) for s in _SHAPES_2D]

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_BINARY_STRATEGIES = ("direct", "explicit_parallel")


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class BinaryStrategyBenchCase:
    """Minimal test harness for binary strategy benchmarks."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        b = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        return a, b


class BinaryStrategyBenchmark(BenchmarkBase):
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


class BinaryStrategyFixture(FixtureBase):
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
                _SHAPES[0], "1024x4096", torch.bfloat16, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[0], "1024x4096", torch.bfloat16, "explicit_parallel",
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
                _SHAPES[1], "1024x10240", torch.bfloat16, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[1], "1024x10240", torch.bfloat16, "explicit_parallel",
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
                _SHAPES[2], "1024x20480", torch.bfloat16, "direct",
                marks=pytest.mark.full,
            ),
            pytest.param(
                _SHAPES[2], "1024x20480", torch.bfloat16, "explicit_parallel",
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
        ]),
    ]


@BinaryStrategyFixture
def test_binary_strategy_bench(
    n_total: int,
    shape_label: str,
    dtype: torch.dtype,
    strategy: str,
) -> None:
    """Benchmark BinaryKernel (add) per strategy to validate DEFAULT_STRATEGY."""
    test = BinaryStrategyBenchCase(n_total, dtype)
    bm = BinaryStrategyBenchmark(test)
    inputs = test.gen_inputs()

    shape = (n_total,)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "binary_strategy",
        locals(),
        result,
        tag=f"add_{strategy}",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
