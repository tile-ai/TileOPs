"""Validate that elementwise strategy benchmark modules satisfy issue #498 AC.

Tests check:
- UnaryKernel benchmark has >= 3 shapes x 3 dtypes x 3 strategies = 27 cases
- BinaryKernel benchmark has >= 3 shapes x 3 dtypes x 2 strategies = 18 cases
- DEFAULT_STRATEGY is a valid member of STRATEGIES for each kernel type
"""

import pytest
import torch

from tileops.kernels.elementwise import BinaryKernel, UnaryKernel


class TestUnaryStrategyBenchStructure:
    """Validate UnaryKernel strategy benchmark coverage."""

    @pytest.mark.smoke
    def test_unary_strategies_count(self):
        """UnaryKernel must support exactly 3 strategies."""
        assert len(UnaryKernel.STRATEGIES) == 3
        assert set(UnaryKernel.STRATEGIES) == {
            "direct",
            "explicit_parallel",
            "register_copy",
        }

    @pytest.mark.full
    def test_unary_default_strategy_is_valid(self):
        """DEFAULT_STRATEGY must be one of the declared STRATEGIES.

        Benchmark evidence (H200): register_copy wins clearly for fp16/bf16
        across all shapes but shows run-to-run variance against
        explicit_parallel for fp32 small shapes (1024x4096). The current
        default (register_copy) is a reasonable choice but not proven
        dominant for every dtype/shape combination.
        """
        assert UnaryKernel.DEFAULT_STRATEGY in UnaryKernel.STRATEGIES

    @pytest.mark.full
    def test_unary_bench_module_parametrize_count(self):
        """bench_unary_strategy must have >= 27 parametrized cases (3x3x3)."""
        from benchmarks.ops.bench_unary_strategy import UnaryStrategyFixture

        _, cases = UnaryStrategyFixture.PARAMS[0]
        assert len(cases) >= 27, (
            f"Expected >= 27 unary strategy benchmark cases, got {len(cases)}"
        )

    @pytest.mark.full
    def test_unary_bench_covers_all_strategies(self):
        """All 3 strategies must appear in the benchmark parameters."""
        from benchmarks.ops.bench_unary_strategy import UnaryStrategyFixture

        _, cases = UnaryStrategyFixture.PARAMS[0]
        strategies_seen = set()
        for case in cases:
            strategies_seen.add(case.values[3])
        assert strategies_seen == {"direct", "explicit_parallel", "register_copy"}

    @pytest.mark.full
    def test_unary_bench_covers_all_dtypes(self):
        """All 3 dtypes (fp16, bf16, fp32) must appear in the benchmark."""
        from benchmarks.ops.bench_unary_strategy import UnaryStrategyFixture

        _, cases = UnaryStrategyFixture.PARAMS[0]
        dtypes_seen = set()
        for case in cases:
            dtypes_seen.add(case.values[2])
        assert dtypes_seen == {torch.float16, torch.bfloat16, torch.float32}

    @pytest.mark.full
    def test_unary_bench_covers_min_shapes(self):
        """At least 3 distinct shapes must appear in the benchmark."""
        from benchmarks.ops.bench_unary_strategy import UnaryStrategyFixture

        _, cases = UnaryStrategyFixture.PARAMS[0]
        shapes_seen = set()
        for case in cases:
            shapes_seen.add(case.values[0])
        assert len(shapes_seen) >= 3, (
            f"Expected >= 3 shapes, got {len(shapes_seen)}: {shapes_seen}"
        )


class TestBinaryStrategyBenchStructure:
    """Validate BinaryKernel strategy benchmark coverage."""

    @pytest.mark.smoke
    def test_binary_strategies_count(self):
        """BinaryKernel must support exactly 2 strategies."""
        assert len(BinaryKernel.STRATEGIES) == 2
        assert set(BinaryKernel.STRATEGIES) == {
            "direct",
            "explicit_parallel",
        }

    @pytest.mark.full
    def test_binary_default_strategy_is_valid(self):
        """DEFAULT_STRATEGY must be one of the declared STRATEGIES.

        Benchmark evidence (H200): explicit_parallel wins consistently
        across all 9 shape/dtype pairs (2.77-4.10 TB/s vs direct
        1.44-2.92 TB/s).
        """
        assert BinaryKernel.DEFAULT_STRATEGY in BinaryKernel.STRATEGIES

    @pytest.mark.full
    def test_binary_bench_module_parametrize_count(self):
        """bench_binary_strategy must have >= 18 parametrized cases (3x3x2)."""
        from benchmarks.ops.bench_binary_strategy import BinaryStrategyFixture

        _, cases = BinaryStrategyFixture.PARAMS[0]
        assert len(cases) >= 18, (
            f"Expected >= 18 binary strategy benchmark cases, got {len(cases)}"
        )

    @pytest.mark.full
    def test_binary_bench_covers_all_strategies(self):
        """Both strategies must appear in the benchmark parameters."""
        from benchmarks.ops.bench_binary_strategy import BinaryStrategyFixture

        _, cases = BinaryStrategyFixture.PARAMS[0]
        strategies_seen = set()
        for case in cases:
            strategies_seen.add(case.values[3])
        assert strategies_seen == {"direct", "explicit_parallel"}

    @pytest.mark.full
    def test_binary_bench_covers_all_dtypes(self):
        """All 3 dtypes (fp16, bf16, fp32) must appear in the benchmark."""
        from benchmarks.ops.bench_binary_strategy import BinaryStrategyFixture

        _, cases = BinaryStrategyFixture.PARAMS[0]
        dtypes_seen = set()
        for case in cases:
            dtypes_seen.add(case.values[2])
        assert dtypes_seen == {torch.float16, torch.bfloat16, torch.float32}

    @pytest.mark.full
    def test_binary_bench_covers_min_shapes(self):
        """At least 3 distinct shapes must appear in the benchmark."""
        from benchmarks.ops.bench_binary_strategy import BinaryStrategyFixture

        _, cases = BinaryStrategyFixture.PARAMS[0]
        shapes_seen = set()
        for case in cases:
            shapes_seen.add(case.values[0])
        assert len(shapes_seen) >= 3, (
            f"Expected >= 3 shapes, got {len(shapes_seen)}: {shapes_seen}"
        )
