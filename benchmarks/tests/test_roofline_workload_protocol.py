"""Unit tests for benchmark capability protocols.

Verifies that ``ShapeDtypeWorkload``, ``InputGeneratingWorkload``, and
``BenchmarkWorkload`` protocols accept duck-typed objects, and that the
generic ``BenchmarkBase`` / ``ManifestBenchmark`` accept workloads through
protocol contracts rather than nominal ``WorkloadBase`` inheritance.
"""

import pytest
import torch

from benchmarks.benchmark import (
    BenchmarkWorkload,
    InputGeneratingWorkload,
    ManifestBenchmark,
    ShapeDtypeWorkload,
    roofline_vars,
)

# ---------------------------------------------------------------------------
# Duck-typed test workloads
# ---------------------------------------------------------------------------


class _DuckShapeDtype:
    """Object with shape and dtype but NOT a WorkloadBase subclass."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


class _DuckInputGen:
    """Object with gen_inputs() only."""

    def gen_inputs(self):
        return (torch.randn(4, 4),)


class _DuckFull:
    """Object satisfying the full BenchmarkWorkload protocol."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self):
        return (torch.randn(*self.shape, dtype=self.dtype),)


class _MissingDtype:
    """Object with shape only -- should NOT satisfy ShapeDtypeWorkload."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class _MissingShape:
    """Object with dtype only -- should NOT satisfy ShapeDtypeWorkload."""

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype


# ---------------------------------------------------------------------------
# ShapeDtypeWorkload protocol tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_roofline_vars_accepts_duck_typed_workload():
    """roofline_vars should work with any object that has shape and dtype."""
    w = _DuckShapeDtype((4, 8, 1024), torch.float16)
    result = roofline_vars(w)
    assert result["M"] == 4 * 8
    assert result["N"] == 1024
    assert result["elem_bytes"] == 2


@pytest.mark.smoke
def test_roofline_vars_with_1d_shape():
    """Single-dimension shape: M should be 1, N should be that dimension."""
    w = _DuckShapeDtype((512,), torch.bfloat16)
    result = roofline_vars(w)
    assert result["M"] == 1
    assert result["N"] == 512
    assert result["elem_bytes"] == 2


@pytest.mark.smoke
def test_shape_dtype_protocol_is_runtime_checkable():
    """ShapeDtypeWorkload should be runtime-checkable for isinstance() use."""
    good = _DuckShapeDtype((4, 8), torch.float32)
    bad_no_dtype = _MissingDtype((4, 8))
    bad_no_shape = _MissingShape(torch.float32)

    assert isinstance(good, ShapeDtypeWorkload)
    assert not isinstance(bad_no_dtype, ShapeDtypeWorkload)
    assert not isinstance(bad_no_shape, ShapeDtypeWorkload)


# ---------------------------------------------------------------------------
# InputGeneratingWorkload protocol tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_input_generating_protocol():
    """InputGeneratingWorkload accepts objects with gen_inputs()."""
    gen = _DuckInputGen()
    assert isinstance(gen, InputGeneratingWorkload)

    no_gen = _DuckShapeDtype((4,), torch.float32)
    assert not isinstance(no_gen, InputGeneratingWorkload)


# ---------------------------------------------------------------------------
# BenchmarkWorkload protocol tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_benchmark_workload_protocol():
    """BenchmarkWorkload requires both shape/dtype and gen_inputs()."""
    full = _DuckFull((4, 8), torch.float16)
    assert isinstance(full, BenchmarkWorkload)
    assert isinstance(full, ShapeDtypeWorkload)
    assert isinstance(full, InputGeneratingWorkload)

    # Partial implementations should not satisfy the full protocol
    shape_only = _DuckShapeDtype((4, 8), torch.float16)
    assert not isinstance(shape_only, BenchmarkWorkload)

    gen_only = _DuckInputGen()
    assert not isinstance(gen_only, BenchmarkWorkload)


# ---------------------------------------------------------------------------
# ManifestBenchmark contract tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_manifest_benchmark_accepts_protocol_workload():
    """ManifestBenchmark should accept any ShapeDtypeWorkload."""
    w = _DuckShapeDtype((4, 8, 1024), torch.float16)
    bm = ManifestBenchmark("TestOp", w)
    assert bm.workload is w
    assert bm._roofline_vars() == {"M": 32, "N": 1024, "elem_bytes": 2}


# ---------------------------------------------------------------------------
# WorkloadBase compatibility
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_workload_base_satisfies_benchmark_workload():
    """Existing WorkloadBase subclasses should satisfy BenchmarkWorkload."""
    from workloads.base import WorkloadBase

    class _ConcreteWorkload(WorkloadBase):
        def __init__(self):
            self.shape = (4, 8)
            self.dtype = torch.float32

        def gen_inputs(self):
            return (torch.randn(*self.shape, dtype=self.dtype),)

    w = _ConcreteWorkload()
    assert isinstance(w, ShapeDtypeWorkload)
    assert isinstance(w, BenchmarkWorkload)

    # Should also work with ManifestBenchmark
    bm = ManifestBenchmark("TestOp", w)
    assert bm._roofline_vars() == {"M": 4, "N": 8, "elem_bytes": 4}
