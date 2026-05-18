"""Unit tests for benchmark capability protocols.

Verifies that ``ShapeDtypeWorkload``, ``InputGeneratingWorkload``, and
``BenchmarkWorkload`` protocols accept duck-typed objects, and that the
generic ``BenchmarkBase`` / ``ManifestBenchmark`` accept workloads through
protocol contracts rather than nominal ``WorkloadBase`` inheritance.
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkWorkload,
    InputGeneratingWorkload,
    ManifestBenchmark,
    ShapeDtypeWorkload,
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


class _FakeRooflineOp:
    """Minimal op-like object for ManifestBenchmark unit tests."""

    def __init__(self, roofline: tuple[int, int] = (128, 256)):
        self.calls = 0
        self._roofline = roofline

    def eval_roofline(self) -> tuple[int, int]:
        self.calls += 1
        return self._roofline


# ---------------------------------------------------------------------------
# ShapeDtypeWorkload protocol tests
# ---------------------------------------------------------------------------


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
    op = _FakeRooflineOp((123, 456))
    bm = ManifestBenchmark("TestOp", op, w)
    assert bm.workload is w
    assert bm.calculate_flops() == 123.0
    assert bm.calculate_memory() == 456.0
    assert op.calls == 1


# ---------------------------------------------------------------------------
# WorkloadBase compatibility
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_workload_base_satisfies_benchmark_workload():
    """Existing WorkloadBase subclasses should satisfy BenchmarkWorkload."""
    from workloads.workload_base import WorkloadBase

    class _ConcreteWorkload(WorkloadBase):
        def __init__(self):
            self.shape = (4, 8)
            self.dtype = torch.float32

        def gen_inputs(self):
            return (torch.randn(*self.shape, dtype=self.dtype),)

    w = _ConcreteWorkload()
    assert isinstance(w, ShapeDtypeWorkload)
    assert isinstance(w, BenchmarkWorkload)

    # Should also work with ManifestBenchmark.
    bm = ManifestBenchmark("TestOp", _FakeRooflineOp((4, 8)), w)
    assert bm.calculate_flops() == 4.0
    assert bm.calculate_memory() == 8.0


# ---------------------------------------------------------------------------
# ManifestBenchmark roofline contract
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_manifest_benchmark_reads_op_eval_roofline_once():
    w = _DuckShapeDtype((2048, 4096), torch.float16)
    op = _FakeRooflineOp((2048, 4096))
    bm = ManifestBenchmark("SumFwdOp", op, w)
    assert bm.calculate_flops() == 2048.0
    assert bm.calculate_memory() == 4096.0
    assert bm.calculate_flops() == 2048.0
    assert op.calls == 1


@pytest.mark.smoke
def test_workloads_to_params_include_extra_propagates_dim():
    """When a workload entry carries ``dim``, ``include_extra=True`` should
    surface it in the pytest param triple.
    """
    from benchmarks.benchmark_base import (
        _workload_extra_params,
        workloads_to_params,
    )

    # Direct unit test on the helper (no manifest mutation required).
    assert _workload_extra_params(
        {"x_shape": [4, 4], "dtypes": ["float16"], "label": "t",
         "dim": 0, "keepdim": True}
    ) == {"dim": 0, "keepdim": True}

    # End-to-end with the manifest: include_extra=True must still yield
    # well-formed triples with the (shape, dtype, extra) mapping. The
    # contract being asserted is per-triple shape/dtype/extra typing; it
    # must not depend on the ordering of SumFwdOp.workloads (which is QA
    # curated and may be reordered without regressing the helper).
    triples = workloads_to_params("SumFwdOp", include_extra=True)
    assert len(triples) > 0
    for p in triples:
        shape, dtype, extra = p.values
        assert isinstance(shape, tuple)
        assert isinstance(dtype, torch.dtype)
        assert isinstance(extra, dict)
    # At least one workload intentionally carries no extras; the harness
    # must expose that as an empty dict rather than omitting the slot.
    assert any(p.values[2] == {} for p in triples)


@pytest.mark.smoke
def test_manifest_benchmark_propagates_op_eval_error():
    w = _DuckShapeDtype((4, 8), torch.float16)

    class _BrokenOp:
        def eval_roofline(self):
            raise RuntimeError("shape not bound")

    bm = ManifestBenchmark("SumFwdOp", _BrokenOp(), w)
    with pytest.raises(RuntimeError, match="shape not bound"):
        bm.calculate_flops()
