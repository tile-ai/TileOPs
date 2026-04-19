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

    # Should also work with ManifestBenchmark
    bm = ManifestBenchmark("TestOp", w)
    assert bm._roofline_vars() == {"M": 4, "N": 8, "elem_bytes": 4}


# ---------------------------------------------------------------------------
# Manifest-driven var resolution (roofline.vars)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_manifest_benchmark_uses_manifest_vars_for_default_dim():
    """For an op with ``roofline.vars`` declared, ManifestBenchmark should
    evaluate those expressions — default ``dim=-1`` matches legacy behaviour.
    """
    w = _DuckShapeDtype((2048, 4096), torch.float16)
    bm = ManifestBenchmark("SumFwdOp", w)
    rv = bm._roofline_vars()
    assert rv["M"] == 2048
    assert rv["N"] == 4096
    assert rv["elem_bytes"] == 2


@pytest.mark.smoke
def test_manifest_benchmark_respects_dim_zero():
    """With ``op_params={"dim": 0}`` the vars must reduce the *first* axis,
    producing M/N that differ from the hardcoded last-axis heuristic.
    """
    w = _DuckShapeDtype((2048, 4096), torch.float16)
    bm = ManifestBenchmark("SumFwdOp", w, op_params={"dim": 0})
    rv = bm._roofline_vars()
    # dim=0 -> N = x.shape[0] = 2048, M = x.shape[1] = 4096
    assert rv["N"] == 2048
    assert rv["M"] == 4096
    # Legacy heuristic would give M=2048, N=4096 — confirm we diverge.
    legacy = roofline_vars(w)
    assert (rv["M"], rv["N"]) != (legacy["M"], legacy["N"])


@pytest.mark.smoke
def test_manifest_benchmark_multi_axis_dim():
    """Tuple ``dim`` collapses multiple axes — M/N reflect that."""
    w = _DuckShapeDtype((4, 8, 16), torch.float32)
    bm = ManifestBenchmark("SumFwdOp", w, op_params={"dim": (0, 2)})
    rv = bm._roofline_vars()
    assert rv["M"] == 8
    assert rv["N"] == 4 * 16
    assert rv["elem_bytes"] == 4


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

    # End-to-end with the manifest: SumFwdOp entries carry no extras today,
    # so include_extra=True must still yield well-formed triples with an
    # empty extras dict, preserving the existing (shape, dtype) ordering.
    triples = workloads_to_params("SumFwdOp", include_extra=True)
    assert len(triples) > 0
    shape, dtype, extra = triples[0].values
    assert isinstance(shape, tuple)
    assert isinstance(dtype, torch.dtype)
    assert extra == {}


@pytest.mark.smoke
def test_manifest_benchmark_falls_back_when_no_vars(monkeypatch):
    """If the manifest entry has no ``roofline.vars``, ManifestBenchmark
    must fall back to the legacy last-axis helper without raising.
    """
    from tileops.manifest import _load_manifest

    _load_manifest.cache_clear()
    real = _load_manifest()
    patched = dict(real)
    patched["_NoVarsOp"] = {
        "roofline": {
            "flops": "M * N",
            "bytes": "(M * N + M) * elem_bytes",
        }
    }
    monkeypatch.setattr("tileops.manifest._load_manifest", lambda: patched)

    w = _DuckShapeDtype((4, 8), torch.float32)
    bm = ManifestBenchmark("_NoVarsOp", w)
    rv = bm._roofline_vars()
    # Falls back to last-axis heuristic.
    assert rv == {"M": 4, "N": 8, "elem_bytes": 4}


@pytest.mark.smoke
def test_manifest_benchmark_propagates_vars_eval_error(monkeypatch):
    """If ``roofline.vars`` is declared but evaluation fails, ManifestBenchmark
    must propagate the error rather than silently falling back to the legacy
    last-axis heuristic — otherwise bad manifest expressions would mask as
    plausible M/N bindings and feed the roofline calculator garbage.
    """
    from tileops.manifest import _load_manifest

    _load_manifest.cache_clear()
    real = _load_manifest()
    patched = dict(real)
    # Copy the SumFwdOp entry but poison its roofline.vars mapping so one
    # expression references a name that is never bound.
    base = dict(real["SumFwdOp"])
    base_roofline = dict(base["roofline"])
    base_roofline["vars"] = {
        "M": "missing_name + 1",
        "N": "x.shape[-1]",
    }
    base["roofline"] = base_roofline
    patched["SumFwdOp"] = base
    monkeypatch.setattr("tileops.manifest._load_manifest", lambda: patched)

    w = _DuckShapeDtype((4, 8), torch.float16)
    bm = ManifestBenchmark("SumFwdOp", w, op_params={"dim": 0})
    with pytest.raises(ValueError, match="Failed to evaluate"):
        bm._roofline_vars()
