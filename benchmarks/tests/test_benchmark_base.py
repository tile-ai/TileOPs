"""Unit tests for benchmarks.benchmark_base.

Verifies that ``ShapeDtypeWorkload``, ``InputGeneratingWorkload``, and
``BenchmarkWorkload`` protocols accept duck-typed objects, and that the
generic ``BenchmarkBase`` / ``ManifestBenchmark`` accept workloads through
protocol contracts rather than nominal ``WorkloadBase`` inheritance.
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    BenchmarkWorkload,
    InputGeneratingWorkload,
    ManifestBenchmark,
    ShapeDtypeWorkload,
    _bench_meta,
    bench_kernel,
    workloads_to_params,
)

# Duck-typed test workloads


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


# ShapeDtypeWorkload protocol tests


@pytest.mark.smoke
def test_shape_dtype_protocol_is_runtime_checkable():
    """ShapeDtypeWorkload should be runtime-checkable for isinstance() use."""
    good = _DuckShapeDtype((4, 8), torch.float32)
    bad_no_dtype = _MissingDtype((4, 8))
    bad_no_shape = _MissingShape(torch.float32)

    assert isinstance(good, ShapeDtypeWorkload)
    assert not isinstance(bad_no_dtype, ShapeDtypeWorkload)
    assert not isinstance(bad_no_shape, ShapeDtypeWorkload)


# InputGeneratingWorkload protocol tests


@pytest.mark.smoke
def test_input_generating_protocol():
    """InputGeneratingWorkload accepts objects with gen_inputs()."""
    gen = _DuckInputGen()
    assert isinstance(gen, InputGeneratingWorkload)

    no_gen = _DuckShapeDtype((4,), torch.float32)
    assert not isinstance(no_gen, InputGeneratingWorkload)


# BenchmarkWorkload protocol tests


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


# ManifestBenchmark contract tests


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


# WorkloadBase compatibility


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


# ManifestBenchmark roofline contract


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
    # End-to-end with the manifest: include_extra=True must still yield
    # well-formed triples with the (shape, dtype, extra) mapping. The
    # contract being asserted is per-triple shape/dtype/extra typing; it
    # must not depend on the ordering of SumFwdOp.workloads (which is QA
    # curated and may be reordered without regressing the helper).
    triples = workloads_to_params("SumFwdOp", include_extra=True)
    assert len(triples) > 0
    assert any("dim" in p.values[2] for p in triples), (
        "at least one SumFwdOp workload must propagate a dim param"
    )
    for p in triples:
        shape, dtype, extra = p.values
        assert isinstance(shape, tuple)
        assert isinstance(dtype, torch.dtype)
        assert isinstance(extra, dict)
    # A workload with no extras must yield an empty dict, not a missing slot.
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


def test_multi_input_op_raises_keyerror():
    """Multi-input ops (q/k/v) raise instead of binding a wrong tensor."""
    with pytest.raises(KeyError, match="exactly one manifest tensor input"):
        workloads_to_params("GroupedQueryAttentionFwdOp")


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_projection_failure_falls_back_to_cuda_events():
    """A callable launching no CUDA kernel projects no annotation windows;
    bench_kernel must fall back and mark the deviating timing method."""
    latency = bench_kernel(lambda: sum(range(64)), n_warmup=1, n_repeat=2, n_trials=1)
    assert latency >= 0.0
    assert _bench_meta.timing == "cuda-events"


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kernel_runtime_error_propagates():
    """Genuine RuntimeErrors must reach the caller, not the fallback path."""
    def boom():
        raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        bench_kernel(boom, n_warmup=0, n_repeat=1, n_trials=1)


@pytest.fixture
def _reset_records():
    """Snapshot and clear ``BenchmarkReport._records`` around each test."""
    saved = BenchmarkReport._records
    BenchmarkReport._records = {}
    try:
        yield
    finally:
        BenchmarkReport._records = saved


class _FakeKernel:
    """Stand-in for ``tileops.kernels.kernel_base.Kernel`` with just a config dict."""

    def __init__(self, config: dict):
        self.config = config


def _result() -> dict:
    return {"latency_ms": 0.01, "tflops": 1.0, "bandwidth_tbs": 0.5}


@pytest.mark.full
@pytest.mark.usefixtures('_reset_records')
def test_record_eager_init_op_keeps_kernel_config():
    """Pattern 1: ``op.kernel`` set in ``__init__`` (GemmOp-style)."""

    class _EagerOp:
        def __init__(self):
            self.kernel = _FakeKernel({"block_m": 128, "block_n": 256})

    BenchmarkReport.record(_EagerOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_EagerOp"]
    assert records[0].get("config") == {"block_m": 128, "block_n": 256}


@pytest.mark.full
@pytest.mark.usefixtures('_reset_records')
def test_record_lazy_with_dummy_kernel_keeps_kernel_config():
    """Pattern 2: dummy ``op.kernel`` plus a populated ``_kernel_cache``."""

    class _LazyDummyOp:
        def __init__(self):
            self.kernel = _FakeKernel({"block_m": 8})
            self._kernel_cache = {1: self.kernel}

    BenchmarkReport.record(_LazyDummyOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_LazyDummyOp"]
    assert records[0].get("config") == {"block_m": 8}


@pytest.mark.full
@pytest.mark.usefixtures('_reset_records')
def test_record_pure_lazy_cache_op_keeps_kernel_config():
    """Pattern 3: only ``_kernel_cache`` is populated."""

    class _PureLazyOp:
        def __init__(self):
            self._kernel_cache = {(32, 256): _FakeKernel({"block_m": 4, "tile_n": 0})}

    BenchmarkReport.record(_PureLazyOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_PureLazyOp"]
    assert records[0].get("config") == {"block_m": 4, "tile_n": 0}


@pytest.mark.full
@pytest.mark.usefixtures('_reset_records')
def test_record_op_with_explicit_config_takes_precedence():
    """A direct ``op.config`` wins over kernel introspection."""

    class _ConfigOp:
        config = {"explicit": True}
        kernel = _FakeKernel({"explicit": False})

    BenchmarkReport.record(_ConfigOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_ConfigOp"]
    assert records[0].get("config") == {"explicit": True}


@pytest.mark.full
@pytest.mark.usefixtures('_reset_records')
def test_record_op_without_any_config_omits_field():
    """Ops with no config sources should not produce a ``config`` field."""

    class _BareOp:
        pass

    BenchmarkReport.record(_BareOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_BareOp"]
    assert "config" not in records[0]


@pytest.mark.full
@pytest.mark.usefixtures('_reset_records')
def test_record_string_name_omits_config_field():
    """When called with a benchmark group name, no config is recorded."""

    BenchmarkReport.record("FA3Baseline", params={}, result=_result(), tag="FA3")
    records = BenchmarkReport._records["FA3Baseline"]
    assert "config" not in records[0]
