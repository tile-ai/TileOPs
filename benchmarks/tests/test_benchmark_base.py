"""Unit tests for benchmarks.benchmark_base.

Verifies that the generic ``BenchmarkBase`` / ``ManifestBenchmark`` accept
any duck-typed workload rather than requiring ``WorkloadBase`` inheritance.
"""

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    _attributed_mean_latency_ms,
    _bench_meta,
    _kernel_span_us,
    _NativeCUPTIAttributionError,
    _select_expected_sequence,
    _ShiftingTensorPool,
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
    """Object carrying shape, dtype and gen_inputs()."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self):
        return (torch.randn(*self.shape, dtype=self.dtype),)


class _FakeRooflineOp:
    """Minimal op-like object for ManifestBenchmark unit tests."""

    def __init__(self, roofline: tuple[int, int] = (128, 256)):
        self.calls = 0
        self._roofline = roofline

    def eval_roofline(self) -> tuple[int, int]:
        self.calls += 1
        return self._roofline


# ManifestBenchmark contract tests


@pytest.mark.smoke
def test_manifest_benchmark_accepts_duck_typed_workload():
    """ManifestBenchmark reads roofline off the op, never off the workload."""
    w = _DuckShapeDtype((4, 8, 1024), torch.float16)
    op = _FakeRooflineOp((123, 456))
    bm = ManifestBenchmark("TestOp", op, w)
    assert bm.workload is w
    assert bm.calculate_flops() == 123.0
    assert bm.calculate_memory() == 456.0
    assert op.calls == 1


# WorkloadBase compatibility


@pytest.mark.smoke
def test_manifest_benchmark_accepts_workload_base_subclass():
    """A nominal WorkloadBase subclass works the same as a duck-typed one."""
    from workloads.workload_base import WorkloadBase

    class _ConcreteWorkload(WorkloadBase):
        def __init__(self):
            self.shape = (4, 8)
            self.dtype = torch.float32

        def gen_inputs(self):
            return (torch.randn(*self.shape, dtype=self.dtype),)

    w = _ConcreteWorkload()
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


def test_kernel_span_uses_activity_envelope():
    kernels = [
        {"name": "first", "start_ns": 1000, "end_ns": 9000},
        {"name": "second", "start_ns": 5000, "end_ns": 7000},
    ]
    assert _kernel_span_us(kernels) == 8.0


def _kernel(name: str, start_ns: int, end_ns: int) -> dict:
    return {"name": name, "start_ns": start_ns, "end_ns": end_ns}


def test_select_expected_sequence_accepts_exact_and_concurrent_reorder():
    exact = [_kernel("a", 1_000, 2_000), _kernel("b", 2_000, 3_000)]
    reordered = [_kernel("b", 1_000, 3_000), _kernel("a", 1_500, 2_500)]

    assert _select_expected_sequence(exact, ("a", "b")) == exact
    assert _select_expected_sequence(reordered, ("a", "b")) == reordered


def test_select_expected_sequence_rejects_serial_reorder():
    reordered = [_kernel("b", 1_000, 2_000), _kernel("a", 2_000, 3_000)]

    assert _select_expected_sequence(reordered, ("a", "b")) is None


def test_select_expected_sequence_handles_duplicate_kernel_names():
    kernels = [
        _kernel("a", 1_000, 2_000),
        _kernel("b", 1_500, 3_000),
        _kernel("a", 2_000, 2_500),
    ]

    assert _select_expected_sequence(kernels, ("a", "a", "b")) == kernels
    assert _select_expected_sequence(kernels, ("a", "b", "b")) is None


@pytest.mark.parametrize(
    "actual",
    [
        [_kernel("a", 1_000, 2_000)],
        [
            _kernel("a", 1_000, 2_000),
            _kernel("b", 2_000, 3_000),
            _kernel("unexpected", 3_000, 4_000),
        ],
        [_kernel("a", 1_000, 2_000), _kernel("a", 2_000, 3_000)],
    ],
)
def test_select_expected_sequence_rejects_incomplete_or_changed_call(actual):
    assert _select_expected_sequence(actual, ("a", "b")) is None


def test_attributed_latency_requires_every_repeat():
    trace = {
        "dropped": 0,
        "kernels": [
            _kernel("a", 2_000, 4_000),
            _kernel("b", 3_000, 8_000),
            _kernel("a", 21_000, 22_000),
            _kernel("b", 23_000, 29_000),
        ],
    }

    latency_ms = _attributed_mean_latency_ms(trace, ("a", "b"), n_repeat=2)

    assert latency_ms == pytest.approx(0.007)
    assert _bench_meta.cupti_sampled_calls == 2
    assert _bench_meta.cupti_expected_kernel_count == 2

    trace["kernels"].append(_kernel("unexpected", 29_000, 29_500))
    with pytest.raises(
        _NativeCUPTIAttributionError,
        match="activity count does not match",
    ):
        _attributed_mean_latency_ms(trace, ("a", "b"), n_repeat=2)


def test_case_sequence_attribution_excludes_prepare_and_preserves_operator_gap():
    trace = {
        "dropped": 0,
        "kernels": [
            _kernel("copy", 1_000, 2_000),
            _kernel("fill", 2_100, 3_000),
            _kernel("op-a", 4_000, 6_000),
            _kernel("op-b", 9_000, 10_000),
            _kernel("copy", 20_000, 21_000),
            _kernel("fill", 21_100, 22_000),
            _kernel("op-a", 23_000, 24_000),
            _kernel("op-b", 29_000, 31_000),
        ],
    }

    latency_ms = _attributed_mean_latency_ms(
        trace,
        ("op-a", "op-b"),
        n_repeat=2,
        expected_prepare_sequence=("copy", "fill"),
    )

    # Operator envelopes are 6 us and 8 us. Prepare kernels are validated but
    # excluded, while the 3/5 us inter-kernel gaps remain part of op latency.
    assert latency_ms == pytest.approx(0.007)


@pytest.mark.parametrize(
    "kernels",
    [
        [
            _kernel("fill", 1_000, 2_000),
            _kernel("op", 3_000, 4_000),
            _kernel("fill", 5_000, 6_000),
        ],
        [
            _kernel("fill", 1_000, 2_000),
            _kernel("op", 3_000, 4_000),
            _kernel("extra", 4_100, 4_200),
            _kernel("fill", 5_000, 6_000),
            _kernel("op", 7_000, 8_000),
        ],
    ],
)
def test_case_sequence_attribution_fails_closed_on_missing_or_extra_activity(kernels):
    with pytest.raises(_NativeCUPTIAttributionError, match="activity count does not match"):
        _attributed_mean_latency_ms(
            {"dropped": 0, "kernels": kernels},
            ("op",),
            n_repeat=2,
            expected_prepare_sequence=("fill",),
        )


def test_attributed_latency_rejects_dropped_records():
    with pytest.raises(_NativeCUPTIAttributionError, match="dropped 1 records"):
        _attributed_mean_latency_ms(
            {"dropped": 1, "kernels": []},
            ("a",),
            n_repeat=1,
        )


def test_shifting_tensor_pool_preserves_layout_values_and_alignment():
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    pool = _ShiftingTensorPool((source, 7), total_iterations=3, seed=123)
    pointers = []

    for _ in range(3):
        shifted, scalar = pool.next_args()
        assert scalar == 7
        assert shifted.stride() == source.stride()
        torch.testing.assert_close(shifted, source)
        pointers.append(shifted.data_ptr())
        shifted.zero_()

    assert len(set(pointers)) == 3
    assert all(
        (pointer - pointers[0]) % _ShiftingTensorPool._POOL_ALIGNMENT == 0
        for pointer in pointers[1:]
    )
    expected = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    torch.testing.assert_close(source, expected)
    with pytest.raises(RuntimeError, match="ShiftingTensorPool exhausted"):
        pool.next_args()


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_cupti_failure_fails_closed_by_default(monkeypatch):
    """A callable launching no CUDA kernel cannot be attributed by CUPTI."""
    monkeypatch.setenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0")
    with pytest.raises(RuntimeError, match="CUDA-events fallback is disabled"):
        bench_kernel(lambda: sum(range(64)), n_warmup=1, n_repeat=2, n_trials=1)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_cupti_failure_falls_back_when_enabled(monkeypatch):
    """CUDA-event fallback remains available for local diagnosis."""
    monkeypatch.setenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "1")
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
def test_record_composite_op_keeps_delegate_kernel_config():
    """A composite that owns no kernels still reports the delegate's config."""

    class _DelegateOp:
        def __init__(self):
            self._kernel_cache = {torch.float16: _FakeKernel({"block_m": 8})}

    class _CompositeOp:
        def __init__(self):
            self._delegate = _DelegateOp()

        @property
        def _kernel_cache(self):
            return self._delegate._kernel_cache

    BenchmarkReport.record(_CompositeOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_CompositeOp"]
    assert records[0].get("config") == {"block_m": 8}


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
