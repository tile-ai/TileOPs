"""Unit tests for ``BenchmarkReport.record`` op-config extraction."""

import pytest

from benchmarks.benchmark_base import BenchmarkReport

pytestmark = pytest.mark.full


@pytest.fixture(autouse=True)
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


def test_record_eager_init_op_keeps_kernel_config():
    """Pattern 1: ``op.kernel`` set in ``__init__`` (GemmOp-style)."""

    class _EagerOp:
        def __init__(self):
            self.kernel = _FakeKernel({"block_m": 128, "block_n": 256})

    BenchmarkReport.record(_EagerOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_EagerOp"]
    assert records[0].get("config") == {"block_m": 128, "block_n": 256}


def test_record_lazy_with_dummy_kernel_keeps_kernel_config():
    """Pattern 2: dummy ``op.kernel`` plus a populated ``_kernel_cache``."""

    class _LazyDummyOp:
        def __init__(self):
            self.kernel = _FakeKernel({"block_m": 8})
            self._kernel_cache = {1: self.kernel}

    BenchmarkReport.record(_LazyDummyOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_LazyDummyOp"]
    assert records[0].get("config") == {"block_m": 8}


def test_record_pure_lazy_cache_op_keeps_kernel_config():
    """Pattern 3: only ``_kernel_cache`` is populated."""

    class _PureLazyOp:
        def __init__(self):
            self._kernel_cache = {(32, 256): _FakeKernel({"block_m": 4, "tile_n": 0})}

    BenchmarkReport.record(_PureLazyOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_PureLazyOp"]
    assert records[0].get("config") == {"block_m": 4, "tile_n": 0}


def test_record_op_with_explicit_config_takes_precedence():
    """A direct ``op.config`` wins over kernel introspection."""

    class _ConfigOp:
        config = {"explicit": True}
        kernel = _FakeKernel({"explicit": False})

    BenchmarkReport.record(_ConfigOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_ConfigOp"]
    assert records[0].get("config") == {"explicit": True}


def test_record_op_without_any_config_omits_field():
    """Ops with no config sources should not produce a ``config`` field."""

    class _BareOp:
        pass

    BenchmarkReport.record(_BareOp(), params={}, result=_result(), tag="t")
    records = BenchmarkReport._records["_BareOp"]
    assert "config" not in records[0]


def test_record_string_name_omits_config_field():
    """When called with a benchmark group name, no config is recorded."""

    BenchmarkReport.record("FA3Baseline", params={}, result=_result(), tag="FA3")
    records = BenchmarkReport._records["FA3Baseline"]
    assert "config" not in records[0]
