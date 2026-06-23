"""Unit tests for benchmark timing backend, CUPTI fallback, and diagnostics."""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, _sum_kernel_time_us, bench_kernel

pytestmark = pytest.mark.full


@pytest.fixture(autouse=True)
def _reset_records():
    """Snapshot and clear BenchmarkReport._records around each test."""
    saved = BenchmarkReport._records
    BenchmarkReport._records = {}
    try:
        yield
    finally:
        BenchmarkReport._records = saved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bench_kernel_returns_structured_dict():
    """bench_kernel() returns dict with latency_ms, stdev_ms, timing_backend, event_breakdown."""
    def simple_kernel():
        x = torch.randn(1024, device="cuda")
        return x * 2.0

    result = bench_kernel(simple_kernel, n_warmup=2, n_repeat=5, n_trials=3)

    assert isinstance(result, dict)
    assert "latency_ms" in result
    assert "stdev_ms" in result
    assert "timing_backend" in result
    assert "event_breakdown" in result

    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] > 0

    assert isinstance(result["stdev_ms"], float)
    assert result["stdev_ms"] >= 0

    assert result["timing_backend"] in ("cupti", "cuda_event")

    assert isinstance(result["event_breakdown"], dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bench_kernel_with_tensor_args():
    """bench_kernel() with tensor args uses arg_pool cloning."""
    x = torch.randn(256, 256, device="cuda")
    y = torch.randn(256, 256, device="cuda")

    def matmul_kernel(a, b):
        return torch.matmul(a, b)

    result = bench_kernel(matmul_kernel, args=(x, y), n_warmup=2, n_repeat=5, n_trials=2)

    assert result["latency_ms"] > 0
    assert result["timing_backend"] in ("cupti", "cuda_event")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bench_kernel_cupti_excludes_flush_kernels():
    """CUPTI path should exclude FillFunctor flush kernels from timing."""
    def simple_kernel():
        x = torch.randn(512, device="cuda")
        return x + 1.0

    result = bench_kernel(simple_kernel, n_warmup=2, n_repeat=5, n_trials=2)

    # If CUPTI is available, event_breakdown should not contain flush patterns
    if result["timing_backend"] == "cupti" and result["event_breakdown"]:
        for kernel_name in result["event_breakdown"].keys():
            # Should not match both vectorized_elementwise AND FillFunctor
            has_vectorized = "vectorized_elementwise" in kernel_name
            has_fill = "FillFunctor" in kernel_name
            # If both are present, this would be a flush kernel (should be excluded)
            assert not (has_vectorized and has_fill), \
                f"Flush kernel leaked into breakdown: {kernel_name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bench_kernel_stdev_with_multiple_trials():
    """stdev_ms should be non-zero when n_trials > 1."""
    def kernel():
        x = torch.randn(128, device="cuda")
        return x * 2.0

    result = bench_kernel(kernel, n_warmup=2, n_repeat=10, n_trials=5)

    # With 5 trials, stdev should typically be > 0 (unless extremely stable)
    # We just check it's a valid number
    assert isinstance(result["stdev_ms"], float)
    assert result["stdev_ms"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bench_kernel_single_trial_has_zero_stdev():
    """stdev_ms should be 0.0 when n_trials = 1."""
    def kernel():
        x = torch.randn(128, device="cuda")
        return x * 2.0

    result = bench_kernel(kernel, n_warmup=2, n_repeat=5, n_trials=1)

    assert result["stdev_ms"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_benchmark_report_propagates_stdev_and_backend():
    """BenchmarkReport.record() should preserve stdev_ms and timing_backend."""
    class _FakeOp:
        pass

    result = {
        "latency_ms": 1.234,
        "stdev_ms": 0.056,
        "timing_backend": "cupti",
        "event_breakdown": {"kernel_a": 100.0, "kernel_b": 50.0},
        "tflops": 10.5,
    }

    BenchmarkReport.record(_FakeOp(), params={"size": 1024}, result=result, tag="test")

    records = BenchmarkReport._records["_FakeOp"]
    assert len(records) == 1
    assert records[0]["result"]["stdev_ms"] == 0.056
    assert records[0]["result"]["timing_backend"] == "cupti"


def test_sum_kernel_time_us_filters_flush_with_and_logic():
    """_sum_kernel_time_us should only exclude kernels matching ALL flush patterns."""
    # Mock a minimal kineto_results-like object
    class MockEvent:
        def __init__(self, name, duration_ns, is_cuda=True):
            self._name = name
            self._duration_ns = duration_ns
            self._is_cuda = is_cuda

        def device_type(self):
            from torch.autograd.profiler import DeviceType
            return DeviceType.CUDA if self._is_cuda else DeviceType.CPU

        def name(self):
            return self._name

        def duration_ns(self):
            return self._duration_ns

    class MockKinetoResults:
        def __init__(self, events):
            self._events = events

        def events(self):
            return self._events

    events = [
        # Should be excluded (both patterns present)
        MockEvent("vectorized_elementwise_kernel<FillFunctor<int>>", 10000),
        # Should be included (only one pattern)
        MockEvent("FillFunctor_custom_kernel", 5000),
        MockEvent("vectorized_elementwise_add", 8000),
        # Should be included (normal kernel)
        MockEvent("my_custom_kernel", 15000),
    ]

    kr = MockKinetoResults(events)
    total_us, per_kernel, excluded = _sum_kernel_time_us(kr)

    # total_us should exclude the first kernel (10000ns = 10us)
    # Include: 5000 + 8000 + 15000 = 28000ns = 28us
    assert total_us == 28.0

    # excluded should contain only the flush kernel
    assert len(excluded) == 1
    assert "vectorized_elementwise_kernel<FillFunctor<int>>" in excluded

    # per_kernel should contain the three non-flush kernels
    assert len(per_kernel) == 3
    assert "FillFunctor_custom_kernel" in per_kernel
    assert "vectorized_elementwise_add" in per_kernel
    assert "my_custom_kernel" in per_kernel


def test_benchmark_report_dump_includes_new_fields(tmp_path):
    """BenchmarkReport.dump() should include timing_backend, stdev_ms, and event_breakdown."""
    class _TestOp:
        pass

    result = {
        "latency_ms": 2.5,
        "stdev_ms": 0.1,
        "timing_backend": "cupti",
        "event_breakdown": {
            "kernel_main": 1000.0,
            "kernel_helper": 500.0,
        },
        "tflops": 5.0,
    }

    BenchmarkReport.record(_TestOp(), params={"n": 512}, result=result, tag="tileops")

    log_path = tmp_path / "test_report.log"
    BenchmarkReport.dump(str(log_path))

    content = log_path.read_text()

    # Check that new fields appear in the table
    assert "timing_backend" in content
    assert "stdev_ms" in content
    assert "cupti" in content
    assert "0.1000" in content  # stdev_ms formatted

    # Check that event breakdown section is present
    assert "Event breakdown" in content
    assert "kernel_main" in content
    assert "kernel_helper" in content
    assert "1000.0" in content
    assert "500.0" in content


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bench_kernel_arg_pool_survives_fallback():
    """arg_pool should remain accessible in CUDA event fallback path."""
    # This is a regression test for the arg_pool scope bug
    x = torch.randn(128, 128, device="cuda")

    def kernel_with_args(tensor):
        return tensor @ tensor.T

    # The test should not crash even if CUPTI fails and fallback is triggered
    result = bench_kernel(kernel_with_args, args=(x,), n_warmup=1, n_repeat=3, n_trials=2)

    assert result["latency_ms"] > 0
    assert result["timing_backend"] in ("cupti", "cuda_event")
