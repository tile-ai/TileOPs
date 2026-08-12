import pytest
import torch

from benchmarks.benchmark_base import workloads_to_params
from benchmarks.timing import (
    _attributed_samples,
    _CUPTIAttributionError,
    bench_kernel,
)


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


def test_multi_input_op_raises_keyerror():
    """Multi-input ops (q/k/v) raise instead of binding a wrong tensor."""
    with pytest.raises(KeyError, match="exactly one manifest tensor input"):
        workloads_to_params("GroupedQueryAttentionFwdOp")


def _kernel(start_ns: int, end_ns: int) -> dict:
    return {"name": "k", "start_ns": start_ns, "end_ns": end_ns}


def test_attribution_separates_execution_from_the_gaps_and_the_prepare():
    """busy counts execution only; latency additionally spans the gaps between."""
    records = [
        _kernel(1_000, 2_000),          # prepare, outside both windows
        _kernel(4_000, 6_000),
        _kernel(9_000, 10_000),
        _kernel(20_000, 21_000),        # prepare
        _kernel(23_000, 24_000),
        _kernel(29_000, 31_000),
    ]
    windows = [(3_000, 19_000), (22_000, 40_000)]

    first, second = _attributed_samples(records, windows, n_repeat=2)

    assert (first.device_busy_ms, first.latency_ms) == pytest.approx((0.003, 0.006))
    assert (second.device_busy_ms, second.latency_ms) == pytest.approx((0.003, 0.008))
    assert (first.n_kernels, second.n_kernels) == (2, 2)


def test_busy_counts_overlapping_kernels_once():
    """Concurrent kernels occupy the device once, however they are summed."""
    records = [_kernel(1_000, 5_000), _kernel(2_000, 9_000)]

    sample, = _attributed_samples(records, [(0, 10_000)], n_repeat=1)

    assert sample.device_busy_ms == pytest.approx(0.008)
    assert sample.latency_ms == pytest.approx(0.008)


def test_attribution_measures_a_call_whose_kernel_count_varies():
    """A dynamic path launching an extra kernel is measured, not rejected."""
    records = [_kernel(1_000, 2_000), _kernel(10_000, 11_000), _kernel(11_500, 13_000)]

    first, second = _attributed_samples(
        records, [(500, 5_000), (9_000, 15_000)], n_repeat=2,
    )

    assert (first.n_kernels, second.n_kernels) == (1, 2)
    assert second.device_busy_ms == pytest.approx(0.0025)
    assert second.latency_ms == pytest.approx(0.003)


def test_attribution_counts_activity_launched_from_another_thread():
    """A window holds the call's activity whichever CPU thread launched it."""
    records = [_kernel(1_000, 2_000), _kernel(3_000, 7_000)]

    sample, = _attributed_samples(records, [(500, 9_000)], n_repeat=1)

    assert sample.n_kernels == 2
    assert sample.device_busy_ms == pytest.approx(0.005)


def test_attribution_fails_closed_when_an_iteration_reaches_no_device():
    with pytest.raises(_CUPTIAttributionError, match="produced no GPU activity"):
        _attributed_samples(
            [_kernel(1_000, 2_000)], [(500, 3_000), (4_000, 5_000)], n_repeat=2,
        )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_cupti_failure_fails_closed_by_default(monkeypatch):
    """A callable launching no CUDA kernel cannot be attributed by CUPTI."""
    monkeypatch.setenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0")
    with pytest.raises(RuntimeError, match="CUDA-events fallback is disabled"):
        bench_kernel(lambda: sum(range(64)))


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kernel_runtime_error_propagates():
    """Genuine RuntimeErrors must reach the caller, not the fallback path."""
    def boom():
        raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        bench_kernel(boom)
