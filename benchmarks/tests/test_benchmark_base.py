import ast
from pathlib import Path

import pytest
import torch

from benchmarks.benchmark_base import ManifestBenchmark, workloads_to_params
from benchmarks.timing import (
    Trace,
    _attributed_samples,
    _bench_meta,
    _capture_bench_meta,
    _collect_attributed,
    _CUPTIAttributionError,
    _CUPTIRecordsLostError,
    _OffThreadLaunchError,
    bench_kernel,
)
from tileops.manifest import load_workloads


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


def test_an_op_class_names_its_own_workloads():
    """A caller holding the Op class does not have to repeat its name as a string."""
    from tileops.ops.reduction.reduce import SumFwdOp

    assert load_workloads(SumFwdOp) == load_workloads("SumFwdOp")
    assert workloads_to_params(SumFwdOp) == workloads_to_params("SumFwdOp")


def test_no_bench_reaches_its_gradients_through_the_autograd_engine():
    """A backward baseline drives its own node; the engine launches on its own thread.

    Cheap where the alternative is not: catching this by running the benchmarks costs
    the nightly 36 minutes, and no PR-stage job executes ``benchmarks/ops`` at all.
    """
    ops_dir = Path(__file__).resolve().parents[1] / "ops"
    offenders = []
    for path in sorted(ops_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            call = ast.unparse(node.func)
            if node.func.attr == "backward" or call.endswith("autograd.grad"):
                offenders.append(f"{path.name}:{node.lineno}: {call}()")

    assert not offenders, (
        "these launch their kernels from autograd's engine thread, where the timer "
        "cannot tell which iteration they belong to; call the backward node instead, "
        "via benchmarks.benchmark_base.backward_of:\n  " + "\n  ".join(offenders)
    )


def test_multi_input_op_raises_keyerror():
    """Multi-input ops (q/k/v) raise instead of binding a wrong tensor."""
    with pytest.raises(KeyError, match="exactly one manifest tensor input"):
        workloads_to_params("GroupedQueryAttentionDenseFwdOp")


def _kernel(start_ns: int, end_ns: int, correlation_id: int = 0) -> dict:
    return {"name": "k", "start_ns": start_ns, "end_ns": end_ns, "correlation_id": correlation_id}


def test_attribution_separates_execution_from_the_gaps_between_kernels():
    """busy counts execution only; latency additionally spans the gaps between."""
    kernels = [
        _kernel(4_000, 6_000, correlation_id=10),
        _kernel(9_000, 10_000, correlation_id=10),
        _kernel(23_000, 24_000, correlation_id=11),
        _kernel(29_000, 31_000, correlation_id=11),
    ]

    first, second = _attributed_samples(kernels, {10: 0, 11: 1}, n_repeat=2)

    assert (first.device_busy_ms, first.latency_ms) == pytest.approx((0.003, 0.006))
    assert (second.device_busy_ms, second.latency_ms) == pytest.approx((0.003, 0.008))
    assert (first.n_kernels, second.n_kernels) == (2, 2)


def test_a_kernel_belongs_to_its_iteration_however_the_timestamps_fall():
    """The whole point of identity: overlap with a neighbour changes nothing."""
    kernels = [
        _kernel(1_000, 9_000, correlation_id=7),
        _kernel(2_000, 3_000, correlation_id=8),
    ]

    first, second = _attributed_samples(kernels, {7: 0, 8: 1}, n_repeat=2)

    assert first.device_busy_ms == pytest.approx(0.008)
    assert second.device_busy_ms == pytest.approx(0.001)


def test_busy_counts_overlapping_kernels_once():
    """Concurrent kernels occupy the device once, however they are summed."""
    kernels = [_kernel(1_000, 5_000, 3), _kernel(2_000, 9_000, 3)]

    (sample,) = _attributed_samples(kernels, {3: 0}, n_repeat=1)

    assert sample.device_busy_ms == pytest.approx(0.008)
    assert sample.latency_ms == pytest.approx(0.008)


def test_attribution_measures_a_call_whose_kernel_count_varies():
    """A dynamic path launching an extra kernel is measured, not rejected."""
    kernels = [
        _kernel(1_000, 2_000, 4),
        _kernel(10_000, 11_000, 5),
        _kernel(11_500, 13_000, 5),
    ]

    first, second = _attributed_samples(kernels, {4: 0, 5: 1}, n_repeat=2)

    assert (first.n_kernels, second.n_kernels) == (1, 2)
    assert second.device_busy_ms == pytest.approx(0.0025)
    assert second.latency_ms == pytest.approx(0.003)


def test_attribution_fails_closed_when_an_iteration_reaches_no_device():
    with pytest.raises(_CUPTIAttributionError, match="CUPTI discarded nothing"):
        _attributed_samples([_kernel(1_000, 2_000, 1)], {1: 0}, n_repeat=2)


@pytest.mark.parametrize(
    "iteration_of",
    [
        {1: 0},  # the second kernel carries an id nothing mapped
        {1: 0, 99: 7},  # ... or one from outside the range this phase pushed
    ],
)
def test_a_kernel_no_iteration_pushed_is_named_as_such(iteration_of):
    """Autograd's engine thread never sees the pushed id, and that is not a lost record."""
    kernels = [_kernel(1_000, 2_000, 1), _kernel(3_000, 4_000, 99)]

    with pytest.raises(_OffThreadLaunchError, match="carry no iteration id"):
        _attributed_samples(kernels, iteration_of, n_repeat=1)


def test_a_discarded_record_is_not_reported_as_a_call_that_missed_the_device():
    """The causes of a missing reading get separate errors, so one can be retried."""
    with pytest.raises(_CUPTIRecordsLostError, match="discarded 7 activity records"):
        _attributed_samples(
            [_kernel(1_000, 2_000, 1)],
            {1: 0},
            n_repeat=2,
            dropped=7,
        )


def test_an_unreadable_drop_count_rules_nothing_out():
    with pytest.raises(_CUPTIAttributionError, match="count is unproven"):
        _attributed_samples(
            [_kernel(1_000, 2_000, 1)],
            {1: 0},
            n_repeat=2,
            dropped=None,
        )


def test_a_phase_that_lost_records_is_measured_again(monkeypatch):
    """A lost reading is re-taken, and the retry asks CUPTI for a larger buffer."""
    lossy = Trace([_kernel(1_000, 2_000, 1)], {1: 0}, 4)
    clean = Trace(
        [_kernel(1_000, 2_000, 1), _kernel(4_100, 4_500, 2)],
        {1: 0, 2: 1},
        0,
    )
    traces = [lossy, clean]
    requested_bytes = []
    monkeypatch.setattr(_bench_meta, "attribution_retries", None, raising=False)

    def fake_collect(run_one, n_repeat, prepare_one, buffer_bytes=0, count_copies=False):
        requested_bytes.append(buffer_bytes)
        return traces.pop(0)

    monkeypatch.setattr("benchmarks.timing.collect_repeats", fake_collect)

    samples = _collect_attributed(lambda i: None, 2, lambda i: None)

    assert len(samples) == 2
    assert requested_bytes[1] == requested_bytes[0] * 4
    assert _capture_bench_meta()["attribution_retries"] == 1


def test_a_call_that_launched_nothing_does_not_spend_the_retries(monkeypatch):
    """Nothing discarded means re-measuring cannot help, so it fails on sight."""
    attempts = []

    def fake_collect(run_one, n_repeat, prepare_one, buffer_bytes=0, count_copies=False):
        attempts.append(buffer_bytes)
        return Trace([], {}, 0)

    monkeypatch.setattr("benchmarks.timing.collect_repeats", fake_collect)

    with pytest.raises(_CUPTIAttributionError, match="CUPTI discarded nothing"):
        _collect_attributed(lambda i: None, 1, lambda i: None)
    assert len(attempts) == 1


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_cupti_failure_fails_closed_by_default(monkeypatch):
    """A callable launching no CUDA kernel cannot be attributed by CUPTI."""
    monkeypatch.setenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "0")
    with pytest.raises(RuntimeError, match="CUDA-events fallback is disabled"):
        bench_kernel(lambda: sum(range(64)))


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_copy_counts_only_where_the_case_asks_for_copies():
    """A copy counts as device work only where the case asks, and is reported otherwise."""
    x = torch.empty(8 * 1024 * 1024, device="cuda", dtype=torch.float16)

    def clone_then_scale():
        y = x.clone()
        y.mul_(2)
        return y

    left_out = bench_kernel(clone_then_scale)
    assert all(s.n_kernels == 1 and s.uncounted_copy_ms > 0 for s in left_out)

    counted = bench_kernel(clone_then_scale, count_copies=True)
    assert all(s.n_kernels == 2 and s.uncounted_copy_ms == 0 for s in counted)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kernel_runtime_error_propagates():
    """Genuine RuntimeErrors must reach the caller, not the fallback path."""

    def boom():
        raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        bench_kernel(boom)


class SumFwdOp:
    """Stands in for the manifest op of that name; only the class name is read."""


class NotAManifestOp:
    """A wrapper of the kind a benchmark must not report under."""


@pytest.mark.smoke
def test_manifest_benchmark_takes_its_name_from_the_op():
    """The report name is the op's class, so it cannot disagree with what ran."""
    assert ManifestBenchmark(SumFwdOp(), object()).op_name == "SumFwdOp"


@pytest.mark.smoke
def test_manifest_benchmark_refuses_an_op_the_manifest_does_not_declare():
    """A wrapper or a subclass would report numbers under a name no spec knows."""
    with pytest.raises(KeyError, match="NotAManifestOp"):
        ManifestBenchmark(NotAManifestOp(), object())
