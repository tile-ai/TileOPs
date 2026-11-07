import ast
import contextlib
from pathlib import Path

import pytest
import torch

from benchmarks.timing import _capture_bench_meta, bench_kernel
from tileops._utils.cupti import (
    AttributionError,
    OffThreadLaunchError,
    RecordsLostError,
    Trace,
    attributed_samples,
    collect_attributed,
)


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
        "via benchmarks.baselines.backward_of:\n  " + "\n  ".join(offenders)
    )


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

    first, second = attributed_samples(kernels, {10: 0, 11: 1}, n_repeat=2)

    assert (first.device_busy_ms, first.latency_ms) == pytest.approx((0.003, 0.006))
    assert (second.device_busy_ms, second.latency_ms) == pytest.approx((0.003, 0.008))
    assert (first.n_kernels, second.n_kernels) == (2, 2)


def test_a_kernel_belongs_to_its_iteration_however_the_timestamps_fall():
    """The whole point of identity: overlap with a neighbour changes nothing."""
    kernels = [
        _kernel(1_000, 9_000, correlation_id=7),
        _kernel(2_000, 3_000, correlation_id=8),
    ]

    first, second = attributed_samples(kernels, {7: 0, 8: 1}, n_repeat=2)

    assert first.device_busy_ms == pytest.approx(0.008)
    assert second.device_busy_ms == pytest.approx(0.001)


def test_busy_counts_overlapping_kernels_once():
    """Concurrent kernels occupy the device once, however they are summed."""
    kernels = [_kernel(1_000, 5_000, 3), _kernel(2_000, 9_000, 3)]

    (sample,) = attributed_samples(kernels, {3: 0}, n_repeat=1)

    assert sample.device_busy_ms == pytest.approx(0.008)
    assert sample.latency_ms == pytest.approx(0.008)


def test_attribution_measures_a_call_whose_kernel_count_varies():
    """A dynamic path launching an extra kernel is measured, not rejected."""
    kernels = [
        _kernel(1_000, 2_000, 4),
        _kernel(10_000, 11_000, 5),
        _kernel(11_500, 13_000, 5),
    ]

    first, second = attributed_samples(kernels, {4: 0, 5: 1}, n_repeat=2)

    assert (first.n_kernels, second.n_kernels) == (1, 2)
    assert second.device_busy_ms == pytest.approx(0.0025)
    assert second.latency_ms == pytest.approx(0.003)


def test_attribution_fails_closed_when_an_iteration_reaches_no_device():
    with pytest.raises(AttributionError, match="CUPTI discarded nothing"):
        attributed_samples([_kernel(1_000, 2_000, 1)], {1: 0}, n_repeat=2)


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

    with pytest.raises(OffThreadLaunchError, match="carry no iteration id"):
        attributed_samples(kernels, iteration_of, n_repeat=1)


def test_a_discarded_record_is_not_reported_as_a_call_that_missed_the_device():
    """The causes of a missing reading get separate errors, so one can be retried."""
    with pytest.raises(RecordsLostError, match="discarded 7 activity records"):
        attributed_samples(
            [_kernel(1_000, 2_000, 1)],
            {1: 0},
            n_repeat=2,
            dropped=7,
        )


def test_an_unreadable_drop_count_rules_nothing_out():
    with pytest.raises(AttributionError, match="count is unproven"):
        attributed_samples(
            [_kernel(1_000, 2_000, 1)],
            {1: 0},
            n_repeat=2,
            dropped=None,
        )


def _collecting(monkeypatch, *traces) -> list:
    """Stub the collector to return *traces* in turn, the last one from then on; returns the
    buffer size each collection asked for."""
    requested = []

    def fake_collect(run_one, n_repeat, prepare_one, buffer_bytes=0):
        requested.append(buffer_bytes)
        return traces[min(len(requested), len(traces)) - 1]

    monkeypatch.setattr("tileops._utils.cupti.collect_repeats", fake_collect)
    return requested


def test_a_phase_that_lost_records_is_measured_again(monkeypatch):
    """A lost reading is re-taken, and the retry asks CUPTI for a larger buffer."""
    lossy = Trace([_kernel(1_000, 2_000, 1)], {1: 0}, 4)
    clean = Trace(
        [_kernel(1_000, 2_000, 1), _kernel(4_100, 4_500, 2)],
        {1: 0, 2: 1},
        0,
    )
    requested_bytes = _collecting(monkeypatch, lossy, clean)
    retries = []

    samples = collect_attributed(lambda i: None, 2, lambda i: None, on_retry=retries.append)

    assert len(samples) == 2
    assert requested_bytes[1] == requested_bytes[0] * 4
    assert retries == [1]


@pytest.mark.parametrize("dropped", [3, None], ids=["counted", "unproven"])
def test_a_discarded_record_no_iteration_misses_keeps_the_readings(monkeypatch, dropped):
    """The timer re-measures only a phase whose loss left a call unmeasured."""
    trace = Trace([_kernel(1_000, 2_000, 1), _kernel(4_100, 4_500, 2)], {1: 0, 2: 1}, dropped)
    attempts = _collecting(monkeypatch, trace)

    samples = collect_attributed(lambda i: None, 2, lambda i: None)

    assert len(samples) == 2
    assert len(attempts) == 1


def test_a_call_that_launched_nothing_does_not_spend_the_retries(monkeypatch):
    """Nothing discarded means re-measuring cannot help, so it fails on sight."""
    attempts = _collecting(monkeypatch, Trace([], {}, 0))

    with pytest.raises(AttributionError, match="CUPTI discarded nothing"):
        collect_attributed(lambda i: None, 1, lambda i: None)
    assert len(attempts) == 1


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_cupti_failure_fails_closed_by_default():
    """A callable launching no CUDA kernel cannot be attributed by CUPTI."""
    with pytest.raises(RuntimeError, match="CUDA-events fallback is disabled"):
        bench_kernel(lambda: sum(range(64)))


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_allow_events_fallback_times_with_events():
    bench_kernel(lambda: sum(range(64)), allow_events_fallback=True)
    assert _capture_bench_meta()["timing"] == "cuda-events"


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


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("events", [False, True])
def test_every_round_resets_then_flushes_then_runs(monkeypatch, events):
    """The reset runs before the flush, so what it writes is evicted, and outside the reading."""
    from benchmarks import timing

    order = []
    flush = timing.flush_l2
    monkeypatch.setattr(timing, "flush_l2", lambda buffer: (order.append("flush"), flush(buffer)))
    x = torch.empty(1 << 24, device="cuda")
    private = torch.empty_like(x)

    def reset():
        order.append("reset")
        private.copy_(x)

    def run():
        order.append("run")
        private.mul_(2)

    if events:
        monkeypatch.setattr(timing, "collect_attributed", _raise_attribution)
    samples = bench_kernel(run, reset=reset, allow_events_fallback=events)
    assert order == ["reset", "flush", "run"] * (len(order) // 3)
    if not events:
        assert all(s.n_kernels == 1 and s.uncounted_copy_ms == 0 for s in samples)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_the_cuda_sweep_reads_every_element_and_writes_none_of_them():
    """A sole nonzero value at block edges, mid-block or in the masked tail reaches the sink."""
    from tileops._utils._l2_sweep import _SINKS, read_sweep

    buffer = torch.zeros(4096 * 3 + 5, dtype=torch.int32, device="cuda")
    for position in (0, 1, 4095, 4096, 2 * 4096 + 17, buffer.numel() - 1):
        buffer.zero_()
        buffer[position] = position + 1
        _SINKS.pop(buffer.device, None)
        before = buffer.clone()

        read_sweep(buffer)

        assert _SINKS[buffer.device].item() == position + 1, position
        assert torch.equal(buffer, before)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_args_on_another_device_than_the_timed_one_are_refused():
    """Every implementation in a row is timed on one device, so a stray tensor is an error."""
    x = torch.empty(16, device="cuda")
    with pytest.raises(ValueError, match="times on cpu"):
        bench_kernel(lambda y: y, args=(x,), device=torch.device("cpu"))


def test_a_flush_resets_persistence_then_reads_the_buffer(monkeypatch):
    """Persisting lines return to normal priority before the read evicts them."""
    from tileops._utils import _l2_sweep, cupti

    order = []
    monkeypatch.setattr(cupti, "reset_persisting_l2_cache", lambda: order.append("persist"))
    monkeypatch.setattr(_l2_sweep, "read_sweep", lambda buffer: order.append(buffer))

    cupti.flush_l2("buffer")

    assert order == ["persist", "buffer"]


def test_the_flush_buffer_of_one_device_is_not_anothers(monkeypatch):
    """Two device types both number their first device 0."""
    from tileops._utils import cupti

    held = torch.empty(1)
    monkeypatch.setattr(cupti, "_L2_FLUSH_BUFFERS", {torch.device("privateuseone", 0): held})
    monkeypatch.setattr(cupti, "_FALLBACK_L2_BYTES", 8)

    assert cupti.l2_flush_buffer(torch.device("cpu")) is not held


def _device_reporting_l2(monkeypatch, l2_bytes: int) -> None:
    properties = type("Properties", (), {"L2_cache_size": l2_bytes})
    module = type("Module", (), {"get_device_properties": staticmethod(lambda i: properties)})
    monkeypatch.setattr(torch, "get_device_module", lambda device_type: module)


@pytest.mark.parametrize("l2_bytes, numel", [(1000, 500), (1001, 501)])
def test_the_flush_buffer_is_twice_l2_of_random_int32_drawn_once(monkeypatch, l2_bytes, numel):
    """Whole int32 elements covering twice L2, drawn without moving the benchmark's RNG."""
    from tileops._utils import cupti

    _device_reporting_l2(monkeypatch, l2_bytes)
    monkeypatch.setattr(cupti, "_L2_FLUSH_BUFFERS", {})
    draws = []
    randint = torch.randint
    monkeypatch.setattr(torch, "randint", lambda *a, **k: draws.append(1) or randint(*a, **k))
    rng = torch.get_rng_state()

    cache = cupti.l2_flush_buffer(torch.device("cpu"))
    again = cupti.l2_flush_buffer(torch.device("cpu"))

    assert torch.equal(torch.get_rng_state(), rng)
    assert (cache.dtype, cache.numel()) == (torch.int32, numel)
    assert cache.unique().numel() > numel // 2
    assert again is cache and len(draws) == 1
    cupti._L2_FLUSH_BUFFERS.clear()
    assert torch.equal(cupti.l2_flush_buffer(torch.device("cpu")), cache)


def test_an_unknown_l2_logs_the_bytes_the_flush_reads(monkeypatch, caplog):
    """The warning names the assumed L2 and the bytes actually allocated, not one for the other."""
    from tileops._utils import cupti

    _device_reporting_l2(monkeypatch, 0)
    monkeypatch.setattr(cupti, "_L2_FLUSH_BUFFERS", {})
    monkeypatch.setattr(cupti, "_FALLBACK_L2_BYTES", 1001)

    with caplog.at_level("WARNING", logger=cupti.__name__):
        cache = cupti.l2_flush_buffer(torch.device("cpu"))

    assert cache.numel() * cache.element_size() == 2004
    assert "assuming 1001 bytes of L2, so each flush reads 2004 bytes" in caplog.text


class _ReadOnlySweep:
    """A flush buffer that answers only the read a flush off CUDA makes, ``max``."""

    def __init__(self, order: list):
        self._order = order

    def max(self):
        self._order.append("torch-max")


class _Event:
    def __init__(self, **_kwargs):
        pass

    def record(self):
        pass

    def elapsed_time(self, _end):
        return 1.0


class _FakeCupti:
    """CUPTI's boundary: the id stack, and a phase's records, which a scenario can spoil.

    ``spoil`` maps a phase number to how it ends: ``lost`` drops a record and counts it,
    ``empty`` returns no record at all, which no retry can mend.
    """

    class ExternalCorrelationKind:
        CUSTOM0 = 0

    def __init__(self, spoil: dict):
        self._ids, self.kernels, self.iteration_of = [], [], {}
        self._spoil, self._phase, self.dropped = spoil, 0, 0

    def activity_push_external_correlation_id(self, _kind, external_id):
        self._ids.append(external_id)

    def activity_pop_external_correlation_id(self, _kind):
        self._ids.pop()

    def launch(self):
        if self._ids:
            correlation_id = len(self.kernels) + 1
            self.kernels.append(_kernel(1_000, 2_000, correlation_id))
            self.iteration_of[correlation_id] = self._ids[-1]

    def flush(self):
        spoil = self._spoil.get(self._phase)
        self._phase += 1
        kernels, iteration_of = self.kernels, self.iteration_of
        self.kernels, self.iteration_of, self.dropped = [], {}, 0
        if spoil == "lost":
            kernels, self.dropped = kernels[:-1], 1
        elif spoil == "empty":
            kernels = []
        return kernels, iteration_of


@pytest.mark.parametrize(
    "device, instrument, spoil, timed_calls, meta",
    [
        ("cuda", "cupti", {}, 2, {"timing": "cupti"}),
        ("cuda", "cupti", {0: "lost"}, 4, {"timing": "cupti", "attribution_retries": 1}),
        ("cuda", "cupti", {0: "empty"}, 4, {"timing": "cuda-events"}),
        ("cuda", "events", {}, 2, {"timing": "events"}),
        ("privateuseone", "events", {}, 2, {"timing": "events"}),
    ],
    ids=["cupti", "retry", "events-fallback", "events", "another-device"],
)
def test_every_call_is_prepared_by_reset_persistence_reset_then_a_read(
    monkeypatch, device, instrument, spoil, timed_calls, meta
):
    """First call, calibration, warm-up, timed repeats, retries and fallback prepare alike.

    Warm-up and timed iterations drain the device between the flush and the call.
    """
    from benchmarks import timing
    from tileops._utils import cupti as collector

    order = []
    cupti = _FakeCupti(spoil)
    monkeypatch.setattr(timing, "l2_flush_buffer", lambda device: _ReadOnlySweep(order))
    monkeypatch.setattr(timing, "flush_l2", lambda buffer: order.append("flush-l2"))
    monkeypatch.setattr(timing, "_instrument_for", lambda device: instrument)
    monkeypatch.setattr(collector, "_load_cupti", lambda: cupti)
    monkeypatch.setattr(collector, "_drop_counter_is_live", lambda: True)
    monkeypatch.setattr(collector, "_read_dropped", lambda: cupti.dropped)
    monkeypatch.setattr(collector, "_phase_session", lambda buffer_bytes: contextlib.nullcontext())
    monkeypatch.setattr(collector, "_flush", cupti.flush)
    monkeypatch.setattr(torch, "Event", _Event)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: order.append("sync"))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: order.append("sync"))

    def run():
        order.append("run")
        cupti.launch()

    timing._bench_kernel(
        torch.device(device, 0),
        run,
        max_iters=2,
        min_iters=2,
        allow_events_fallback=True,
        reset=lambda: order.append("reset"),
    )

    on_cuda = device == "cuda"
    prepared = ["reset", "flush-l2"] if on_cuda else ["reset", "torch-max"]
    assert [step for step in order if step != "sync"] == [*prepared, "run"] * (
        1 + 3 + 2 + timed_calls
    )
    # The first call and the three calibration calls come before warm-up.
    calls = [i for i, step in enumerate(order) if step == "run"]
    assert all(order[i - 1] == "sync" for i in calls[4:])
    recorded = timing._capture_bench_meta()
    assert (recorded.pop("fallback_reason", None) is not None) == (meta["timing"] == "cuda-events")
    flush = timing._L2_FLUSH_CUDA if on_cuda else timing._L2_FLUSH_OTHER
    assert recorded == {**meta, "l2_flush": flush}


def _raise_attribution(*_args, **_kwargs):
    raise AttributionError("forced")
