import ast
import contextlib
import itertools
import statistics
import time
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

    def fake_collect(run_one, n_repeat, prepare_one, buffer_bytes=0, finish_one=None):
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
def test_a_slow_host_leaves_no_device_idle_between_samples():
    """A call that spends 2 ms on the host before its kernel still runs its samples back to
    back: each starts when the previous one ends, not once the host has launched it. Every
    sample carries the SM clock it ran at."""
    x = torch.empty(1 << 20, device="cuda")
    starts = []

    def reset():
        starts.append(torch.Event(enable_timing=True))
        starts[-1].record()

    def run():
        time.sleep(2e-3)
        x.mul_(2)

    samples = bench_kernel(run, reset=reset)
    torch.accelerator.synchronize()

    gaps = [a.elapsed_time(b) for a, b in itertools.pairwise(starts[-len(samples) :])]
    # Had the device waited for the host, every gap would hold the host's 2 ms.
    assert statistics.median(gaps) < 1.0
    clocks = [s.sm_clock_hz for s in samples]
    assert all(hz is not None and hz > 0 for hz in clocks)
    # One sample spans a call of a few microseconds, so the step of the global timer leaves
    # its clock a few percent off either way; the median carries no such error.
    peak_hz = torch.cuda.get_device_properties(x.device).clock_rate * 1e3
    assert statistics.median(clocks) <= peak_hz * 1.01


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_the_host_sees_a_blocker_time_out_while_the_stream_still_runs():
    """A block the host releases did not time out; one it leaves held releases itself, and
    the host reads that before draining the stream, which is how it counts what was queued."""
    from benchmarks.cold import StreamBlocker

    blocker = StreamBlocker(timeout_s=0.05)
    blocker.block()
    blocker.unblock()
    torch.accelerator.synchronize()
    assert not blocker.timed_out()

    blocker.block()
    # Work still queued behind it: a second blocker, released by the host.
    behind = StreamBlocker(timeout_s=5.0)
    behind.block()
    deadline = time.monotonic() + 5.0
    while not blocker.timed_out() and time.monotonic() < deadline:
        time.sleep(1e-3)
    seen, running = blocker.timed_out(), not torch.cuda.current_stream().query()
    behind.unblock()
    torch.accelerator.synchronize()
    assert seen and running and not behind.timed_out()


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
    from tileops._utils import _l2_sweep

    order = []
    sweep = _l2_sweep.read_sweep
    monkeypatch.setattr(
        _l2_sweep, "read_sweep", lambda buffer: (order.append("flush"), sweep(buffer))
    )
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
def test_the_cuda_sweep_reads_evict_last_then_demotes_its_lines_to_normal():
    """A plain read leaves a previous kernel's evict-last lines in L2; an evict-last read
    replaces them, and its own lines must not stay evict-last for the timed call."""
    from tileops._utils._l2_sweep import _L2_LINE_BYTES, _fold_max

    buffer = torch.zeros(4096 * 2, dtype=torch.int32, device="cuda")
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    ptx = _fold_max[(2,)](
        buffer,
        sink,
        buffer.numel(),
        buffer.numel() // 2,
        BLOCK=4096,
        LINE=_L2_LINE_BYTES // buffer.element_size(),
    ).asm["ptx"]

    loads = [line for line in ptx.splitlines() if "ld.global" in line]
    assert loads and all("L2::cache_hint" in line for line in loads)
    assert "createpolicy.fractional.L2::evict_last" in ptx
    assert "applypriority.global.L2::evict_normal" in ptx


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_args_on_another_device_than_the_timed_one_are_refused():
    """Every implementation in a row is timed on one device, so a stray tensor is an error."""
    x = torch.empty(16, device="cuda")
    with pytest.raises(ValueError, match="times on cpu"):
        bench_kernel(lambda y: y, args=(x,), device=torch.device("cpu"))


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
    ``counted`` counts a dropped record while every call kept one,
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
        elif spoil == "counted":
            self.dropped = 1
        elif spoil == "empty":
            kernels = []
        return kernels, iteration_of


class _Blocker:
    """The blocking kernel's host side, logging into *order*. The queue behind it takes
    *fits* samples, or behind each block in turn the next of a list of them, counting the
    block that loads it before warm-up; queueing one more makes it release on its own, as a
    full queue does. Each block whose entry in *late* is true releases on its own after the
    host queued every sample, before the host releases it."""

    def __init__(self, order: list, fits: int | list | None = None, late: list | None = None):
        self._order, self._fits = order, iter(fits) if isinstance(fits, list) else fits
        self.held, self.queued, self._timed_out, self._room = False, 0, False, None
        self._late, self._releases_late = iter(late or []), False

    def block(self):
        self._order.append("block")
        self.held, self.queued, self._timed_out = True, 0, False
        self._room = next(self._fits, None) if hasattr(self._fits, "__next__") else self._fits
        self._releases_late = next(self._late, False)

    def unblock(self):
        if self.held:
            self._order.append("unblock")
            self._timed_out = self._timed_out or self._releases_late
        self.held = False

    def timed_out(self):
        return self._timed_out

    def queue(self):
        if self.held:
            self.queued += 1
            if self._room is not None and self.queued > self._room:
                self._timed_out = True


class _Clocks:
    """SM 0's clocks, each sample read at 1.5 GHz."""

    def __init__(self, n, _device):
        pass

    def clear(self):
        pass

    def start(self, _i):
        pass

    def stop(self, _i):
        pass

    def read_hz(self, n):
        return [1.5e9] * n


def _time_on_fakes(
    monkeypatch, device, spoil, fits=None, call=None, repeats=2, late=None, start_block=None
):
    """Run ``_bench_kernel`` on stand-ins for every device-facing piece, timing *call* (by
    default one that only launches) *repeats* times; returns the steps it took in order, and
    its samples."""
    from benchmarks import cold, timing
    from tileops._utils import _l2_sweep
    from tileops._utils import cupti as collector

    order = []
    cupti = _FakeCupti(spoil)
    blocker = _Blocker(order, fits, late)
    monkeypatch.setattr(timing, "l2_flush_buffer", lambda device: _ReadOnlySweep(order))
    monkeypatch.setattr(_l2_sweep, "read_sweep", lambda buffer: order.append("read"))
    monkeypatch.setattr(timing, "reset_persisting_l2_cache", lambda: order.append("persist"))
    monkeypatch.setattr(timing, "_BLOCKERS", {})
    monkeypatch.setattr(cold, "StreamBlocker", lambda timeout_s: blocker)
    monkeypatch.setattr(cold, "SmClocks", _Clocks)
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
        if call is not None:
            call()
        order.append("run")
        blocker.queue()
        cupti.launch()

    samples = timing._bench_kernel(
        torch.device(device, 0),
        run,
        max_iters=repeats,
        min_iters=repeats,
        allow_events_fallback=True,
        reset=lambda: order.append("reset"),
        start_block=start_block,
    )
    return order, samples


def _held_spans(order: list) -> list[tuple[int, int]]:
    """Each block and the unblock that ends it, as indices into *order*."""
    blocks = [i for i, step in enumerate(order) if step == "block"]
    return [(b, order.index("unblock", b)) for b in blocks]


@pytest.mark.parametrize(
    "spoil, timed_calls, meta",
    [
        ({}, 2, {"timing": "cupti"}),
        ({0: "lost"}, 4, {"timing": "cupti", "attribution_retries": 1}),
        ({0: "counted"}, 4, {"timing": "cupti", "attribution_retries": 1}),
        ({0: "empty"}, 4, {"timing": "cuda-events"}),
    ],
    ids=["cupti", "retry", "retry-short-call", "events-fallback"],
)
def test_every_call_is_prepared_by_reset_then_a_read(monkeypatch, spoil, timed_calls, meta):
    """First call, calibration, warm-up, timed repeats, retries and fallback prepare alike.

    Every timed call is queued behind the blocking kernel with no drain between calls; the
    kernel loads before warm-up, whose calls it never holds and which are drained before and
    after; persisting L2 is reset before each block.
    """
    from benchmarks import timing

    order, _ = _time_on_fakes(monkeypatch, "cuda", spoil)

    steps = [step for step in order if step not in ("sync", "persist", "block", "unblock")]
    assert steps == ["reset", "read", "run"] * (1 + 3 + 2 + timed_calls)
    calls = [i for i, step in enumerate(order) if step == "run"]
    # The first call and the three calibration calls come before the two warm-up calls;
    # nvbench's warm-up drains each call before and after, never behind the blocker.
    assert all(order[c - 1] == order[c + 1] == "sync" for c in calls[4:6])
    spans = _held_spans(order)
    assert [c for c in calls if any(b < c < u for b, u in spans)] == calls[6:]
    for block, unblock in spans:
        # Neither a drain nor a persisting-L2 reset while the stream is held: under CUPTI the
        # reset stalls the next launch until the blocker times out.
        assert "sync" not in order[block:unblock] and "persist" not in order[block:unblock]
        held_calls = [c for c in calls if block < c < unblock]
        assert not held_calls or order[block - 1] == "persist"
    recorded = timing._capture_bench_meta()
    assert (recorded.pop("fallback_reason", None) is not None) == (meta["timing"] == "cuda-events")
    assert recorded == {**meta, "l2_flush": timing._L2_FLUSH_CUDA}


def test_another_device_prepares_every_call_by_reset_then_a_read_with_no_drain(monkeypatch):
    """Off CUDA, events bracket each call: no call is held, and none is drained between the
    warm-up calls or between the timed ones."""
    from benchmarks import timing

    order, _ = _time_on_fakes(monkeypatch, "privateuseone", {})

    assert [step for step in order if step != "sync"] == ["reset", "torch-max", "run"] * 8
    calls = [i for i, step in enumerate(order) if step == "run"]
    warmup, timed = calls[4:6], calls[6:]
    assert "sync" not in order[warmup[0] : warmup[-1]] + order[timed[0] : timed[-1]]
    assert timing._capture_bench_meta() == {
        "timing": "events",
        "l2_flush": timing._L2_FLUSH_OTHER,
    }


@pytest.mark.parametrize("start_block", [None, 2])
def test_a_block_the_queue_cannot_hold_is_measured_again_in_what_it_took(monkeypatch, start_block):
    """A blocker released by its timeout leaves samples run while the host was still
    queueing, so the block is discarded and its samples measured in blocks of what the queue
    took. A given start block is where the reading starts, not a size it keeps."""
    from benchmarks import timing

    order, samples = _time_on_fakes(
        monkeypatch, "cuda", {}, fits=1, repeats=3, start_block=start_block
    )
    discarded = start_block or 3

    assert len(samples) == 3
    assert all(s.sm_clock_hz == 1.5e9 for s in samples)
    recorded = timing._capture_bench_meta()
    assert recorded["block_samples"] == 1 and recorded["l2_flush"] == timing._L2_FLUSH_CUDA
    calls = [i for i, step in enumerate(order) if step == "run"]
    # 4 calls before warm-up, 3 warm-up calls, the discarded block, then the retaken 3, one
    # per block.
    assert len(calls) == 4 + 3 + discarded + 3
    retaken = calls[7 + discarded :]
    spans = _held_spans(order)
    held = [[c for c in retaken if b < c < u] for b, u in spans]
    assert [h for h in held if h] == [[c] for c in retaken]


def test_a_block_that_times_out_after_its_last_sample_is_retaken_one_smaller(monkeypatch):
    """The host saw every sample of the block of 3 queued before the blocker released on
    its own, so no sample says how many fitted: the block is retaken in blocks of 2."""
    from benchmarks import timing

    # The load block, then the block of 3 that releases itself after its last sample.
    order, samples = _time_on_fakes(monkeypatch, "cuda", {}, repeats=3, late=[False, True])

    assert len(samples) == 3
    assert timing._capture_bench_meta()["block_samples"] == 2
    calls = [i for i, step in enumerate(order) if step == "run"]
    # 4 calls before warm-up, 3 warm-up calls, the discarded block of 3, then 2 and 1.
    assert len(calls) == 4 + 3 + 3 + 3
    retaken = calls[10:]
    held = [[c for c in retaken if b < c < u] for b, u in _held_spans(order)]
    assert [h for h in held if h] == [retaken[:2], retaken[2:]]


def test_a_row_counts_the_retries_of_the_blocks_it_keeps(monkeypatch):
    """The discarded block of 3 lost records once; of the kept blocks of 1, the first and
    the last lost records once. The row reports the kept blocks' two retries."""
    from benchmarks import timing

    _, samples = _time_on_fakes(
        monkeypatch, "cuda", {0: "lost", 2: "lost", 5: "lost"}, fits=1, repeats=3
    )

    assert len(samples) == 3
    assert timing._capture_bench_meta()["attribution_retries"] == 2


def test_a_queue_that_takes_no_sample_discards_the_held_ones(monkeypatch):
    """Smaller blocks succeed until one takes no sample: the reading starts over without
    the blocker, so no reading mixes held samples with unheld ones."""
    from benchmarks import timing

    # The load block, the block of 2 that took 1, a block of 1, then one that took none.
    order, samples = _time_on_fakes(monkeypatch, "cuda", {}, fits=[None, 1, None, 0])

    assert len(samples) == 2
    recorded = timing._capture_bench_meta()
    assert recorded["block_samples"] == 0
    assert recorded["l2_flush"] == timing._L2_FLUSH_CUDA_DRAINED
    calls = [i for i, step in enumerate(order) if step == "run"]
    # 4 calls before warm-up, 2 warm-up calls, 2 + 1 + 1 discarded, then 2 unheld.
    assert len(calls) == 4 + 2 + 4 + 2
    # Without the blocker a sample is drained before and after, as nvbench's timer does.
    assert all(order[c - 1] == order[c + 1] == "sync" for c in calls[-2:])
    assert not any(b < c < u for b, u in _held_spans(order) for c in calls[-2:])


def test_a_reading_given_its_block_size_queues_no_block_that_times_out(monkeypatch):
    """A queue that takes one sample, given a start of one: no block is discarded, and the
    row records the block size it was read in."""
    from benchmarks import timing

    order, samples = _time_on_fakes(monkeypatch, "cuda", {}, fits=1, repeats=3, start_block=1)

    assert len(samples) == 3
    recorded = timing._capture_bench_meta()
    assert recorded["block_samples"] == 1 and recorded["l2_flush"] == timing._L2_FLUSH_CUDA
    calls = [i for i, step in enumerate(order) if step == "run"]
    # 4 calls before warm-up, 3 warm-up calls, then 3 blocks of 1 and no discarded block.
    assert len(calls) == 4 + 3 + 3
    held = [[c for c in calls[7:] if b < c < u] for b, u in _held_spans(order)]
    assert [h for h in held if h] == [[c] for c in calls[7:]]


def test_a_reading_given_no_block_drains_every_sample_unheld(monkeypatch):
    """A start of 0 is a call the queue held no sample of: every sample is drained before
    and after, and none is queued behind the blocker first."""
    from benchmarks import timing

    order, samples = _time_on_fakes(monkeypatch, "cuda", {}, fits=0, start_block=0)

    assert len(samples) == 2
    recorded = timing._capture_bench_meta()
    assert recorded["block_samples"] == 0
    assert recorded["l2_flush"] == timing._L2_FLUSH_CUDA_DRAINED
    calls = [i for i, step in enumerate(order) if step == "run"]
    # 4 calls before warm-up, 2 warm-up calls, then 2 unheld.
    assert len(calls) == 4 + 2 + 2
    assert all(order[c - 1] == order[c + 1] == "sync" for c in calls[-2:])
    assert not any(b < c < u for b, u in _held_spans(order) for c in calls)


def test_a_start_of_every_sample_reads_as_no_start(monkeypatch):
    """A start at the sample count, the boundary, holds every sample in one block and records
    no block size, as a reading given none does."""
    from benchmarks import timing

    order, samples = _time_on_fakes(monkeypatch, "cuda", {}, repeats=3, start_block=3)

    assert len(samples) == 3
    assert "block_samples" not in timing._capture_bench_meta()
    calls = [i for i, step in enumerate(order) if step == "run"]
    held = [[c for c in calls[7:] if b < c < u] for b, u in _held_spans(order)]
    assert [h for h in held if h] == [calls[7:]]


def test_an_events_fallback_reports_only_its_own_blocks(monkeypatch):
    """CUPTI's reading went without the blocker and then failed; the events reading that
    replaces it was held, and says so."""
    from benchmarks import timing

    order, samples = _time_on_fakes(monkeypatch, "cuda", {1: "empty"}, fits=[None, 0, None])

    assert len(samples) == 2
    recorded = timing._capture_bench_meta()
    assert recorded["timing"] == "cuda-events" and "block_samples" not in recorded
    assert recorded["l2_flush"] == timing._L2_FLUSH_CUDA
    calls = [i for i, step in enumerate(order) if step == "run"]
    block, unblock = _held_spans(order)[-1]
    assert [c for c in calls if block < c < unblock] == calls[-2:]


def test_a_call_that_raises_while_held_releases_the_stream(monkeypatch):
    """The error reaches the caller, and the stream is not left held behind it."""
    calls = []

    def raise_on_the_timed_call():
        calls.append(None)
        # A first call, three calibration calls and one warm-up call come before.
        if len(calls) == 6:
            raise RuntimeError("kernel failure")

    from benchmarks import timing

    with pytest.raises(RuntimeError, match="kernel failure"):
        _time_on_fakes(monkeypatch, "cuda", {}, call=raise_on_the_timed_call, repeats=1)
    (blocker,) = timing._BLOCKERS.values()
    assert not blocker.held


def test_a_clock_is_read_only_from_two_written_records():
    """SM 0 runs 3000 cycles in 2000 ns: 1.5 GHz. A record it never wrote is still zero, and
    such a sample's clock is unknown rather than a ratio of the other record's counters."""
    from benchmarks.cold import SmClocks

    clocks = SmClocks.__new__(SmClocks)
    clocks._stamps = torch.tensor(
        [
            [[1_000, 10_000], [3_000, 13_000]],
            [[0, 0], [3_000, 13_000]],
            [[1_000, 10_000], [0, 0]],
        ]
    )

    assert clocks.read_hz(3) == [pytest.approx(1.5e9), None, None]


def _raise_attribution(*_args, **_kwargs):
    raise AttributionError("forced")
