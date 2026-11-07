"""CUDA autotune timing, cache invalidation, collector failures and concurrency."""

import inspect
import shutil
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest
import tilelang
import tilelang.language as T
import torch
from tilelang.autotuner.tuner import AutoTuner

from tileops._kernels import autotune as tileops_autotune
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.reduction._primitives import tune_by_forward
from tileops._utils import cupti
from workloads.device import run_device

_EXTENT = 1 << 20
_BLOCK = 256
# Above any candidate's real kernel time; a candidate's inverted reading is this less it.
_CEILING_NS = 10_000_000


def _accumulating_builder(target: str = "auto"):
    """A program whose kernel takes longer the more ``loops`` it runs."""
    extent, block = _EXTENT, _BLOCK

    @tilelang.jit(out_idx=[1], target=target)
    def program(loops: int = 1):
        @T.prim_func
        def main(x: T.Tensor((extent,), "float32"), y: T.Tensor((extent,), "float32")):
            with T.Kernel(T.ceildiv(extent, block), threads=block) as bx:
                for i in T.Parallel(block):
                    y[bx * block + i] = 0.0
                for _ in T.serial(loops):
                    for i in T.Parallel(block):
                        y[bx * block + i] = y[bx * block + i] + x[bx * block + i] * 2.0

        return main

    return program


class _AccumulatingKernel(Kernel):
    autotune_configs = [{"loops": 1}, {"loops": 256}]

    def __init__(self, target: str = "auto") -> None:
        super().__init__()
        self.kernel = _accumulating_builder(target)
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {"loops": 1}

    @property
    def autotune_supply_prog(self):
        def supply_prog(params):
            return [torch.ones(_EXTENT, dtype=torch.float32, device=run_device())]

        return supply_prog

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel(**self.config)(x)

    def tune(self) -> object:
        return self.tune_jit_kernel(self.kernel, self.autotune_configs, warmup=2, rep=5)


@pytest.fixture
def uncached(monkeypatch):
    """Tune every candidate, whatever an earlier run stored."""
    monkeypatch.setattr(AutoTuner, "_memory_cache", {})
    monkeypatch.setattr(tilelang.env, "is_autotune_cache_disabled", lambda: True)


@pytest.fixture
def cache_dir(monkeypatch):
    """A cache of stored results that starts empty, in TileLang's temporary directory."""
    Path(tilelang.env.TILELANG_TMP_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=tilelang.env.TILELANG_TMP_DIR))
    monkeypatch.setattr(AutoTuner, "cache_dir", property(lambda self: path))
    monkeypatch.setattr(AutoTuner, "_memory_cache", {})
    monkeypatch.setattr(tilelang.env, "is_cache_enabled", lambda: True)
    monkeypatch.setattr(tilelang.env, "is_autotune_cache_disabled", lambda: False)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _kernel_record(start_ns: int, end_ns: int, correlation_id: int) -> dict:
    return {"name": "k", "start_ns": start_ns, "end_ns": end_ns, "correlation_id": correlation_id}


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
def test_a_reading_is_the_median_of_each_calls_kernel_union(monkeypatch):
    """Count overlapping intervals once, exclude preparation, and use the median."""
    n_repeat = 5
    prepared = []

    def fake_collect(run_one, n_repeat_, prepare_one, buffer_bytes=0):
        kernels, iteration_of = [], {}
        for i in range(n_repeat_):
            prepare_one(i)
            prepared.append(i)
            run_one(i)
            base = i * 1_000_000
            iteration_of[i] = i
            iteration_of[100 + i] = cupti._PREPARE_ID
            kernels += [
                _kernel_record(base - 500_000, base - 1_000, 100 + i),  # the L2 flush
                _kernel_record(base, base + 4_000, i),
                _kernel_record(base + 2_000, base + 9_000, i),
                _kernel_record(base + 20_000, base + 21_000, i),
            ]
        # The first call is slow, which a mean would carry and the median does not.
        kernels.append(_kernel_record(0, 500_000, 0))
        return cupti.Trace(kernels, iteration_of, 0)

    monkeypatch.setattr(cupti, "collect_repeats", fake_collect)
    x = torch.zeros(16, device="cuda")

    reading = tileops_autotune.candidate_busy_ms(lambda: x.add_(1), 2, n_repeat)

    assert reading == pytest.approx(0.010)
    assert prepared == list(range(n_repeat))


def _short_reading_collect(lossy_attempts: int, attempts: list, dropped):
    """Simulate incomplete traces followed by a complete 100-us reading."""

    def collect(run_one, n_repeat, prepare_one, buffer_bytes=0):
        attempts.append(buffer_bytes)
        lossy = len(attempts) <= lossy_attempts
        length = 1_000 if lossy else 100_000
        kernels = [
            _kernel_record(i * 1_000_000, i * 1_000_000 + length, i) for i in range(n_repeat)
        ]
        return cupti.Trace(kernels, {i: i for i in range(n_repeat)}, dropped if lossy else 0)

    return collect


# A known loss, and a count the counter could not prove.
_LOSSES = pytest.mark.parametrize("dropped", [1, None], ids=["counted", "unproven"])


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
@_LOSSES
def test_a_reading_that_lost_any_record_is_taken_again(monkeypatch, dropped):
    """Retry incomplete traces with a larger buffer."""
    attempts = []
    monkeypatch.setattr(cupti, "collect_repeats", _short_reading_collect(1, attempts, dropped))
    x = torch.zeros(16, device="cuda")

    reading = tileops_autotune.candidate_busy_ms(lambda: x.add_(1), 1, 3)

    assert reading == pytest.approx(0.1)
    assert attempts[1] == attempts[0] * 4


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
@_LOSSES
def test_a_reading_that_keeps_losing_records_fails(monkeypatch, dropped):
    """Records lost on every attempt fail the reading instead of ranking a short one."""
    attempts = []
    monkeypatch.setattr(cupti, "collect_repeats", _short_reading_collect(99, attempts, dropped))
    x = torch.zeros(16, device="cuda")

    with pytest.raises(cupti.RecordsLostError, match="read short"):
        tileops_autotune.candidate_busy_ms(lambda: x.add_(1), 1, 3)
    assert len(attempts) == 3


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
def test_the_tuner_keeps_the_candidate_with_the_lowest_busy_reading(monkeypatch, uncached):
    """With every busy reading inverted, the candidate that runs longest wins, which an
    event timing would never choose."""
    real_collect = cupti.collect_repeats

    def inverted_collect(run_one, n_repeat, prepare_one, buffer_bytes=0):
        trace = real_collect(run_one, n_repeat, prepare_one, buffer_bytes)
        kernels = [
            kernel
            if trace.iteration_of.get(kernel["correlation_id"]) == cupti._PREPARE_ID
            else {
                **kernel,
                "end_ns": kernel["start_ns"]
                + _CEILING_NS
                - (kernel["end_ns"] - kernel["start_ns"]),
            }
            for kernel in trace.kernels
        ]
        return trace._replace(kernels=kernels)

    monkeypatch.setattr(cupti, "collect_repeats", inverted_collect)

    tuned = _AccumulatingKernel().tune()

    assert tuned.config == {"loops": 256}
    assert tuned.latency > 0.9 * _CEILING_NS * 1e-6


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
@pytest.mark.parametrize("fail_at, caching", [(1, "uncached"), (2, "cache_dir"), (2, "uncached")])
def test_a_reading_lost_for_one_candidate_stores_no_result(monkeypatch, request, fail_at, caching):
    """A ranking missing a candidate is neither returned nor stored, so the next tuning
    measures every candidate again."""
    stored = request.getfixturevalue(caching)
    real_collect = cupti.collect_repeats
    calls = []

    def collector_fails(*args, **kwargs):
        calls.append(None)
        if len(calls) >= fail_at:
            raise cupti.CUPTIError("collector failed to start")
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(cupti, "collect_repeats", collector_fails)

    with pytest.raises(cupti.CUPTIError, match="failed to start"):
        _AccumulatingKernel().tune()

    assert len(calls) == 2
    if caching == "cache_dir":
        assert not any(stored.iterdir())
        assert not AutoTuner._memory_cache


def _stored_key(target: str) -> str:
    """The key under which a tuning of the accumulating kernel for *target* is stored."""
    tuner = tileops_autotune.autotune(configs=_AccumulatingKernel.autotune_configs)(
        _accumulating_builder(target)
    ).get_tunner()
    return tuner.generate_cache_key(inspect.signature(tuner.fn).parameters, {})


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("target", "policy"),
    # A compile target the key reads, not a device anything is placed on.
    [("cuda", "_DEVICE_BUSY"), ("hip", "_TILELANG_EVENTS")],  # lint: device-literal allow: target
)
def test_a_result_ranked_by_another_reading_is_not_reused(monkeypatch, target, policy):
    """The policy that read a target's candidates is in the key of the stored result."""
    before = _stored_key(target)
    monkeypatch.setattr(tileops_autotune, policy, "another reading")
    assert _stored_key(target) != before


@pytest.mark.smoke
def test_a_candidate_for_another_target_keeps_tilelangs_reading():
    """Off CUDA there is no CUPTI, and the tuner reads candidates as TileLang does."""
    tuner = tileops_autotune.autotune(configs=_AccumulatingKernel.autotune_configs)(
        _accumulating_builder("hip")
    ).get_tunner()
    assert tuner._benchmark_target.__func__ is AutoTuner._benchmark_target


class _LoopingKernel(Kernel):
    """A kernel ``tune_by_forward`` sweeps, whose call takes longer the more ``loops``."""

    autotune_configs = [{"loops": 1}, {"loops": 64}]

    def __init__(self) -> None:
        super().__init__()
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {"loops": 1}

    def forward(self, x: torch.Tensor) -> None:
        for _ in range(self.config["loops"]):
            x.mul_(1.0)


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
def test_a_sweep_through_calls_keeps_the_lowest_busy_reading(monkeypatch):
    """``tune_by_forward`` ranks by the tuner's reading: inverted, the longest call wins."""
    real_collect = cupti.collect_repeats

    def inverted_collect(run_one, n_repeat, prepare_one, buffer_bytes=0):
        trace = real_collect(run_one, n_repeat, prepare_one, buffer_bytes)
        claimed = {}
        for kernel in trace.kernels:
            iteration = trace.iteration_of.get(kernel["correlation_id"])
            if iteration != cupti._PREPARE_ID:
                claimed.setdefault(iteration, []).append(kernel)
        # Each call becomes one kernel lasting the ceiling less its real busy time.
        kernels = [k for k in trace.kernels if k not in sum(claimed.values(), [])]
        for group in claimed.values():
            busy = sum(k["end_ns"] - k["start_ns"] for k in group)
            start = group[0]["start_ns"]
            kernels.append({**group[0], "end_ns": start + _CEILING_NS - busy})
        return trace._replace(kernels=kernels)

    monkeypatch.setattr(cupti, "collect_repeats", inverted_collect)
    kernel = _LoopingKernel()

    tune_by_forward(kernel, torch.ones(1 << 16, device=run_device()), warmup=2, rep=5)

    assert kernel.config == {"loops": 64}


class _Stub:
    """Any builder: every call returns the stub, so a candidate compiles and launches nothing."""

    def __call__(self, *args, **kwargs):
        return self


def _linear_attention_kernels() -> list:
    """Linear-attention kernels with custom tuning loops."""
    from tileops._kernels.linear_attention.deltanet import recurrent as deltanet
    from tileops._kernels.linear_attention.gla import chunk_bwd, chunk_fwd
    from tileops._kernels.linear_attention.gla import recurrent as gla

    return [
        pytest.param(
            deltanet,
            lambda: deltanet.DeltaNetDecodeKernel(1, 2, 64, 64, "bfloat16"),
            id="deltanet-decode",
        ),
        pytest.param(
            deltanet,
            lambda: deltanet.DeltaNetDecodeWarpKernel(1, 2, 128, 128, "bfloat16"),
            id="deltanet-decode-warp",
        ),
        pytest.param(
            deltanet,
            lambda: deltanet.DeltaNetDecodeFP32Kernel(1, 2, 64, 64),
            id="deltanet-decode-fp32",
        ),
        pytest.param(
            gla, lambda: gla.GLADecodeKernel(1, 2, 64, 64, 0.125, "bfloat16"), id="gla-decode"
        ),
        pytest.param(
            gla, lambda: gla.GLADecodeFP32Kernel(1, 2, 64, 64, 0.125), id="gla-decode-fp32"
        ),
        pytest.param(
            chunk_fwd,
            lambda: chunk_fwd.GLAChunkFwdKernel(1, 128, 2, 64, 64, 64, 0.125),
            id="gla-chunk-fwd",
        ),
        pytest.param(
            chunk_bwd,
            lambda: chunk_bwd.GLAChunkBwdKernel(1, 128, 2, 64, 64, 64, 0.125, torch.float16),
            id="gla-chunk-bwd",
        ),
    ]


def _stubbed(monkeypatch, module, make) -> tuple:
    """*make*'s kernel with its builders and launches stubbed, and the configs it publishes."""
    kernel = make()
    for name in [n for n in vars(module) if n.endswith("_tl")]:
        monkeypatch.setattr(module, name, _Stub())
    monkeypatch.setattr(kernel, "forward", _Stub())
    published = []
    monkeypatch.setattr(kernel, "_publish", lambda config, build, *facts: published.append(config))
    return kernel, published


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
@pytest.mark.parametrize(("module", "make"), _linear_attention_kernels())
def test_a_linear_attention_sweep_keeps_the_lowest_busy_reading(monkeypatch, module, make):
    """Every candidate is read by the tuner's reading, and the lowest reading is published."""
    kernel, published = _stubbed(monkeypatch, module, make)
    swept = []
    candidates = kernel.tuning_candidates

    def recorded(*args, **kwargs):
        swept.extend(candidates(*args, **kwargs))
        return list(swept)

    monkeypatch.setattr(kernel, "tuning_candidates", recorded)
    readings = []

    def reading(fn, n_warmup, n_repeat):
        # Lowest at the middle candidate, so neither sweep end wins by its position.
        readings.append(abs(len(readings) - len(swept) // 2) + 1.0)
        return readings[-1]

    monkeypatch.setattr(module, "candidate_busy_ms", reading)

    kernel.autotune()

    assert len(swept) > 1
    assert len(readings) == len(swept)
    assert published == [swept[readings.index(min(readings))]]


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
@pytest.mark.parametrize(("module", "make"), _linear_attention_kernels())
def test_a_linear_attention_sweep_fails_when_no_reading_can_be_taken(monkeypatch, module, make):
    """A loop skips a candidate that fails, but not a reading the collector failed to take."""
    kernel, published = _stubbed(monkeypatch, module, make)

    def unavailable(fn, n_warmup, n_repeat):
        raise cupti.CUPTIError("cupti-python is unavailable")

    monkeypatch.setattr(module, "candidate_busy_ms", unavailable)

    with pytest.raises(cupti.CUPTIError, match="unavailable"):
        kernel.autotune()
    assert not published


def _batch_norm_split_kernel(monkeypatch) -> tuple:
    """BatchNorm's split training forward at a small shape, its launches stubbed, and the
    configs it publishes."""
    from tileops._kernels.norm.batch_norm import BatchNormFwdTrainSplitKernel

    kernel = BatchNormFwdTrainSplitKernel(1, 4, 1 << 16, torch.float16, 1e-5, 0.1, 128)
    monkeypatch.setattr(kernel, "_launch_of", lambda splits, threads: _Stub())
    published = []
    monkeypatch.setattr(kernel, "_publish_tuned", published.append)
    return kernel, published


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
def test_the_batch_norm_split_sweep_keeps_the_lowest_busy_reading(monkeypatch):
    """Every ``(splits, threads)`` pair is read by the tuner's reading, and the lowest is
    published."""
    from tileops._kernels.norm import batch_norm

    kernel, published = _batch_norm_split_kernel(monkeypatch)
    swept = kernel.tuning_candidates(kernel.autotune_configs)
    readings = []

    def reading(fn, n_warmup, n_repeat):
        # Lowest at the middle candidate, so neither sweep end wins by its position.
        readings.append(abs(len(readings) - len(swept) // 2) + 1.0)
        return readings[-1]

    monkeypatch.setattr(batch_norm, "candidate_busy_ms", reading)

    kernel.autotune()

    lowest = swept[readings.index(min(readings))]
    assert len(readings) == len(swept) > 1
    assert published == [{"splits": lowest["splits"], "threads": lowest["threads"]}]


@pytest.mark.smoke
@pytest.mark.cuda_only("cuda-kernel")
def test_the_batch_norm_split_sweep_fails_when_no_reading_can_be_taken(monkeypatch):
    """The sweep skips a pair the layout refuses, but not a reading the collector failed to
    take."""
    from tileops._kernels.norm import batch_norm

    kernel, published = _batch_norm_split_kernel(monkeypatch)

    def unavailable(fn, n_warmup, n_repeat):
        raise cupti.CUPTIError("cupti-python is unavailable")

    monkeypatch.setattr(batch_norm, "candidate_busy_ms", unavailable)

    with pytest.raises(cupti.CUPTIError, match="unavailable"):
        kernel.autotune()
    assert not published


class _Overlap:
    """How many callers are inside at once, at most."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._inside = 0
        self.most = 0

    def enter(self) -> None:
        with self._guard:
            self._inside += 1
            self.most = max(self.most, self._inside)
        time.sleep(0.05)
        with self._guard:
            self._inside -= 1


def _from_threads(call, count: int = 4) -> None:
    threads = [threading.Thread(target=call) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


@pytest.mark.smoke
def test_concurrent_benchmark_workers_take_their_readings_in_turn(monkeypatch):
    """TileLang's benchmark workers are threads; a candidate's benchmark step runs alone,
    as CUPTI records every thread's launches and collects one session at a time."""
    overlap = _Overlap()

    def benchmark_target(self, jit_kernel, **kwargs):
        overlap.enter()

    monkeypatch.setattr(AutoTuner, "_benchmark_target", benchmark_target)
    tuner = tileops_autotune.autotune(configs=_AccumulatingKernel.autotune_configs)(
        _accumulating_builder("cuda")  # lint: device-literal allow: a compile target
    ).get_tunner()

    _from_threads(lambda: tuner._benchmark_target(jit_kernel=None, warmup=1, rep=1))

    assert overlap.most == 1


@pytest.mark.smoke
def test_concurrent_readings_are_taken_in_turn(monkeypatch):
    """Two threads reading candidates at once queue for the one collector."""
    overlap = _Overlap()

    def collect_attributed(run_one, n_repeat, prepare_one, **kwargs):
        overlap.enter()
        return [cupti.Sample(device_busy_ms=1.0, latency_ms=1.0, n_kernels=1)]

    # No device: the reading's preparation is stubbed, and the collector records overlap.
    cuda = types.SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda device: types.SimpleNamespace(L2_cache_size=1),
        synchronize=lambda: None,
    )
    monkeypatch.setattr(
        tileops_autotune,
        "torch",
        types.SimpleNamespace(cuda=cuda, int8=None, empty=lambda *a, **k: _Stub()),
    )
    monkeypatch.setattr(tileops_autotune, "reset_persisting_l2_cache", lambda: None)
    monkeypatch.setattr(tileops_autotune, "collect_attributed", collect_attributed)

    _from_threads(lambda: tileops_autotune.candidate_busy_ms(lambda: None, 0, 1))

    assert overlap.most == 1
