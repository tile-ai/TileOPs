import logging
import subprocess
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Optional, Tuple

import torch
from torch.autograd.profiler import DeviceType
from tilelang.profiler import do_bench

from tests.test_base import TestBase

_logger = logging.getLogger("tileops.bench")

# Thread-local storage for conftest hook to pick up per-test bench results.
# A single test function may call record() multiple times (tileops + baseline).
_bench_results = threading.local()

# ---------------------------------------------------------------------------
# Shared CUPTI profiler session
# ---------------------------------------------------------------------------

_shared_cupti_session = None


def _sum_kernel_time_us(kineto_results):
    """Extract total CUDA kernel time directly from C++ Kineto events.

    Bypasses ``profiler.key_averages()`` which triggers expensive Python
    event parsing (~120ms) and tree building (~10ms) for large traces.
    Direct C++ iteration is ~16x faster for n_repeat=1280.
    """
    total_us = 0.0
    for evt in kineto_results.events():
        if evt.device_type() == DeviceType.CUDA:
            name = evt.name()
            if "vectorized_elementwise" in name and "FillFunctor" in name:
                continue
            total_us += evt.duration_ns() / 1000.0
    return total_us


class CuptiSession:
    """Persistent CUPTI profiler session shared across multiple benchmarks.

    Keeps a single ``torch.profiler.profile()`` instance alive so that
    CUPTI init/teardown and event-parsing overhead is paid only once per
    session instead of once per ``do_bench(backend='cupti')`` call.

    Usage::

        with CuptiSession():
            # All do_bench(backend='cupti') calls inside this block
            # automatically use the shared session via the monkey-patch.
            latency = do_bench(fn, backend='cupti')
    """

    def __init__(self):
        self._profiler = None
        self._results: list[float] = []
        self._current_n_repeat: int = 1

    # -- callback -----------------------------------------------------------

    def _on_trace_ready(self, prof):
        kr = prof.profiler.kineto_results
        kernel_time_us = _sum_kernel_time_us(kr) / self._current_n_repeat
        self._results.append(kernel_time_us * 1e-3)

    # -- context manager ----------------------------------------------------

    def __enter__(self):
        global _shared_cupti_session
        from tilelang.profiler.bench import suppress_stdout_stderr

        schedule = torch.profiler.schedule(
            wait=0, warmup=1, active=1, repeat=0,
        )

        # Suppress CUPTI/Kineto init warnings during session creation.
        with suppress_stdout_stderr():
            self._profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
                on_trace_ready=self._on_trace_ready,
            )
            self._profiler.__enter__()

            # Prime with a dummy warmup+active cycle so the very first real
            # measurement does not absorb one-time CUPTI overhead.
            dummy = torch.empty(1, device="cuda")
            dummy.zero_()
            self._profiler.step()  # warmup (discarded)
            dummy.zero_()
            self._profiler.step()  # active  → on_trace_ready (dummy)

        self._results.clear()
        _shared_cupti_session = self
        return self

    def __exit__(self, *exc_info):
        global _shared_cupti_session
        _shared_cupti_session = None

        from tilelang.profiler.bench import suppress_stdout_stderr

        with suppress_stdout_stderr():
            self._profiler.__exit__(*exc_info)

        self._profiler = None
        return False

    # -- public API ---------------------------------------------------------

    def measure(self, fn, cache, n_repeat):
        """Run one benchmark measurement using the shared CUPTI session.

        Called from the monkey-patched ``_bench_with_cupti`` when a shared
        session is active.  Returns kernel time in milliseconds.
        """
        self._current_n_repeat = n_repeat
        idx = len(self._results)

        # warmup step (data discarded by schedule)
        for _ in range(n_repeat):
            cache.zero_()
            fn()
        self._profiler.step()

        # active step (triggers on_trace_ready)
        for _ in range(n_repeat):
            cache.zero_()
            fn()
        self._profiler.step()

        if len(self._results) > idx:
            return self._results[idx]
        return 0.0  # callback didn't fire; caller will fall back


def _patch_cupti_filter():
    """Narrow tilelang's CUPTI kernel-name exclusion to cache-clearing only.

    tilelang's ``_bench_with_cupti`` (bench.py) excludes all CUDA kernels
    whose name contains ``"at::native::vectorized_elementwise"``.  The
    intent is to strip the ``cache.zero_()`` overhead inserted between
    benchmark iterations, but ``cache.zero_()`` produces a kernel with
    ``FillFunctor`` in its name — the same ``vectorized_elementwise``
    family that PyTorch uses for add, mul, where, maximum, etc.

    The overly broad substring match therefore removes **all** PyTorch
    elementwise kernel time, making baseline latencies near-zero for
    elementwise benchmarks.

    This patch replaces the upstream function with one that only excludes
    kernels containing **both** ``vectorized_elementwise`` and
    ``FillFunctor`` (i.e. the actual cache-clearing kernel).
    """
    import tilelang.profiler.bench as _bench_mod

    if not hasattr(_bench_mod, "_bench_with_cupti"):
        import warnings
        warnings.warn(
            "tilelang CUPTI patch skipped: _bench_with_cupti not found. "
            "The upstream API may have changed — elementwise baseline "
            "latencies measured via CUPTI may be inaccurate.",
            stacklevel=2,
        )
        return

    def _bench_with_cupti_fixed(fn, cache, n_repeat):
        # Fast path: use the shared CUPTI session if one is active.
        if _shared_cupti_session is not None:
            return _shared_cupti_session.measure(fn, cache, n_repeat)

        # Slow path: per-call profiler (original behaviour).
        from tilelang.profiler.bench import suppress_stdout_stderr

        with suppress_stdout_stderr():
            # Use an explicit warmup phase so torch.profiler does not warn
            # about skipped warmup and the measured active step is steadier.
            schedule = torch.profiler.schedule(
                wait=1, warmup=1, active=1, repeat=1,
            )
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
            )
            with profiler:
                for _ in range(3):
                    for _ in range(n_repeat):
                        cache.zero_()
                        fn()
                    profiler.step()

        kr = profiler.profiler.kineto_results
        kernel_time_us = _sum_kernel_time_us(kr) / n_repeat
        return kernel_time_us * 1e-3

    _bench_mod._bench_with_cupti = _bench_with_cupti_fixed


_patch_cupti_filter()

# ---------------------------------------------------------------------------
# L2 cache flush buffer (sized to actual L2, allocated lazily)
# ---------------------------------------------------------------------------

_l2_flush_cache: Optional[torch.Tensor] = None


def _get_l2_flush_cache() -> torch.Tensor:
    global _l2_flush_cache
    if _l2_flush_cache is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if l2_bytes <= 0:
            l2_bytes = int(256e6)  # fallback
        _l2_flush_cache = torch.empty(l2_bytes // 4, dtype=torch.int, device="cuda")
    return _l2_flush_cache


# ---------------------------------------------------------------------------
# NVIDIA SOL-ExecBench–style benchmark
# ---------------------------------------------------------------------------

def bench_kernel(
    fn: Callable,
    args: Tuple[torch.Tensor, ...] = (),
    n_warmup: int = 10,
    n_repeat: int = 50,
    n_trials: int = 3,
) -> float:
    """Benchmark a GPU kernel with pure kernel timing via CUPTI.

    Protocol (adapted from NVIDIA SOL-ExecBench, arxiv.org/abs/2603.19173):
      1. Lock GPU clocks externally (nvidia-smi).
      2. Run *n_warmup* un-timed iterations with L2 flush.
      3. For each of *n_trials* trials, profile *n_repeat* iterations
         under CUPTI to get pure kernel execution time (no launch overhead).
         L2 is flushed before every iteration.  Input tensors are cloned
         each iteration so the kernel always sees fresh addresses.
      4. Report the median trial mean (robust to outlier trials).

    Uses CUPTI via torch.profiler for accurate kernel-only timing, with
    direct Kineto C++ event iteration to avoid Python parsing overhead.
    Falls back to CUDA events if CUPTI is unavailable.

    Args:
        fn: Callable to benchmark.  If *args* is provided, called as
            ``fn(*cloned_args)``; otherwise called as ``fn()``.
        args: Tensor arguments to clone each iteration.  Non-tensor
            values are passed through unchanged.
        n_warmup: Warmup iterations (default 10).
        n_repeat: Timed iterations per trial (default 50).
        n_trials: Independent trials (default 3).

    Returns:
        Kernel latency in **milliseconds**.
    """
    from tilelang.profiler.bench import suppress_stdout_stderr

    cache = _get_l2_flush_cache()
    has_args = len(args) > 0

    # Pre-clone a small pool of input tensors so the kernel sees different
    # addresses across iterations.  Skip cloning if total tensor memory
    # exceeds 1 GB to avoid OOM on large workloads.
    _N_CLONES = 3
    _MAX_CLONE_BYTES = 1 << 30  # 1 GB
    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(a.nelement() * a.element_size()
                          for a, m in zip(args, tensor_mask) if m)
        if total_bytes * _N_CLONES <= _MAX_CLONE_BYTES:
            arg_pool = [
                tuple(a.clone() if m else a for a, m in zip(args, tensor_mask))
                for _ in range(_N_CLONES)
            ]
            def _run(i):
                return fn(*arg_pool[i % _N_CLONES])
        else:
            arg_pool = None
            def _run(i):
                return fn(*args)
    else:
        arg_pool = None
        def _run(i):
            return fn()

    # Warmup (no profiling)
    for i in range(n_warmup):
        cache.zero_()
        _run(i % n_repeat)
    torch.cuda.synchronize()

    # Timed trials with CUPTI (single profiler, n_trials cycles)
    trial_means: list[float] = []

    def _on_trace_ready(prof):
        kr = prof.profiler.kineto_results
        kernel_us = _sum_kernel_time_us(kr) / n_repeat
        trial_means.append(kernel_us * 1e-3)

    try:
        with suppress_stdout_stderr():
            schedule = torch.profiler.schedule(
                wait=0, warmup=1, active=1, repeat=n_trials,
            )
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
                on_trace_ready=_on_trace_ready,
            )
            with profiler:
                for _ in range(n_trials):
                    # Warmup step (discarded by schedule)
                    for _ in range(n_repeat):
                        cache.zero_()
                        _run(0)
                    profiler.step()
                    # Active step (measured → _on_trace_ready)
                    for i in range(n_repeat):
                        cache.zero_()
                        _run(i)
                    profiler.step()
    except RuntimeError:
        pass

    # Fallback to CUDA events if CUPTI failed
    if not trial_means:
        for _ in range(n_trials):
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
            for i in range(n_repeat):
                cache.zero_()
                start_events[i].record()
                _run(i)
                end_events[i].record()
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            trial_means.append(sum(times) / len(times))

    # Free the arg pool and release cached GPU memory to prevent
    # accumulation across hundreds of benchmark calls.
    if arg_pool is not None:
        del arg_pool
    torch.cuda.empty_cache()

    trial_means.sort()
    return trial_means[len(trial_means) // 2]


def _get_env_metadata() -> list[str]:
    """Collect GPU model, driver version, CUDA version, and torch version."""
    lines = []
    lines.append(f"- **Torch version**: {torch.__version__}")
    lines.append(f"- **CUDA version (torch)**: {torch.version.cuda or 'N/A'}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        lines.append(f"- **GPU model**: {gpu_name}")
    else:
        lines.append("- **GPU model**: N/A (no CUDA device)")

    # Try to get NVIDIA driver version from nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        driver = result.stdout.strip().split("\n")[0] if result.returncode == 0 else "N/A"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        driver = "N/A"
    lines.append(f"- **Driver version**: {driver}")

    return lines


class BenchmarkBase(ABC):
    """Abstract base class for op benchmarking.

    Takes a TestBase instance to share gen_inputs().
    Subclass must implement calculate_flops() and calculate_memory().
    """

    def __init__(self, test: TestBase):
        self.test = test

    @abstractmethod
    def calculate_flops(self) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def calculate_memory(self) -> Optional[float]:
        raise NotImplementedError

    def profile(self,
                functor: Any,
                *inputs: Tuple[torch.Tensor]) -> dict:
        """Profile a callable and return structured results.

        Uses the NVIDIA SOL-ExecBench protocol: CUPTI kernel timing,
        10 warmup, 50 repeats × 3 trials, L2 flush sized to actual
        cache, input tensors cloned each iteration.
        """
        with torch.no_grad():
            latency = bench_kernel(functor, args=inputs)

        result = {"latency_ms": latency}
        flops = self.calculate_flops()
        if flops is not None:
            result["tflops"] = flops / latency * 1e-9
        memory = self.calculate_memory()
        if memory is not None:
            result["bandwidth_tbs"] = memory / latency * 1e-9
        return result


class BenchmarkReport:
    """Collects benchmark results and dumps a markdown report.

    All methods are static — use as BenchmarkReport.record(...).
    Call clear() at session start, dump() at session end.
    """
    _records: dict = {}

    @staticmethod
    def record(op_or_name, params: dict, result: dict, tag: str = "tileops") -> None:
        """Record a benchmark result.

        Args:
            op_or_name: Op instance or benchmark group name string.
                If an Op instance, class name and module are extracted automatically.
            params: Parameter dict (typically from locals())
            result: Dict with latency_ms, tflops, bandwidth_tbs
            tag: Label to distinguish implementations (e.g. "tileops", "FA3", "fla")
        """
        if isinstance(op_or_name, str):
            name = op_or_name
            op_module = None
        else:
            name = op_or_name.__class__.__name__
            op_module = op_or_name.__class__.__module__

        # Filter params to only include serializable benchmark parameters
        filtered_params = {
            k: v for k, v in params.items()
            if k not in ("test", "bm", "op", "inputs", "result", "result_bl",
                         "baseline_fn", "tune")
            and not k.startswith("_")
            and isinstance(v, (int, float, bool, str, torch.dtype))
        }
        BenchmarkReport._records.setdefault(name, []).append({
            "params": filtered_params,
            "result": result,
            "tag": tag,
        })

        # Accumulate in thread-local for conftest hook.
        if not hasattr(_bench_results, "entries"):
            _bench_results.entries = []
        entry = {"tag": tag, "op": name, **result}
        if op_module:
            entry["op_module"] = op_module
        _bench_results.entries.append(entry)

        _logger.info("op=%s module=%s tag=%s latency_ms=%.4f tflops=%.2f",
                      name, op_module or "N/A", tag,
                      result.get("latency_ms", 0),
                      result.get("tflops", 0))

    @staticmethod
    def dump(path: str) -> None:
        """Write all collected results to a markdown-formatted log file."""
        if not BenchmarkReport._records:
            return

        lines = [
            "# TileOPs Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Environment",
            "",
        ]
        lines.extend(_get_env_metadata())
        lines.append("")

        result_keys = ["latency_ms", "tflops", "bandwidth_tbs"]

        for name, entries in BenchmarkReport._records.items():
            if not entries:
                continue

            lines.append(f"## {name}")
            lines.append("")

            # Group by tag
            tag_entries = {}
            for entry in entries:
                tag_entries.setdefault(entry["tag"], []).append(entry)

            for tag, tag_group in tag_entries.items():
                lines.append(f"### {tag}")
                lines.append("")

                param_keys = list(tag_group[0]["params"].keys())
                header_parts = param_keys + result_keys
                lines.append("| " + " | ".join(header_parts) + " |")
                lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

                for entry in tag_group:
                    row = [str(entry["params"].get(k, "")) for k in param_keys]
                    for rk in result_keys:
                        val = entry["result"].get(rk)
                        row.append(f"{val:.2f}" if val is not None else "N/A")
                    lines.append("| " + " | ".join(row) + " |")

                lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        print(f"Benchmark report saved to {path}")

    @staticmethod
    def clear() -> None:
        """Clear all collected records."""
        BenchmarkReport._records.clear()
