"""Benchmark framework — NPU/CUDA portable kernel timing and reporting.

Core abstractions:
  - ``bench_kernel()``: SOL-ExecBench-style timing with L2-flush, input
    cloning, warmup, and multi-trial median.  Uses backend-agnostic
    timing events (NPU or CUDA) instead of CUPTI.
  - ``BenchmarkBase`` / ``ManifestBenchmark``: wraps an Op + workload,
    computes roofline-derived TFLOPS / bandwidth.
  - ``BenchmarkReport``: collects results and dumps a markdown report.
  - ``workload_field_params()``: turns manifest workload dicts into
    pytest params.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Generic, Optional, TypeVar

import pytest
import torch

from manifest import load_workloads
from utils import (
    device_str,
    empty_cache,
    get_device_name,
    is_available,
    is_cuda,
    is_npu,
    manual_seed_all,
    synchronize,
    timing_event,
)

_logger = logging.getLogger("npu_bench")

W = TypeVar("W")

_bench_results = threading.local()
_bench_meta = threading.local()

_l2_flush_cache: Optional[torch.Tensor] = None


def _get_l2_flush_cache() -> Optional[torch.Tensor]:
    """Allocate a cache-flush buffer sized to L2 (best-effort on NPU)."""
    global _l2_flush_cache
    if _l2_flush_cache is not None:
        return _l2_flush_cache

    flush_env = os.getenv("NPU_BENCH_FLUSH_CACHE", "1")
    if flush_env != "1":
        _logger.info("L2 cache flush disabled (NPU_BENCH_FLUSH_CACHE=0)")
        return None

    try:
        props = None
        if is_npu():
            props = torch.npu.get_device_properties(0)
        elif is_cuda():
            props = torch.cuda.get_device_properties(0)
        l2_bytes = getattr(props, "L2_cache_size", 0) if props else 0
    except Exception:
        l2_bytes = 0

    if l2_bytes <= 0:
        l2_bytes = int(256e6)
        _logger.warning("L2 size unknown; using 256 MB flush buffer")

    _l2_flush_cache = torch.empty(l2_bytes // 4, dtype=torch.int, device=device_str())
    return _l2_flush_cache


def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    n_warmup: int = 10,
    n_repeat: int = 50,
    n_trials: int = 3,
) -> float:
    """Benchmark a callable with event-based timing.

    Protocol:
      1. Run *n_warmup* iterations with L2 flush.
      2. For each of *n_trials* trials, time *n_repeat* iterations
         using backend timing events (NPU or CUDA Event).
         L2 is flushed before every iteration; inputs are cloned
         from a small pool so the kernel sees fresh addresses.
      3. Report the median trial mean.

    Args:
        fn: Callable to benchmark.
        args: Tensor arguments (cloned each iteration).
        n_warmup: Warmup iterations (default 10).
        n_repeat: Timed iterations per trial (default 50).
        n_trials: Independent trials (default 3).

    Returns:
        Kernel latency in **milliseconds**.
    """
    if not isinstance(args, tuple):
        raise TypeError(f"bench_kernel expects tuple args, got {type(args).__name__}")

    cache = _get_l2_flush_cache()
    has_args = len(args) > 0

    _N_CLONES = 3
    _MAX_CLONE_BYTES = 1 << 30  # 1 GB
    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(
            a.nelement() * a.element_size()
            for a, m in zip(args, tensor_mask, strict=True) if m)
        if total_bytes * _N_CLONES <= _MAX_CLONE_BYTES:
            arg_pool = [
                tuple(a.clone() if m else a
                      for a, m in zip(args, tensor_mask, strict=True))
                for _ in range(_N_CLONES)
            ]
            def _run(i):
                return fn(*arg_pool[i % _N_CLONES])
        else:
            _logger.warning(
                "inputs total %.2f GiB; skipping per-iteration cloning",
                total_bytes / (1 << 30))
            arg_pool = None
            def _run(i):
                return fn(*args)
    else:
        arg_pool = None
        def _run(i):
            return fn()

    _bench_meta.inputs_cloned = arg_pool is not None or not has_args

    # Warmup
    for i in range(n_warmup):
        if cache is not None:
            cache.zero_()
        _run(i)
    synchronize()

    # Timed trials
    trial_means: list[float] = []
    for _ in range(n_trials):
        start_events = [timing_event(enable_timing=True) for _ in range(n_repeat)]
        end_events = [timing_event(enable_timing=True) for _ in range(n_repeat)]

        for i in range(n_repeat):
            if cache is not None:
                cache.zero_()
            synchronize()
            start_events[i].record()
            _run(i)
            end_events[i].record()
        synchronize()

        times = [s.elapsed_time(e)
                 for s, e in zip(start_events, end_events, strict=True)]
        trial_means.append(sum(times) / len(times))

    _bench_meta.timing = "events"

    if arg_pool is not None:
        del arg_pool
    empty_cache()

    trial_means.sort()
    return trial_means[len(trial_means) // 2]


class BenchmarkBase(Generic[W], ABC):
    """Abstract base for op benchmarking."""

    def __init__(self, workload: W):
        self.workload = workload

    @abstractmethod
    def calculate_flops(self) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def calculate_memory(self) -> Optional[float]:
        raise NotImplementedError

    def profile(self, functor: Any, *inputs: Any) -> dict:
        with torch.no_grad():
            latency = bench_kernel(functor, args=inputs)
        return self._build_result(latency)

    def _build_result(self, latency: float) -> dict:
        result = {"latency_ms": latency}
        timing = getattr(_bench_meta, "timing", None)
        if timing is not None:
            result["timing"] = timing
        if getattr(_bench_meta, "inputs_cloned", True) is False:
            result["inputs_cloned"] = False
        flops = self.calculate_flops()
        if flops is not None:
            result["tflops"] = flops / latency * 1e-9
        memory = self.calculate_memory()
        if memory is not None:
            result["bandwidth_tbs"] = memory / latency * 1e-9
        return result


class ManifestBenchmark(BenchmarkBase[Any]):
    """Benchmark that reads FLOP/byte counts from an Op's eval_roofline()."""

    def __init__(self, op_name: str, op: Any, workload: Any):
        super().__init__(workload)
        self._op_name = op_name
        self._op = op
        self._roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def workload_field_params(workloads: list, keys: tuple) -> list:
    """Turn manifest workload dicts into pytest params.

    First workload is marked ``smoke``, the rest ``full``.
    Keys ending in ``dtype`` are resolved to ``torch.dtype``.
    """
    params = []
    for i, w in enumerate(workloads):
        args = [getattr(torch, w[k]) if k.endswith("dtype") else w[k] for k in keys]
        params.append(
            pytest.param(
                *args,
                marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                id=w["label"],
            )
        )
    return params


def _get_env_metadata() -> list[str]:
    lines = [f"- **Torch version**: {torch.__version__}"]
    if is_available():
        lines.append(f"- **Device**: {get_device_name(0)}")
        lines.append(f"- **Backend**: {device_str()}")
    else:
        lines.append("- **Device**: N/A")
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "board", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines()[:10]:
                lines.append(f"- {line.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return lines


class BenchmarkReport:
    """Collects benchmark results and dumps a markdown report."""

    _records: dict = {}

    @staticmethod
    def record(op_or_name, params: dict, result: dict,
               tag: str = "kernel") -> None:
        if isinstance(op_or_name, str):
            name = op_or_name
            op_module = None
            op_config = None
        else:
            name = op_or_name.__class__.__name__
            op_module = op_or_name.__class__.__module__
            kernel = getattr(op_or_name, "kernel", None)
            op_config = getattr(kernel, "config", None) if kernel else None

        def _is_serializable(v: Any) -> bool:
            if isinstance(v, (int, float, bool, str, torch.dtype)):
                return True
            if isinstance(v, tuple):
                return all(_is_serializable(x) for x in v)
            return False

        filtered_params = {
            k: v for k, v in params.items()
            if k not in ("test", "bm", "op", "inputs", "result", "result_bl",
                         "baseline_fn", "tune")
            and not k.startswith("_")
            and _is_serializable(v)
        }
        entry = {
            "params": filtered_params,
            "result": result,
            "tag": tag,
        }
        if op_config:
            entry["config"] = op_config
        BenchmarkReport._records.setdefault(name, []).append(entry)

        if not hasattr(_bench_results, "entries"):
            _bench_results.entries = []
        _bench_results.entries.append({
            "tag": tag, "op": name, **result,
            **({"op_module": op_module} if op_module else {}),
        })

        _logger.info("op=%s tag=%s latency_ms=%.4f tflops=%.2f",
                     name, tag, result.get("latency_ms", 0),
                     result.get("tflops", 0))

    @staticmethod
    def dump(path: str) -> None:
        if not BenchmarkReport._records:
            return

        lines = [
            "# NPU Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Environment",
            "",
        ]
        lines.extend(_get_env_metadata())
        lines.append("")

        default_keys = ["latency_ms", "tflops", "bandwidth_tbs"]

        for name, entries in BenchmarkReport._records.items():
            if not entries:
                continue
            lines.append(f"## {name}")
            lines.append("")
            tag_entries: dict[str, list] = {}
            for entry in entries:
                tag_entries.setdefault(entry["tag"], []).append(entry)

            result_keys = list(default_keys)
            for entry in entries:
                for key in entry["result"]:
                    if key not in result_keys:
                        result_keys.append(key)

            for tag, group in tag_entries.items():
                lines.append(f"### {tag}")
                lines.append("")
                param_keys = list(group[0]["params"].keys())
                has_config = any("config" in e for e in group)
                header = param_keys + result_keys
                if has_config:
                    header.append("config")
                lines.append("| " + " | ".join(header) + " |")
                lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                for entry in group:
                    row = [str(entry["params"].get(k, "")) for k in param_keys]
                    for rk in result_keys:
                        val = entry["result"].get(rk)
                        if val is None:
                            row.append("N/A")
                        elif isinstance(val, (int, float)) and not isinstance(val, bool):
                            row.append(f"{val:.4f}")
                        else:
                            row.append(str(val))
                    if has_config:
                        row.append(str(entry.get("config", "")))
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"Benchmark report saved to {path}")

    @staticmethod
    def clear() -> None:
        BenchmarkReport._records.clear()
