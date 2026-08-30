"""Where a measurement goes: the run's records, and the markdown report.

Holds no opinion on how a number was produced -- it is handed results and turns them
into the report a run leaves behind.
"""

import logging
import subprocess
import threading
from datetime import datetime
from typing import Any, Iterator, Optional

import torch

_logger = logging.getLogger("tileops.bench")

# Thread-local storage for conftest hook to pick up per-test bench results.
# A single test function may call record() multiple times (tileops + baseline).
_bench_results = threading.local()


def _current_case_rows() -> list:
    """The rows recorded by the case running on this thread."""
    if not hasattr(_bench_results, "entries"):
        _bench_results.entries = []
    return _bench_results.entries


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

    # Try to get NVIDIA driver version and clocks from nvidia-smi.
    gpu_query_fields = [
        "driver_version",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.applications.graphics",
        "clocks.applications.memory",
    ]
    gpu_query_values = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(gpu_query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_query_values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    driver = gpu_query_values[0] if len(gpu_query_values) == len(gpu_query_fields) else "N/A"
    lines.append(f"- **Driver version**: {driver}")

    if len(gpu_query_values) == len(gpu_query_fields):
        sm_clock, mem_clock, app_sm_clock, app_mem_clock = gpu_query_values[1:]
        lines.append(
            "- **GPU clocks**: "
            f"SM current {sm_clock} MHz, memory current {mem_clock} MHz, "
            f"application SM {app_sm_clock} MHz, "
            f"application memory {app_mem_clock} MHz"
        )

    return lines


def _op_kernels(op: object) -> Iterator[object]:
    """Yield the kernels *op* holds.

    An ``Op`` enumerates them itself through ``iter_kernels`` — slot entries and
    a directly bound ``op.kernel`` alike. A baseline that is not an ``Op``
    exposes at most a single ``kernel`` attribute.
    """
    iter_kernels = getattr(op, "iter_kernels", None)
    if callable(iter_kernels):
        yield from iter_kernels()
        return
    kernel = getattr(op, "kernel", None)
    if kernel is not None:
        yield kernel


def _extract_op_config(op: object) -> Optional[dict]:
    """Return the kernel config for an Op instance, or None if unavailable.

    A direct ``op.config`` attribute (explicit override) takes precedence over
    kernel introspection. Otherwise the first config among the kernels the op
    holds: kernels an op built share dtype and op kind, so the first is
    sufficient for a report that records one entry per call.
    """
    op_config = getattr(op, "config", None)
    if op_config:
        return op_config

    # FIXME(staged-rollout): reads `config` off the kernel object.
    #
    # Broken invariant: callers reach an op by calling it, not by reading its
    #   kernel's attributes.
    # Why: nothing reports what a call ran with; enumeration names the kernels
    #   but not what they were built for.
    # Cleanup: read the op's own execution metadata once it reports any.
    for kernel in _op_kernels(op):
        op_config = getattr(kernel, "config", None)
        if op_config:
            return op_config

    return None


class BenchmarkReport:
    """Collects benchmark results and dumps a markdown report.

    All methods are static — use as BenchmarkReport.record(...).
    Call clear() at session start, dump() at session end.
    """

    _records: dict = {}

    @staticmethod
    def record(op, params: dict, result: dict, tag: str = "tileops") -> None:
        """Record a benchmark result.

        Args:
            op: the Op the row measured. Every row of the report is one op's
                measurement, which is what lets a consumer group by op and read
                a name as a manifest entry. A comparison whose subject is not an
                op — a kernel strategy, a field of library implementations —
                decides something rather than tracking it, and belongs to
                ``benchmarks/studies/``, which the nightly sweep does not reach.
            params: Parameter dict (typically from locals())
            result: Dict with device_busy_ms, latency_ms, tflops, bandwidth_tbs
            tag: Label to distinguish implementations (e.g. "tileops", "FA3", "fla")
        """
        if isinstance(op, str):
            raise TypeError(
                f"record() takes the Op the row measured, not the name {op!r}. A "
                "comparison whose subject is not an op belongs in "
                "benchmarks/studies/, which the nightly does not sweep."
            )
        name = op.__class__.__name__
        op_module = op.__class__.__module__
        op_config = _extract_op_config(op)

        # Filter params to only include serializable benchmark parameters.
        # Tuples of primitives (e.g. ``shape=(4096, 4096)``) are preserved
        # verbatim so the profile log carries the original input geometry
        # rather than a flattened element count.
        def _is_serializable(v: Any) -> bool:
            if isinstance(v, (int, float, bool, str, torch.dtype)):
                return True
            if isinstance(v, tuple):
                return all(_is_serializable(x) for x in v)
            return False

        filtered_params = {
            k: v
            for k, v in params.items()
            if k not in ("test", "bm", "op", "inputs", "result", "result_bl", "baseline_fn", "tune")
            and not k.startswith("_")
            and _is_serializable(v)
        }
        record_entry = {
            "op": name,
            "tag": tag,
            "params": filtered_params,
            "result": result,
        }
        if op_module:
            record_entry["op_module"] = op_module
        if op_config:
            record_entry["config"] = op_config
        dtype = filtered_params.get("dtype")
        if isinstance(dtype, torch.dtype):
            record_entry["dtype"] = str(dtype).removeprefix("torch.")
        BenchmarkReport._records.setdefault(name, []).append(record_entry)
        # The same row, handed to the pytest hook that turns the running case's
        # rows into XML properties. One row recorded once: the log and the XML
        # read it, rather than each getting a copy that can drift from the other.
        _current_case_rows().append(record_entry)

        _logger.info(
            "op=%s module=%s tag=%s device_busy_ms=%.4f tflops=%.2f",
            name,
            op_module or "N/A",
            tag,
            result.get("device_busy_ms", 0),
            result.get("tflops", 0),
        )

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

        # device_busy_ms leads: it is the column implementations are compared on.
        default_result_keys = [
            "device_busy_ms",
            "latency_ms",
            "gap_ms",
            "n_kernels",
            "tflops",
            "bandwidth_tbs",
        ]

        for name, entries in BenchmarkReport._records.items():
            if not entries:
                continue

            lines.append(f"## {name}")
            lines.append("")

            # Group by tag
            tag_entries = {}
            for entry in entries:
                tag_entries.setdefault(entry["tag"], []).append(entry)
            result_keys = list(default_result_keys)
            for entry in entries:
                for key in entry["result"]:
                    if key not in result_keys:
                        result_keys.append(key)

            for tag, tag_group in tag_entries.items():
                lines.append(f"### {tag}")
                lines.append("")

                param_keys = list(tag_group[0]["params"].keys())
                has_config = any("config" in e for e in tag_group)
                header_parts = param_keys + result_keys
                if has_config:
                    header_parts.append("config")
                lines.append("| " + " | ".join(header_parts) + " |")
                lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

                for entry in tag_group:
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
                        cfg = entry.get("config")
                        row.append(str(cfg) if cfg else "")
                    lines.append("| " + " | ".join(row) + " |")

                lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        print(f"Benchmark report saved to {path}")

    @staticmethod
    def clear() -> None:
        """Clear all collected records."""
        BenchmarkReport._records.clear()
