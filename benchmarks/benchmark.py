import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Tuple

import torch
from tilelang.profiler import do_bench

from tests.test_base import TestBase


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
                *inputs: Tuple[torch.Tensor],
                warmup: int = 100,
                rep: int = 100) -> dict:
        """Profile a callable and return structured results.

        Works for both tileops ops and baseline implementations.
        """
        def bench_fn():
            return functor(*inputs)

        with torch.no_grad():
            latency = do_bench(bench_fn, warmup=warmup, rep=rep, backend='cupti')
            if latency <= 0:
                latency = do_bench(bench_fn, warmup=warmup, rep=rep, backend='event')

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
    def record(name: str, params: dict, result: dict, tag: str = "tileops") -> None:
        """Record a benchmark result.

        Args:
            name: Benchmark group name (e.g. "gemm", "mha_fwd")
            params: Parameter dict (typically from locals())
            result: Dict with latency_ms, tflops, bandwidth_tbs
            tag: Label to distinguish implementations (e.g. "tileops", "baseline")
        """
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
