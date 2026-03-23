"""CSV output utilities."""

import csv
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Common CSV fields
ENV_FIELDS = [
    "gpu_name", "sm", "mig_mode", "driver", "cuda", "torch", "tilelang",
    "clock_sm_mhz", "clock_mem_mhz",
]
CASE_FIELDS = [
    "benchmark", "backend", "dtype", "shape", "size_bytes",
    "working_set_bytes", "stride_bytes", "block_dim", "num_warps",
]
METRIC_FIELDS = [
    "warmup", "rep", "latency_us", "latency_ms",
    "bandwidth_gbs", "bandwidth_tbs", "tflops", "achieved_pct_of_peak",
]
AUX_FIELDS = ["notes", "timestamp"]

ALL_FIELDS = ENV_FIELDS + CASE_FIELDS + METRIC_FIELDS + AUX_FIELDS


class CSVWriter:
    """Simple CSV writer that appends rows to a file."""

    def __init__(self, filename, fieldnames):
        self.filepath = os.path.join(RESULTS_DIR, filename)
        self.fieldnames = fieldnames
        self._file = open(self.filepath, "w", newline="")  # noqa: SIM115
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def writerow(self, row_dict):
        full_row = {k: row_dict.get(k, "") for k in self.fieldnames}
        self._writer.writerow(full_row)
        self._file.flush()

    def close(self):
        self._file.close()

    @property
    def path(self):
        return self.filepath


def make_csv(benchmark_name):
    """Create a CSVWriter for a benchmark."""
    return CSVWriter(f"{benchmark_name}.csv", ALL_FIELDS)


def make_row(env_info, **kwargs):
    """Build a row dict from env info + keyword args."""
    row = {}
    for k in ENV_FIELDS:
        row[k] = env_info.get(k, "")
    row["timestamp"] = datetime.now().isoformat()
    row.update(kwargs)
    return row
