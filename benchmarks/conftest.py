import os
from pathlib import Path

import pytest
import torch

from benchmarks.benchmark import BenchmarkReport


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def pytest_sessionstart(session):
    BenchmarkReport.clear()


def pytest_sessionfinish(session, exitstatus):
    report_path = Path(os.environ.get("BENCHMARK_REPORT_PATH", "profile_run.log"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    BenchmarkReport.dump(str(report_path))
