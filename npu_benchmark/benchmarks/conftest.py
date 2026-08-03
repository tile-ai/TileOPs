"""Pytest hooks for benchmark sessions.

- ``pytest_sessionstart``: clear report state, seed RNG.
- ``pytest_runtest_call``: attach perf data to test item properties.
- ``pytest_sessionfinish``: dump markdown report.
"""

from __future__ import annotations

import gc

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, _bench_results
from utils import empty_cache, is_available, manual_seed_all


def _release_cache() -> None:
    gc.collect()
    if is_available():
        empty_cache()


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if is_available():
        manual_seed_all(1235)


def pytest_sessionstart(session):
    BenchmarkReport.clear()


def pytest_sessionfinish(session, exitstatus):
    BenchmarkReport.dump("profile_run.log")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _bench_results.entries = []
    try:
        yield
        entries = getattr(_bench_results, "entries", [])
        if not entries:
            return

        kernel_entry = None
        baseline_entries = []
        for e in entries:
            if e["tag"].startswith("kernel"):
                if kernel_entry is None:
                    kernel_entry = e
            else:
                baseline_entries.append(e)

        if kernel_entry:
            item.user_properties.append(("op", kernel_entry["op"]))
            if "op_module" in kernel_entry:
                item.user_properties.append(("op_module", kernel_entry["op_module"]))
            item.user_properties.append(("kernel_latency_ms",
                                         f"{kernel_entry.get('latency_ms', 0):.4f}"))
            tflops = kernel_entry.get("tflops")
            if tflops is not None:
                item.user_properties.append(("kernel_tflops", f"{tflops:.2f}"))
            bw = kernel_entry.get("bandwidth_tbs")
            if bw is not None:
                item.user_properties.append(("kernel_bandwidth_tbs", f"{bw:.2f}"))

        for idx, be in enumerate(baseline_entries):
            tag = be["tag"]
            bl_latency = be.get("latency_ms", 0)
            bl_tflops = be.get("tflops")

            if idx == 0:
                item.user_properties.append(("baseline_tag", tag))
                item.user_properties.append(("baseline_latency_ms", f"{bl_latency:.4f}"))
                if bl_tflops is not None:
                    item.user_properties.append(("baseline_tflops", f"{bl_tflops:.2f}"))
                if kernel_entry:
                    tl = kernel_entry.get("latency_ms", 0)
                    if tl > 0 and bl_latency > 0:
                        item.user_properties.append(("baseline_ratio",
                                                     f"{bl_latency / tl:.4f}"))

            item.user_properties.append((f"{tag}_latency_ms", f"{bl_latency:.4f}"))
            if bl_tflops is not None:
                item.user_properties.append((f"{tag}_tflops", f"{bl_tflops:.2f}"))
            if kernel_entry:
                tl = kernel_entry.get("latency_ms", 0)
                if tl > 0 and bl_latency > 0:
                    item.user_properties.append((f"{tag}_ratio",
                                                 f"{bl_latency / tl:.4f}"))
    finally:
        _bench_results.entries = []
        _release_cache()
