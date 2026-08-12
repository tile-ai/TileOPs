import gc

import pytest
import torch

from benchmarks.report import BenchmarkReport, _bench_results


def _release_cuda_cache_after_case() -> None:
    """Drop per-case Python references and cached CUDA blocks between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def pytest_sessionstart(session):
    BenchmarkReport.clear()


def pytest_sessionfinish(session, exitstatus):
    BenchmarkReport.dump("profile_run.log")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """After bench test execution, attach perf data to the item as properties."""
    _bench_results.entries = []
    try:
        yield
        entries = getattr(_bench_results, "entries", [])
        if not entries:
            return

        # Separate tileops entry (tag starts with "tileops") from baselines.
        tileops_entry = None
        baseline_entries = []
        for e in entries:
            if e["tag"].startswith("tileops"):
                if tileops_entry is None:
                    tileops_entry = e
            else:
                baseline_entries.append(e)

        if tileops_entry:
            item.user_properties.append(("op", tileops_entry["op"]))
            if "op_module" in tileops_entry:
                item.user_properties.append(("op_module", tileops_entry["op_module"]))
            tag = tileops_entry["tag"]
            if tag != "tileops" and tag.startswith("tileops_"):
                item.user_properties.append(("tileops_variant", tag[len("tileops_"):]))
            item.user_properties.append(("tileops_latency_ms",
                                         f"{tileops_entry.get('latency_ms', 0):.4f}"))
            tflops = tileops_entry.get("tflops")
            if tflops is not None:
                item.user_properties.append(("tileops_tflops", f"{tflops:.2f}"))
            bw = tileops_entry.get("bandwidth_tbs")
            if bw is not None:
                item.user_properties.append(("tileops_bandwidth_tbs", f"{bw:.2f}"))
            # Trust metadata for the reported median: which timer produced it,
            # and how wide the samples it summarizes were.
            for key in ("latency_p10_ms", "latency_p90_ms"):
                value = tileops_entry.get(key)
                if value is not None:
                    item.user_properties.append((f"tileops_{key}", f"{value:.4f}"))
            n_samples = tileops_entry.get("n_samples")
            if n_samples is not None:
                item.user_properties.append(("tileops_n_samples", str(n_samples)))
            timing = tileops_entry.get("timing")
            if timing is not None:
                item.user_properties.append(("tileops_timing", str(timing)))

        # Write all baselines into JUnit XML properties.
        # The first baseline uses the legacy unprefixed names (baseline_tag, etc.)
        # for backward compatibility.  Additional baselines use "{tag}_latency_ms",
        # "{tag}_tflops", "{tag}_ratio" so the report can display multiple columns.
        for idx, be in enumerate(baseline_entries):
            tag = be["tag"]
            bl_latency = be.get("latency_ms", 0)
            bl_tflops = be.get("tflops")

            if idx == 0:
                # Legacy unprefixed keys — consumed by existing nightly_report.py
                item.user_properties.append(("baseline_tag", tag))
                item.user_properties.append(("baseline_latency_ms", f"{bl_latency:.4f}"))
                if bl_tflops is not None:
                    item.user_properties.append(("baseline_tflops", f"{bl_tflops:.2f}"))
                if tileops_entry:
                    tl = tileops_entry.get("latency_ms", 0)
                    if tl > 0 and bl_latency > 0:
                        item.user_properties.append(("baseline_ratio",
                                                     f"{bl_latency / tl:.4f}"))

            # Tag-prefixed keys — always written for every baseline
            item.user_properties.append((f"{tag}_latency_ms", f"{bl_latency:.4f}"))
            if bl_tflops is not None:
                item.user_properties.append((f"{tag}_tflops", f"{bl_tflops:.2f}"))
            if tileops_entry:
                tl = tileops_entry.get("latency_ms", 0)
                if tl > 0 and bl_latency > 0:
                    item.user_properties.append((f"{tag}_ratio", f"{bl_latency / tl:.4f}"))
    finally:
        _bench_results.entries = []
        _release_cuda_cache_after_case()
