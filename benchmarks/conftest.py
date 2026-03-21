import pytest
import torch

from benchmarks.benchmark import BenchmarkReport, _bench_results


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
    yield
    entries = getattr(_bench_results, "entries", [])
    if not entries:
        return

    # Find tileops entry and baseline entries.
    tileops_entry = None
    baseline_entries = []
    for e in entries:
        if e["tag"] == "tileops":
            tileops_entry = e
        else:
            baseline_entries.append(e)

    if tileops_entry:
        item.user_properties.append(("op", tileops_entry["op"]))
        if "op_module" in tileops_entry:
            item.user_properties.append(("op_module", tileops_entry["op_module"]))
        item.user_properties.append(("tileops_latency_ms",
                                     f"{tileops_entry.get('latency_ms', 0):.4f}"))
        tflops = tileops_entry.get("tflops")
        if tflops is not None:
            item.user_properties.append(("tileops_tflops", f"{tflops:.2f}"))
        bw = tileops_entry.get("bandwidth_tbs")
        if bw is not None:
            item.user_properties.append(("tileops_bandwidth_tbs", f"{bw:.2f}"))

    for be in baseline_entries:
        tag = be["tag"]
        item.user_properties.append(("baseline_tag", tag))
        item.user_properties.append(("baseline_latency_ms",
                                     f"{be.get('latency_ms', 0):.4f}"))
        bl_tflops = be.get("tflops")
        if bl_tflops is not None:
            item.user_properties.append(("baseline_tflops", f"{bl_tflops:.2f}"))

        # Compute baseline ratio if both tileops and baseline have latency.
        if tileops_entry:
            tl = tileops_entry.get("latency_ms", 0)
            bl = be.get("latency_ms", 0)
            if tl > 0 and bl > 0:
                ratio = bl / tl
                item.user_properties.append(("baseline_ratio", f"{ratio:.4f}"))

    _bench_results.entries = []
