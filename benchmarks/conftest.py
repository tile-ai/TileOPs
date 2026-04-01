import pytest
import torch

from benchmarks.benchmark import BenchmarkReport, CuptiSession, _bench_results

# Skip NSA benchmarks until op failures are resolved (see #696).
collect_ignore_glob = [
    "ops/bench_deepseek_nsa_*.py",
]


@pytest.fixture(scope="session", autouse=True)
def cupti_session():
    """Shared CUPTI profiler session that lives for the entire pytest run.

    All ``do_bench(backend='cupti')`` calls automatically use the shared
    session (via the monkey-patch in benchmark.py), eliminating per-call
    CUPTI init/teardown overhead.
    """
    if not torch.cuda.is_available():
        yield None
        return
    with CuptiSession() as session:
        yield session


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

    _bench_results.entries = []
