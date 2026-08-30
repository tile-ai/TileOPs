import gc

import pytest
import torch

# Imported for its side effect: arming the guard that keeps flag_gems from
# reaching torch's op registry before vllm. See benchmarks.baselines.
import benchmarks.baselines  # noqa: F401
from benchmarks.report import BenchmarkReport, _bench_results


def pytest_make_parametrize_id(config, val, argname):
    """Render the values pytest would otherwise collect as `shape0`, `dtype0`.

    A case id is the workload's name everywhere it is read later — the nightly
    report, the published page, the perf history key. This covers the values
    with no readable repr; it does not invent a name for the case, which is the
    author's job (see .claude/domain-rules/benchmark.md).
    """
    if isinstance(val, torch.dtype):
        return str(val).removeprefix("torch.")
    if isinstance(val, tuple) and val and all(isinstance(v, int) for v in val):
        return "x".join(str(v) for v in val)
    if isinstance(val, bool):
        name = argname
        for prefix in ("has_", "is_", "use_", "with_", "num_", "n_"):
            name = name.removeprefix(prefix)
        name = name.replace("_", "")
        return name if val else f"no{name}"
    return None


# What a row carries besides its measurements.
_NOT_A_MEASUREMENT = frozenset({"tag", "op", "op_module", "ops", "params", "run_config", "result"})


def _prop(value) -> str:
    """Format one measurement for the XML.

    Significant digits rather than fixed decimals: rates across the suite span
    six orders of magnitude, and a sub-microsecond kernel loses several percent
    to four decimal places.
    """
    if isinstance(value, (bool, int)):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.6g}"
    return str(value)


def _emit(item, tag: str, entry: dict) -> None:
    """Publish every measurement an implementation recorded.

    Generic over the keys: a measurement added to the benchmark layer reaches
    the XML, and the consumers that parse `<tag>_<metric>`, without a change
    here. Hand-listing them is how the report came to publish a quantity the
    benchmark had stopped comparing.
    """
    measurements = {**entry["result"], "dtype": entry.get("dtype")}
    for key, value in measurements.items():
        if key in _NOT_A_MEASUREMENT or value is None:
            continue
        item.user_properties.append((f"{tag}_{key}", _prop(value)))


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
            # Every op this case benchmarked, for the coverage gate: one case
            # may time more than one, and the row properties above describe
            # only the first.
            benchmarked = sorted({e["op"] for e in entries if e["tag"].startswith("tileops")})
            item.user_properties.append(("ops", ",".join(benchmarked)))
            if "op_module" in tileops_entry:
                item.user_properties.append(("op_module", tileops_entry["op_module"]))
            tag = tileops_entry["tag"]
            if tag != "tileops" and tag.startswith("tileops_"):
                item.user_properties.append(("tileops_variant", tag[len("tileops_") :]))
            _emit(item, "tileops", tileops_entry)

        # Every baseline is written under its own tag. The first also uses the
        # unprefixed legacy names that scripts/nightly_report.py reads.
        for idx, be in enumerate(baseline_entries):
            tag = be["tag"]
            _emit(item, tag, be)
            if idx == 0:
                item.user_properties.append(("baseline_tag", tag))
                _emit(item, "baseline", be)
            if not tileops_entry:
                continue
            # Ratios compare device_busy_ms: two implementations need not have
            # the same number of gaps between kernels.
            tl = tileops_entry["result"].get("device_busy_ms", 0)
            bl = be["result"].get("device_busy_ms", 0)
            if tl > 0 and bl > 0:
                item.user_properties.append((f"{tag}_ratio", f"{bl / tl:.4f}"))
                if idx == 0:
                    item.user_properties.append(("baseline_ratio", f"{bl / tl:.4f}"))
    finally:
        _bench_results.entries = []
        _release_cuda_cache_after_case()
