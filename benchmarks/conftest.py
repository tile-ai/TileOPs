import gc
import json

import pytest
import torch

import benchmarks.baselines  # noqa: F401

# Imported for its side effect: arming the guard that keeps flag_gems from
# reaching torch's op registry before vllm. See benchmarks.baselines.
from benchmarks import api
from benchmarks.report import BenchmarkReport, _bench_results
from benchmarks.timing import events_fallback_allowed, set_events_fallback_allowed
from tileops.utils.device import current_accelerator_type

# What a row carries besides its measurements.
_NOT_A_MEASUREMENT = frozenset(
    {"tag", "op", "op_module", "ops", "params", "kernel_config", "result"}
)


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
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _emit(item, tag: str, entry: dict) -> None:
    """Publish every measurement an implementation recorded.

    Generic over the keys: a measurement added to the benchmark layer reaches
    the XML, and the consumers that parse `<tag>_<metric>`, without a change
    here. Hand-listing them is how the report came to publish a quantity the
    benchmark had stopped comparing.
    """
    measurements = {
        **entry["result"],
        "unverified": (entry.get("params") or {}).get("unverified"),
        "no_ratio": (entry.get("params") or {}).get("no_ratio"),
    }
    for key, value in measurements.items():
        if key in _NOT_A_MEASUREMENT or value is None:
            continue
        item.user_properties.append((f"{tag}_{key}", _prop(value)))


def _release_device_cache_after_case() -> None:
    """Drop per-case Python references and the device's cached blocks between benchmarks."""
    gc.collect()
    if current_accelerator_type() is not None:
        torch.accelerator.empty_cache()


@pytest.fixture(autouse=True)
def setup() -> None:
    # Seeds every device's generator, CUDA's and a backend's alike.
    torch.manual_seed(1235)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register whether a benchmark run may time with CUDA events where CUPTI fails."""
    parser.addoption(
        "--tileops-allow-events-fallback",
        action="store_true",
        help="Time a case with CUDA events when CUPTI cannot attribute it, instead of "
        "failing it. Off by default: the report would mix two timing methods.",
    )
    parser.addoption(
        "--tileops-verify",
        action="store_true",
        help="Run correctness warmup only, omitting timing. Normal benchmarks also verify "
        "before sampling.",
    )


# The settings in force before configure, put back at unconfigure.
_OUTER_EVENTS_FALLBACK = pytest.StashKey[bool]()
_OUTER_VERIFYING = pytest.StashKey[bool]()


def pytest_configure(config):
    """Refuse a run device that is neither CUDA nor the process's accelerator, and pin the
    events-fallback setting.

    A CUDA device is timed through CUPTI and another accelerator with its device events.
    CUDA is accepted on a host without one, where the benchmarks' own tests collect.
    """
    device = torch.device(config.getoption("--tileops-device"))
    if device.type not in ("cuda", current_accelerator_type()):
        raise pytest.UsageError(
            f"--tileops-device={device}: benchmarks time on CUDA or on this process's "
            f"accelerator, which is {current_accelerator_type()}"
        )
    config.stash[_OUTER_EVENTS_FALLBACK] = events_fallback_allowed()
    config.stash[_OUTER_VERIFYING] = api.verifying()
    api.set_verifying(config.getoption("--tileops-verify"))
    set_events_fallback_allowed(config.getoption("--tileops-allow-events-fallback"))


def pytest_unconfigure(config):
    """Put back the settings this file replaced."""
    if _OUTER_EVENTS_FALLBACK in config.stash:
        set_events_fallback_allowed(config.stash[_OUTER_EVENTS_FALLBACK])
    if _OUTER_VERIFYING in config.stash:
        api.set_verifying(config.stash[_OUTER_VERIFYING])


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
            # A ratio states that two implementations of one function differ in speed. Where
            # either side carries an exception, that is not what it would state.
            if (be.get("params") or {}).get("no_ratio") or (tileops_entry.get("params") or {}).get(
                "no_ratio"
            ):
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
        for value in getattr(getattr(item, "callspec", None), "params", {}).values():
            if isinstance(value, api.Case):
                value._release()
        _release_device_cache_after_case()
