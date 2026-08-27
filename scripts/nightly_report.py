#!/usr/bin/env python3
"""Generate an Op-level nightly report from pytest JUnit XML outputs.

Usage:
    python scripts/nightly_report.py \
        --test-xml test_results.xml \
        --bench-xml bench_results.xml \
        [--history perf_history.json] \
        --output nightly_report.md \
        [--history-out perf_history_updated.json]
"""

import argparse
import contextlib
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGRESSION_THRESHOLD = 0.10  # 10% latency change => regression or improvement
NOISE_MULTIPLE = 2.5  # a delta within 2.5x the row's p90-p10 spread is noise
REGRESSION_ABS_MIN = 0.01  # noise floor for rows recorded without percentiles

# Measurement properties carried from the benchmark XML through to the report.
# Parsing and aggregation must read the same set.
_PERF_KEYS = (
    "tileops_device_busy_ms",
    "tileops_latency_ms",
    "tileops_gap_ms",
    "tileops_n_kernels",
    "tileops_tflops",
    "tileops_flops",
    "tileops_bytes",
    "tileops_bandwidth_tbs",
    "tileops_compute_roof",
    "tileops_uncounted_copy_ms",
    "tileops_variant",
    "tileops_timing",
    "tileops_device_busy_p10_ms",
    "tileops_device_busy_p90_ms",
    "tileops_n_samples",
    "baseline_tag",
    "baseline_device_busy_ms",
    "baseline_latency_ms",
    "baseline_tflops",
    "baseline_ratio",
)
BASELINE_RATIO_ALERT = 0.80  # tileops slower than baseline by >25%
HISTORY_RETENTION_DAYS = 14

# Algorithmic speed-of-light (SOL) verdict lines. Efficiency is
# sol_time / measured_time against the *calibrated* (effective) ceilings;
# see docs/design/roofline.md §1.2 and §4.3.
#
# The HBM ceiling is the envelope over access mixes, and a kernel's own mix
# caps lower (a perfect 2R:1W kernel reaches ~90% of it, a perfect 1R:1W ~87%);
# the green line must sit below every mix's personal ceiling.
SOL_GREEN_MEMORY = 0.80
SOL_GREEN_COMPUTE = 0.80  # tensor-core sustained calibration is noisier
SOL_ANOMALY = 1.05  # above the effective ceiling: formula or profile is wrong
# Below both floors the roofline has no traction: launch overhead and wave
# quantization dominate, so the row is labeled instead of judged. Regression
# detection still covers it.
LATENCY_BOUND_SOL_MS = 0.003
LATENCY_BOUND_MEASURED_MS = 0.020

# ── Emoji constants ───────────────────────────────────────────────────────
_PASS = "\u2705"  # ✅
_FAIL = "\u274c"  # ❌
_WARN = "\u26a0\ufe0f"  # ⚠️
_PARTY = "\U0001f389"  # 🎉
_RED = "\U0001f534"  # 🔴
_YELLOW = "\U0001f7e1"  # 🟡
_BLUE = "\U0001f535"  # 🔵
_GREEN = "\U0001f7e2"  # 🟢

# ---------------------------------------------------------------------------
# JUnit XML parsing
# ---------------------------------------------------------------------------


def _get_properties(testcase: ET.Element) -> dict[str, str]:
    """Extract user properties from a JUnit testcase element."""
    props = {}
    for ps in testcase.iter("properties"):
        for p in ps.iter("property"):
            props[p.attrib["name"]] = p.attrib.get("value", "")
    return props


def parse_test_xml(path: str) -> list[dict]:
    """Parse correctness test results, returning per-testcase dicts."""
    tree = ET.parse(path)
    results = []
    for tc in tree.iter("testcase"):
        props = _get_properties(tc)
        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")
        if skipped is not None:
            outcome = "skipped"
        elif failure is not None or error is not None:
            outcome = "failed"
        else:
            outcome = "passed"
        results.append(
            {
                "nodeid": f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}",
                "name": tc.attrib.get("name", ""),
                "outcome": outcome,
                "op": props.get("op"),
                "op_module": props.get("op_module"),
                "max_abs_err": props.get("max_abs_err"),
                "failure_message": (
                    failure.attrib.get("message", "")
                    if failure is not None
                    else error.attrib.get("message", "")
                    if error is not None
                    else None
                ),
            }
        )
    return results


def parse_bench_xml(path: str) -> list[dict]:
    """Parse benchmark results, returning per-testcase dicts."""
    tree = ET.parse(path)
    results = []
    for tc in tree.iter("testcase"):
        props = _get_properties(tc)
        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")
        if skipped is not None:
            outcome = "skipped"
        elif failure is not None or error is not None:
            outcome = "failed"
        else:
            outcome = "passed"

        entry = {
            "nodeid": f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}",
            "name": tc.attrib.get("name", ""),
            "outcome": outcome,
            "op": props.get("op"),
            "op_module": props.get("op_module"),
            "failure_message": (
                failure.attrib.get("message", "")
                if failure is not None
                else error.attrib.get("message", "")
                if error is not None
                else None
            ),
        }
        # Perf data
        for key in _PERF_KEYS:
            if key in props:
                try:
                    entry[key] = float(props[key])
                except ValueError:
                    entry[key] = props[key]

        # Collect tag-prefixed baselines written by conftest (e.g. fa3_latency_ms,
        # flashinfer_latency_ms).  Each baseline becomes a dict in "baselines".
        baselines = {}
        for pkey, pval in props.items():
            if pkey.endswith("_device_busy_ms") and pkey not in (
                "tileops_device_busy_ms",
                "baseline_device_busy_ms",
            ):
                tag = pkey.removesuffix("_device_busy_ms")
                baselines.setdefault(tag, {})["device_busy_ms"] = _try_float(pval)
            elif pkey.endswith("_latency_ms") and pkey not in (
                "tileops_latency_ms",
                "baseline_latency_ms",
            ):
                tag = pkey.removesuffix("_latency_ms")
                baselines.setdefault(tag, {})["latency_ms"] = _try_float(pval)
            elif pkey.endswith("_tflops") and pkey not in ("tileops_tflops", "baseline_tflops"):
                tag = pkey.removesuffix("_tflops")
                baselines.setdefault(tag, {})["tflops"] = _try_float(pval)
            elif pkey.endswith("_ratio") and pkey not in ("baseline_ratio",):
                tag = pkey.removesuffix("_ratio")
                baselines.setdefault(tag, {})["ratio"] = _try_float(pval)
        if baselines:
            entry["baselines"] = baselines

        results.append(entry)
    return results


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def parse_coverage_xml(path: str) -> list[dict] | None:
    """Read a coverage.py XML report into one record per ``src/tileops`` file.

    Each record carries the path relative to ``src/tileops/`` plus covered and
    total counts for statements and branches. Interpretation is left to
    ``_coverage_section`` — the raw per-file numbers mean different things
    inside and outside ``kernels/``.
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None

    files: list[dict] = []
    for cls in root.iter("class"):
        filename = cls.get("filename") or ""
        marker = "src/tileops/"
        if marker not in filename:
            continue

        stmts = covered = branches = branches_hit = 0
        for line in cls.iter("line"):
            stmts += 1
            covered += int(line.get("hits") or 0) > 0
            condition = line.get("condition-coverage") or ""
            if line.get("branch") == "true" and "(" in condition:
                hit, total = condition.split("(", 1)[1].rstrip(")").split("/")
                branches += int(total)
                branches_hit += int(hit)
        if stmts:
            files.append(
                {
                    "path": filename.split(marker, 1)[1],
                    "stmts": stmts,
                    "covered": covered,
                    "branches": branches,
                    "branches_hit": branches_hit,
                }
            )

    return files or None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_test_results(results: list[dict]) -> dict:
    """Group test results by Op."""
    ops = defaultdict(
        lambda: {
            "module": None,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "max_abs_err": 0.0,
            "failing_tests": [],
        }
    )
    for r in results:
        op = r.get("op")
        if not op:
            continue
        d = ops[op]
        if not d["module"]:
            d["module"] = r.get("op_module")
        d[r["outcome"]] += 1
        err = r.get("max_abs_err")
        if err:
            with contextlib.suppress(ValueError):
                d["max_abs_err"] = max(d["max_abs_err"], float(err))
        if r["outcome"] == "failed":
            d["failing_tests"].append(r["name"])
    return dict(ops)


def aggregate_bench_results(results: list[dict]) -> dict:
    """Group bench results by Op, keeping per-config perf data."""
    ops = defaultdict(lambda: {"module": None, "configs": []})
    for r in results:
        op = r.get("op")
        if not op or r["outcome"] != "passed":
            continue
        d = ops[op]
        if not d["module"]:
            d["module"] = r.get("op_module")
        config_entry = {"name": r["name"]}
        for key in (*_PERF_KEYS, "baselines"):
            if key in r:
                config_entry[key] = r[key]
        d["configs"].append(config_entry)
    return dict(ops)


def count_bench_skips(results: list[dict]) -> int:
    """Skipped cases, which are neither configs nor failures: without a count they vanish."""
    return sum(1 for r in results if r["outcome"] == "skipped")


def collect_bench_failures(results: list[dict]) -> list[dict]:
    """Collect failed benchmark results for reporting."""
    return [
        {
            "name": r["name"],
            "failure_message": r.get("failure_message"),
        }
        for r in results
        if r["outcome"] == "failed"
    ]


# ---------------------------------------------------------------------------
# History & regression detection
# ---------------------------------------------------------------------------


def load_history(path: str | None) -> list[dict]:
    """Load perf history JSON, returning list of runs."""
    if not path or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    return data.get("runs", [])


def prune_history(runs: list[dict], retention_days: int = HISTORY_RETENTION_DAYS) -> list[dict]:
    """Remove runs older than retention_days."""
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    return [r for r in runs if r.get("date", "") >= cutoff]


# Verdicts are drawn on device execution time, not on the span that also covers the
# gaps between a call's kernels.
_CONCLUSION_KEY = "device_busy_ms"


def _conclusion(cfg: dict) -> tuple[float | None, str]:
    """The reading a verdict is drawn on, and which key it came from.

    Runs recorded before the switch carry only ``latency_ms``. The key travels with
    the value so history is compared like with like: for an op launching several
    kernels the two differ by the gaps between them, and comparing across the pair
    would read as a change the op did not make.
    """
    busy = cfg.get(f"tileops_{_CONCLUSION_KEY}")
    if busy is not None:
        return busy, _CONCLUSION_KEY
    return cfg.get("tileops_latency_ms"), "latency_ms"


def _conclusion_ms(cfg: dict) -> float | None:
    return _conclusion(cfg)[0]


#: Absorbs the rounding a count recovered from ``tflops`` carries.
_DERIVED_COUNT_RTOL = 1e-3


class _WorkCounts(NamedTuple):
    flops: float | None
    nbytes: float | None
    #: False when ``flops`` was recovered from ``tflops`` rather than recorded.
    flops_recorded: bool


def _work_counts(reading: dict, ms: float | None) -> _WorkCounts:
    """Return the FLOP and byte counts of the work a reading timed."""
    nbytes, flops, tflops = reading.get("bytes"), reading.get("flops"), reading.get("tflops")
    if flops is None and tflops is not None and ms is not None:
        return _WorkCounts(tflops * ms * 1e9, nbytes, False)
    return _WorkCounts(flops, nbytes, True)


def _same_count(current: float | None, historical: float | None, rtol: float) -> bool:
    """Whether two counts are the same work, taking an unknown count as a match."""
    if current is None or historical is None:
        return True
    if current <= 0 or historical <= 0:
        return current == historical
    return abs(current - historical) / max(current, historical) <= rtol


def _same_workload(current: _WorkCounts, historical: _WorkCounts) -> bool:
    """Whether two readings timed the same workload."""
    rtol = 0.0 if current.flops_recorded and historical.flops_recorded else _DERIVED_COUNT_RTOL
    return _same_count(current.flops, historical.flops, rtol) and _same_count(
        current.nbytes, historical.nbytes, 0.0
    )


class _Reading(NamedTuple):
    ms: float
    spread: float | None  # p90 - p10 of the run that produced ``ms``


def _spread(props: dict) -> float | None:
    lo, hi = props.get("device_busy_p10_ms"), props.get("device_busy_p90_ms")
    return hi - lo if lo is not None and hi is not None else None


def _alias_name(runs: list[dict], op: str, config_name: str, work: _WorkCounts) -> str | None:
    """The unique prior display name of this row in history, or None.

    A name is adopted only when its recorded FLOP and byte counts equal the
    current row's exactly and it never shares a run with the current name: a
    renamed row and its new name never co-occur, while another variant of the
    same workload does.
    """
    if not work.flops_recorded or work.flops is None or work.nbytes is None:
        return None
    candidates: set[str] = set()
    excluded: set[str] = set()
    for run in runs:
        cfgs = run.get("ops", {}).get(op, {})
        has_current = config_name in cfgs
        for name, entry in cfgs.items():
            if name == config_name:
                continue
            if has_current:
                excluded.add(name)
                continue
            tileops_data = entry.get("tileops", {})
            ms = tileops_data.get(_CONCLUSION_KEY, tileops_data.get("latency_ms"))
            hist = _work_counts(tileops_data, ms)
            if hist.flops_recorded and hist.flops == work.flops and hist.nbytes == work.nbytes:
                candidates.add(name)
    candidates -= excluded
    return candidates.pop() if len(candidates) == 1 else None


def _config_readings(
    runs: list[dict], op: str, config_name: str, key: str, work: _WorkCounts
) -> list[_Reading]:
    """Workload-matched positive readings for one row, oldest first.

    Per run the current display name wins; a run that recorded the row only
    under its prior name (see ``_alias_name``) contributes that reading.
    """
    alias = _alias_name(runs, op, config_name, work)
    readings = []
    for run in runs:
        cfgs = run.get("ops", {}).get(op, {})
        for name in (config_name, alias):
            if name is None or name not in cfgs:
                continue
            tileops_data = cfgs[name].get("tileops", {})
            ms = tileops_data.get(key)
            if ms is None or ms <= 0 or not _same_workload(work, _work_counts(tileops_data, ms)):
                continue
            readings.append(_Reading(ms, _spread(tileops_data)))
            break
    return readings


def _reportable(delta_ms: float, base: _Reading, curr_spread: float | None) -> bool:
    """Whether a move of ``delta_ms`` against ``base`` clears both gates.

    Relative gate: ``REGRESSION_THRESHOLD`` of the baseline. Noise gate:
    ``NOISE_MULTIPLE`` times the wider of the two runs' p90-p10 spreads, or
    ``REGRESSION_ABS_MIN`` when neither run recorded percentiles.
    """
    spreads = [s for s in (curr_spread, base.spread) if s is not None]
    floor = NOISE_MULTIPLE * max(spreads) if spreads else REGRESSION_ABS_MIN
    return delta_ms / base.ms > REGRESSION_THRESHOLD and delta_ms > floor


def _verdict_inputs(bench_ops: dict, history_runs: list[dict]):
    """Yield one verdict input per config with a positive reading and history."""
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            lat, key = _conclusion(cfg)
            if lat is None or lat <= 0:
                continue
            props = {k.removeprefix("tileops_"): v for k, v in cfg.items()}
            work = _work_counts(props, lat)
            readings = _config_readings(history_runs, op, cfg["name"], key, work)
            if readings:
                yield op, cfg, lat, _spread(props), readings


def _record(op: str, cfg: dict, base_ms: float, curr_ms: float) -> dict:
    return {
        "op": op,
        "config": cfg["name"],
        "base_ms": base_ms,
        "curr_ms": curr_ms,
        "delta_pct": (curr_ms - base_ms) / base_ms * 100,
        "tflops": cfg.get("tileops_tflops"),
    }


def detect_regressions(bench_ops: dict, history_runs: list[dict]) -> list[dict]:
    """Rows slower than their 14-day median by the threshold and the noise gate.

    Why the median: the window minimum is a lucky extremum of one-sample
    nights and would alarm on every later normal night.
    """
    out = []
    for op, cfg, lat, curr_spread, readings in _verdict_inputs(bench_ops, history_runs):
        base = sorted(readings, key=lambda r: r.ms)[(len(readings) - 1) // 2]
        if _reportable(lat - base.ms, base, curr_spread):
            out.append(_record(op, cfg, base.ms, lat))
    return out


def detect_improvements(bench_ops: dict, history_runs: list[dict]) -> list[dict]:
    """Rows faster than every 14-day reading by the threshold and the noise gate."""
    out = []
    for op, cfg, lat, curr_spread, readings in _verdict_inputs(bench_ops, history_runs):
        base = min(readings, key=lambda r: r.ms)
        if _reportable(base.ms - lat, base, curr_spread):
            out.append(_record(op, cfg, base.ms, lat))
    return out


def detect_previous_run_shifts(bench_ops: dict, history_runs: list[dict]) -> list[dict]:
    """Rows that moved either way since their most recent comparable reading.

    A fix that returns a row to its old level reads as 0% against the 14-day
    best; this lens reports it.
    """
    out = []
    for op, cfg, lat, curr_spread, readings in _verdict_inputs(bench_ops, history_runs):
        base = readings[-1]
        if _reportable(abs(lat - base.ms), base, curr_spread):
            out.append(_record(op, cfg, base.ms, lat))
    return out


def detect_baseline_alerts(bench_ops: dict) -> list[dict]:
    """Find configs where tileops is slower than its strongest baseline: one alert each."""
    alerts = []
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            primary = cfg.get("baseline_tag", "baseline")
            candidates = [
                (
                    cfg.get("baseline_ratio"),
                    primary,
                    cfg.get(f"baseline_{_CONCLUSION_KEY}", cfg.get("baseline_latency_ms")),
                )
            ]
            candidates += [
                (bl.get("ratio"), tag, bl.get(_CONCLUSION_KEY, bl.get("latency_ms")))
                for tag, bl in cfg.get("baselines", {}).items()
                if tag != primary
            ]
            timed = [c for c in candidates if c[0] is not None]
            if not timed:
                continue
            ratio, tag, baseline_ms = min(timed, key=lambda c: c[0])
            if ratio < BASELINE_RATIO_ALERT:
                alerts.append(
                    {
                        "op": op,
                        "config": cfg["name"],
                        "tileops_ms": _conclusion_ms(cfg),
                        "baseline_ms": baseline_ms,
                        "ratio": ratio,
                        "baseline_tag": tag,
                    }
                )
    return alerts


# ---------------------------------------------------------------------------
# History update
# ---------------------------------------------------------------------------


def build_history_entry(bench_ops: dict, coverage: list[dict] | None = None) -> dict:
    """Build a history entry from current bench results.

    Coverage sits in a key of its own beside ``ops``, which regression
    detection and pruning do not read, so entries written before it existed
    stay readable.
    """
    commit = _get_git_commit()
    gpu = _get_gpu_name()
    ops_data = {}
    for op, data in (bench_ops or {}).items():
        cfg_data = {}
        for cfg in data["configs"]:
            entry = {}
            lat = cfg.get("tileops_latency_ms")
            busy = cfg.get("tileops_device_busy_ms")
            if lat is not None or busy is not None:
                entry["tileops"] = {}
                for key, value in (("latency_ms", lat), (_CONCLUSION_KEY, busy)):
                    if value is not None:
                        entry["tileops"][key] = value
                for name in (
                    "tflops",
                    "flops",
                    "bytes",
                    "compute_roof",
                    "device_busy_p10_ms",
                    "device_busy_p90_ms",
                ):
                    value = cfg.get(f"tileops_{name}")
                    if value is not None:
                        entry["tileops"][name] = value
                sol = cfg.get("sol")
                if sol is not None:
                    entry["tileops"]["sol"] = {
                        "efficiency": round(sol["efficiency"], 4),
                        "bound": sol["bound"],
                        "latency_bound": sol["latency_bound"],
                    }
            bl_lat = cfg.get("baseline_latency_ms")
            bl_busy = cfg.get(f"baseline_{_CONCLUSION_KEY}")
            if bl_lat is not None or bl_busy is not None:
                tag = cfg.get("baseline_tag", "baseline")
                if isinstance(tag, str):
                    entry[tag] = {}
                    for key, value in (("latency_ms", bl_lat), (_CONCLUSION_KEY, bl_busy)):
                        if value is not None:
                            entry[tag][key] = value
                    bl_tflops = cfg.get("baseline_tflops")
                    if bl_tflops is not None:
                        entry[tag]["tflops"] = bl_tflops
            # Additional baselines
            for btag, bl in cfg.get("baselines", {}).items():
                if btag == cfg.get("baseline_tag"):
                    continue  # already recorded above
                bl_entry = {}
                for bl_key in ("latency_ms", _CONCLUSION_KEY, "tflops"):
                    if bl.get(bl_key) is not None:
                        bl_entry[bl_key] = bl[bl_key]
                if bl_entry:
                    entry[btag] = bl_entry
            if entry:
                cfg_data[cfg["name"]] = entry
        if cfg_data:
            ops_data[op] = cfg_data
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "commit": commit,
        "gpu": gpu,
        "ops": ops_data,
    }
    if coverage:
        entry["coverage"] = _coverage_snapshot(coverage)
    return entry


def _coverage_snapshot(files: list[dict]) -> dict:
    """The three tracked coverage quantities, as plain numbers for history."""
    s = _coverage_signals(files)
    return {
        "never_built": len(s["never_built"]),
        "roofline_untested": s["roofline_untested"],
        "op_untested": s["op_untested"],
        "op_branches_hit": s["op_branches_hit"],
        "op_branches": s["op_branches"],
    }


def _previous_coverage(runs: list[dict]) -> dict | None:
    """The most recent recorded coverage snapshot, or None before any exists."""
    for run in reversed(runs):
        snapshot = run.get("coverage")
        if snapshot:
            return {**snapshot, "date": run.get("date", "")}
    return None


def _delta(current: int | float, previous: int | float | None, unit: str = "") -> str:
    """A signed change against the previous run, empty when it held.

    ``unit`` is ``"pp"`` for percentage points, otherwise a plain count. The
    footnote under the table names the run compared against, so an empty
    result reads as steady rather than as missing history.
    """
    if previous is None:
        return ""
    change = current - previous
    if abs(change) < (0.05 if unit == "pp" else 1):
        return ""
    sign = "+" if change > 0 else "−"
    magnitude = f"{abs(change):.1f}" if unit == "pp" else f"{abs(change):.0f}"
    return f" **{sign}{magnitude}{unit}**"


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _get_gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "N/A"


# ---------------------------------------------------------------------------
# Speed-of-Light (M5)
# ---------------------------------------------------------------------------


def _load_gpu_profile(gpu_name: str) -> dict | None:
    """Profile matching the measured device, or None when none claims it."""
    try:
        from tileops.perf.profile import find_profile
    except ImportError:
        return None
    try:
        return find_profile(gpu_name)
    except Exception:
        return None


def _compute_sol(cfg: dict, profile: dict) -> dict | None:
    """Algorithmic SOL efficiency for one benchmark row, or None.

    Returns None when the row lacks any input the model needs: (flops,
    bytes) from ``op.eval_roofline()``, a declared compute roof, a CUPTI
    reading, and a calibrated profile section for both ceilings. A missing
    input leaves the SOL column blank rather than guessing.
    """
    from tileops.perf.profile import resolve_roof

    flops = cfg.get("tileops_flops")
    nbytes = cfg.get("tileops_bytes")
    busy = cfg.get("tileops_device_busy_ms")
    roof_key = cfg.get("tileops_compute_roof")
    if not all(isinstance(v, (int, float)) for v in (flops, nbytes, busy)):
        return None
    if nbytes <= 0 or busy <= 0 or not isinstance(roof_key, str):
        return None
    if cfg.get("tileops_timing") != "cupti":
        return None
    hbm = profile.get("hbm")
    roof = resolve_roof(profile, roof_key)
    if not isinstance(hbm, dict) or "effective" not in hbm or roof is None:
        return None
    # Copies the reading excluded are device work the op's algorithm issued;
    # the roofline denominator has to carry them or efficiency inflates.
    denom_ms = busy + (cfg.get("tileops_uncounted_copy_ms") or 0.0)
    mem_ms = nbytes / hbm["effective"] * 1e3
    comp_ms = flops / roof["effective"] * 1e3
    sol_ms = max(mem_ms, comp_ms)
    return {
        "efficiency": sol_ms / denom_ms,
        "bound": "memory" if mem_ms >= comp_ms else "compute",
        "latency_bound": (sol_ms < LATENCY_BOUND_SOL_MS and denom_ms < LATENCY_BOUND_MEASURED_MS),
        "roof": roof_key,
        # Physically impossible rates: the formula (or roof) is wrong, not fast.
        "impossible": [
            signal
            for signal, rate, ceiling in (
                ("bytes/s over HBM theoretical", nbytes / denom_ms * 1e3, hbm["theoretical"]),
                ("FLOP/s over roof theoretical", flops / denom_ms * 1e3, roof["theoretical"]),
            )
            if rate > ceiling
        ],
    }


def annotate_sol(bench_ops: dict, profile: dict | None) -> list[dict]:
    """Attach a ``sol`` reading to every config row; return the anomalies.

    FAIL anomalies are physically impossible rates (a broken formula or
    roof); WARN anomalies exceed the calibrated ceiling by more than
    ``SOL_ANOMALY`` allows. Both exclude the row from green verdicts.
    """
    anomalies = []
    if profile is None:
        return anomalies
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            sol = _compute_sol(cfg, profile)
            if sol is None:
                continue
            cfg["sol"] = sol
            for signal in sol["impossible"]:
                anomalies.append(
                    {"level": "FAIL", "op": op, "config": cfg["name"], "signal": signal}
                )
            if not sol["impossible"] and sol["efficiency"] > SOL_ANOMALY:
                anomalies.append(
                    {
                        "level": "WARN",
                        "op": op,
                        "config": cfg["name"],
                        "signal": f"{sol['efficiency']:.0%} of the calibrated ceiling",
                    }
                )
    return anomalies


def _sol_cell(sol: dict | None) -> str:
    """Render one row's SOL reading for the benchmark table."""
    if sol is None:
        return "-"
    eff = sol["efficiency"]
    bound = "M" if sol["bound"] == "memory" else "C"
    if sol["impossible"] or eff > SOL_ANOMALY:
        return f"{_WARN} {eff:.0%} {bound}"
    if sol["latency_bound"]:
        return "<sub>lat-bound</sub>"
    green = SOL_GREEN_MEMORY if sol["bound"] == "memory" else SOL_GREEN_COMPUTE
    mark = f"{_PASS} " if eff >= green else ""
    return f"{mark}{eff:.0%} {bound}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _ratio_emoji(ratio: float) -> str:
    """Return an emoji indicator for a baseline ratio value."""
    if ratio >= 1.5:
        return _GREEN
    if ratio >= 1.0:
        return _BLUE
    if ratio >= BASELINE_RATIO_ALERT:
        return _YELLOW
    return _RED


def generate_report(
    test_ops: dict | None,
    bench_ops: dict | None,
    bench_failures: list[dict],
    regressions: list[dict],
    improvements: list[dict],
    baseline_alerts: list[dict],
    coverage: list[dict] | None = None,
    coverage_prev: dict | None = None,
    bench_skips: int = 0,
    previous_run_shifts: list[dict] | None = None,
    sol_anomalies: list[dict] | None = None,
    have_gpu_profile: bool = False,
) -> str:
    """Generate markdown report."""
    lines = []
    commit = _get_git_commit()
    gpu = _get_gpu_name()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sol_fails = [a for a in (sol_anomalies or []) if a["level"] == "FAIL"]

    # ── Header ────────────────────────────────────────────────────────────
    n_test_ops = len(test_ops) if test_ops else 0
    n_bench_ops = len(bench_ops) if bench_ops else 0
    n_failures = sum(1 for d in (test_ops or {}).values() if d["failed"] > 0)
    total_tests = sum(d["passed"] + d["failed"] + d["skipped"] for d in (test_ops or {}).values())
    total_passed = sum(d["passed"] for d in (test_ops or {}).values())

    health = (
        _PASS
        if (n_failures == 0 and not regressions and not bench_failures and not sol_fails)
        else _FAIL
    )
    lines.append(f"# {health} TileOPs Nightly Report")
    lines.append("")
    lines.append(f"> **{now}** &ensp;|&ensp; `{commit}` &ensp;|&ensp; {gpu}")
    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────
    corr_icon = _PASS if n_failures == 0 else f"{_FAIL} {n_failures} failed"
    bench_fail_icon = f"{_FAIL} {len(bench_failures)}" if bench_failures else f"{_PASS} None"
    if bench_skips:
        bench_fail_icon += f" &ensp;|&ensp; {_WARN} {bench_skips} skipped"
    reg_icon = f"{_PASS} None" if not regressions else f"{_WARN} {len(regressions)}"
    alert_icon = f"{_WARN} {len(baseline_alerts)}" if baseline_alerts else f"{_PASS} None"

    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(
        f"| **Correctness** | {corr_icon}"
        f" &ensp; ({total_passed}/{total_tests} tests across"
        f" {n_test_ops} ops) |"
    )
    lines.append(f"| **Benchmarked Ops** | {n_bench_ops} |")
    lines.append(f"| **Benchmark Failures** | {bench_fail_icon} |")
    lines.append(f"| **Regressions** (vs 14-day median) | {reg_icon} |")
    lines.append(f"| **Baseline Alerts** (< {BASELINE_RATIO_ALERT:.0%}) | {alert_icon} |")
    if have_gpu_profile:
        sol_icon = (
            f"{_FAIL} {len(sol_fails)} impossible"
            if sol_fails
            else f"{_WARN} {len(sol_anomalies)}"
            if sol_anomalies
            else f"{_PASS} None"
        )
        lines.append(f"| **Roofline anomalies** | {sol_icon} |")
    if improvements:
        lines.append(f"| **Improvements** (vs 14-day best) | {_PARTY} {len(improvements)} |")
    if previous_run_shifts:
        lines.append(f"| **Moved since previous run** | {_BLUE} {len(previous_run_shifts)} |")
    if coverage:
        # Repeated here because the Coverage section sits below the benchmark
        # tables, which run to hundreds of rows. One row per concern: a single
        # untested-line total would double-count ops/ against its own branch
        # figure and bury perf/, where a wrong number fails silently.
        sig = _coverage_signals(coverage)
        prev = coverage_prev or {}
        sep = " &ensp;·&ensp; "
        if sig["never_built"]:
            worst = sig["never_built"][0]
            lines.append(
                f"| **Never-built kernels** | {_WARN} "
                f"{len(sig['never_built'])} files"
                f"{_delta(len(sig['never_built']), prev.get('never_built'))}"
                f"{sep}`{worst['path']}` at "
                f"{_pct(worst['covered'], worst['stmts'])} |"
            )
        else:
            lines.append(f"| **Never-built kernels** | {_PASS} None |")
        rl_worst = sig["roofline_worst"]
        rl_hint = (
            f"{sep}`{rl_worst['path']}` at {_pct(rl_worst['covered'], rl_worst['stmts'])}"
            if rl_worst
            else ""
        )
        lines.append(
            f"| **Untested roofline math** | {sig['roofline_untested']} lines"
            f" in `perf/`"
            f"{_delta(sig['roofline_untested'], prev.get('roofline_untested'))}"
            f"{rl_hint} |"
        )
        op_branch_pct = (
            100 * sig["op_branches_hit"] / sig["op_branches"] if sig["op_branches"] else 0.0
        )
        prev_branch_pct = (
            100 * prev.get("op_branches_hit", 0) / prev["op_branches"]
            if prev.get("op_branches")
            else None
        )
        lines.append(
            f"| **Untested op logic** | {sig['op_untested']} lines in `ops/`"
            f"{_delta(sig['op_untested'], prev.get('op_untested'))}"
            f"{sep}{op_branch_pct:.1f}% of branches taken"
            f"{_delta(op_branch_pct, prev_branch_pct, 'pp')} |"
        )
        if prev.get("date"):
            lines.append(
                "| | <sub>coverage compared against the "
                f"{prev['date']} run; no figure means it held</sub> |"
            )
    lines.append("")

    # ── Test Failures (only if any) ───────────────────────────────────────
    failed_ops = {op: d for op, d in (test_ops or {}).items() if d["failed"] > 0}
    if failed_ops:
        lines.append(f"## {_FAIL} Test Failures")
        lines.append("")
        lines.append("| Op | Module | Failed / Total | Failing Tests |")
        lines.append("|:---|:-------|:--------------:|:--------------|")
        for op, d in sorted(failed_ops.items()):
            total = d["passed"] + d["failed"] + d["skipped"]
            tests_str = ", ".join(d["failing_tests"][:3])
            if len(d["failing_tests"]) > 3:
                tests_str += f", ... (+{len(d['failing_tests']) - 3})"
            lines.append(
                f"| **{op}** | `{d['module'] or 'N/A'}` | {d['failed']}/{total} | {tests_str} |"
            )
        lines.append("")

    # ── Benchmark Failures (only if any) ─────────────────────────────────
    if bench_failures:
        lines.append(f"## {_FAIL} Benchmark Failures")
        lines.append("")
        lines.append("| Test | Error |")
        lines.append("|:-----|:------|")
        for f in bench_failures:
            name = f["name"]
            msg = f.get("failure_message") or ""
            if len(msg) > 120:
                msg = msg[:120] + "..."
            msg = msg.replace("|", "\\|")
            lines.append(f"| {name} | {msg} |")
        lines.append("")

    # ── Regressions ───────────────────────────────────────────────────────
    if regressions:
        lines.append(f"## {_WARN} Performance Regressions (vs 14-day median)")
        lines.append("")
        lines.append("| Op | Config | Median (ms) | Current (ms) | Delta | TFLOPS |")
        lines.append("|:---|:-------|------------:|-----------:|------:|-------:|")
        for r in sorted(regressions, key=lambda x: -x["delta_pct"]):
            tflops_str = f"{r['tflops']:.2f}" if r.get("tflops") else "-"
            lines.append(
                f"| **{r['op']}** | {r['config']} "
                f"| {r['base_ms']:.4f} | {r['curr_ms']:.4f} "
                f"| +{r['delta_pct']:.1f}% | {tflops_str} |"
            )
        lines.append("")

    # ── Improvements ──────────────────────────────────────────────────────
    if improvements:
        lines.append(f"## {_PARTY} Performance Improvements (vs 14-day best)")
        lines.append("")
        lines.append("| Op | Config | Prev Best (ms) | Current (ms) | Delta | TFLOPS |")
        lines.append("|:---|:-------|---------------:|-----------:|------:|-------:|")
        for r in sorted(improvements, key=lambda x: x["delta_pct"]):
            tflops_str = f"{r['tflops']:.2f}" if r.get("tflops") else "-"
            lines.append(
                f"| **{r['op']}** | {r['config']} "
                f"| {r['base_ms']:.4f} | {r['curr_ms']:.4f} "
                f"| {r['delta_pct']:.1f}% | {tflops_str} |"
            )
        lines.append("")

    # ── Moves since the previous run ──────────────────────────────────────
    if previous_run_shifts:
        lines.append(f"## {_BLUE} Moved Since Previous Run")
        lines.append("")
        lines.append(
            "> Moves against the most recent reading. A row restored to its"
            " old level appears only here: returning is not a new 14-day record."
        )
        lines.append("")
        lines.append("| Op | Config | Previous (ms) | Current (ms) | Delta | TFLOPS |")
        lines.append("|:---|:-------|--------------:|-----------:|------:|-------:|")
        for r in sorted(previous_run_shifts, key=lambda x: x["delta_pct"]):
            tflops_str = f"{r['tflops']:.2f}" if r.get("tflops") else "-"
            lines.append(
                f"| **{r['op']}** | {r['config']} "
                f"| {r['base_ms']:.4f} | {r['curr_ms']:.4f} "
                f"| {r['delta_pct']:+.1f}% | {tflops_str} |"
            )
        lines.append("")

    # ── Roofline anomalies ────────────────────────────────────────────────
    if sol_anomalies:
        lines.append(f"## {_WARN} Roofline Model Anomalies")
        lines.append("")
        lines.append(
            "> A FAIL row implies a rate above the hardware's theoretical"
            " ceiling: its (flops, bytes) formula or declared roof is wrong,"
            " and its SOL reading cannot be trusted. A WARN row exceeds the"
            " calibrated ceiling; recheck the formula or the calibration."
        )
        lines.append("")
        lines.append("| Level | Op | Config | Signal |")
        lines.append("|:------|:---|:-------|:-------|")
        for a in sorted(sol_anomalies, key=lambda x: (x["level"] != "FAIL", x["op"])):
            lines.append(f"| {a['level']} | **{a['op']}** | {a['config']} | {a['signal']} |")
        lines.append("")

    # ── Baseline Alerts ───────────────────────────────────────────────────
    if baseline_alerts:
        lines.append(f"## {_RED} Baseline Performance Alerts")
        lines.append("")
        lines.append(
            "> TileOPs is slower than baseline"
            f" (ratio < {BASELINE_RATIO_ALERT:.0%})."
            " Ratio = baseline device-busy / tileops device-busy."
        )
        lines.append("")
        lines.append("| | Op | Config | TileOPs (ms) | Baseline (ms) | Ratio | Via |")
        lines.append("|:-|:---|:-------|------------:|-------------:|------:|:----|")
        for a in sorted(baseline_alerts, key=lambda x: x.get("ratio", 1)):
            emoji = _ratio_emoji(a["ratio"])
            lines.append(
                f"| {emoji} | **{a['op']}** | {a['config']} "
                f"| {a['tileops_ms']:.4f} | {a['baseline_ms']:.4f} "
                f"| {a['ratio']:.1%} | {a['baseline_tag']} |"
            )
        lines.append("")

    # ── Full Correctness Results (collapsible) ────────────────────────────
    if test_ops:
        lines.append("<details>")
        lines.append(
            f"<summary><strong>Full Correctness Results ({n_test_ops} ops)</strong></summary>"
        )
        lines.append("")
        lines.append("| | Op | Module | Pass | Fail | Skip | Max Error |")
        lines.append("|:-|:---|:-------|-----:|-----:|-----:|----------:|")
        for op in sorted(test_ops):
            d = test_ops[op]
            err_str = f"{d['max_abs_err']:.2e}" if d["max_abs_err"] else "-"
            icon = _PASS if d["failed"] == 0 else _FAIL
            lines.append(
                f"| {icon} | {op} | `{d['module'] or 'N/A'}` "
                f"| {d['passed']} | {d['failed']} | {d['skipped']} "
                f"| {err_str} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # ── Full Benchmark Results (collapsible) ──────────────────────────────
    if bench_ops:
        n_configs = sum(len(d["configs"]) for d in bench_ops.values())
        lines.append("<details>")
        lines.append(
            f"<summary><strong>Full Benchmark Results"
            f" ({n_configs} configs across"
            f" {n_bench_ops} ops)</strong></summary>"
        )
        lines.append("")
        if have_gpu_profile:
            lines.append(
                "> SOL = algorithmic speed-of-light efficiency:"
                " `max(bytes/BW, flops/roof) / device-busy` against the"
                " calibrated ceilings. `bytes` is the algorithm's minimum"
                " traffic (not measured DRAM bytes); `flops` follows the"
                " TileOPs counting convention; the roof is the unit an"
                " optimal implementation would use, not the running kernel's."
                f" M/C = memory/compute-bound; {_PASS} at"
                f" ≥{SOL_GREEN_MEMORY:.0%} (M) / ≥{SOL_GREEN_COMPUTE:.0%} (C);"
                " lat-bound rows are too small for the model to judge."
            )
            lines.append("")
        lines.append(
            "| | Op | Config | Device busy (ms) | TFLOPS | BW (TB/s) | SOL | Via | Ratio |"
        )
        lines.append("|:-|:---|:-------|------------:|-------:|----------:|----:|:----|------:|")
        for op in sorted(bench_ops):
            for cfg in bench_ops[op]["configs"]:
                lat = _conclusion_ms(cfg)
                tflops = cfg.get("tileops_tflops")
                bw = cfg.get("tileops_bandwidth_tbs")
                variant = cfg.get("tileops_variant")
                lat_str = f"{lat:.4f}" if lat is not None else "-"
                tflops_str = f"{tflops:.2f}" if tflops is not None else "-"
                bw_str = f"{bw:.2f}" if bw is not None else "-"
                sol_str = _sol_cell(cfg.get("sol"))

                # Collect all baselines for this config into rows
                bl_rows = []
                bl_tag = cfg.get("baseline_tag", "")
                ratio = cfg.get("baseline_ratio")
                if bl_tag:
                    bl_rows.append((bl_tag, ratio))
                for tag, bl in cfg.get("baselines", {}).items():
                    if tag == bl_tag:
                        continue
                    bl_rows.append((tag, bl.get("ratio")))

                if not bl_rows:
                    bl_str = f"strategy: {variant}" if variant else "-"
                    lines.append(
                        f"|  | {op} | {cfg['name']} "
                        f"| {lat_str} | {tflops_str} | {bw_str} | {sol_str} "
                        f"| {bl_str} | - |"
                    )
                else:
                    via_parts = []
                    for btag, bratio in bl_rows:
                        r_str = f"{bratio:.1%}" if bratio is not None else "-"
                        via_parts.append(f"{btag} {r_str}")
                    via_str = ", ".join(via_parts)
                    # Strongest baseline = fastest = lowest ratio.
                    rows = [r for _, r in bl_rows if r is not None]
                    emoji = _ratio_emoji(min(rows)) if rows else ""
                    lines.append(
                        f"| {emoji} | {op} | {cfg['name']} "
                        f"| {lat_str} | {tflops_str} | {bw_str} | {sol_str} "
                        f"| {via_str} | - |"
                    )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # ── Coverage ──────────────────────────────────────────────────────────
    if coverage:
        lines.extend(_coverage_section(sig))

    return "\n".join(lines)


def _pct(hit: int, total: int) -> str:
    return f"{100 * hit / total:.1f}%" if total else "-"


# A kernel file below this share of executed lines was never constructed by any
# test. The lowest genuinely-built kernel file measures 36.9%, so the threshold
# has room before it starts catching built kernels.
_KERNEL_BUILT_PCT = 25
# Below this, one statement swings the percentage too far to read.
_COVERAGE_MIN_STMTS = 20
_COVERAGE_WORST_N = 15  # rows in the least-covered file list


def _coverage_signals(files: list[dict]) -> dict:
    """Reduce per-file coverage to the three numbers worth acting on.

    A ``kernels/`` line counts as covered once the kernel is traced into IR, so
    its percentage says the kernel was built, not that its generated code ran.
    Only the never-built case carries information there, and it is returned as
    a file list rather than a rate. Elsewhere the ordinary reading holds.
    """
    kernels = [f for f in files if f["path"].startswith("kernels/")]
    pure = [f for f in files if not f["path"].startswith("kernels/")]

    never_built = sorted(
        (
            f
            for f in kernels
            if f["stmts"] >= _COVERAGE_MIN_STMTS
            and 100 * f["covered"] / f["stmts"] < _KERNEL_BUILT_PCT
        ),
        key=lambda f: f["covered"] / f["stmts"],
    )
    untested = sorted(
        (f for f in pure if f["stmts"] - f["covered"] > 0), key=lambda f: f["covered"] - f["stmts"]
    )

    ops = [f for f in files if f["path"].startswith("ops/")]
    roofline = [f for f in pure if f["path"].startswith("perf/")]
    return {
        "never_built": never_built,
        "untested": untested,
        "untested_lines": sum(f["stmts"] - f["covered"] for f in pure),
        "roofline_untested": sum(f["stmts"] - f["covered"] for f in roofline),
        "roofline_worst": min(
            (f for f in roofline if f["stmts"] >= _COVERAGE_MIN_STMTS),
            key=lambda f: f["covered"] / f["stmts"],
            default=None,
        ),
        "op_untested": sum(f["stmts"] - f["covered"] for f in ops),
        "op_branches_hit": sum(f["branches_hit"] for f in ops),
        "op_branches": sum(f["branches"] for f in ops),
    }


def _coverage_section(signals: dict, worst_n: int = _COVERAGE_WORST_N) -> list[str]:
    """Three explicit signals, each with what it means and what to do about it."""
    s = signals
    lines = ["## Coverage", ""]
    lines.append("| Signal | Value | What it means | What a bad number costs |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| Never-built kernels | {len(s['never_built'])} files "
        "| no test constructs these kernels "
        "| the kernel stops compiling and nothing says so until someone runs it |"
    )
    lines.append(
        f"| Untested roofline math | {s['roofline_untested']} lines in `perf/` "
        "| cost-model statements that never executed "
        "| benchmarks report wrong TFLOPS while every correctness test passes |"
    )
    lines.append(
        f"| Untested op logic | {s['op_untested']} lines in `ops/`, "
        f"{_pct(s['op_branches_hit'], s['op_branches'])} of branches "
        "| validation and dispatch paths not taken "
        "| a reversed shape or dtype check returns a wrong result instead of raising |"
    )
    lines.append("")
    lines.append(
        f"Everything outside `kernels/` accounts for {s['untested_lines']} untested "
        "lines; the two rows above carry the ones with an owner. Track the "
        "direction, not the absolute value. Smoke-only cases run in "
        "`gpu-smoke.yml`, so code reached solely by them counts as untested here."
    )
    lines.append("")

    if s["never_built"]:
        lines.append("### Never-built kernels")
        lines.append("")
        lines.append("| File | Executed |")
        lines.append("| --- | --- |")
        for f in s["never_built"]:
            lines.append(f"| `{f['path']}` | {_pct(f['covered'], f['stmts'])} |")
        lines.append("")

    if s["untested"]:
        lines.append("<details>")
        lines.append(f"<summary>Untested pure Python, worst {worst_n} files</summary>")
        lines.append("")
        lines.append("| File | Uncovered | Executed |")
        lines.append("| --- | --- | --- |")
        for f in s["untested"][:worst_n]:
            lines.append(
                f"| `{f['path']}` | {f['stmts'] - f['covered']} "
                f"| {_pct(f['covered'], f['stmts'])} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append(
        "Per-line detail is in the `htmlcov/` directory of this run's `tileops_op_test` artifact."
    )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate TileOPs nightly Op-level report")
    parser.add_argument("--test-xml", help="Path to correctness test JUnit XML")
    parser.add_argument("--bench-xml", help="Path to benchmark JUnit XML")
    parser.add_argument("--history", help="Path to perf_history.json (input)")
    parser.add_argument("--output", required=True, help="Output markdown report path")
    parser.add_argument("--history-out", help="Path to write updated perf_history.json")
    parser.add_argument("--coverage-xml", help="Path to coverage.py XML report")
    args = parser.parse_args()

    # Parse results
    test_ops = None
    if args.test_xml and Path(args.test_xml).exists():
        test_results = parse_test_xml(args.test_xml)
        test_ops = aggregate_test_results(test_results)

    bench_ops = None
    bench_failures = []
    bench_skips = 0
    if args.bench_xml and Path(args.bench_xml).exists():
        bench_results = parse_bench_xml(args.bench_xml)
        bench_ops = aggregate_bench_results(bench_results)
        bench_failures = collect_bench_failures(bench_results)
        bench_skips = count_bench_skips(bench_results)

    # Prune first: the carried-over artifact can hold entries older than the
    # window when a run gap exceeds the retention period, and the verdicts below
    # are labelled with the 14-day window.
    gpu_profile = _load_gpu_profile(_get_gpu_name())
    sol_anomalies = annotate_sol(bench_ops, gpu_profile) if bench_ops else []

    history_runs = prune_history(load_history(args.history))
    regressions = detect_regressions(bench_ops, history_runs) if bench_ops else []
    improvements = detect_improvements(bench_ops, history_runs) if bench_ops else []
    previous_run_shifts = detect_previous_run_shifts(bench_ops, history_runs) if bench_ops else []
    baseline_alerts = detect_baseline_alerts(bench_ops) if bench_ops else []

    coverage = None
    if args.coverage_xml and Path(args.coverage_xml).exists():
        coverage = parse_coverage_xml(args.coverage_xml)
    # Read before this run is appended, so the comparison is against a prior run.
    coverage_prev = _previous_coverage(history_runs)

    # Generate report
    report = generate_report(
        test_ops,
        bench_ops,
        bench_failures,
        regressions,
        improvements,
        baseline_alerts,
        coverage,
        coverage_prev,
        bench_skips,
        previous_run_shifts,
        sol_anomalies,
        have_gpu_profile=gpu_profile is not None,
    )
    Path(args.output).write_text(report)
    print(f"Report written to {args.output}")

    # Recorded on coverage alone too, so a night the benchmark job produced
    # nothing does not drop a reading and leave the next run comparing stale.
    if args.history_out and (bench_ops or coverage):
        entry = build_history_entry(bench_ops, coverage)
        history_runs.append(entry)
        Path(args.history_out).write_text(json.dumps({"runs": history_runs}, indent=2))
        print(f"History updated: {args.history_out}")


if __name__ == "__main__":
    main()
