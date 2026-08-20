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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGRESSION_THRESHOLD = 0.10  # 10% latency change => regression or improvement
REGRESSION_ABS_MIN = 0.01  # ignore regressions < 0.01 ms

# Measurement properties carried from the benchmark XML through to the report.
# Parsing and aggregation must read the same set.
_PERF_KEYS = (
    "tileops_device_busy_ms",
    "tileops_latency_ms",
    "tileops_gap_ms",
    "tileops_n_kernels",
    "tileops_tflops",
    "tileops_bandwidth_tbs",
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


def find_best_latency(
    runs: list[dict],
    op: str,
    config_name: str,
    key: str = _CONCLUSION_KEY,
) -> float | None:
    """Find the best (lowest) tileops reading for an op+config across history."""
    best = None
    for run in runs:
        tileops_data = run.get("ops", {}).get(op, {}).get(config_name, {}).get("tileops", {})
        lat = tileops_data.get(key)
        if lat is not None and (best is None or lat < best):
            best = lat
    return best


def _history_deltas(bench_ops: dict, history_runs: list[dict]):
    """Yield ``(record, delta)`` per config with both a reading and a history best.

    ``delta`` is the fractional change against that best: positive is slower.
    """
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            lat, key = _conclusion(cfg)
            if lat is None:
                continue
            best = find_best_latency(history_runs, op, cfg["name"], key)
            if best is None:
                continue
            yield (
                {
                    "op": op,
                    "config": cfg["name"],
                    "best_ms": best,
                    "curr_ms": lat,
                    "delta_pct": (lat - best) / best * 100,
                    "tflops": cfg.get("tileops_tflops"),
                },
                (lat - best) / best,
            )


def detect_regressions(bench_ops: dict, history_runs: list[dict]) -> list[dict]:
    """Detect performance regressions vs 14-day best."""
    return [
        record
        for record, delta in _history_deltas(bench_ops, history_runs)
        if delta > REGRESSION_THRESHOLD
        and (record["curr_ms"] - record["best_ms"]) > REGRESSION_ABS_MIN
    ]


def detect_improvements(bench_ops: dict, history_runs: list[dict]) -> list[dict]:
    """Detect performance improvements vs 14-day best."""
    return [
        record
        for record, delta in _history_deltas(bench_ops, history_runs)
        if delta < -REGRESSION_THRESHOLD
    ]


def detect_baseline_alerts(bench_ops: dict) -> list[dict]:
    """Find ops where tileops is significantly slower than any baseline."""
    alerts = []
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            # Check legacy primary baseline
            ratio = cfg.get("baseline_ratio")
            if ratio is not None and ratio < BASELINE_RATIO_ALERT:
                alerts.append(
                    {
                        "op": op,
                        "config": cfg["name"],
                        "tileops_ms": _conclusion_ms(cfg),
                        "baseline_ms": cfg.get(
                            f"baseline_{_CONCLUSION_KEY}", cfg.get("baseline_latency_ms")
                        ),
                        "ratio": ratio,
                        "baseline_tag": cfg.get("baseline_tag", "baseline"),
                    }
                )
            # Check additional baselines
            for tag, bl in cfg.get("baselines", {}).items():
                bl_ratio = bl.get("ratio")
                if bl_ratio is not None and bl_ratio < BASELINE_RATIO_ALERT:
                    # Skip if this is the same as the primary baseline
                    if tag == cfg.get("baseline_tag"):
                        continue
                    alerts.append(
                        {
                            "op": op,
                            "config": cfg["name"],
                            "tileops_ms": _conclusion_ms(cfg),
                            "baseline_ms": bl.get(_CONCLUSION_KEY, bl.get("latency_ms")),
                            "ratio": bl_ratio,
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
                tflops = cfg.get("tileops_tflops")
                if tflops is not None:
                    entry["tileops"]["tflops"] = tflops
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


def _strongest_baseline_ratio(bl_rows: list[tuple[str, float | None]]) -> float | None:
    """Return the fastest baseline ratio for a config.

    Ratio is ``baseline_latency / tileops_latency``. Smaller means the baseline
    is faster and therefore the stronger competitor for table coloring.
    """
    ratios = [ratio for _, ratio in bl_rows if ratio is not None]
    return min(ratios, default=None)


def generate_report(
    test_ops: dict | None,
    bench_ops: dict | None,
    bench_failures: list[dict],
    regressions: list[dict],
    improvements: list[dict],
    baseline_alerts: list[dict],
    coverage: list[dict] | None = None,
    coverage_prev: dict | None = None,
) -> str:
    """Generate markdown report."""
    lines = []
    commit = _get_git_commit()
    gpu = _get_gpu_name()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ────────────────────────────────────────────────────────────
    n_test_ops = len(test_ops) if test_ops else 0
    n_bench_ops = len(bench_ops) if bench_ops else 0
    n_failures = sum(1 for d in (test_ops or {}).values() if d["failed"] > 0)
    total_tests = sum(d["passed"] + d["failed"] + d["skipped"] for d in (test_ops or {}).values())
    total_passed = sum(d["passed"] for d in (test_ops or {}).values())

    health = _PASS if (n_failures == 0 and not regressions and not bench_failures) else _FAIL
    lines.append(f"# {health} TileOPs Nightly Report")
    lines.append("")
    lines.append(f"> **{now}** &ensp;|&ensp; `{commit}` &ensp;|&ensp; {gpu}")
    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────
    corr_icon = _PASS if n_failures == 0 else f"{_FAIL} {n_failures} failed"
    bench_fail_icon = f"{_FAIL} {len(bench_failures)}" if bench_failures else f"{_PASS} None"
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
    lines.append(f"| **Regressions** (vs 14-day best) | {reg_icon} |")
    lines.append(f"| **Baseline Alerts** (< {BASELINE_RATIO_ALERT:.0%}) | {alert_icon} |")
    if improvements:
        lines.append(f"| **Improvements** (vs 14-day best) | {_PARTY} {len(improvements)} |")
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
        lines.append(f"## {_WARN} Performance Regressions (vs 14-day best)")
        lines.append("")
        lines.append("| Op | Config | Best (ms) | Current (ms) | Delta | TFLOPS |")
        lines.append("|:---|:-------|----------:|-----------:|------:|-------:|")
        for r in sorted(regressions, key=lambda x: -x["delta_pct"]):
            tflops_str = f"{r['tflops']:.2f}" if r.get("tflops") else "-"
            lines.append(
                f"| **{r['op']}** | {r['config']} "
                f"| {r['best_ms']:.4f} | {r['curr_ms']:.4f} "
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
                f"| {r['best_ms']:.4f} | {r['curr_ms']:.4f} "
                f"| {r['delta_pct']:.1f}% | {tflops_str} |"
            )
        lines.append("")

    # ── Baseline Alerts ───────────────────────────────────────────────────
    if baseline_alerts:
        lines.append(f"## {_RED} Baseline Performance Alerts")
        lines.append("")
        lines.append(
            "> TileOPs is slower than baseline"
            f" (ratio < {BASELINE_RATIO_ALERT:.0%})."
            " Ratio = baseline_latency / tileops_latency."
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
        lines.append("| | Op | Config | Latency (ms) | TFLOPS | BW (TB/s) | Via | Ratio |")
        lines.append("|:-|:---|:-------|------------:|-------:|----------:|:----|------:|")
        for op in sorted(bench_ops):
            for cfg in bench_ops[op]["configs"]:
                lat = _conclusion_ms(cfg)
                tflops = cfg.get("tileops_tflops")
                bw = cfg.get("tileops_bandwidth_tbs")
                variant = cfg.get("tileops_variant")
                lat_str = f"{lat:.4f}" if lat else "-"
                tflops_str = f"{tflops:.2f}" if tflops else "-"
                bw_str = f"{bw:.2f}" if bw else "-"

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
                        f"| {lat_str} | {tflops_str} | {bw_str} "
                        f"| {bl_str} | - |"
                    )
                else:
                    via_parts = []
                    for btag, bratio in bl_rows:
                        r_str = f"{bratio:.1%}" if bratio else "-"
                        via_parts.append(f"{btag} {r_str}")
                    via_str = ", ".join(via_parts)
                    best_ratio = _strongest_baseline_ratio(bl_rows)
                    emoji = _ratio_emoji(best_ratio) if best_ratio else ""
                    lines.append(
                        f"| {emoji} | {op} | {cfg['name']} "
                        f"| {lat_str} | {tflops_str} | {bw_str} "
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
    if args.bench_xml and Path(args.bench_xml).exists():
        bench_results = parse_bench_xml(args.bench_xml)
        bench_ops = aggregate_bench_results(bench_results)
        bench_failures = collect_bench_failures(bench_results)

    # Prune first: the carried-over artifact can hold entries older than the
    # window when a run gap exceeds the retention period, and the verdicts below
    # are labelled "vs 14-day best".
    history_runs = prune_history(load_history(args.history))
    regressions = detect_regressions(bench_ops, history_runs) if bench_ops else []
    improvements = detect_improvements(bench_ops, history_runs) if bench_ops else []
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
