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

REGRESSION_THRESHOLD = 0.10  # 10% latency increase => regression
REGRESSION_ABS_MIN = 0.01  # ignore regressions < 0.01 ms
NOISE_FLOOR = 0.05  # ignore <=5% fluctuations (measurement noise)
BASELINE_RATIO_ALERT = 0.80  # tileops slower than baseline by >25%
HISTORY_RETENTION_DAYS = 14

# ── Emoji constants ───────────────────────────────────────────────────────
_PASS = "\u2705"          # ✅
_FAIL = "\u274c"          # ❌
_WARN = "\u26a0\ufe0f"   # ⚠️
_PARTY = "\U0001f389"     # 🎉
_RED = "\U0001f534"       # 🔴
_YELLOW = "\U0001f7e1"    # 🟡
_BLUE = "\U0001f535"      # 🔵
_GREEN = "\U0001f7e2"     # 🟢

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
        results.append({
            "nodeid": f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}",
            "name": tc.attrib.get("name", ""),
            "outcome": outcome,
            "op": props.get("op"),
            "op_module": props.get("op_module"),
            "max_abs_err": props.get("max_abs_err"),
            "failure_message": (failure.attrib.get("message", "") if failure is not None
                                else error.attrib.get("message", "") if error is not None
                                else None),
        })
    return results


def parse_bench_xml(path: str) -> list[dict]:
    """Parse benchmark results, returning per-testcase dicts."""
    tree = ET.parse(path)
    results = []
    for tc in tree.iter("testcase"):
        props = _get_properties(tc)
        failure = tc.find("failure")
        skipped = tc.find("skipped")
        if skipped is not None:
            outcome = "skipped"
        elif failure is not None:
            outcome = "failed"
        else:
            outcome = "passed"

        entry = {
            "nodeid": f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}",
            "name": tc.attrib.get("name", ""),
            "outcome": outcome,
            "op": props.get("op"),
            "op_module": props.get("op_module"),
            "failure_message": (failure.attrib.get("message", "") if failure is not None
                                else None),
        }
        # Perf data
        for key in ("tileops_latency_ms", "tileops_tflops", "tileops_bandwidth_tbs",
                     "tileops_variant",
                     "baseline_tag", "baseline_latency_ms", "baseline_tflops",
                     "baseline_ratio"):
            if key in props:
                try:
                    entry[key] = float(props[key])
                except ValueError:
                    entry[key] = props[key]
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_test_results(results: list[dict]) -> dict:
    """Group test results by Op."""
    ops = defaultdict(lambda: {
        "module": None, "passed": 0, "failed": 0, "skipped": 0,
        "max_abs_err": 0.0, "failing_tests": [],
    })
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
        for key in ("tileops_latency_ms", "tileops_tflops", "tileops_bandwidth_tbs",
                     "tileops_variant",
                     "baseline_tag", "baseline_latency_ms", "baseline_tflops",
                     "baseline_ratio"):
            if key in r:
                config_entry[key] = r[key]
        d["configs"].append(config_entry)
    return dict(ops)


def collect_bench_failures(results: list[dict]) -> list[dict]:
    """Collect failed benchmark results for reporting."""
    return [
        {
            "nodeid": r["nodeid"],
            "name": r["name"],
            "op": r.get("op"),
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
    with open(path) as f:
        data = json.load(f)
    return data.get("runs", [])


def prune_history(runs: list[dict], retention_days: int = HISTORY_RETENTION_DAYS) -> list[dict]:
    """Remove runs older than retention_days."""
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    return [r for r in runs if r.get("date", "") >= cutoff]


def find_best_latency(runs: list[dict], op: str, config_name: str) -> float | None:
    """Find the best (lowest) tileops latency for an op+config across history."""
    best = None
    for run in runs:
        op_data = run.get("ops", {}).get(op, {})
        cfg_data = op_data.get(config_name, {})
        tileops_data = cfg_data.get("tileops", {})
        lat = tileops_data.get("latency_ms")
        if lat is not None and (best is None or lat < best):
            best = lat
    return best


def detect_regressions(bench_ops: dict, history_runs: list[dict]) -> list[dict]:
    """Detect performance regressions vs 14-day best."""
    regressions = []
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            lat = cfg.get("tileops_latency_ms")
            if lat is None:
                continue
            best = find_best_latency(history_runs, op, cfg["name"])
            if best is None:
                continue
            delta = (lat - best) / best
            if (delta > REGRESSION_THRESHOLD
                    and delta > NOISE_FLOOR
                    and (lat - best) > REGRESSION_ABS_MIN):
                regressions.append({
                    "op": op,
                    "config": cfg["name"],
                    "best_ms": best,
                    "curr_ms": lat,
                    "delta_pct": delta * 100,
                    "tflops": cfg.get("tileops_tflops"),
                })
    return regressions


def detect_improvements(bench_ops: dict, history_runs: list[dict]) -> list[dict]:
    """Detect performance improvements vs 14-day best."""
    improvements = []
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            lat = cfg.get("tileops_latency_ms")
            if lat is None:
                continue
            best = find_best_latency(history_runs, op, cfg["name"])
            if best is None:
                continue
            delta = (lat - best) / best
            if delta < -REGRESSION_THRESHOLD and abs(delta) > NOISE_FLOOR:
                improvements.append({
                    "op": op,
                    "config": cfg["name"],
                    "best_ms": best,
                    "curr_ms": lat,
                    "delta_pct": delta * 100,
                    "tflops": cfg.get("tileops_tflops"),
                })
    return improvements


def detect_baseline_alerts(bench_ops: dict) -> list[dict]:
    """Find ops where tileops is significantly slower than baseline."""
    alerts = []
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            ratio = cfg.get("baseline_ratio")
            if ratio is not None and ratio < BASELINE_RATIO_ALERT:
                alerts.append({
                    "op": op,
                    "config": cfg["name"],
                    "tileops_ms": cfg.get("tileops_latency_ms"),
                    "baseline_ms": cfg.get("baseline_latency_ms"),
                    "ratio": ratio,
                    "baseline_tag": cfg.get("baseline_tag", "baseline"),
                })
    return alerts


# ---------------------------------------------------------------------------
# History update
# ---------------------------------------------------------------------------


def build_history_entry(bench_ops: dict) -> dict:
    """Build a history entry from current bench results."""
    commit = _get_git_commit()
    gpu = _get_gpu_name()
    ops_data = {}
    for op, data in bench_ops.items():
        cfg_data = {}
        for cfg in data["configs"]:
            entry = {}
            lat = cfg.get("tileops_latency_ms")
            if lat is not None:
                entry["tileops"] = {"latency_ms": lat}
                tflops = cfg.get("tileops_tflops")
                if tflops is not None:
                    entry["tileops"]["tflops"] = tflops
            bl_lat = cfg.get("baseline_latency_ms")
            if bl_lat is not None:
                tag = cfg.get("baseline_tag", "baseline")
                if isinstance(tag, str):
                    entry[tag] = {"latency_ms": bl_lat}
                    bl_tflops = cfg.get("baseline_tflops")
                    if bl_tflops is not None:
                        entry[tag]["tflops"] = bl_tflops
            if entry:
                cfg_data[cfg["name"]] = entry
        if cfg_data:
            ops_data[op] = cfg_data
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "commit": commit,
        "gpu": gpu,
        "ops": ops_data,
    }


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
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


def generate_report(
    test_ops: dict | None,
    bench_ops: dict | None,
    bench_failures: list[dict],
    regressions: list[dict],
    improvements: list[dict],
    baseline_alerts: list[dict],
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
    total_tests = sum(d["passed"] + d["failed"] + d["skipped"]
                      for d in (test_ops or {}).values())
    total_passed = sum(d["passed"] for d in (test_ops or {}).values())

    health = _PASS if (n_failures == 0 and not regressions
                       and not bench_failures) else _FAIL
    lines.append(f"# {health} TileOPs Nightly Report")
    lines.append("")
    lines.append(f"> **{now}** &ensp;|&ensp; `{commit}` &ensp;|&ensp; {gpu}")
    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────
    corr_icon = _PASS if n_failures == 0 else f"{_FAIL} {n_failures} failed"
    bench_fail_icon = (f"{_FAIL} {len(bench_failures)}"
                       if bench_failures else f"{_PASS} None")
    reg_icon = f"{_PASS} None" if not regressions else f"{_WARN} {len(regressions)}"
    alert_icon = (f"{_WARN} {len(baseline_alerts)}"
                  if baseline_alerts else f"{_PASS} None")

    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| **Correctness** | {corr_icon}"
                 f" &ensp; ({total_passed}/{total_tests} tests across"
                 f" {n_test_ops} ops) |")
    lines.append(f"| **Benchmarked Ops** | {n_bench_ops} |")
    lines.append(f"| **Benchmark Failures** | {bench_fail_icon} |")
    lines.append(f"| **Regressions** (vs 14-day best) | {reg_icon} |")
    lines.append(f"| **Baseline Alerts** (< {BASELINE_RATIO_ALERT:.0%}) |"
                 f" {alert_icon} |")
    if improvements:
        lines.append(f"| **Improvements** (vs 14-day best) |"
                     f" {_PARTY} {len(improvements)} |")
    lines.append("")

    # ── Test Failures (only if any) ───────────────────────────────────────
    if test_ops:
        failed_ops = {op: d for op, d in test_ops.items() if d["failed"] > 0}
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
                lines.append(f"| **{op}** | `{d['module'] or 'N/A'}` "
                             f"| {d['failed']}/{total} | {tests_str} |")
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
            lines.append(f"| **{r['op']}** | {r['config']} "
                         f"| {r['best_ms']:.4f} | {r['curr_ms']:.4f} "
                         f"| +{r['delta_pct']:.1f}% | {tflops_str} |")
        lines.append("")

    # ── Improvements ──────────────────────────────────────────────────────
    if improvements:
        lines.append(f"## {_PARTY} Performance Improvements (vs 14-day best)")
        lines.append("")
        lines.append("| Op | Config | Prev Best (ms) | Current (ms) | Delta | TFLOPS |")
        lines.append("|:---|:-------|---------------:|-----------:|------:|-------:|")
        for r in sorted(improvements, key=lambda x: x["delta_pct"]):
            tflops_str = f"{r['tflops']:.2f}" if r.get("tflops") else "-"
            lines.append(f"| **{r['op']}** | {r['config']} "
                         f"| {r['best_ms']:.4f} | {r['curr_ms']:.4f} "
                         f"| {r['delta_pct']:.1f}% | {tflops_str} |")
        lines.append("")

    # ── Baseline Alerts ───────────────────────────────────────────────────
    if baseline_alerts:
        lines.append(f"## {_RED} Baseline Performance Alerts")
        lines.append("")
        lines.append("> TileOPs is slower than baseline"
                     f" (ratio < {BASELINE_RATIO_ALERT:.0%})."
                     " Ratio = baseline_latency / tileops_latency.")
        lines.append("")
        lines.append("| | Op | Config | TileOPs (ms) | Baseline (ms)"
                     " | Ratio | Via |")
        lines.append("|:-|:---|:-------|------------:|-------------:"
                     "|------:|:----|")
        for a in sorted(baseline_alerts, key=lambda x: x.get("ratio", 1)):
            emoji = _ratio_emoji(a["ratio"])
            lines.append(f"| {emoji} | **{a['op']}** | {a['config']} "
                         f"| {a['tileops_ms']:.4f} | {a['baseline_ms']:.4f} "
                         f"| {a['ratio']:.1%} | {a['baseline_tag']} |")
        lines.append("")

    # ── Full Correctness Results (collapsible) ────────────────────────────
    if test_ops:
        lines.append("<details>")
        lines.append(f"<summary><strong>Full Correctness Results"
                     f" ({n_test_ops} ops)</strong></summary>")
        lines.append("")
        lines.append("| | Op | Module | Pass | Fail | Skip | Max Error |")
        lines.append("|:-|:---|:-------|-----:|-----:|-----:|----------:|")
        for op in sorted(test_ops):
            d = test_ops[op]
            err_str = f"{d['max_abs_err']:.2e}" if d["max_abs_err"] else "-"
            icon = _PASS if d["failed"] == 0 else _FAIL
            lines.append(f"| {icon} | {op} | `{d['module'] or 'N/A'}` "
                         f"| {d['passed']} | {d['failed']} | {d['skipped']} "
                         f"| {err_str} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # ── Full Benchmark Results (collapsible) ──────────────────────────────
    if bench_ops:
        n_configs = sum(len(d["configs"]) for d in bench_ops.values())
        lines.append("<details>")
        lines.append(f"<summary><strong>Full Benchmark Results"
                     f" ({n_configs} configs across"
                     f" {n_bench_ops} ops)</strong></summary>")
        lines.append("")
        lines.append("| | Op | Config | Latency (ms) | TFLOPS | BW (TB/s)"
                     " | Via | Ratio |")
        lines.append("|:-|:---|:-------|------------:|-------:|----------:"
                     "|:----|------:|")
        for op in sorted(bench_ops):
            for cfg in bench_ops[op]["configs"]:
                lat = cfg.get("tileops_latency_ms")
                tflops = cfg.get("tileops_tflops")
                bw = cfg.get("tileops_bandwidth_tbs")
                bl_tag = cfg.get("baseline_tag", "")
                variant = cfg.get("tileops_variant")
                ratio = cfg.get("baseline_ratio")
                lat_str = f"{lat:.4f}" if lat else "-"
                tflops_str = f"{tflops:.2f}" if tflops else "-"
                bw_str = f"{bw:.2f}" if bw else "-"
                if bl_tag:
                    bl_str = str(bl_tag)
                elif variant:
                    bl_str = f"strategy: {variant}"
                else:
                    bl_str = "-"
                ratio_str = f"{ratio:.1%}" if ratio else "-"
                emoji = _ratio_emoji(ratio) if ratio else ""
                lines.append(f"| {emoji} | {op} | {cfg['name']} "
                             f"| {lat_str} | {tflops_str} | {bw_str} "
                             f"| {bl_str} | {ratio_str} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


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

    # Load history and detect regressions
    history_runs = load_history(args.history)
    regressions = detect_regressions(bench_ops, history_runs) if bench_ops else []
    improvements = detect_improvements(bench_ops, history_runs) if bench_ops else []
    baseline_alerts = detect_baseline_alerts(bench_ops) if bench_ops else []

    # Generate report
    report = generate_report(test_ops, bench_ops, bench_failures,
                             regressions, improvements, baseline_alerts)
    Path(args.output).write_text(report)
    print(f"Report written to {args.output}")

    # Update history
    if args.history_out and bench_ops:
        entry = build_history_entry(bench_ops)
        history_runs.append(entry)
        history_runs = prune_history(history_runs)
        Path(args.history_out).write_text(json.dumps({"runs": history_runs}, indent=2))
        print(f"History updated: {args.history_out}")


if __name__ == "__main__":
    main()
