#!/usr/bin/env python3
"""Generate an Op-level nightly report from pytest JUnit XML outputs.

Usage:
    python scripts/nightly_report.py \
        --test-xml test_results.xml \
        --bench-xml bench_results.xml \
        [--history perf_history.json] \
        --output nightly_report.md
"""

import argparse
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.outcomes import UNAVAILABLE_PROPERTY

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
    "tileops_roof_key",
    "tileops_roofline_data_terms",
    "tileops_uncounted_copy_ms",
    "tileops_timing",
    "tileops_l2_flush",
    "tileops_device_busy_p10_ms",
    "tileops_device_busy_p90_ms",
    "tileops_n_samples",
    "baseline_tag",
    "baseline_device_busy_ms",
    "baseline_latency_ms",
    "baseline_tflops",
    "baseline_ratio",
)
# Row properties stating what established the row, and whether anything did, for every tag.
_EVIDENCE_SUFFIXES = ("_no_ratio", "_unverified")
# A row below this ratio is behind its fastest baseline and raises an alert; from it up to
# 1.0 the two are level, as the tracker issues count them.
BASELINE_RATIO_ALERT = 0.99
BASELINE_ALERT_WORST_N = 10  # alerts shown open; the rest collapse
HISTORY_RETENTION_DAYS = 14

# Algorithmic speed-of-light (SOL) efficiency is sol_time / measured_time against the
# calibrated (effective) ceilings; see docs/design/roofline.md §1.2 and §4.3.
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
_NEW = "\U0001f195"  # 🆕
_GREEN = "\U0001f7e2"  # 🟢
_UP = "\u2191"  # ↑
_DOWN = "\u2193"  # ↓

# ---------------------------------------------------------------------------
# JUnit XML parsing
# ---------------------------------------------------------------------------


# A dtype name as a case id spells it: the id ends in its dtype cases and dtype parameters.
_DTYPE_TOKEN = re.compile(r"bfloat16|bool|u?int\d+|float\d+(?:_[a-z0-9]+)?|complex\d+")
# A kernel file below this share of executed lines was never constructed by any test.
_KERNEL_BUILT_PCT = 25
# Unattributed failures listed before the table says how many it left out. The
# attributed table above caps its per-op test names at three for the same reason.
_UNATTRIBUTED_SHOWN = 10

# Below this, one statement swings the percentage too far to read.
_COVERAGE_MIN_STMTS = 20
_COVERAGE_WORST_N = 15  # rows in the least-covered file list


def testcase_properties(testcase: ET.Element) -> dict[str, str]:
    """The user properties of a JUnit testcase element."""
    return {p.attrib["name"]: p.attrib.get("value", "") for p in testcase.iter("property")}


def testcase_fields(tc: ET.Element, props: dict[str, str]) -> dict:
    """The fields every parsed testcase carries: its id, outcome, op and failure message."""
    failure = tc.find("failure")
    error = tc.find("error")
    if failure is not None or error is not None:
        # A teardown error after a skip writes both elements; the error wins.
        outcome = "failed"
    elif tc.find("skipped") is not None and UNAVAILABLE_PROPERTY in props:
        # A row no implementation serves yet (benchmarks/conftest.py): neither a skip to
        # warn of nor a failure.
        outcome = "unavailable"
    elif tc.find("skipped") is not None:
        outcome = "skipped"
    else:
        outcome = "passed"
    reported = failure if failure is not None else error
    return {
        "nodeid": f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}",
        "name": tc.attrib.get("name", ""),
        "outcome": outcome,
        "op": props.get("op"),
        "op_module": props.get("op_module"),
        "failure_message": reported.attrib.get("message", "") if reported is not None else None,
    }


def parse_test_xml(path: str) -> list[dict]:
    """Parse correctness test results, returning per-testcase dicts."""
    results = []
    for tc in ET.parse(path).iter("testcase"):
        props = testcase_properties(tc)
        entry = testcase_fields(tc, props)
        # Present only where a comparison completed, which is what
        # distinguishes establishing a value from merely running.
        entry["compared"] = int(props.get("checked_outputs", "0")) > 0
        results.append(entry)
    return results


def _data_terms(text: str) -> dict:
    """A row's ``roofline_data_terms`` JSON property; an empty mapping where it is not one,
    so one unreadable row does not fail the whole history build."""
    try:
        terms = json.loads(text)
    except ValueError:
        return {}
    return terms if isinstance(terms, dict) else {}


def parse_bench_xml(path: str) -> list[dict]:
    """Parse benchmark results, returning per-testcase dicts."""
    results = []
    for tc in ET.parse(path).iter("testcase"):
        props = testcase_properties(tc)
        entry = testcase_fields(tc, props)
        entry |= {k: v for k, v in props.items() if k.endswith(_EVIDENCE_SUFFIXES)}
        entry |= {key: _try_float(props[key]) for key in _PERF_KEYS if key in props}
        terms = props.get("tileops_roofline_data_terms")
        if terms is not None:
            entry["tileops_roofline_data_terms"] = _data_terms(terms)

        # Every other ``<tag>_<metric>`` the benchmark conftest writes is one baseline's.
        baselines = {}
        for pkey, pval in props.items():
            if pkey.endswith("_no_ratio"):
                continue
            for metric in ("device_busy_ms", "latency_ms", "tflops", "ratio"):
                tag = pkey.removesuffix(f"_{metric}")
                if tag != pkey and tag not in ("tileops", "baseline"):
                    value = _try_float(pval)
                    if metric != "ratio" or isinstance(value, float):
                        baselines.setdefault(tag, {})[metric] = value
                    break
        if baselines:
            entry["baselines"] = baselines

        results.append(entry)
    return results


def _try_float(v: str) -> float | str:
    try:
        return float(v)
    except ValueError:
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


def _tally(results: list[dict], key_of) -> dict[str, dict]:
    """Group testcases by ``key_of``, counting each group's outcomes and naming its failing
    tests. A testcase ``key_of`` maps to nothing is left out."""
    groups: dict[str, dict] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "skipped": 0, "failing_tests": []}
    )
    for r in results:
        key = key_of(r)
        if key:
            d = groups[key]
            d[r["outcome"]] += 1
            if r["outcome"] == "failed":
                d["failing_tests"].append(r["name"])
    return dict(groups)


def aggregate_test_results(results: list[dict]) -> dict:
    """Group test results by Op."""
    ops = _tally(results, lambda r: r.get("op"))
    for d in ops.values():
        d.update(module=None, compared=0)
    for r in results:
        d = ops.get(r.get("op"))
        if d is not None:
            d["module"] = d["module"] or r.get("op_module")
            d["compared"] += r["outcome"] == "passed" and bool(r.get("compared"))
    return ops


def _model_of(result: dict) -> str:
    """The model a testcase tests: ``<model>`` of the ``tests.models.test_<model>`` module in
    its classname, or in its name where a collection error leaves the classname empty."""
    source = result["nodeid"].split("::")[0] or result["name"]
    parts = source.split(".")
    module = parts[parts.index("models") + 1] if "models" in parts[:-1] else source
    return module.removeprefix("test_")


def aggregate_model_results(results: list[dict]) -> dict[str, dict]:
    """Group model test results by model, the ``tests/models/test_<model>.py`` file each
    testcase comes from."""
    return _tally(results, _model_of)


def _table_cell(text: str) -> str:
    """``text`` as one markdown table cell: whitespace runs, newlines among them, become one
    space, pipes are escaped, and at most 120 characters remain."""
    return " ".join(text.split()).replace("|", "\\|")[:120]


def _throughput_section(throughput: dict[str, dict], models: dict[str, dict] | None) -> list[str]:
    """The ``## Model Throughput`` section: per model, the tokens/s ``scripts/model_throughput.py``
    wrote for each prefill length and decode context, or the error its run ended with. A model
    whose tests did not all pass, given ``models``, has its numbers withheld."""
    lines = ["## Model Throughput", "", "Batch 1 on one GPU with random weights, in tokens/s.", ""]
    if not throughput:
        return [*lines, f"{_FAIL} The timing run wrote no results.", ""]
    columns = [
        (key, n)
        for key in ("prefill_tokens_per_s", "decode_tokens_per_s")
        for n in sorted({int(n) for d in throughput.values() for n in d.get(key, {})})
    ]
    head = [
        f"Prefill {n}" if key.startswith("prefill") else f"Decode ctx {n}" for key, n in columns
    ]
    lines.append("| " + " | ".join(["Model", "Run", *head]) + " |")
    lines.append("|:------|:----|" + "------:|" * len(columns))
    for model, d in sorted(throughput.items()):
        if "error" in d:
            cells = [f"{_FAIL} **{model}**", _table_cell(d["error"])] + ["—"] * len(columns)
        elif models is not None and models.get(model, {"failed": 1})["failed"]:
            cells = [f"{_FAIL} **{model}**", "withheld: its model tests did not pass"]
            cells += ["—"] * len(columns)
        else:
            cells = [f"**{model}**", d["run"]]
            for key, n in columns:
                value = d[key].get(str(n))
                fmt = ",.0f" if key.startswith("prefill") else ".2f"
                cells.append("—" if value is None else format(value, fmt))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


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
        config_entry |= {
            key: value
            for key, value in r.items()
            if key in _PERF_KEYS or key == "baselines" or key.endswith(_EVIDENCE_SUFFIXES)
        }
        d["configs"].append(config_entry)
    return dict(ops)


def case_id(name: str) -> str:
    """The parametrized part of a testcase name, which is the workload's own id and keys the
    row's history: renaming a test function keeps it. It is unique per op, which
    ``scripts/check_bench_coverage.py`` holds."""
    if name.endswith("]") and "[" in name:
        return name[name.index("[") + 1 : -1]
    return name


def _dtypes_of(name: str) -> tuple[str, ...]:
    """The dtype names in a row's case id, in order."""
    return tuple(t for t in case_id(name).split("-") if _DTYPE_TOKEN.fullmatch(t))


def count_bench_outcome(results: list[dict], outcome: str) -> int:
    """Cases with *outcome*, skipped or unavailable, which are neither configs nor failures:
    without a count they vanish."""
    return sum(1 for r in results if r["outcome"] == outcome)


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
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    return data.get("runs", [])


def history_window(runs: list[dict], retention_days: int = HISTORY_RETENTION_DAYS) -> list[dict]:
    """The runs the verdicts compare against, ordered by date: the last run of each date
    within the period, so a day run twice does not weigh twice."""
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    kept = {d: r for r in runs if (d := r.get("date") or "") >= cutoff}
    return sorted(kept.values(), key=lambda r: r["date"])


class _WorkCounts(NamedTuple):
    """The FLOP and byte counts of the work a reading timed."""

    flops: float | None
    nbytes: float | None


def _work_counts(reading: dict) -> _WorkCounts:
    return _WorkCounts(reading.get("flops"), reading.get("bytes"))


def _same_workload(current: _WorkCounts, historical: _WorkCounts) -> bool:
    """Whether two readings timed the same workload, taking an unknown count as a match."""
    return all(c is None or h is None or c == h for c, h in zip(current, historical, strict=True))


class _Reading(NamedTuple):
    ms: float
    spread: float | None  # p90 - p10 of the run that produced ``ms``


def _spread(props: dict) -> float | None:
    lo, hi = props.get("device_busy_p10_ms"), props.get("device_busy_p90_ms")
    return hi - lo if lo is not None and hi is not None else None


def _alias_name(
    runs: list[dict], op: str, config_name: str, work: _WorkCounts, current: frozenset[str]
) -> str | None:
    """The unique prior name of this row in history, or None.

    A prior name has the row's exact FLOP and byte counts and dtypes, and shares no run with
    the current name, in history or in *current*, the case ids of this run: a renamed row and
    its new name never co-occur, while another variant of the workload does.
    """
    if work.flops is None or work.nbytes is None:
        return None
    dtypes = _dtypes_of(config_name)
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
            if _work_counts(entry.get("tileops", {})) == work and _dtypes_of(name) == dtypes:
                candidates.add(name)
    candidates -= excluded | current
    return candidates.pop() if len(candidates) == 1 else None


def _config_readings(
    runs: list[dict],
    op: str,
    config_name: str,
    work: _WorkCounts,
    current: frozenset[str],
    l2_flush: str | None,
) -> list[_Reading]:
    """Workload-matched positive readings for one row, oldest first.

    Per run the current display name wins; a run that recorded the row only
    under its prior name (see ``_alias_name``) contributes that reading. Only matching
    cache preparation methods provide comparable baselines.
    """
    alias = _alias_name(runs, op, config_name, work, current)
    readings = []
    for run in runs:
        cfgs = run.get("ops", {}).get(op, {})
        for name in (config_name, alias):
            if name is None or name not in cfgs:
                continue
            tileops_data = cfgs[name].get("tileops", {})
            ms = tileops_data.get("device_busy_ms")
            if ms is None or ms <= 0 or not _same_workload(work, _work_counts(tileops_data)):
                continue
            if tileops_data.get("l2_flush") != l2_flush:
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
        current = frozenset(case_id(cfg["name"]) for cfg in data["configs"])
        for cfg in data["configs"]:
            lat = cfg.get("tileops_device_busy_ms")
            if lat is None or lat <= 0:
                continue
            props = {k.removeprefix("tileops_"): v for k, v in cfg.items()}
            readings = _config_readings(
                history_runs,
                op,
                case_id(cfg["name"]),
                _work_counts(props),
                current,
                props.get("l2_flush"),
            )
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


def detect_new_ops(bench_ops: dict, history_runs: list[dict]) -> list[dict]:
    """Rows of the ops no run in the window measured.

    Going from unmeasured to measured is progress with no 14-day best to beat, so these
    rows sit beside the improvements instead of in them. A row needs a positive reading,
    and so does a history entry for its op to count as measured. A renamed op reads as
    new for the night of its rename. Empty without history, where every op would read
    as new.
    """
    if not history_runs:
        return []
    measured = set()
    for run in history_runs:
        for op, cfgs in run.get("ops", {}).items():
            for entry in cfgs.values():
                ms = entry.get("tileops", {}).get("device_busy_ms")
                if ms is not None and ms > 0:
                    measured.add(op)
    out = []
    for op, data in bench_ops.items():
        if op in measured:
            continue
        for cfg in data["configs"]:
            lat = cfg.get("tileops_device_busy_ms")
            if lat is None or lat <= 0:
                continue
            out.append(
                {
                    "op": op,
                    "config": cfg["name"],
                    "curr_ms": lat,
                    "tflops": cfg.get("tileops_tflops"),
                }
            )
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


def _unverified_rows(bench_ops: dict) -> list[tuple[str, str, str]]:
    """Rows no reference checked, as (op, config, note).

    A row carries ``no_ratio`` where nothing established it: the workload has no reference,
    the baseline computes a different function, or no reference runs at this shape. A
    `Partial` row is excluded; its reference checked part of the result and its note names
    what that leaves open.
    """
    found = []
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            for key in cfg:
                if not key.endswith("_no_ratio") or not cfg[key]:
                    continue
                tag = key[: -len("_no_ratio")]
                # The alias conftest writes beside the first baseline's own tag.
                if tag == "baseline" and f"{cfg.get('baseline_tag', '')}_no_ratio" in cfg:
                    continue
                found.append(
                    (op, cfg.get("config", ""), cfg.get(f"{tag}_unverified", "unestablished"))
                )
    return found


def _rated_configs(bench_ops: dict):
    """Yield ``(op, cfg, ratio, tag, baseline_ms)`` per row that has a timed baseline.

    The ratio is the row's lowest, so every consumer judges a row against the
    strongest baseline that was timed for it rather than against whichever one
    the manifest happens to name first.
    """
    for op, data in bench_ops.items():
        for cfg in data["configs"]:
            primary = cfg.get("baseline_tag", "baseline")
            candidates = [(cfg.get("baseline_ratio"), primary, cfg.get("baseline_device_busy_ms"))]
            candidates += [
                (bl.get("ratio"), tag, bl.get("device_busy_ms"))
                for tag, bl in cfg.get("baselines", {}).items()
                if tag != primary
            ]
            timed = [c for c in candidates if c[0] is not None]
            if timed:
                ratio, tag, baseline_ms = min(timed, key=lambda c: c[0])
                yield op, cfg, ratio, tag, baseline_ms


def baseline_standing(bench_ops: dict) -> tuple[int, int]:
    """``(rows ahead of or level with baseline, rows behind it)``, over rows that have one.

    Rows behind are the rows that alert. The two cover every comparable row, so
    their sum is the population the alert count is drawn from.
    """
    ratios = [ratio for _, _, ratio, _, _ in _rated_configs(bench_ops)]
    ahead_or_level = sum(1 for r in ratios if r >= BASELINE_RATIO_ALERT)
    return ahead_or_level, len(ratios) - ahead_or_level


# Share of rated rows ahead of or level with their fastest baseline, at and above
# which the README badge reads green, and the brighter green when no row is behind.
BADGE_GREEN_SHARE = 0.95

# A row whose fastest comparator takes less than this counts against the badge
# only if it was also behind the night before: rows that small move about twice
# as much from night to night as larger ones.
BADGE_NOISE_MS = 0.010


def fastest_comparator(props: dict) -> tuple[str, float] | None:
    """``(tag, ms)`` of the fastest comparator among a row's properties, or ``None``.

    Every ``<tag>_device_busy_ms`` property except ``tileops`` and ``baseline`` is a
    comparator, including tags marked ``no_ratio``.
    """
    timed = [
        (name[: -len("_device_busy_ms")], float(value))
        for name, value in props.items()
        if name.endswith("_device_busy_ms")
        and name not in ("tileops_device_busy_ms", "baseline_device_busy_ms")
    ]
    return min(timed, key=lambda item: item[1]) if timed else None


def case_reading(case: ET.Element) -> tuple[dict, tuple[float, float] | None]:
    """A benchmark testcase's properties and its ``(tileops_ms, fastest_ms)`` reading.

    The reading is ``None`` for a failed row, or one without a nonzero TileOPs time or a
    comparator; such a row is judged nowhere.
    """
    props = testcase_properties(case)
    failed = any(child.tag in ("failure", "error") for child in case)
    fastest = fastest_comparator(props)
    tileops = float(props.get("tileops_device_busy_ms") or 0)
    if failed or fastest is None or tileops == 0:
        return props, None
    return props, (tileops, fastest[1])


def badge_rows(bench_xml: str) -> dict:
    """``{(classname, name): reading}`` for the badge, by ``case_reading``."""
    rows = {}
    for case in ET.parse(bench_xml).getroot().iter("testcase"):
        props, reading = case_reading(case)
        if props.get("op"):
            rows[(case.get("classname"), case.get("name"))] = reading
    return rows


def row_behind(reading: tuple[float, float], earlier: tuple[float, float] | None) -> bool:
    """Whether a ``(tileops_ms, fastest_ms)`` row is behind its fastest comparator.

    A row is behind when its ratio is below ``BASELINE_RATIO_ALERT``. A row whose
    fastest comparator takes under ``BADGE_NOISE_MS`` is behind only if *earlier*, the
    same row in the previous night, was also below that ratio.
    """

    def ratio(pair):
        return pair[1] / pair[0]

    if ratio(reading) >= BASELINE_RATIO_ALERT:
        return False
    if reading[1] >= BADGE_NOISE_MS:
        return True
    return earlier is not None and ratio(earlier) < BASELINE_RATIO_ALERT


def badge_standing(rows: dict, previous: dict | None) -> tuple[int, int]:
    """``(rows ahead of or level with their fastest comparator, rows behind)`` for the badge.

    Each row is judged by ``row_behind`` against the same row in *previous*.
    """
    ahead = behind = 0
    for key, reading in rows.items():
        if reading is None:
            continue
        if row_behind(reading, (previous or {}).get(key)):
            behind += 1
        else:
            ahead += 1
    return ahead, behind


def baseline_badge(ahead_or_level: int, behind: int) -> dict:
    """The shields.io endpoint payload for the README's vs-baseline badge.

    The share is rounded down to one decimal, so a night with one row behind never
    reads 100%.
    """
    total = ahead_or_level + behind
    share = ahead_or_level / total
    percent = math.floor(share * 1000) / 10
    color = "brightgreen" if behind == 0 else "green" if share >= BADGE_GREEN_SHARE else "yellow"
    return {
        "schemaVersion": 1,
        "label": "vs fastest baseline",
        "message": f"{percent:g}% ahead",
        "color": color,
    }


def detect_baseline_alerts(bench_ops: dict) -> list[dict]:
    """Find configs where tileops is slower than its strongest baseline: one alert each."""
    return [
        {
            "op": op,
            "config": cfg["name"],
            "tileops_ms": cfg.get("tileops_device_busy_ms"),
            "baseline_ms": baseline_ms,
            "ratio": ratio,
            "baseline_tag": tag,
        }
        for op, cfg, ratio, tag, baseline_ms in _rated_configs(bench_ops)
        if ratio < BASELINE_RATIO_ALERT
    ]


# ---------------------------------------------------------------------------
# History window
# ---------------------------------------------------------------------------


def build_history_entry(bench_ops: dict, run: dict, coverage: list[dict] | None = None) -> dict:
    """Build the history entry for one nightly run.

    ``run`` states where the readings came from: ``date``, ``commit``, ``gpu``
    and, where the snapshot recorded one, ``run_id``. Coverage sits in a key of
    its own beside ``ops``, which regression detection does not read.
    """
    tileops_keys = (
        "latency_ms",
        "device_busy_ms",
        "tflops",
        "flops",
        "bytes",
        "roof_key",
        "roofline_data_terms",
        "device_busy_p10_ms",
        "device_busy_p90_ms",
        "l2_flush",
    )
    baseline_keys = ("latency_ms", "device_busy_ms", "tflops")
    ops_data = {}
    for op, data in (bench_ops or {}).items():
        cfg_data = {}
        for cfg in data["configs"]:
            entry = {}
            tileops = {
                k: cfg[f"tileops_{k}"] for k in tileops_keys if cfg.get(f"tileops_{k}") is not None
            }
            if "latency_ms" in tileops or "device_busy_ms" in tileops:
                sol = cfg.get("sol")
                if sol is not None:
                    tileops["sol"] = {
                        "efficiency": round(sol["efficiency"], 4),
                        "bound": sol["bound"],
                        "latency_bound": sol["latency_bound"],
                    }
                entry["tileops"] = tileops
            # The unprefixed ``baseline_*`` properties are the first baseline's, under its tag.
            baselines = dict(cfg.get("baselines", {}))
            tag = cfg.get("baseline_tag", "baseline")
            if isinstance(tag, str):
                baselines[tag] = {k: cfg.get(f"baseline_{k}") for k in baseline_keys}
            for btag, bl in baselines.items():
                reading = {k: bl[k] for k in baseline_keys if bl.get(k) is not None}
                if reading:
                    entry[btag] = reading
            if entry:
                cfg_data[case_id(cfg["name"])] = entry
        if cfg_data:
            ops_data[op] = cfg_data
    entry = {
        "date": run["date"],
        "commit": run["commit"],
        "gpu": run["gpu"],
        "ops": ops_data,
    }
    if run.get("run_id"):
        entry["run_id"] = run["run_id"]
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


def _window_caption(runs: list[dict], read: bool) -> str:
    """The span the verdicts were drawn from, as the report captions it.

    An unread window leaves every verdict unmade; an empty one says no prior
    run falls in the period.
    """
    if not read:
        return "not read"
    if not runs:
        return "empty"
    return f"{len(runs)} runs, {runs[0]['date']} to {runs[-1]['date']}"


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


def _ops_verified(
    test_ops: dict | None, bench_ops: dict | None, implemented: set[str] | None
) -> tuple[int, int] | None:
    """``(ops with a verified result, implemented ops)``, or None without *implemented*.

    An op is verified when a unit test compared one of its outputs or a benchmark row
    published a ratio against a baseline.
    """
    if implemented is None:
        return None
    # A constructor-rejection test passes without comparing any of the op's values.
    passed = {op for op, d in (test_ops or {}).items() if d.get("compared", 0) > 0}
    # A noncomparable baseline still records its timing; only a ratio is evidence.
    compared = {
        op
        for op, data in (bench_ops or {}).items()
        if any(
            cfg.get("baseline_ratio") is not None
            or any(bl.get("ratio") is not None for bl in cfg.get("baselines", {}).values())
            for cfg in data.get("configs", ())
        )
    }
    return len((passed | compared) & implemented), len(implemented)


def _kernel_count() -> int | None:
    """Concrete ``Kernel`` classes in the ``__all__`` of ``tileops._kernels`` and its
    subpackages, or None when they will not import."""
    try:
        import importlib
        import inspect
        import pkgutil

        import tileops._kernels as kernels_pkg
        from tileops._kernels.kernel_base import Kernel

        def exported(module) -> set:
            names = getattr(module, "__all__", ())
            return {
                attribute
                for attribute in (getattr(module, name, None) for name in names)
                if inspect.isclass(attribute)
                and issubclass(attribute, Kernel)
                and not inspect.isabstract(attribute)
            }

        # Packages only: a private module's own ``__all__`` is not an export.
        found = exported(kernels_pkg)
        for module in pkgutil.walk_packages(kernels_pkg.__path__, prefix="tileops._kernels."):
            if not module.ispkg:
                continue
            try:
                found |= exported(importlib.import_module(module.name))
            except Exception:
                continue
    except Exception:
        return None
    return len(found)


def _load_manifest() -> dict | None:
    """The installed op manifest, or None when it will not load.

    Returns None rather than raising: a night whose benchmark data is intact still
    has a report to render, and the rows the manifest feeds are dropped instead.
    """
    try:
        from tileops.manifest import load_manifest

        return load_manifest()
    except Exception:
        return None


def _library_scale(manifest: dict | None) -> dict[str, int] | None:
    """Implemented ``operators``, ``specs``, ``workloads`` of implemented ops and, where they
    import, ``kernels``, counted as the manifest stats job counts them; None without
    *manifest*."""
    if manifest is None:
        return None
    try:
        from manifest_stats import collect_stats

        stats = collect_stats(manifest)
    except Exception:
        return None
    scale = {
        "operators": stats["by_status"].get("implemented", 0),
        "specs": stats["by_status"].get("spec-only", 0),
        "workloads": stats["workloads_implemented_total"],
    }
    kernels = _kernel_count()
    if kernels is not None:
        scale["kernels"] = kernels
    return scale


def git_commit() -> str:
    """The checkout's short commit hash, or ``unknown``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    return (result.returncode == 0 and result.stdout.strip()) or "unknown"


def _get_gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "N/A"


# ---------------------------------------------------------------------------
# Speed-of-Light
# ---------------------------------------------------------------------------


def load_gpu_profile(gpu_name: str) -> dict | None:
    """Profile matching the measured device, or None when none claims it."""
    try:
        from tileops._perf.profile import find_profile
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
    from tileops._perf.profile import resolve_roof

    flops = cfg.get("tileops_flops")
    nbytes = cfg.get("tileops_bytes")
    busy = cfg.get("tileops_device_busy_ms")
    roof_key = cfg.get("tileops_roof_key")
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


def _shift_table(
    title: str,
    base_label: str,
    rows: list[dict],
    sort_key,
    delta_fmt: str,
    note: str | None = None,
) -> list[str]:
    """One movement table: the op, the config, both readings, the delta."""
    lines = [title, ""]
    if note:
        lines += [note, ""]
    lines.append(f"| Op | Config | {base_label} (ms) | Current (ms) | Delta | TFLOPS |")
    lines.append("|:---|:-------|------------:|-----------:|------:|-------:|")
    for r in sorted(rows, key=sort_key):
        tflops_str = f"{r['tflops']:.2f}" if r.get("tflops") else "-"
        lines.append(
            f"| **{r['op']}** | {r['config']} "
            f"| {r['base_ms']:.4f} | {r['curr_ms']:.4f} "
            f"| {delta_fmt.format(r['delta_pct'])} | {tflops_str} |"
        )
    lines.append("")
    return lines


def _alert_rows(alerts: list[dict]) -> list[str]:
    """One baseline-alert table for the given rows, worst first."""
    lines = ["| | Op | Config | TileOPs (ms) | Baseline (ms) | Ratio | Via |"]
    lines.append("|:-|:---|:-------|------------:|-------------:|------:|:----|")
    for a in alerts:
        emoji = _ratio_emoji(a["ratio"])
        lines.append(
            f"| {emoji} | **{a['op']}** | {a['config']} "
            f"| {a['tileops_ms']:.4f} | {a['baseline_ms']:.4f} "
            f"| {a['ratio']:.1%} | {a['baseline_tag']} |"
        )
    lines.append("")
    return lines


def _ratio_emoji(ratio: float) -> str:
    """Return an emoji indicator for a baseline ratio value."""
    if ratio >= 1.5:
        return _GREEN
    if ratio >= 1.0:
        return _BLUE
    if ratio >= BASELINE_RATIO_ALERT:
        return _YELLOW
    return _RED


def _failing_tests_cell(names: list[str]) -> str:
    """The first three failing test names, then a count of the rest."""
    shown = 3
    cell = ", ".join(names[:shown])
    if len(names) > shown:
        cell += f", ... (+{len(names) - shown})"
    return cell


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
    bench_unavailable: int = 0,
    previous_run_shifts: list[dict] | None = None,
    sol_anomalies: list[dict] | None = None,
    have_gpu_profile: bool = False,
    history_window: str = "not read",
    test_results: list[dict] | None = None,
    scale: dict[str, int] | None = None,
    implemented: set[str] | None = None,
    new_ops: list[dict] | None = None,
    models: dict[str, dict] | None = None,
    throughput: dict[str, dict] | None = None,
) -> str:
    """Generate markdown report."""
    lines = []
    commit = git_commit()
    gpu = _get_gpu_name()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sol_fails = [a for a in (sol_anomalies or []) if a["level"] == "FAIL"]
    sig = _coverage_signals(coverage) if coverage else None

    # ── Header ────────────────────────────────────────────────────────────
    # Every testcase, not only those attributed to an op: the rest fail the job just the same.
    outcomes = [r["outcome"] for r in (test_results or [])]
    n_bench_ops = len(bench_ops) if bench_ops else 0
    n_failures = sum(1 for d in (test_ops or {}).values() if d["failed"] > 0)
    if outcomes:
        total_tests = len(outcomes)
        total_passed = sum(1 for o in outcomes if o == "passed")
        n_failures = sum(1 for o in outcomes if o == "failed")
    else:
        total_tests = sum(
            d["passed"] + d["failed"] + d["skipped"] for d in (test_ops or {}).values()
        )
        total_passed = sum(d["passed"] for d in (test_ops or {}).values())

    # An empty result is a model run that wrote no testcase: it crashed or collected nothing.
    model_failures = sum(d["failed"] for d in (models or {}).values()) or int(models == {})
    health = (
        _PASS
        if (
            n_failures == 0
            and model_failures == 0
            and not regressions
            and not bench_failures
            and not sol_fails
        )
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
    if bench_unavailable:
        bench_fail_icon += f" &ensp;|&ensp; {bench_unavailable} unavailable"
    reg_icon = f"{_PASS} None" if not regressions else f"{_WARN} {len(regressions)}"
    alert_icon = f"{_WARN} {len(baseline_alerts)}" if baseline_alerts else f"{_PASS} None"

    lines.append("| | |")
    lines.append("|---|---|")
    # Ahead of the run's own figures: these two say how large the library is,
    # and the rows under them say how it did tonight.
    if scale:
        lines.append(f"| **Operators** | {scale['operators']} |")
        if "kernels" in scale:
            lines.append(f"| **Kernels** | {scale['kernels']} |")
        lines.append(f"| **Specs** | {scale['specs']} |")
        lines.append(f"| **Workloads** | {scale['workloads']} |")
    # Ops first: the question the row answers is how much of the library was
    # established tonight, and the test count is the evidence, not the answer.
    verified = _ops_verified(test_ops, bench_ops, implemented)
    scope = f"{verified[0]}/{verified[1]} ops verified, " if verified else ""
    lines.append(
        f"| **Correctness** | {corr_icon} &ensp; ({scope}{total_passed}/{total_tests} tests) |"
    )
    if models is not None:
        model_tests = sum(d["passed"] + d["failed"] + d["skipped"] for d in models.values())
        model_passed = sum(d["passed"] for d in models.values())
        model_icon = (
            f"{_FAIL} no results"
            if not models
            else _PASS
            if model_failures == 0
            else f"{_FAIL} {model_failures} failed"
        )
        lines.append(
            f"| **Models** | {model_icon} &ensp; ({len(models)} models, "
            f"{model_passed}/{model_tests} tests) |"
        )
    lines.append(f"| **Benchmarked Ops** | {n_bench_ops} |")
    lines.append(f"| **Benchmark Failures** | {bench_fail_icon} |")
    lines.append(f"| **Regressions** (vs 14-day median) | {reg_icon} |")
    lines.append(f"| **Baseline Alerts** (< {BASELINE_RATIO_ALERT:.0%}) | {alert_icon} |")
    # The alert count is the rows behind; the split below it adds the rows ahead of
    # or level with their baseline. The total is stated because it is not the workload
    # count above it: a workload is benchmarked once per dtype it declares, so these
    # rows run about 1.6x the manifest entries they come from.
    ahead_or_level, behind = baseline_standing(bench_ops or {})
    if ahead_or_level or behind:
        lines.append(
            f"| **vs Baseline** | {_UP} {ahead_or_level} &ensp;·&ensp; {_DOWN} {behind}"
            f" &ensp;/&ensp; {ahead_or_level + behind} rows |"
        )
    unverified = _unverified_rows(bench_ops or {})
    if unverified:
        lines.append(f"| **Rows no reference checked** | {_WARN} {len(unverified)} |")
    lines.append(f"| **History window** | {history_window} |")
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
    if new_ops:
        n_ops = len({r["op"] for r in new_ops})
        lines.append(
            f"| **New ops** (no 14-day history) | {_NEW} {n_ops} ops · {len(new_ops)} rows |"
        )
    if previous_run_shifts:
        lines.append(f"| **Moved since previous run** | {_BLUE} {len(previous_run_shifts)} |")
    if coverage:
        # One row per concern: a single untested-line total would double-count
        # ops/ against its own branch figure and bury perf/, where a wrong
        # number fails silently.
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

    # ── Models ────────────────────────────────────────────────────────────
    if models:
        lines.append("## Models")
        lines.append("")
        lines.append("| Model | Passed | Failed | Skipped | Failing Tests |")
        lines.append("|:------|:------:|:------:|:-------:|:--------------|")
        for model, d in sorted(models.items()):
            icon = _PASS if d["failed"] == 0 else _FAIL
            lines.append(
                f"| {icon} **{model}** | {d['passed']} | {d['failed']} | {d['skipped']} "
                f"| {_failing_tests_cell(d['failing_tests'])} |"
            )
        lines.append("")

    if throughput is not None:
        lines += _throughput_section(throughput, models)

    # ── Test Failures (only if any) ───────────────────────────────────────
    failed_ops = {op: d for op, d in (test_ops or {}).items() if d["failed"] > 0}
    if failed_ops:
        lines.append(f"## {_FAIL} Test Failures")
        lines.append("")
        lines.append("| Op | Module | Failed / Total | Failing Tests |")
        lines.append("|:---|:-------|:--------------:|:--------------|")
        for op, d in sorted(failed_ops.items()):
            total = d["passed"] + d["failed"] + d["skipped"]
            lines.append(
                f"| **{op}** | `{d['module'] or 'N/A'}` | {d['failed']}/{total} "
                f"| {_failing_tests_cell(d['failing_tests'])} |"
            )
        lines.append("")

    # Failures no reference comparison attributed to an op. They are counted in
    # Correctness above, so leaving them out here reports a number with nothing
    # behind it: the reader sees N failed and a table that does not hold N rows.
    unattributed = [r for r in (test_results or []) if r["outcome"] == "failed" and not r.get("op")]
    if unattributed:
        lines.append(f"## {_FAIL} Test Failures (no op attributed)")
        lines.append("")
        lines.append("| Test | Message |")
        lines.append("|:-----|:--------|")
        for r in sorted(unattributed, key=lambda r: r["nodeid"])[:_UNATTRIBUTED_SHOWN]:
            message = _table_cell(r.get("failure_message") or "")
            lines.append(f"| `{r['nodeid']}` | {message} |")
        if len(unattributed) > _UNATTRIBUTED_SHOWN:
            lines.append(f"| ... | (+{len(unattributed) - _UNATTRIBUTED_SHOWN} more) |")
        lines.append("")

    # ── Benchmark Failures (only if any) ─────────────────────────────────
    if bench_failures:
        lines.append(f"## {_FAIL} Benchmark Failures")
        lines.append("")
        lines.append("| Test | Error |")
        lines.append("|:-----|:------|")
        for f in bench_failures:
            lines.append(f"| {f['name']} | {_table_cell(f.get('failure_message') or '')} |")
        lines.append("")

    # ── Movement tables ───────────────────────────────────────────────────
    if regressions:
        lines += _shift_table(
            f"## {_WARN} Performance Regressions (vs 14-day median)",
            "Median",
            regressions,
            lambda x: -x["delta_pct"],
            "+{:.1f}%",
        )
    if improvements:
        lines += _shift_table(
            f"## {_PARTY} Performance Improvements (vs 14-day best)",
            "Prev Best",
            improvements,
            lambda x: x["delta_pct"],
            "{:.1f}%",
        )
    if new_ops:
        lines.append(f"## {_NEW} New Ops (no 14-day history)")
        lines.append("")
        lines.append("| Op | Config | Current (ms) | TFLOPS |")
        lines.append("|:---|:-------|-----------:|-------:|")
        for r in new_ops:
            tflops_str = f"{r['tflops']:.2f}" if r.get("tflops") else "-"
            lines.append(f"| **{r['op']}** | {r['config']} | {r['curr_ms']:.4f} | {tflops_str} |")
        lines.append("")
    if previous_run_shifts:
        lines += _shift_table(
            f"## {_BLUE} Moved Since Previous Run",
            "Previous",
            previous_run_shifts,
            lambda x: x["delta_pct"],
            "{:+.1f}%",
            note=(
                "> Moves against the most recent reading. A row restored to its"
                " old level appears only here: returning is not a new 14-day record."
            ),
        )

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
        ranked = sorted(baseline_alerts, key=lambda x: x.get("ratio", 1))
        worst, rest = ranked[:BASELINE_ALERT_WORST_N], ranked[BASELINE_ALERT_WORST_N:]
        lines.extend(_alert_rows(worst))
        if rest:
            lines.append("<details>")
            lines.append(f"<summary><strong>{len(rest)} more alerts</strong></summary>")
            lines.append("")
            lines.extend(_alert_rows(rest))
            lines.append("</details>")
            lines.append("")

    # ── Coverage ──────────────────────────────────────────────────────────
    if sig:
        lines.extend(_coverage_section(sig))

    return "\n".join(lines)


def _pct(hit: int, total: int) -> str:
    return f"{100 * hit / total:.1f}%" if total else "-"


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
    parser.add_argument("--model-test-xml", help="Path to the model tests' JUnit XML")
    parser.add_argument(
        "--model-throughput-json",
        help="Path to the model tokens/s JSON scripts/model_throughput.py writes",
    )
    parser.add_argument("--bench-xml", help="Path to benchmark JUnit XML")
    parser.add_argument("--history", help="Path to perf_history.json (input)")
    parser.add_argument("--output", required=True, help="Output markdown report path")
    parser.add_argument("--coverage-xml", help="Path to coverage.py XML report")
    parser.add_argument(
        "--badge-output",
        help="Path to write the vs-baseline badge JSON (shields.io endpoint) to",
    )
    parser.add_argument(
        "--previous-bench-xml",
        help="The previous nightly's benchmark JUnit XML, for the badge's noise rule",
    )
    parser.add_argument(
        "--regressions-output",
        help="Path to write the regressed rows to as JSON; not written without history",
    )
    args = parser.parse_args()

    if args.history and not Path(args.history).exists():
        parser.error(f"--history file does not exist: {args.history}")

    test_ops = None
    test_results = None
    if args.test_xml and Path(args.test_xml).exists():
        test_results = parse_test_xml(args.test_xml)
        test_ops = aggregate_test_results(test_results)

    # Given but absent, the model run died before pytest wrote it: no results, not a pass.
    models = None
    if args.model_test_xml:
        path = Path(args.model_test_xml)
        models = aggregate_model_results(parse_test_xml(str(path))) if path.exists() else {}

    throughput = None
    if args.model_throughput_json:
        path = Path(args.model_throughput_json)
        throughput = json.loads(path.read_text()) if path.exists() else {}

    bench_ops = None
    bench_failures = []
    bench_skips = bench_unavailable = 0
    if args.bench_xml and Path(args.bench_xml).exists():
        bench_results = parse_bench_xml(args.bench_xml)
        bench_ops = aggregate_bench_results(bench_results)
        bench_failures = collect_bench_failures(bench_results)
        bench_skips = count_bench_outcome(bench_results, "skipped")
        bench_unavailable = count_bench_outcome(bench_results, "unavailable")

    # The rebuild can reach past the period when a run gap exceeds it, and the
    # verdicts below are labelled with the 14-day window.
    gpu_profile = load_gpu_profile(_get_gpu_name())
    sol_anomalies = annotate_sol(bench_ops, gpu_profile) if bench_ops else []

    history_runs = history_window(load_history(args.history))
    regressions = detect_regressions(bench_ops, history_runs) if bench_ops else []
    improvements = detect_improvements(bench_ops, history_runs) if bench_ops else []
    previous_run_shifts = detect_previous_run_shifts(bench_ops, history_runs) if bench_ops else []
    new_ops = detect_new_ops(bench_ops, history_runs) if bench_ops else []
    baseline_alerts = detect_baseline_alerts(bench_ops) if bench_ops else []

    coverage = None
    if args.coverage_xml and Path(args.coverage_xml).exists():
        coverage = parse_coverage_xml(args.coverage_xml)
    # Read before this run is appended, so the comparison is against a prior run.
    coverage_prev = _previous_coverage(history_runs)
    manifest = _load_manifest()
    implemented = (
        None
        if manifest is None
        else {name for name, op in manifest.items() if op.get("status") == "implemented"}
    )

    report = generate_report(
        test_ops=test_ops,
        bench_ops=bench_ops,
        bench_failures=bench_failures,
        regressions=regressions,
        improvements=improvements,
        baseline_alerts=baseline_alerts,
        coverage=coverage,
        coverage_prev=coverage_prev,
        bench_skips=bench_skips,
        bench_unavailable=bench_unavailable,
        previous_run_shifts=previous_run_shifts,
        sol_anomalies=sol_anomalies,
        have_gpu_profile=gpu_profile is not None,
        history_window=_window_caption(history_runs, bool(args.history)),
        test_results=test_results,
        scale=_library_scale(manifest),
        implemented=implemented,
        new_ops=new_ops,
        models=models,
        throughput=throughput,
    )
    Path(args.output).write_text(report)
    print(f"Report written to {args.output}")

    if args.regressions_output and bench_ops and history_runs:
        Path(args.regressions_output).write_text(json.dumps(regressions, indent=2) + "\n")
        print(f"{len(regressions)} regressed rows written to {args.regressions_output}")

    if args.badge_output and args.bench_xml and Path(args.bench_xml).exists():
        previous = None
        if args.previous_bench_xml and Path(args.previous_bench_xml).exists():
            previous = badge_rows(args.previous_bench_xml)
        ahead_or_level, behind = badge_standing(badge_rows(args.bench_xml), previous)
        if ahead_or_level + behind:
            badge = baseline_badge(ahead_or_level, behind)
            Path(args.badge_output).write_text(json.dumps(badge, indent=2) + "\n")
            print(f"Badge written to {args.badge_output}: {badge['message']}")


if __name__ == "__main__":
    main()
