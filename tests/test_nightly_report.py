"""Tests for scripts/nightly_report.py.

Covers the verdict rules a wrong comparison can mislead: workload identity,
baseline choice, the noise gate, rename recovery, and the previous-run lens.
"""

import importlib.util
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_SCRIPT = REPO_ROOT / "scripts" / "nightly_report.py"

_OP = "FooFwdOp"
_CONFIG = "test_foo_bench[row-bfloat16]"
# History keys a row by its case id, the bracketed part of the testcase name.
_KEY = "row-bfloat16"
_RUN = {"date": "2026-09-16", "commit": "abc1234", "gpu": "NVIDIA H200", "run_id": "42"}


@pytest.fixture(scope="module")
def report():
    """Import nightly_report as a module."""
    spec = importlib.util.spec_from_file_location("nightly_report", REPORT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _history_run(busy_ms, name=_KEY, p10=None, p90=None, **counts):
    tileops = {"device_busy_ms": busy_ms, **counts}
    if p10 is not None:
        tileops["device_busy_p10_ms"] = p10
        tileops["device_busy_p90_ms"] = p90
    return {"ops": {_OP: {name: {"tileops": tileops}}}}


def _row_xml(**properties: str) -> str:
    """A benchmark report holding the FooFwdOp row read at 0.1 ms, with *properties* added."""
    props = {"op": _OP, "tileops_device_busy_ms": "0.1", **properties}
    rendered = "".join(
        f"<property name={quoteattr(k)} value={quoteattr(v)}/>" for k, v in props.items()
    )
    return (
        '<testsuites><testsuite name="bench"><testcase classname="benchmarks.test_foo"'
        f' name="{_CONFIG}"><properties>{rendered}</properties></testcase></testsuite></testsuites>'
    )


def _history_reading(report, tmp_path, **properties: str) -> dict:
    """The history entry's tileops reading of the row ``_row_xml`` writes."""
    xml = tmp_path / "bench.xml"
    xml.write_text(_row_xml(**properties))
    ops = report.aggregate_bench_results(report.parse_bench_xml(str(xml)))
    return report.build_history_entry(ops, _RUN)["ops"][_OP][_KEY]["tileops"]


def _bench_ops(busy_ms, p10=None, p90=None, **counts):
    config = {"name": _CONFIG, "tileops_device_busy_ms": busy_ms}
    if p10 is not None:
        config["tileops_device_busy_p10_ms"] = p10
        config["tileops_device_busy_p90_ms"] = p90
    config.update({f"tileops_{k}": v for k, v in counts.items()})
    return {_OP: {"configs": [config]}}


def test_history_reading_at_another_work_size_is_not_a_baseline(report):
    """A run whose FLOP count differs is a different workload, not a faster one."""
    runs = [_history_run(0.1, flops=5e9, bytes=1e6)]
    assert report.detect_regressions(_bench_ops(0.4, flops=2e10, bytes=4e6), runs) == []


def test_history_reading_at_the_same_work_size_still_compares(report):
    """The same counts keep their history: this is where a regression shows."""
    runs = [_history_run(0.1, flops=5e9, bytes=1e6)]
    (found,) = report.detect_regressions(_bench_ops(0.4, flops=5e9, bytes=1e6), runs)
    assert found["base_ms"] == 0.1


def test_history_reading_of_another_dtype_at_equal_flops_is_not_a_baseline(report):
    """Half the bytes at the same FLOP count is a different row under one name."""
    runs = [_history_run(0.1, flops=5e9, bytes=1e6)]
    assert report.detect_regressions(_bench_ops(0.4, flops=5e9, bytes=5e5), runs) == []


@pytest.mark.parametrize(
    "history_flush, compares", [(None, False), ("zero-2x-l2", False), ("read-2x-l2", True)]
)
def test_history_reading_after_another_l2_flush_is_not_a_baseline(report, history_flush, compares):
    """Another cache preparation is another starting state, not a slower op."""
    flush = {} if history_flush is None else {"l2_flush": history_flush}
    runs = [_history_run(0.1, **flush)]
    found = report.detect_regressions(_bench_ops(0.4, l2_flush="read-2x-l2"), runs)
    assert len(found) == compares


def test_the_l2_flush_of_a_reading_reaches_the_history_entry(report, tmp_path):
    """The next night's comparison needs to know how this one was prepared."""
    tileops = _history_reading(report, tmp_path, tileops_l2_flush="read-2x-l2")
    assert tileops["l2_flush"] == "read-2x-l2"


def test_history_reading_without_counts_falls_back_to_tflops(report):
    """A reading carrying only ``tflops`` still recovers a comparable FLOP count."""
    runs = [_history_run(0.1, tflops=50.0)]
    (found,) = report.detect_regressions(_bench_ops(0.4, flops=5e9), runs)
    assert found["base_ms"] == 0.1


def test_history_reading_without_counts_at_another_work_size_is_not_a_baseline(report):
    """A FLOP count recovered from ``tflops`` separates workloads too."""
    runs = [_history_run(0.1, tflops=50.0)]
    assert report.detect_regressions(_bench_ops(0.4, flops=2e10), runs) == []


def test_sub_floor_change_with_percentiles_is_reported(report):
    """Once a spread is recorded, the noise gate replaces the fixed floor."""
    runs = [_history_run(0.010, p10=0.0099, p90=0.0101)]
    (found,) = report.detect_improvements(_bench_ops(0.005, p10=0.0049, p90=0.0051), runs)
    assert found["base_ms"] == 0.010


def test_change_within_noise_spread_is_not_reported(report):
    """A 20% move inside 2.5x the row's p90-p10 spread is noise."""
    runs = [_history_run(0.0010, p10=0.0009, p90=0.0011)]
    assert report.detect_regressions(_bench_ops(0.0012), runs) == []


def test_row_without_percentiles_keeps_the_absolute_floor(report):
    """Legacy rows with no spread on either side gate on the fixed floor."""
    runs = [_history_run(0.010)]
    assert report.detect_regressions(_bench_ops(0.015), runs) == []


def test_regression_baseline_is_the_median_not_the_minimum(report):
    """One lucky fast night must not alarm every later night."""
    runs = [_history_run(0.10), _history_run(0.20), _history_run(0.21)]
    assert report.detect_regressions(_bench_ops(0.21), runs) == []


def test_improvement_baseline_stays_the_minimum(report):
    """An improvement is a new record, not a move against the typical night."""
    runs = [_history_run(0.10), _history_run(0.20), _history_run(0.21)]
    assert report.detect_improvements(_bench_ops(0.15), runs) == []
    (found,) = report.detect_improvements(_bench_ops(0.05), runs)
    assert found["base_ms"] == 0.10


def test_restored_row_shows_as_moved_since_previous_run(report):
    """A fix back to the old level is no new record; the previous-run lens reports it."""
    runs = [_history_run(0.066), _history_run(0.082)]
    assert report.detect_improvements(_bench_ops(0.066), runs) == []
    (found,) = report.detect_previous_run_shifts(_bench_ops(0.066), runs)
    assert found["base_ms"] == 0.082


def test_non_positive_reading_is_not_a_measurement(report):
    """A zero on either side is rejected, not reported as a 100% move."""
    assert report.detect_improvements(_bench_ops(0.0), [_history_run(0.10)]) == []
    assert report.detect_regressions(_bench_ops(0.4), [_history_run(0.0)]) == []


def test_renamed_row_keeps_its_history(report):
    """History under the row's prior display name still baselines it."""
    runs = [_history_run(0.1, name="test_foo_bench[old-bfloat16]", flops=5e9, bytes=1e6)]
    (found,) = report.detect_regressions(_bench_ops(0.4, flops=5e9, bytes=1e6), runs)
    assert found["base_ms"] == 0.1


def test_prior_name_of_another_dtype_is_not_a_rename(report):
    """fp16 and bf16 rows of one shape have equal counts; only the case id tells them apart."""
    runs = [_history_run(0.1, name="old-float16", flops=5e9, bytes=1e6)]
    assert report.detect_regressions(_bench_ops(0.4, flops=5e9, bytes=1e6), runs) == []


def test_name_that_ever_shared_a_run_with_the_current_name_is_not_a_rename(report):
    """Co-occurrence disqualifies a prior name even when its counts differ there."""
    shared_run = {
        "ops": {
            _OP: {
                _KEY: {"tileops": {"device_busy_ms": 0.39, "flops": 5e9, "bytes": 1e6}},
                "test_foo_bench[other]": {
                    "tileops": {"device_busy_ms": 0.20, "flops": 9e9, "bytes": 9e6}
                },
            }
        }
    }
    runs = [_history_run(0.10, name="test_foo_bench[other]", flops=5e9, bytes=1e6), shared_run]
    assert report.detect_regressions(_bench_ops(0.40, flops=5e9, bytes=1e6), runs) == []


def test_variant_added_beside_its_sibling_is_not_a_rename(report):
    """A row new in this run does not inherit a sibling that is still running.

    The sibling is absent from no historical run, so only the current run's own row
    set tells the two apart, and a new row with no history of its own has no verdict.
    """
    sibling = "test_foo_bench[sibling-bfloat16]"
    runs = [_history_run(0.1, name="sibling-bfloat16", flops=5e9, bytes=1e6)]
    bench_ops = _bench_ops(0.4, flops=5e9, bytes=1e6)
    bench_ops[_OP]["configs"].append(
        {"name": sibling, "tileops_device_busy_ms": 0.1, "tileops_flops": 5e9, "tileops_bytes": 1e6}
    )
    assert [r["config"] for r in report.detect_regressions(bench_ops, runs)] == []


def _named_ops(*ops):
    """bench_ops of ``(op, [case ids])`` pairs, every row read at 0.1 ms."""
    return {
        op: {
            "configs": [
                {"name": f"test_x_bench[{c}]", "tileops_device_busy_ms": 0.1} for c in cases
            ]
        }
        for op, cases in ops
    }


def _named_run(*ops):
    return {
        "ops": {op: {c: {"tileops": {"device_busy_ms": 0.1}} for c in cases} for op, cases in ops}
    }


def test_an_op_the_window_never_measured_lists_every_row(report):
    """Unmeasured to measured is progress, with no 14-day best to beat."""
    bench = _named_ops(("OldOp", ["a"]), ("NewOp", ["x", "y"]))
    found = report.detect_new_ops(bench, [_named_run(("OldOp", ["a"]))])
    assert [(r["op"], r["config"]) for r in found] == [
        ("NewOp", "test_x_bench[x]"),
        ("NewOp", "test_x_bench[y]"),
    ]


def test_a_new_row_of_a_measured_op_is_not_a_new_op(report):
    bench = _named_ops(("OldOp", ["a", "b"]))
    assert report.detect_new_ops(bench, [_named_run(("OldOp", ["a"]))]) == []


def test_a_renamed_op_reads_as_new(report):
    """Renames are not detected; the op reads as new for the night of its rename."""
    bench = _named_ops(("GLAFwdOp", ["a"]))
    runs = [_named_run(("GLAInferenceFwdOp", ["a"]))]
    assert [r["op"] for r in report.detect_new_ops(bench, runs)] == ["GLAFwdOp"]


def test_an_op_history_holds_only_without_a_positive_reading_is_new(report):
    """A zero or a baseline-only entry is not a measurement of the op."""
    zero = {"ops": {"OldOp": {"x": {"tileops": {"device_busy_ms": 0.0}}}}}
    baseline_only = {"ops": {"OldOp": {"x": {"tileops": {}, "baselines": {"torch": 0.1}}}}}
    bench = _named_ops(("OldOp", ["x"]))
    assert [r["op"] for r in report.detect_new_ops(bench, [zero, baseline_only])] == ["OldOp"]


def test_without_history_no_op_is_new(report):
    """With nothing to compare against, every op would read as new."""
    assert report.detect_new_ops(_named_ops(("NewOp", ["x"])), []) == []


def test_a_new_op_row_without_a_positive_reading_is_not_listed(report):
    bench = _named_ops(("NewOp", ["x"]))
    bench["NewOp"]["configs"][0]["tileops_device_busy_ms"] = 0.0
    assert report.detect_new_ops(bench, [_named_run(("OldOp", ["a"]))]) == []


def test_history_is_keyed_by_case_id(report):
    """A renamed test function keeps its history key."""
    row = {
        "name": "test_foo_bench[row-bfloat16]",
        "op": _OP,
        "outcome": "passed",
        "tileops_device_busy_ms": 0.1,
    }
    ops = report.aggregate_bench_results([row])
    assert set(report.build_history_entry(ops, _RUN)["ops"][_OP]) == {"row-bfloat16"}


def test_every_unchecked_tag_is_counted(report):
    """A case timing three implementations without a ratio publishes three rows.

    The first baseline is written twice, under its own tag and under the unprefixed alias
    the perf keys read; counting both would double it.
    """
    row = {
        "name": "test_foo_bench[row-bfloat16]",
        "op": _OP,
        "outcome": "passed",
        "tileops_device_busy_ms": 0.1,
        "tileops_no_ratio": "True",
        "tileops_unverified": "unestablished: the workload carries no reference",
        "baseline_tag": "torch",
        "baseline_no_ratio": "True",
        "torch_no_ratio": "True",
        "torch-compile_no_ratio": "True",
    }
    ops = report.aggregate_bench_results([row])
    assert len(report._unverified_rows(ops)) == 3


def test_history_entry_records_the_percentiles(report):
    """The noise gate needs each run's spread persisted with its reading."""
    entry = report.build_history_entry(_bench_ops(0.010, p10=0.0099, p90=0.0101), _RUN)
    tileops = entry["ops"][_OP][_KEY]["tileops"]
    assert tileops["device_busy_p10_ms"] == 0.0099
    assert tileops["device_busy_p90_ms"] == 0.0101


@pytest.mark.parametrize(
    "terms", ['{"active_experts": 12}', "{'active_experts': 12}"], ids=["json", "python-repr"]
)
def test_roofline_data_terms_reach_the_history_entry(report, tmp_path, terms):
    """What decided a reading's bytes is kept with the reading, as a mapping, whether the
    snapshot recorded it as JSON or as a Python repr."""
    tileops = _history_reading(report, tmp_path, tileops_roofline_data_terms=terms)
    assert tileops["roofline_data_terms"] == {"active_experts": 12}


def test_unreadable_data_terms_drop_the_terms_not_the_row(report):
    assert report._data_terms("{'active_experts':") == {}
    assert report._data_terms("12") == {}


# ---------------------------------------------------------------------------
# Speed-of-Light
# ---------------------------------------------------------------------------


@pytest.fixture
def profile():
    """Each test owns a synthetic profile whose expected efficiencies are exact."""
    return {
        "gpu": "TestGPU",
        "hbm": {"theoretical": 5e12, "effective": 4e12},
        "cuda_core": {"fp32": {"theoretical": 6e13, "effective": 5e13}},
        "tensor_core": {"bf16": {"theoretical": 1e15, "effective": 7e14}},
    }


def _sol_row(**overrides):
    row = {
        "name": _CONFIG,
        "tileops_flops": 2e9,
        "tileops_bytes": 4e9,  # 1 ms at effective HBM
        "tileops_device_busy_ms": 1.0,
        "tileops_roof_key": "cuda_core.fp32",
        "tileops_timing": "cupti",
    }
    row.update(overrides)
    return row


def test_sol_efficiency_is_one_at_the_effective_ceiling(report, profile):
    sol = report._compute_sol(_sol_row(), profile)
    assert sol["bound"] == "memory"
    assert sol["efficiency"] == pytest.approx(1.0)
    assert not sol["impossible"] and not sol["latency_bound"]


def test_sol_compute_bound_uses_the_declared_roof(report, profile):
    # 7e11 FLOPs on the bf16 tensor-core roof: sol_time = 1 ms; measured 2 ms.
    sol = report._compute_sol(
        _sol_row(
            tileops_flops=7e11,
            tileops_bytes=1e6,
            tileops_device_busy_ms=2.0,
            tileops_roof_key="tensor_core.bf16",
        ),
        profile,
    )
    assert sol["bound"] == "compute"
    assert sol["efficiency"] == pytest.approx(0.5)


def test_sol_rate_above_theoretical_is_impossible_not_fast(report, profile):
    sol = report._compute_sol(_sol_row(tileops_device_busy_ms=0.5), profile)
    assert sol["impossible"] == ["bytes/s over HBM theoretical"]


def test_sol_uncounted_copies_enter_the_denominator(report, profile):
    sol = report._compute_sol(_sol_row(tileops_uncounted_copy_ms=1.0), profile)
    assert sol["efficiency"] == pytest.approx(0.5)


def test_sol_skips_rows_the_model_cannot_judge(report, profile):
    assert report._compute_sol(_sol_row(tileops_timing="cuda-events"), profile) is None
    no_timing = _sol_row()
    del no_timing["tileops_timing"]
    assert report._compute_sol(no_timing, profile) is None
    assert report._compute_sol(_sol_row(tileops_bytes=0), profile) is None
    assert report._compute_sol(_sol_row(tileops_roof_key="tensor_core.fp8"), profile) is None


def test_sol_latency_bound_needs_both_floors(report, profile):
    tiny = _sol_row(tileops_flops=1e3, tileops_bytes=1e3, tileops_device_busy_ms=0.004)
    assert report._compute_sol(tiny, profile)["latency_bound"]
    # A tiny lower bound with a large measured time is a slow kernel, not noise.
    slow = _sol_row(tileops_flops=1e3, tileops_bytes=1e3, tileops_device_busy_ms=0.5)
    assert not report._compute_sol(slow, profile)["latency_bound"]


def test_annotate_sol_reports_anomalies_and_tags_rows(report, profile):
    bench_ops = {
        _OP: {
            "module": "m",
            "configs": [_sol_row(), _sol_row(name="impossible", tileops_device_busy_ms=0.5)],
        }
    }
    anomalies = report.annotate_sol(bench_ops, profile)
    assert [a["level"] for a in anomalies] == ["FAIL"]
    assert bench_ops[_OP]["configs"][0]["sol"]["efficiency"] == pytest.approx(1.0)


def test_history_entry_records_the_sol_reading(report, profile):
    bench_ops = {_OP: {"module": "m", "configs": [_sol_row()]}}
    report.annotate_sol(bench_ops, profile)
    entry = report.build_history_entry(bench_ops, _RUN)
    tileops = entry["ops"][_OP][_KEY]["tileops"]
    assert tileops["roof_key"] == "cuda_core.fp32"
    assert tileops["sol"] == {"efficiency": 1.0, "bound": "memory", "latency_bound": False}


# The window is rebuilt from the snapshot repository on every run, so what a
# snapshot cannot supply must not be taken from the machine doing the rebuild.

BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_perf_history.py"

_BENCH_XML = _row_xml()


_COVERAGE_XML = (
    "<coverage><packages><package><classes>"
    '<class filename="src/tileops/_ops/foo.py"><lines>'
    '<line number="1" hits="1"/><line number="2" hits="0"/>'
    "</lines></class></classes></package></packages></coverage>"
)


def _snapshot_repo(tmp_path, runs, coverage=None):
    """A clone-shaped repository with one commit per run, oldest first."""
    repo = tmp_path / "snapshots"
    repo.mkdir()
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q", "-b", "snapshots"], check=True)
    subprocess.run([*git, "config", "user.email", "t@t"], check=True)
    subprocess.run([*git, "config", "user.name", "t"], check=True)
    for meta, bench in runs:
        (repo / "meta.json").write_text(json.dumps(meta))
        (repo / "bench_results.xml").write_text(bench)
        if coverage is not None:
            (repo / "coverage.xml").write_text(coverage)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", meta["run_id"]], check=True)
    return repo


def _build_window(tmp_path, repo, *more):
    out = tmp_path / "perf_history.json"
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--repo", str(repo), "--out", str(out), *more],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(out.read_text())["runs"]


def _days_ago(n):
    """A date the retention cutoff still admits, whenever the suite runs."""
    return (date.today() - timedelta(days=n)).isoformat()


def _meta(day_offset, run_id):
    return {
        "date": _days_ago(day_offset),
        "commit": f"c{run_id}",
        "gpu": "NVIDIA H200",
        "run_id": run_id,
    }


def test_window_entries_carry_the_snapshot_provenance(tmp_path):
    """Reading the local checkout instead would label every run with today."""
    repo = _snapshot_repo(tmp_path, [(_meta(3, "1001"), _BENCH_XML)], coverage=_COVERAGE_XML)
    runs = _build_window(tmp_path, repo)
    assert [(r["date"], r["commit"], r["run_id"]) for r in runs] == [
        (_meta(3, "1001")["date"], "c1001", "1001")
    ]
    assert runs[0]["coverage"]["op_branches"] == 0


def test_a_rerun_leaves_its_own_snapshot_out_of_the_window(tmp_path):
    """A rerun after its snapshot was published must not judge tonight against itself."""
    runs = [(_meta(2, "1001"), _BENCH_XML), (_meta(0, "1002"), _BENCH_XML)]
    repo = _snapshot_repo(tmp_path, runs)
    window = _build_window(tmp_path, repo, "--exclude-run-id", "1002")
    assert [r["run_id"] for r in window] == ["1001"]


def test_the_window_is_the_period_and_one_run_per_date(tmp_path):
    """A count alone decides neither end, and a twice-run day must not weigh twice."""
    repo = _snapshot_repo(
        tmp_path,
        [
            (_meta(40, "900"), _BENCH_XML),
            (_meta(2, "901"), _BENCH_XML),
            (_meta(1, "902"), _BENCH_XML),
            (_meta(1, "903"), _BENCH_XML),
        ],
    )
    assert [r["run_id"] for r in _build_window(tmp_path, repo)] == ["901", "903"]


def test_a_snapshot_without_benchmark_readings_is_skipped(tmp_path):
    """A run that published no numbers must not end the window early."""
    repo = _snapshot_repo(
        tmp_path,
        [
            (_meta(2, "910"), _BENCH_XML),
            (_meta(1, "911"), "<testsuites/>"),
        ],
    )
    assert [r["run_id"] for r in _build_window(tmp_path, repo)] == ["910"]


def test_the_window_is_ordered_by_date_whatever_the_file_order(report):
    """Newest-first input would make the previous-run comparison the oldest one."""
    newest_first = [
        {"date": _days_ago(1), "run_id": "3"},
        {"date": _days_ago(2), "run_id": "2"},
    ]
    assert [r["run_id"] for r in report.history_window(newest_first)] == ["2", "3"]


def test_an_unread_window_does_not_read_as_an_empty_one(report):
    """A failed rebuild leaves every verdict unmade; an empty window does not."""
    assert report._window_caption([], read=False) == "not read"
    assert report._window_caption([], read=True) == "empty"
    assert report._window_caption([{"date": "2026-09-05"}, {"date": "2026-09-18"}], read=True) == (
        "2 runs, 2026-09-05 to 2026-09-18"
    )


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPORT_SCRIPT), *args], capture_output=True, text=True
    )


def test_a_named_history_that_does_not_exist_is_refused(tmp_path):
    """A window the builder failed to write must not read as no history."""
    result = _run_script(
        "--output", str(tmp_path / "report.md"), "--history", str(tmp_path / "absent.json")
    )
    assert result.returncode != 0
    assert "--history file does not exist" in result.stderr


def _report(report, **overrides):
    """The whole report over one passing FooFwdOp test, with *overrides* to its arguments."""
    args = {
        "test_ops": {_OP: {"module": "m", "failed": 0, "passed": 1, "skipped": 0, "tests": []}},
        "bench_ops": None,
        "bench_failures": [],
        "regressions": [],
        "improvements": [],
        "baseline_alerts": [],
    }
    return report.generate_report(**{**args, **overrides})


def _summary_report(report, scale):
    return _report(report, scale=scale, bench_ops=_bench_ops(1.0))


def test_the_summary_names_the_scale_rows_verbatim(report):
    """The Lark card looks these rows up by key, from another repository.

    Renaming either key drops a column from the card and breaks nothing here,
    so the names are pinned rather than left to the renderer.
    """
    md = _summary_report(report, {"operators": 186, "kernels": 272, "specs": 19, "workloads": 1135})
    assert "| **Operators** | 186 |" in md
    assert "| **Kernels** | 272 |" in md
    assert "| **Specs** | 19 |" in md
    assert "| **Workloads** | 1135 |" in md


def test_the_new_ops_section_and_summary_row_are_named_verbatim(report):
    """The Lark card parses this heading and table from another repository."""
    new_ops = [
        {"op": "NewOp", "config": "test_x_bench[x]", "curr_ms": 0.0123, "tflops": 1.5},
        {"op": "NewOp", "config": "test_x_bench[y]", "curr_ms": 0.5, "tflops": None},
    ]
    md = _report(report, test_ops=None, bench_ops=_bench_ops(1.0), new_ops=new_ops)
    assert "| **New ops** (no 14-day history) | \U0001f195 1 ops · 2 rows |" in md
    section = md[md.index("## \U0001f195 New Ops (no 14-day history)") :]
    assert "| Op | Config | Current (ms) | TFLOPS |" in section
    assert "| **NewOp** | test_x_bench[x] | 0.0123 | 1.50 |" in section
    assert "| **NewOp** | test_x_bench[y] | 0.5000 | - |" in section


def test_no_new_op_adds_neither_section_nor_row(report):
    md = _summary_report(report, None)
    assert "New Ops" not in md and "New ops" not in md


def test_a_manifest_that_will_not_load_drops_the_rows_not_the_report(report):
    """Benchmark data stands on its own; an unreadable manifest costs the four scale rows."""
    md = _summary_report(report, None)
    assert "**Operators**" not in md
    assert "**Kernels**" not in md
    assert "**Specs**" not in md
    assert "**Workloads**" not in md
    assert "| **Correctness** |" in md


def _result(outcome, op, name="test_x", compared=True):
    """One parse_test_xml row, with the fields that function always fills."""
    return {
        "outcome": outcome,
        "op": op,
        "op_module": None,
        "nodeid": f"tests/ops/test_f.py::{name}",
        "name": name,
        "compared": compared,
        "failure_message": "boom" if outcome == "failed" else None,
    }


def test_unattributed_passes_and_failures_are_counted_and_failures_named(report):
    """A test with no op property fails the job, so the report must not lose it.

    Counting it and leaving it out of the failure table reports a number with
    nothing behind it: the reader sees N failed and a table holding fewer rows.
    """
    row = _result("failed", None, "test_sum")
    md = _report(report, test_results=[_result("passed", _OP), _result("passed", None), row])

    assert "(2/3 tests)" in md
    assert report._FAIL in md
    assert row["nodeid"] in md


def test_ops_verified_unions_both_verifiers(report):
    """Unit tests and benchmark rows verify different ops; neither alone is coverage."""
    implemented = {"A", "B"}

    tested = {"A": {"passed": 1, "failed": 0, "compared": 1}}
    benched = {"B": {"configs": [{"baseline_ratio": 0.9}]}}

    assert report._ops_verified(tested, benched, implemented) == (2, 2)
    assert report._ops_verified(tested, None, implemented) == (1, 2)
    assert report._ops_verified(
        tested, {"C": {"configs": [{"baseline_ratio": 0.9}]}}, implemented
    ) == (1, 2)
    # A name alone is not evidence: every test for this op failed.
    assert report._ops_verified(
        {"A": {"passed": 0, "failed": 3, "compared": 0}}, None, implemented
    ) == (0, 2)
    # Nor is a passing test that compared nothing, such as a constructor-rejection test.
    rejection_only = {"A": {"passed": 4, "failed": 0, "compared": 0}}
    assert report._ops_verified(rejection_only, None, implemented) == (0, 2)
    # Nor is a benchmark row with no baseline to compare against.
    assert report._ops_verified(None, {"B": {"configs": [{}]}}, implemented) == (0, 2)
    # A timed but noncomparable baseline leaves a populated dict and no ratio:
    # the conftest writes timing before deciding whether a tag may publish one.
    noncomparable = {"B": {"configs": [{"baselines": {"torch": {"latency_ms": 1.0}}}]}}
    assert report._ops_verified(None, noncomparable, implemented) == (0, 2)
    rated = {"B": {"configs": [{"baselines": {"torch": {"ratio": 0.9}}}]}}
    assert report._ops_verified(None, rated, implemented) == (1, 2)


def test_a_case_that_compared_nothing_is_not_evidence(report):
    """A rejection or dispatch test owns the op and establishes none of its values."""
    results = [
        _result("passed", _OP, "compares"),
        _result("passed", _OP, "rejects", compared=False),
        _result("failed", _OP, "breaks"),
    ]
    aggregated = report.aggregate_test_results(results)

    assert aggregated[_OP]["passed"] == 2
    assert aggregated[_OP]["compared"] == 1


def test_an_exclusion_reason_is_not_a_ratio(report, tmp_path):
    """`<tag>_no_ratio` says why a tag published none; it is not a comparison."""
    xml = tmp_path / "bench.xml"
    xml.write_text(
        '<testsuite><testcase classname="b" name="t"><properties>'
        '<property name="op" value="AddFwdOp"/>'
        '<property name="tileops_no_ratio" value="no reference"/>'
        '<property name="tileops_unverified" value="no reference"/>'
        "</properties></testcase></testsuite>",
        encoding="utf-8",
    )
    rows = report.parse_bench_xml(str(xml))
    bench_ops = report.aggregate_bench_results(rows)

    assert "tileops_no" not in (rows[0].get("baselines") or {})
    assert report.baseline_standing(bench_ops) == (0, 0)


@pytest.mark.parametrize("count, expected", [("0", False), ("1", True)])
def test_junit_uses_explicit_coverage_even_for_zero_error(report, tmp_path, count, expected):
    xml = tmp_path / "results.xml"
    xml.write_text(f"""<testsuites><testsuite><testcase name="exact" classname="tests.ops.test_x">
      <properties><property name="max_abs_err" value="0"/>
      <property name="checked_outputs" value="{count}"/></properties>
    </testcase></testsuite></testsuites>""")
    assert report.parse_test_xml(str(xml))[0]["compared"] is expected


def test_a_row_from_0_99_is_level_and_below_it_is_behind(report):
    """The standing split and the alerts draw the same line: 0.99 of the fastest baseline."""
    row = {"baseline_tag": "torch", "baseline_device_busy_ms": 1.0, "tileops_device_busy_ms": 1.0}
    bench_ops = {
        "AddFwdOp": {
            "configs": [
                {**row, "name": "level", "baseline_ratio": 0.995},
                {**row, "name": "behind", "baseline_ratio": 0.985},
            ]
        }
    }

    assert report.baseline_standing(bench_ops) == (1, 1)
    assert [a["config"] for a in report.detect_baseline_alerts(bench_ops)] == ["behind"]


@pytest.mark.parametrize(
    "ahead, behind, message, color",
    [
        pytest.param(9999, 1, "99.9% ahead", "green", id="rounds-down-below-full"),
        pytest.param(10, 0, "100% ahead", "brightgreen", id="none-behind"),
        pytest.param(90, 10, "90% ahead", "yellow", id="below-green-share"),
    ],
)
def test_badge_share_and_color(report, ahead, behind, message, color):
    """The share rounds down, so a row behind never reads full."""
    badge = report.baseline_badge(ahead, behind)
    assert (badge["message"], badge["color"]) == (message, color)


def _bench_xml(tmp_path, name, cases):
    """Write a benchmark JUnit XML; each case is ``(test, tileops_ms, {tag: ms})``."""
    body = ""
    for test, tileops_ms, comparators in cases:
        props = {"op": "XOp", "tileops_device_busy_ms": tileops_ms}
        props.update({f"{tag}_device_busy_ms": ms for tag, ms in comparators.items()})
        rendered = "".join(f'<property name="{k}" value="{v}"/>' for k, v in props.items())
        body += (
            f'<testcase classname="b" name="{test}"><properties>{rendered}</properties></testcase>'
        )
    path = tmp_path / name
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    return str(path)


@pytest.mark.parametrize(
    "tileops_ms, comparators, expected",
    [
        (1.0, {"torch": 2.0, "vllm": 0.9}, (0, 1)),  # a no_ratio comparator still counts
        (1.0, {"torch": 0.98}, (0, 1)),  # a large row is behind on its first night
        (1.0, {}, (0, 0)),  # no comparator: judged nowhere
        (0, {"torch": 1.0}, (0, 0)),  # no TileOPs time: judged nowhere
    ],
)
def test_badge_judges_a_row_against_every_timed_comparator(
    report, tmp_path, tileops_ms, comparators, expected
):
    rows = report.badge_rows(_bench_xml(tmp_path, "now.xml", [("t[a]", tileops_ms, comparators)]))
    assert report.badge_standing(rows, None) == expected


def test_badge_counts_a_small_row_behind_only_when_it_was_behind_before(report, tmp_path):
    now = [("t[a]", 0.005, {"torch": 0.0049}), ("t[b]", 0.005, {"torch": 0.0049})]
    before = [("t[a]", 0.005, {"torch": 0.0049}), ("t[b]", 0.005, {"torch": 0.0051})]
    rows = report.badge_rows(_bench_xml(tmp_path, "now.xml", now))
    previous = report.badge_rows(_bench_xml(tmp_path, "before.xml", before))
    assert report.badge_standing(rows, previous) == (1, 1)
    assert report.badge_standing(rows, None) == (2, 0)


@pytest.mark.parametrize(
    "extra, outcome",
    [("", "unavailable"), ('<error message="teardown"/>', "failed")],
)
def test_an_unavailable_row_is_neither_a_skip_nor_a_failure(report, tmp_path, extra, outcome):
    """A row no implementation serves is its own outcome; an error after the skip outranks it."""
    xml = tmp_path / "bench.xml"
    xml.write_text(
        '<testsuite><testcase classname="b" name="t[row-bfloat16]">'
        '<skipped message="unavailable: no implementation serves this call"/>'
        f"{extra}<properties>"
        '<property name="unavailable" value="no implementation serves this call"/>'
        '<property name="ops" value="AddFwdOp"/>'
        "</properties></testcase></testsuite>",
        encoding="utf-8",
    )
    rows = report.parse_bench_xml(str(xml))

    assert rows[0]["outcome"] == outcome
    assert report.count_bench_outcome(rows, "skipped") == 0
    assert report.count_bench_outcome(rows, "unavailable") == (outcome == "unavailable")
    assert len(report.collect_bench_failures(rows)) == (outcome == "failed")


def _model_case(model: str, name: str, outcome: str) -> dict:
    return {"nodeid": f"tests.models.test_{model}::{name}", "name": name, "outcome": outcome}


def test_model_results_group_by_model_file(report):
    models = report.aggregate_model_results(
        [
            _model_case("qwen3_8", "test_a", "passed"),
            _model_case("qwen3_8", "test_b", "failed"),
            _model_case("deepseek_v4", "test_c", "skipped"),
        ]
    )
    assert models["qwen3_8"] == {
        "passed": 1,
        "failed": 1,
        "skipped": 0,
        "failing_tests": ["test_b"],
    }
    assert models["deepseek_v4"]["skipped"] == 1


@pytest.mark.parametrize(
    ("nodeid", "name"),
    [
        pytest.param("tests.models.test_qwen3_8.TestLoad::test_a", "test_a", id="class-test"),
        pytest.param("::tests.models.test_qwen3_8", "tests.models.test_qwen3_8", id="collection"),
    ],
)
def test_a_model_is_named_by_its_test_module(report, nodeid, name):
    """A class test's classname ends in its class, and a collection error leaves the classname
    empty and puts the module in the name; both still name the model."""
    case = {"nodeid": nodeid, "name": name, "outcome": "failed"}
    assert list(report.aggregate_model_results([case])) == ["qwen3_8"]


def test_models_are_their_own_summary_row_and_section(report):
    md = _report(
        report, models={"qwen3_8": {"passed": 30, "failed": 0, "skipped": 0, "failing_tests": []}}
    )
    assert "| **Models** | ✅ &ensp; (1 models, 30/30 tests) |" in md
    assert "| ✅ **qwen3_8** | 30 | 0 | 0 |  |" in md[md.index("## Models") :]
    assert md.startswith("# ✅")


def test_a_failing_model_test_fails_the_report(report):
    md = _report(
        report,
        models={"qwen3_8": {"passed": 29, "failed": 1, "skipped": 0, "failing_tests": ["test_x"]}},
    )
    assert md.startswith("# ❌")
    assert "| **Models** | ❌ 1 failed" in md
    assert "| ❌ **qwen3_8** | 29 | 1 | 0 | test_x |" in md


def test_a_run_without_model_results_has_no_models_row(report):
    assert "**Models**" not in _report(report)


def test_a_missing_model_results_file_fails_the_report(tmp_path):
    """A model run killed before pytest writes its JUnit file reports no results, not a pass."""
    out = tmp_path / "report.md"
    missing = tmp_path / "model_test_results.xml"
    _run_script("--model-test-xml", str(missing), "--output", str(out)).check_returncode()
    md = out.read_text()
    assert md.startswith("# ❌")
    assert "| **Models** | ❌ no results" in md


def test_regressions_output_holds_the_regressed_rows_only_with_history(tmp_path):
    """The board's regressions file lists the rows slower than their 14-day median, and is
    not written without history, where no row can be judged."""
    bench = tmp_path / "bench.xml"
    bench.write_text(_row_xml())
    today = date.today().isoformat()
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"runs": [{**_history_run(0.05), "date": today}]}))
    out = tmp_path / "regressions.json"
    common = ["--bench-xml", str(bench), "--output", str(tmp_path / "report.md")]
    _run_script(*common, "--regressions-output", str(out)).check_returncode()
    assert not out.exists()
    _run_script(
        *common, "--history", str(history), "--regressions-output", str(out)
    ).check_returncode()
    assert [(r["op"], r["config"], r["delta_pct"]) for r in json.loads(out.read_text())] == [
        (_OP, _CONFIG, pytest.approx(100.0))
    ]
