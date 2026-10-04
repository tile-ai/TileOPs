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
    """A case timing five implementations publishes five rows, not two.

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


# ---------------------------------------------------------------------------
# Speed-of-Light (M5)
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
        "tileops_compute_roof": "cuda_core.fp32",
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
            tileops_compute_roof="tensor_core.bf16",
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
    assert report._compute_sol(_sol_row(tileops_compute_roof="tensor_core.fp8"), profile) is None


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
    assert tileops["compute_roof"] == "cuda_core.fp32"
    assert tileops["sol"] == {"efficiency": 1.0, "bound": "memory", "latency_bound": False}


# The window is rebuilt from the snapshot repository on every run, so what a
# snapshot cannot supply must not be taken from the machine doing the rebuild.

BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_perf_history.py"

_BENCH_XML = (
    '<testsuites><testsuite name="bench"><testcase classname="benchmarks.test_foo"'
    f' name="{_CONFIG}"><properties>'
    f'<property name="op" value="{_OP}"/>'
    '<property name="tileops_device_busy_ms" value="0.1"/>'
    "</properties></testcase></testsuite></testsuites>"
)


_COVERAGE_XML = (
    "<coverage><packages><package><classes>"
    '<class filename="src/tileops/ops/foo.py"><lines>'
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


def _build_window(tmp_path, repo):
    out = tmp_path / "perf_history.json"
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--repo", str(repo), "--out", str(out)],
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
    assert report._history_window([], read=False) == "not read"
    assert report._history_window([], read=True) == "empty"
    assert report._history_window([{"date": "2026-09-05"}, {"date": "2026-09-18"}], read=True) == (
        "2 runs, 2026-09-05 to 2026-09-18"
    )


def test_a_named_history_that_does_not_exist_is_refused(tmp_path):
    """A window the builder failed to write must not read as no history."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--output",
            str(tmp_path / "report.md"),
            "--history",
            str(tmp_path / "absent.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--history file does not exist" in result.stderr


def _summary_report(report, scale):
    return report.generate_report(
        scale=scale,
        test_ops={_OP: {"module": "m", "failed": 0, "passed": 1, "skipped": 0, "tests": []}},
        bench_ops=_bench_ops(1.0),
        bench_failures=[],
        regressions=[],
        improvements=[],
        baseline_alerts=[],
    )


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
    # An entry that is only a specification is not something the library offers.
    assert "| **Operators** | 205 |" not in md


def test_a_manifest_that_will_not_load_drops_the_rows_not_the_report(report):
    """Benchmark data stands on its own; an unreadable manifest costs two rows."""
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


def _counted(report, results):
    """The whole report, built over *results* as the suite that ran."""
    md = report.generate_report(
        test_ops={_OP: {"module": "m", "failed": 0, "passed": 1, "skipped": 0, "tests": []}},
        bench_ops=None,
        bench_failures=[],
        regressions=[],
        improvements=[],
        baseline_alerts=[],
        test_results=results,
    )
    return md


def test_correctness_counts_every_test_not_only_the_attributed_ones(report):
    """Counting only tests a reference attributed reported a rate over half the suite."""
    results = [_result("passed", _OP)] + [_result("passed", None) for _ in range(3)]

    assert "(4/4 tests)" in _counted(report, results)


def test_an_unattributed_failure_is_counted_and_named(report):
    """A test with no op property fails the job, so the report must not lose it.

    Counting it and leaving it out of the failure table reports a number with
    nothing behind it: the reader sees N failed and a table holding fewer rows.
    """
    row = _result("failed", None, "test_sum")
    md = _counted(report, [_result("passed", _OP), row])

    assert "(1/2 tests)" in md
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
    # Nor is a passing test that compared nothing. After ownership becomes a
    # declaration, a constructor-rejection test carries the op name and no value.
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


def test_the_kernel_count_is_what_packages_export(report):
    """Concrete Kernel subclasses a package re-exports, by type and at any depth.

    Three ways to get it wrong, and the figure looks plausible after each: read
    it off the manifest and it becomes the op count; walk every module and the
    private implementation bases under `_base` join it; filter on the name and
    `IndexedExpertGemmTemplate` drops out.
    """
    import importlib
    import inspect
    import pkgutil

    import tileops.kernels as kernels_pkg
    from tileops.kernels.kernel_base import Kernel

    def exported(module):
        return {
            attribute
            for attribute in (getattr(module, n, None) for n in getattr(module, "__all__", ()))
            if inspect.isclass(attribute)
            and issubclass(attribute, Kernel)
            and not inspect.isabstract(attribute)
        }

    every_module = exported(kernels_pkg)
    off_name = set()
    for module in pkgutil.walk_packages(kernels_pkg.__path__, prefix="tileops.kernels."):
        try:
            package = importlib.import_module(module.name)
        except Exception:
            continue
        every_module |= exported(package)
        if module.ispkg:
            off_name |= {c for c in exported(package) if not c.__name__.endswith("Kernel")}

    count = report._kernel_count()

    assert off_name, "no off-name kernel left; the name filter would now be equivalent"
    # Walking every module pulls in the private bases under `_base`.
    assert count < len(every_module)


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
