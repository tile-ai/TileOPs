"""Tests for scripts/nightly_report.py.

Covers the verdict rules a wrong comparison can mislead: workload identity,
baseline choice, the noise gate, rename recovery, and the previous-run lens.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_SCRIPT = REPO_ROOT / "scripts" / "nightly_report.py"

_OP = "FooFwdOp"
_CONFIG = "test_foo_bench[row-bfloat16]"


@pytest.fixture(scope="module")
def report():
    """Import nightly_report as a module."""
    spec = importlib.util.spec_from_file_location("nightly_report", REPORT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _history_run(busy_ms, name=_CONFIG, p10=None, p90=None, **counts):
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


def test_name_that_ever_shared_a_run_with_the_current_name_is_not_a_rename(report):
    """Co-occurrence disqualifies a prior name even when its counts differ there."""
    shared_run = {
        "ops": {
            _OP: {
                _CONFIG: {"tileops": {"device_busy_ms": 0.39, "flops": 5e9, "bytes": 1e6}},
                "test_foo_bench[other]": {
                    "tileops": {"device_busy_ms": 0.20, "flops": 9e9, "bytes": 9e6}
                },
            }
        }
    }
    runs = [_history_run(0.10, name="test_foo_bench[other]", flops=5e9, bytes=1e6), shared_run]
    assert report.detect_regressions(_bench_ops(0.40, flops=5e9, bytes=1e6), runs) == []


def test_history_entry_records_the_percentiles(report):
    """The noise gate needs each run's spread persisted with its reading."""
    entry = report.build_history_entry(_bench_ops(0.010, p10=0.0099, p90=0.0101))
    tileops = entry["ops"][_OP][_CONFIG]["tileops"]
    assert tileops["device_busy_p10_ms"] == 0.0099
    assert tileops["device_busy_p90_ms"] == 0.0101


# ---------------------------------------------------------------------------
# Speed-of-Light (M5)
# ---------------------------------------------------------------------------

# A fixed synthetic profile keeps the expected efficiencies exact; the real
# h200.yaml numbers are calibration data, not arithmetic under test.
_PROFILE = {
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


def test_sol_efficiency_is_one_at_the_effective_ceiling(report):
    sol = report._compute_sol(_sol_row(), _PROFILE)
    assert sol["bound"] == "memory"
    assert sol["efficiency"] == pytest.approx(1.0)
    assert not sol["impossible"] and not sol["latency_bound"]


def test_sol_compute_bound_uses_the_declared_roof(report):
    # 7e11 FLOPs on the bf16 tensor-core roof: sol_time = 1 ms; measured 2 ms.
    sol = report._compute_sol(
        _sol_row(
            tileops_flops=7e11,
            tileops_bytes=1e6,
            tileops_device_busy_ms=2.0,
            tileops_compute_roof="tensor_core.bf16",
        ),
        _PROFILE,
    )
    assert sol["bound"] == "compute"
    assert sol["efficiency"] == pytest.approx(0.5)


def test_sol_rate_above_theoretical_is_impossible_not_fast(report):
    sol = report._compute_sol(_sol_row(tileops_device_busy_ms=0.5), _PROFILE)
    assert sol["impossible"] == ["bytes/s over HBM theoretical"]


def test_sol_uncounted_copies_enter_the_denominator(report):
    sol = report._compute_sol(_sol_row(tileops_uncounted_copy_ms=1.0), _PROFILE)
    assert sol["efficiency"] == pytest.approx(0.5)


def test_sol_skips_rows_the_model_cannot_judge(report):
    assert report._compute_sol(_sol_row(tileops_timing="cuda-events"), _PROFILE) is None
    no_timing = _sol_row()
    del no_timing["tileops_timing"]
    assert report._compute_sol(no_timing, _PROFILE) is None
    assert report._compute_sol(_sol_row(tileops_bytes=0), _PROFILE) is None
    assert report._compute_sol(_sol_row(tileops_compute_roof="tensor_core.fp8"), _PROFILE) is None


def test_sol_latency_bound_needs_both_floors(report):
    tiny = _sol_row(tileops_flops=1e3, tileops_bytes=1e3, tileops_device_busy_ms=0.004)
    assert report._compute_sol(tiny, _PROFILE)["latency_bound"]
    # A tiny lower bound with a large measured time is a slow kernel, not noise.
    slow = _sol_row(tileops_flops=1e3, tileops_bytes=1e3, tileops_device_busy_ms=0.5)
    assert not report._compute_sol(slow, _PROFILE)["latency_bound"]


def test_annotate_sol_reports_anomalies_and_tags_rows(report):
    bench_ops = {
        _OP: {
            "module": "m",
            "configs": [_sol_row(), _sol_row(name="impossible", tileops_device_busy_ms=0.5)],
        }
    }
    anomalies = report.annotate_sol(bench_ops, _PROFILE)
    assert [a["level"] for a in anomalies] == ["FAIL"]
    assert bench_ops[_OP]["configs"][0]["sol"]["efficiency"] == pytest.approx(1.0)


def test_history_entry_records_the_sol_reading(report):
    bench_ops = {_OP: {"module": "m", "configs": [_sol_row()]}}
    report.annotate_sol(bench_ops, _PROFILE)
    entry = report.build_history_entry(bench_ops)
    tileops = entry["ops"][_OP][_CONFIG]["tileops"]
    assert tileops["compute_roof"] == "cuda_core.fp32"
    assert tileops["sol"] == {"efficiency": 1.0, "bound": "memory", "latency_bound": False}
