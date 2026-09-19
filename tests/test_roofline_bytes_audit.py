"""Verdict logic of scripts/validate_roofline_bytes.py (roofline.md §4.5)."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_SPEC = importlib.util.spec_from_file_location(
    "validate_roofline_bytes",
    Path(__file__).resolve().parents[1] / "scripts" / "validate_roofline_bytes.py",
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

# Measured: SiluFwdOp, x_shape=[2048, 14336] fp16, H200. Reads match the
# formula's read half; the same run's writes read 37,068,800 of 58,720,256
# because L2 still held them when the kernel ended.
MEASURED_READ = 58_731_520
READ_BYTES = 58_720_256


class TestReadSideVerdict:
    def test_matching_read_traffic_passes_despite_the_write_shortfall(self):
        assert audit.read_side_verdict(MEASURED_READ, READ_BYTES) == "PASS"

    def test_read_traffic_below_the_declared_half_fails(self):
        assert audit.read_side_verdict(READ_BYTES * 0.5, READ_BYTES) == "FAIL"

    def test_multi_pass_read_traffic_warns(self):
        assert audit.read_side_verdict(READ_BYTES * 2, READ_BYTES) == "WARN"

    def test_an_undeclared_read_half_yields_no_verdict(self):
        assert audit.read_side_verdict(MEASURED_READ, None) == "NO-VERDICT"

    def test_a_zero_read_half_is_a_broken_declaration(self):
        assert audit.read_side_verdict(MEASURED_READ, 0) == "ERROR"


class TestDeclaredReadHalf:
    """No fallback: an undeclared read half must never become a number."""

    class _Undeclared:
        def eval_roofline_read_bytes(self):
            return NotImplemented

    class _Declared:
        def eval_roofline_read_bytes(self):
            return READ_BYTES

    def test_an_undeclared_read_half_is_none_not_an_input_sum(self):
        assert audit._declared_read_bytes(self._Undeclared()) is None

    def test_a_declared_read_half_is_returned(self):
        assert audit._declared_read_bytes(self._Declared()) == READ_BYTES


_CSV_HEADER = '"ID","Kernel Name","Metric Name","Metric Value"'


class TestNcuCsvParsing:
    def test_reads_and_writes_sum_separately_across_kernels(self, tmp_path):
        csv_path = tmp_path / "two_kernels.csv"
        csv_path.write_text(
            "\n".join(
                [
                    _CSV_HEADER,
                    '"0","k0","dram__bytes_read.sum","1,000"',
                    '"0","k0","dram__bytes_write.sum","10"',
                    '"1","k1","dram__bytes_read.sum","500"',
                    '"1","k1","dram__bytes_write.sum","5"',
                ]
            )
        )
        assert audit._parse_ncu_csv(csv_path) == ((1500.0, 15.0), 2)

    def test_an_unreadable_metric_is_not_read_as_zero(self, tmp_path):
        csv_path = tmp_path / "na.csv"
        csv_path.write_text(
            "\n".join(
                [
                    _CSV_HEADER,
                    '"0","k0","dram__bytes_read.sum","n/a"',
                    '"0","k0","dram__bytes_write.sum","10"',
                ]
            )
        )
        measured, _ = audit._parse_ncu_csv(csv_path)
        assert measured is None


class TestExitCode:
    def test_a_no_verdict_row_fails_the_run(self):
        assert audit.exit_code({"PASS": 10, "NO-VERDICT": 1}) == 1

    def test_warn_and_skipped_stay_green(self):
        assert audit.exit_code({"PASS": 2, "WARN": 1, "SKIPPED": 3}) == 0
