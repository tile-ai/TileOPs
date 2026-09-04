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
