"""Verdict logic of scripts/validate_roofline_bytes.py (roofline.md §4.5)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.smoke

_SPEC = importlib.util.spec_from_file_location(
    "validate_roofline_bytes",
    Path(__file__).resolve().parents[1] / "scripts" / "validate_roofline_bytes.py",
)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_base import BenchmarkBase  # noqa: E402

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


class TestReadHalfAfterACall:
    """The audit reads the declaration off an op it has just called."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="the audit calls the op")
    def test_an_op_that_keeps_no_input_shape_still_declares_its_read_half(self):
        """An op keeps what its own `eval_roofline` needs: `DropoutFwdOp` keeps an
        element count and a dtype, never a shape. The write half is priced from the
        output shapes, which the call's input shapes decide, so `Op.__call__` records
        them."""
        from tileops.ops.dropout import DropoutFwdOp
        from tileops.ops.op_base import record_roofline_calls

        op = DropoutFwdOp(p=0.5)
        x = torch.rand(2048, 4096, dtype=torch.float16, device="cuda")
        record_roofline_calls()
        try:
            op(x)
        finally:
            record_roofline_calls(False)
        assert op.eval_roofline_read_bytes() == x.numel() * x.element_size()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="the audit calls the op")
    def test_a_call_outside_the_audit_records_nothing(self):
        """The recording costs about a microsecond, which is a fifth of a small
        kernel's launch, so every other call must stay clear of it."""
        from tileops.ops.dropout import DropoutFwdOp
        from tileops.ops.op_base import record_roofline_calls

        record_roofline_calls(False)
        op = DropoutFwdOp(p=0.5)
        op(torch.rand(64, 64, dtype=torch.float16, device="cuda"))
        assert getattr(op, "_roofline_call_tensors", None) is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="the audit calls the op")
    def test_the_gemm_case_builds_the_layout_its_row_names(self):
        """`trans_a` / `trans_b` decide which axis of each operand contracts, and a
        row that names them describes operands stored that way."""
        from tileops.manifest import load_manifest

        entry = load_manifest()["GemmFwdOp"]
        row = {"m": 64, "n": 128, "k": 256, "trans_a": False, "trans_b": True}
        op, (a, b) = audit._gemm_case("GemmFwdOp", entry, row, torch.float16)
        assert (tuple(a.shape), tuple(b.shape)) == ((64, 256), (128, 256))


class TestRooflineInputsRecording:
    """What decided a call's bytes travels with the reading (roofline.md 4.7)."""

    @staticmethod
    def _reported(reader):
        """What a benchmark over an op with this `roofline_inputs` records."""
        return BenchmarkBase._roofline_inputs(
            SimpleNamespace(op=SimpleNamespace(roofline_inputs=reader))
        )

    def test_a_reading_carries_what_the_op_reports(self):
        assert self._reported(lambda: {"active_experts": 96}) == {"active_experts": 96}

    def test_a_diagnostic_that_raises_does_not_fail_the_measurement(self):
        def explode():
            raise RuntimeError("routing was never bound")

        assert self._reported(explode) == {}
