"""Verdict logic of scripts/validate_roofline_bytes.py (roofline.md §4.5)."""

import importlib.util
import json
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

    def test_a_call_that_reads_nothing_has_no_ratio_to_be_judged_by(self):
        """Dropout at p == 1 writes zeros and reads none of its input, which is a
        read half of zero: a legitimate declaration, and not one a ratio judges."""
        assert audit.read_side_verdict(MEASURED_READ, 0) == "SKIPPED"

    def test_a_negative_read_half_is_a_broken_declaration(self):
        assert audit.read_side_verdict(MEASURED_READ, -1) == "ERROR"

    def test_a_waived_shortfall_is_reported_not_judged(self):
        """A kernel may predicate away the load of a position whose value decides
        nothing, so reading less than the call binds is not the formula's fault."""
        assert audit.read_side_verdict(READ_BYTES * 0.5, READ_BYTES, bound=False) == "EXEMPT"


class TestReadBoundException:
    """The exception covers the calls its condition names and no others."""

    ENTRY = {
        "signature": {"params": {"p": {"default": 0.5}, "training": {"default": True}}},
        "roofline": {
            "read_bound_exception": {
                "when": "training and 0.0 < p < 1.0",
                "reason": "the mask can predicate away a dropped position's load",
            }
        },
    }

    def test_a_row_inside_the_condition_is_waived(self):
        assert audit.read_bound_exception(self.ENTRY, {"p": 0.5, "training": True})

    def test_a_row_outside_it_is_judged(self):
        """Eval mode copies the input, so the full read really is required."""
        assert audit.read_bound_exception(self.ENTRY, {"p": 0.5, "training": False}) == ""
        assert audit.read_bound_exception(self.ENTRY, {"p": 0.0, "training": True}) == ""

    def test_a_row_that_omits_the_key_falls_back_to_the_param_default(self):
        assert audit.read_bound_exception(self.ENTRY, {"p": 0.5})

    def test_the_condition_reads_the_element_type_the_row_expands_to(self, tmp_path):
        """A row states a dtype axis; the call runs one of them, and whether a load
        can be predicated away can follow it. The audited row is what the condition
        sees, so this goes through the run that produces a verdict."""
        entry = {
            "signature": {"params": {}},
            "roofline": {
                "read_bound_exception": {
                    "when": "dtype == 'float16'",
                    "reason": "the packed load covers a dropped position",
                }
            },
            "workloads": [
                {"x_shape": [64], "dtypes": ["float16", "float32"], "label": "row"},
            ],
        }
        verdicts = {}
        for dtype_str in ("float16", "float32"):
            rows = self._audited(entry, dtype_str, tmp_path)
            verdicts[dtype_str] = rows[0]["verdict"]
        assert verdicts == {"float16": "EXEMPT", "float32": "FAIL"}

    @staticmethod
    def _audited(entry: dict, dtype_str: str, tmp_path) -> list[dict]:
        """Run audit_one over one dtype, with the profiler and its CSV stubbed.

        The child would need a GPU and ncu needs counters no test has, so what
        is exercised here is what the audit does with a measurement: which row
        and dtype reach the condition.
        """
        import subprocess
        from unittest import mock

        declared, measured = 1024, 512  # a shortfall, whatever the dtype
        emitted = json.dumps(
            {"formula_flops": 0, "formula_bytes": declared * 2, "read_bytes": declared}
        )
        row = dict(entry["workloads"][0], dtypes=[dtype_str])
        with (
            mock.patch.object(audit, "_pick_workloads", return_value=[(row, dtype_str)]),
            mock.patch.object(
                subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, stdout=emitted, stderr=""),
            ),
            mock.patch.object(audit, "_parse_ncu_csv", return_value=((measured, 0.0), 1)),
        ):
            return audit.audit_one("Op", entry, tmp_path)

    def test_an_op_whose_every_row_is_waived_is_named(self):
        """Its read half went unjudged, which a run has to say rather than count as
        checked."""
        results = [
            {"op": "Waived", "verdict": "EXEMPT"},
            {"op": "Waived", "verdict": "SKIPPED"},
            {"op": "Judged", "verdict": "EXEMPT"},
            {"op": "Judged", "verdict": "PASS"},
        ]
        assert audit.fully_waived(results) == ["Waived"]

    def test_an_entry_without_the_exception_waives_nothing(self):
        assert audit.read_bound_exception({"roofline": {}}, {"p": 0.5}) == ""

    def test_the_manifest_states_both_halves_wherever_it_waives(self):
        from tileops.manifest import load_manifest

        for name, entry in load_manifest().items():
            stated = (entry.get("roofline") or {}).get("read_bound_exception")
            if stated is None:
                continue
            assert stated.get("when", "").strip(), name
            assert stated.get("reason", "").strip(), name


class TestDeclaredReadHalf:
    """No fallback: an undeclared read half must never become a number."""

    class _Undeclared:
        def eval_roofline_read_bytes(self):
            return None

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

    def test_warn_skipped_and_exempt_stay_green(self):
        assert audit.exit_code({"PASS": 2, "WARN": 1, "SKIPPED": 3, "EXEMPT": 2}) == 0

    def test_a_verdict_nothing_reads_is_not_a_pass(self):
        """Green is an allowlist: a run whose rows carry a spelling this file does
        not know has not been judged, whatever that spelling was meant to say."""
        assert audit.exit_code({"PASS": 2, "EXEMPTED": 1}) == 1


class TestReadHalfAfterACall:
    """The audit reads the declaration off an op it has just called."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="the audit calls the op")
    def test_an_op_that_keeps_no_input_shape_still_declares_its_read_half(self):
        """An op keeps what its own `eval_roofline` needs: `DropoutFwdOp` keeps an
        element count and a dtype, never a shape. The write half is priced from the
        output shapes, which the call's input shapes decide, so `Op.__call__` records
        them."""
        from tileops.ops.dropout import DropoutFwdOp
        from tileops.ops.op_base import _recording_roofline_calls

        op = DropoutFwdOp(p=0.5)
        x = torch.rand(2048, 4096, dtype=torch.float16, device="cuda")
        with _recording_roofline_calls():
            op(x)
        assert op.eval_roofline_read_bytes() == x.numel() * x.element_size()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="the audit calls the op")
    def test_a_call_outside_the_block_records_nothing(self):
        """The recording costs about a microsecond a call, which every other caller
        must stay clear of."""
        from tileops.ops.dropout import DropoutFwdOp
        from tileops.ops.op_base import _recording_roofline_calls

        op = DropoutFwdOp(p=0.5)
        x = torch.rand(64, 64, dtype=torch.float16, device="cuda")
        with _recording_roofline_calls():
            op(x)
        op(x)
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
    """What decided a call's bytes travels with the reading."""

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
