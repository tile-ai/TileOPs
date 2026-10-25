"""The benchmark-coverage gate's verdicts.

Which op a benchmark measured is read out of a run's JUnit report, so the cases here are
reports: one per state an op's manifest calls can be in.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_bench_coverage", REPO_ROOT / "scripts" / "check_bench_coverage.py"
)
coverage = importlib.util.module_from_spec(_spec)
sys.modules["check_bench_coverage"] = coverage
_spec.loader.exec_module(coverage)

_ENTRY = yaml.safe_load((REPO_ROOT / "tests" / "manifest_cases.yaml").read_text())["entries"][
    "SiLUAndMulFwdOp"
]
MANIFEST = {
    "FooFwdOp": {**_ENTRY, "status": "implemented"},
    "BarFwdOp": {**_ENTRY, "status": "implemented"},
    "SpecFwdOp": {**_ENTRY, "status": "spec-only"},
}
CASE = "siluandmul-0-bfloat16"


def _report(tmp_path: Path, *testcases: str) -> Path:
    body = "".join(testcases)
    path = tmp_path / "bench_results.xml"
    path.write_text(f'<?xml version="1.0"?><testsuites><testsuite>{body}</testsuite></testsuites>')
    return path


def _passed(name: str, ops: str, prop: str = "ops") -> str:
    """A passing testcase recording *ops* under the property *prop*."""
    props = f'<properties><property name="{prop}" value="{ops}"/></properties>'
    return f'<testcase classname="benchmarks.ops.bench_foo" name="{name}">{props}</testcase>'


def _verdicts(path: Path) -> dict[str, str]:
    recorded, _ = coverage.parse_run(path)
    return {op: verdict for op, verdict, _ in coverage.verdicts(recorded, MANIFEST)}


class TestVerdicts:
    """One case per verdict, over the implemented entries only."""

    def test_a_recorded_case_passes_and_an_unrecorded_one_is_not_run(self, tmp_path):
        path = _report(tmp_path, _passed(f"test_foo[{CASE}]", "FooFwdOp"))
        assert _verdicts(path) == {"FooFwdOp": coverage.OK, "BarFwdOp": coverage.NOT_RUN}

    def test_a_case_id_recorded_twice_fails(self, tmp_path):
        """The case id keys the op's history, so it names one row."""
        path = _report(
            tmp_path,
            _passed(f"test_foo_kn[{CASE}]", "FooFwdOp"),
            _passed(f"test_foo_nk[{CASE}]", "FooFwdOp"),
        )
        assert _verdicts(path)["FooFwdOp"] == coverage.FAIL

    def test_one_testcase_may_benchmark_several_ops(self, tmp_path):
        """``op`` names the first; the gate reads them all off ``ops``, or ``op`` without it."""
        both = _report(tmp_path, _passed(f"test_both[{CASE}]", "BarFwdOp,FooFwdOp"))
        assert set(_verdicts(both).values()) == {coverage.OK}
        first = _report(tmp_path, _passed(f"test_foo[{CASE}]", "FooFwdOp", prop="op"))
        assert _verdicts(first)["FooFwdOp"] == coverage.OK


class TestExitCodes:
    """An audit that reached no conclusion must not read as a passed one."""

    @pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
    def test_unsuccessful_cases_fail_even_with_complete_coverage(
        self, tmp_path, monkeypatch, outcome
    ):
        monkeypatch.setattr(coverage, "load_manifest", lambda: MANIFEST)
        path = _report(
            tmp_path,
            _passed(f"test_a[{CASE}]", "BarFwdOp,FooFwdOp"),
            f'<testcase name="test_other"><{outcome}/></testcase>',
        )
        assert coverage.main(["--bench-xml", str(path)]) == coverage.EXIT_GAP

    @pytest.mark.parametrize("state", ["empty", "missing", "skipped", "unverified"])
    def test_incomplete_evidence_fails(self, tmp_path, monkeypatch, state):
        monkeypatch.setattr(coverage, "load_manifest", lambda: MANIFEST)
        case = _passed(f"test_a[{CASE}]", "BarFwdOp,FooFwdOp")
        if state == "empty":
            case = ""
        elif state == "missing":
            case = _passed(f"test_a[{CASE}]", "FooFwdOp")
        elif state == "skipped":
            case = case.replace("</testcase>", '<skipped message="unavailable"/></testcase>')
        else:
            case = case.replace(
                "</properties>",
                '<property name="tileops_unverified" value="no reference"/>'
                '<property name="tileops_no_ratio" value="True"/></properties>',
            )
        assert coverage.main(["--bench-xml", str(_report(tmp_path, case))]) == coverage.EXIT_GAP

    def test_declared_partial_verification_counts_as_evidence(self, tmp_path, monkeypatch):
        monkeypatch.setattr(coverage, "load_manifest", lambda: MANIFEST)
        case = _passed(f"test_a[{CASE}]", "BarFwdOp,FooFwdOp").replace(
            "</properties>",
            '<property name="tileops_unverified" value="intermediates have no oracle"/>'
            "</properties>",
        )
        assert coverage.main(["--bench-xml", str(_report(tmp_path, case))]) == coverage.EXIT_OK

    def test_exit_codes_follow_the_worst_verdict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(coverage, "load_manifest", lambda: MANIFEST)
        twice = _report(
            tmp_path,
            _passed(f"test_a[{CASE}]", "FooFwdOp"),
            _passed(f"test_b[{CASE}]", "FooFwdOp"),
        )
        assert coverage.main(["--bench-xml", str(twice)]) == coverage.EXIT_GAP
        once = _report(tmp_path, _passed(f"test_a[{CASE}]", "BarFwdOp,FooFwdOp"))
        out = tmp_path / "coverage.md"
        assert coverage.main(["--bench-xml", str(once), "--output", str(out)]) == coverage.EXIT_OK
        assert "2 OK" in out.read_text()

    @pytest.mark.parametrize("content", [None, "<testsuites><broken"])
    def test_a_missing_or_unusable_report_is_not_a_pass(self, tmp_path, content):
        path = tmp_path / "bench_results.xml"
        if content is not None:
            path.write_text(content)
        assert coverage.main(["--bench-xml", str(path)]) == coverage.EXIT_NO_REPORT
