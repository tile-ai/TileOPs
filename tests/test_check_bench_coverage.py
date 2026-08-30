"""The benchmark-coverage gate's verdicts.

Which op a bench file measured is read out of a run's JUnit report, so the
cases here are reports: one per state a declared benchmark can be in.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_bench_coverage", REPO_ROOT / "scripts" / "check_bench_coverage.py"
)
coverage = importlib.util.module_from_spec(_spec)
sys.modules["check_bench_coverage"] = coverage
_spec.loader.exec_module(coverage)

MANIFEST = {
    "FooOp": {
        "status": "implemented",
        "source": {"bench": "benchmarks/ops/bench_foo.py"},
        "workloads": [{"label": "foo-case", "x_shape": [8], "dtypes": ["float16"]}],
    },
    "BarOp": {"status": "implemented", "source": {"bench": "benchmarks/ops/bench_foo.py"}},
    "SpecOp": {"status": "spec-only", "source": {"bench": "benchmarks/ops/bench_foo.py"}},
    "NoBenchOp": {"status": "implemented", "source": {}},
}


def _report(tmp_path: Path, *testcases: str) -> Path:
    """A JUnit report whose testcases all belong to ``bench_foo``."""
    body = "".join(testcases)
    path = tmp_path / "bench_results.xml"
    path.write_text(f'<?xml version="1.0"?><testsuites><testsuite>{body}</testsuite></testsuites>')
    return path


def _passed(name: str, op: str | None = None, classname: str = "benchmarks.ops.bench_foo") -> str:
    """A passing testcase; *op* may name several ops the case benchmarked."""
    props = ""
    if op:
        first = op.split(",")[0]
        props = (
            "<properties>"
            f'<property name="op" value="{first}"/>'
            f'<property name="ops" value="{op}"/>'
            "</properties>"
        )
    return f'<testcase classname="{classname}" name="{name}">{props}</testcase>'


def _skipped(name: str, message: str) -> str:
    return (
        f'<testcase classname="benchmarks.ops.bench_foo" name="{name}">'
        f'<skipped message="{message}"/></testcase>'
    )


def _failed(name: str, message: str, tag: str = "failure") -> str:
    return (
        f'<testcase classname="benchmarks.ops.bench_foo" name="{name}">'
        f'<{tag} message="{message}"/></testcase>'
    )


def _verdicts(path: Path) -> dict[str, str]:
    rows = coverage.verdicts(coverage.parse_run(path), MANIFEST)
    return {op: verdict for op, _, verdict, _ in rows}


class TestVerdicts:
    """One case per state of the table in scripts/check_bench_coverage.py."""

    def test_recorded_op_passes_and_an_unrecorded_sibling_fails(self, tmp_path):
        """A green file that never records an op it is declared for is the gap."""
        path = _report(
            tmp_path, _passed("test_foo[foo-case-float16]", op="FooOp"), _passed("test_other")
        )
        assert _verdicts(path) == {"FooOp": coverage.OK, "BarOp": coverage.FAIL}

    def test_a_failing_testcase_leaves_no_verdict(self, tmp_path):
        """The benchmark job's own exit code already reports a failed run."""
        for tag in ("failure", "error"):
            path = _report(
                tmp_path,
                _passed("test_foo[foo-case-float16]", op="FooOp"),
                _failed("test_bar", "boom", tag),
            )
            assert _verdicts(path)["BarOp"] == coverage.NO_VERDICT

    def test_all_skipped_states_its_reason(self, tmp_path):
        path = _report(tmp_path, _skipped("test_foo", "flag_gems missing"))
        assert _verdicts(path) == {"FooOp": coverage.SKIPPED, "BarOp": coverage.SKIPPED}

    def test_a_skip_beside_a_pass_is_not_a_gap(self, tmp_path):
        """A skipped testcase carries no op name, so it could have been this one.

        27 bench files are shared by several ops; a sibling passing there says
        nothing about the op whose case was skipped.
        """
        path = _report(
            tmp_path,
            _passed("test_foo[foo-case-float16]", op="FooOp"),
            _skipped("test_bar", "bar dependency missing"),
        )
        assert _verdicts(path) == {"FooOp": coverage.OK, "BarOp": coverage.SKIPPED}

    def test_a_file_absent_from_the_report_never_ran(self, tmp_path):
        path = _report(
            tmp_path, _passed("test_x", op="OtherOp", classname="benchmarks.ops.bench_x")
        )
        assert _verdicts(path) == {"FooOp": coverage.NOT_RUN, "BarOp": coverage.NOT_RUN}

    def test_a_class_holds_its_testcases_for_the_module(self, tmp_path):
        """pytest names a test inside a class ``module.ClassName``."""
        path = _report(
            tmp_path,
            _passed(
                "test_foo[foo-case-float16]",
                op="FooOp",
                classname="benchmarks.ops.bench_foo.TestFoo",
            ),
            _passed("test_bar", op="BarOp", classname="benchmarks.ops.bench_foo.TestBar"),
        )
        assert _verdicts(path) == {"FooOp": coverage.OK, "BarOp": coverage.OK}

    def test_rows_that_miss_every_declared_workload_fail(self, tmp_path):
        """A run on shapes the manifest never declared is not this op's coverage.

        L4 stopped matching the op name in the source; this is what catches a
        file reading workloads declared under other names. A sibling declaring
        the same labels is indistinguishable, and its shapes are this op's too.
        """
        path = _report(tmp_path, _passed("test_foo[other-case-float16]", op="FooOp"))
        assert _verdicts(path)["FooOp"] == coverage.FAIL

    def test_a_longer_id_does_not_borrow_a_shorter_declared_one(self, tmp_path):
        """``foo-case`` must not answer for a row of ``foo-case-wide``."""
        path = _report(tmp_path, _passed("test_foo[foo-case-wide-float16]", op="FooOp"))
        assert _verdicts(path)["FooOp"] == coverage.FAIL

    def test_a_declared_workload_in_the_case_id_is_the_coverage(self, tmp_path):
        path = _report(tmp_path, _passed("test_foo[foo-case-float16]", op="FooOp"))
        assert _verdicts(path)["FooOp"] == coverage.OK

    def test_one_testcase_may_benchmark_several_ops(self, tmp_path):
        """``op`` alone names the first; the gate reads them all off ``ops``."""
        path = _report(tmp_path, _passed("test_both[foo-case-float16]", op="BarOp,FooOp"))
        assert _verdicts(path) == {"FooOp": coverage.OK, "BarOp": coverage.OK}

    def test_a_report_without_the_ops_property_still_attributes(self, tmp_path):
        """A report predating ``ops`` carries ``op`` only."""
        case = (
            '<testcase classname="benchmarks.ops.bench_foo" name="test_foo[foo-case-float16]">'
            '<properties><property name="op" value="FooOp"/></properties></testcase>'
        )
        path = _report(tmp_path, case)
        assert _verdicts(path)["FooOp"] == coverage.OK

    def test_only_implemented_ops_declaring_a_bench_are_expected(self, tmp_path):
        """A spec-only op and an op without a bench pointer are not rows."""
        path = _report(tmp_path, _passed("test_foo[foo-case-float16]", op="FooOp"))
        assert set(_verdicts(path)) == {"FooOp", "BarOp"}


class TestExitCodes:
    """An audit that reached no conclusion must not read as a passed one."""

    def test_exit_codes_follow_the_worst_verdict(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(coverage, "load_manifest", lambda: MANIFEST)

        gap = _report(tmp_path, _passed("test_foo[foo-case-float16]", op="FooOp"))
        assert coverage.main(["--bench-xml", str(gap)]) == coverage.EXIT_GAP

        covered = _report(
            tmp_path,
            _passed("test_foo[foo-case-float16]", op="FooOp"),
            _passed("test_bar", op="BarOp"),
        )
        out = tmp_path / "coverage.md"
        assert (
            coverage.main(["--bench-xml", str(covered), "--output", str(out)]) == coverage.EXIT_OK
        )
        assert "2 OK" in out.read_text()

    @pytest.mark.parametrize("content", [None, "<testsuites><broken"])
    def test_a_missing_or_unusable_report_is_not_a_pass(self, tmp_path, content):
        path = tmp_path / "bench_results.xml"
        if content is not None:
            path.write_text(content)
        assert coverage.main(["--bench-xml", str(path)]) == coverage.EXIT_NO_REPORT
