"""Collection holds every ``cuda_only`` mark to a reason, and reports what a run dropped."""

import csv
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import workloads.device
from tests import conftest
from tests.conftest import (
    _DROPPED,
    UnservedError,
    _cuda_only_errors,
    pytest_collection_finish,
    pytest_collection_modifyitems,
    pytest_runtest_makereport,
)
from tileops.backend import OpNotAvailableError

pytestmark = pytest.mark.smoke


def _item(*marks: pytest.MarkDecorator, path: str = "tests/test_foo.py") -> SimpleNamespace:
    """A collected test carrying *marks*, plus the tier collection asks for."""
    carried = [mark.mark for mark in (pytest.mark.smoke, *marks)]
    return SimpleNamespace(
        nodeid=f"{path}::test_foo",
        path=Path(path),
        location=(path, 0, "test_foo"),
        name="test_foo",
        originalname="test_foo",
        callspec=None,
        config=SimpleNamespace(
            stash={}, hook=SimpleNamespace(pytest_deselected=lambda items: None)
        ),
        iter_markers=lambda name=None: iter(m for m in carried if name in (None, m.name)),
        get_closest_marker=lambda name: next((m for m in carried if m.name == name), None),
        add_marker=lambda mark: carried.append(mark.mark),
    )


@pytest.mark.parametrize(
    "mark",
    [
        pytest.mark.cuda_only,
        pytest.mark.cuda_only("needs-a-gpu"),
        pytest.mark.cuda_only(reason="cuda-graph"),
    ],
    ids=["none", "unknown", "keyword"],
)
def test_a_mark_without_one_known_reason_fails_collection(mark):
    with pytest.raises(pytest.UsageError, match="cuda_only takes one reason"):
        pytest_collection_modifyitems([_item(mark)])


def test_an_architecture_mark_already_needs_cuda():
    (error,) = _cuda_only_errors(_item(pytest.mark.cuda_only("cuda-kernel")), archs=["sm90"])
    assert "sm90 already needs CUDA" in error


def test_in_tests_ops_in_tree_kernels_already_drops_a_test_off_cuda():
    marks = (pytest.mark.cuda_only("cuda-reference"), pytest.mark.in_tree_kernels)
    (error,) = _cuda_only_errors(_item(*marks, path="tests/ops/test_foo.py"), archs=[])
    assert "in_tree_kernels already drops it" in error
    assert _cuda_only_errors(_item(*marks), archs=[]) == [], "outside tests/ops a cpu run keeps it"


def test_an_in_tree_kernel_in_tests_ops_is_marked_in_tree_kernels():
    item = _item(pytest.mark.cuda_only("cuda-kernel"), path="tests/ops/test_foo.py")
    (error,) = _cuda_only_errors(item, archs=[])
    assert "is in_tree_kernels" in error


@pytest.fixture
def checking_claims_off_cuda(monkeypatch):
    """A cpu run with ``--tileops-cuda-only-must-fail``."""
    monkeypatch.setattr(conftest, "_cuda_only_must_fail", True)
    monkeypatch.setattr(workloads.device, "_device", "cpu")


def test_checking_the_claims_keeps_only_cuda_only_tests_each_expected_to_fail(
    checking_claims_off_cuda,
):
    claimed, other = _item(pytest.mark.cuda_only("cuda-kernel")), _item()
    items = [claimed, other]

    pytest_collection_modifyitems(items)

    assert items == [claimed]
    xfail = claimed.get_closest_marker("xfail")
    assert xfail is not None and xfail.kwargs["strict"]


def test_checking_the_claims_fails_a_cuda_only_test_that_skips(checking_claims_off_cuda):
    report = SimpleNamespace(skipped=True, outcome="skipped", longrepr=None)
    wrapper = pytest_runtest_makereport(_item(pytest.mark.cuda_only("cuda-kernel")), None)
    next(wrapper)

    with pytest.raises(StopIteration):
        wrapper.send(SimpleNamespace(get_result=lambda: report))

    assert report.outcome == "failed"


def test_the_dropped_report_counts_each_file_s_collected_and_dropped_tests(tmp_path):
    def located(file):
        return SimpleNamespace(location=(file, 0, "test"))

    report = tmp_path / "dropped.csv"
    kept = [located("tests/test_a.py"), located("tests/test_b.py")]
    config = SimpleNamespace(
        getoption=lambda name: str(report),
        stash={_DROPPED: [located("tests/test_a.py"), located("tests/test_c.py")]},
    )

    pytest_collection_finish(SimpleNamespace(config=config, items=kept))

    with open(report, newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        ["file", "collected", "dropped"],
        ["tests/test_a.py", "2", "1"],
        ["tests/test_b.py", "1", "0"],
        ["tests/test_c.py", "1", "1"],
    ]


@pytest.mark.parametrize(
    "error, reason, accepted",
    [
        (RuntimeError("Found no NVIDIA driver on your system"), "cuda-kernel", True),
        (RuntimeError("CUDA error: driver unavailable"), "cuda-graph", True),
        (RuntimeError("0 active drivers ([]). There should only be one."), "cuda-reference", True),
        (RuntimeError("shape mismatch"), "cuda-kernel", False),
        (
            ValueError("No registered target detector found an available target."),
            "cuda-kernel",
            True,
        ),
        (
            ValueError("No registered target detector found an available target."),
            "cuda-reference",
            False,
        ),
        (
            ValueError("No registered target detector found an available target. Tried: broken"),
            "cuda-kernel",
            False,
        ),
        (
            subprocess.CalledProcessError(
                1,
                "python",
                stderr="ValueError: No registered target detector found an available target.\n",
            ),
            "cuda-kernel",
            True,
        ),
        (
            subprocess.CalledProcessError(
                1,
                "python",
                stderr="ValueError: No registered target detector found an available target.\n",
            ),
            "cuda-reference",
            False,
        ),
        (AssertionError("CUDA result differs"), "cuda-kernel", False),
        (OpNotAvailableError("Op's in-tree kernels do not run on test:0"), "cuda-kernel", True),
        (OpNotAvailableError("Op's in-tree kernels do not run on test:0"), "cuda-reference", False),
        (
            subprocess.CalledProcessError(1, "python", stderr="CUDA error: no device"),
            "cuda-kernel",
            True,
        ),
        (subprocess.CalledProcessError(1, "python", stderr="AssertionError"), "cuda-kernel", False),
    ],
)
def test_only_cuda_execution_failures_substantiate_a_cuda_only_mark(
    checking_claims_off_cuda, error, reason, accepted
):
    item = _item(pytest.mark.cuda_only(reason))
    pytest_collection_modifyitems([item])
    expected = item.get_closest_marker("xfail").kwargs["raises"]
    assert expected.matches(error) is accepted


def test_a_fixture_failure_does_not_substantiate_cuda_dependence(checking_claims_off_cuda):
    report = SimpleNamespace(
        skipped=True, outcome="skipped", longrepr=None, when="setup", wasxfail="needs CUDA"
    )
    wrapper = pytest_runtest_makereport(_item(pytest.mark.cuda_only("cuda-kernel")), None)
    next(wrapper)
    with pytest.raises(StopIteration):
        wrapper.send(SimpleNamespace(get_result=lambda: report))
    assert report.outcome == "failed" and not hasattr(report, "wasxfail")


def _report_of(item: SimpleNamespace, error: Exception) -> pytest.TestReport:
    """The report of *item*, whose test body raised *error*."""
    report = pytest.TestReport(item.nodeid, item.location, {}, "failed", None, "call")
    call = SimpleNamespace(
        when="call",
        excinfo=SimpleNamespace(value=error, errisinstance=lambda kind: isinstance(error, kind)),
    )
    wrapper = pytest_runtest_makereport(item, call)
    next(wrapper)
    with pytest.raises(StopIteration):
        wrapper.send(SimpleNamespace(get_result=lambda: report))
    return report


def test_a_call_the_test_target_refuses_is_unvalidated():
    report = _report_of(_item(), UnservedError("not served"))
    assert report.outcome == "skipped" and report.unvalidated
    assert report.longrepr[2] == "unvalidated: not served"


def test_a_refusal_the_shared_verifier_renames_is_still_the_test_target_s():
    try:
        raise OpNotAvailableError("RMSNormFwdOp: not served") from UnservedError("not served")
    except OpNotAvailableError as renamed:
        report = _report_of(_item(), renamed)
    assert report.outcome == "skipped" and report.unvalidated


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError("no device support"),
        OpNotAvailableError("in-tree kernels a test pinned"),
    ],
    ids=["not-a-refusal", "the-op-layer-s-refusal"],
)
def test_only_the_test_target_s_refusal_is_unvalidated(error):
    """An unsupported-operation error, or the op layer's own refusal, fails the test."""
    report = _report_of(_item(), error)
    assert report.outcome == "failed" and not hasattr(report, "unvalidated")


def test_an_unvalidated_cuda_only_test_is_not_a_failed_claim(checking_claims_off_cuda):
    report = _report_of(_item(pytest.mark.cuda_only("cuda-reference")), UnservedError("x"))
    assert report.outcome == "skipped" and report.unvalidated
