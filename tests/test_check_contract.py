"""Tests for what TestBase.check() establishes, and what it must refuse to.

A helper that reports a pass where it compared nothing is worse than no helper:
the op's row then carries a result that stands for nothing.
"""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests import test_base
from tests.test_base import TestBase, allclose_compare
from tileops.ops.elementwise import AbsFwdOp, NegFwdOp

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def check_result(monkeypatch):
    """Keep synthetic comparisons out of the recorder the pytest hook reads.

    The framework's own tests execute fixed results, not the named ops. Give
    them a private recorder for the whole test, including successive checks.
    """
    result = threading.local()
    monkeypatch.setattr(test_base, "_check_result", result)
    return result


class _Reference:
    """Supplies check() its reference and nothing else.

    Not a workload: inputs are passed to check() directly, so this never
    authors gen_inputs, which belongs to the workloads layer.
    """

    def __init__(self, outputs):
        self._outputs = outputs

    def ref_program(self, *_):
        return self._outputs


def _compare(reference, returned):
    """Compare *returned* against *reference* through check().

    The Op is incidental -- it names what the result is filed under, while
    ``runs`` supplies the values under test. Returns the error the run
    recorded, or None where it recorded no comparison.
    """
    TestBase.check(_Reference(reference), AbsFwdOp(), torch.zeros(1), runs=lambda *_: returned)
    return getattr(test_base._check_result, "max_abs_err", None)


def test_a_wrong_shape_carrying_right_values_is_not_close():
    """Broadcasting the two first accepted (1, 3) where the reference is (2, 3)."""
    with pytest.raises(AssertionError):
        allclose_compare(torch.ones(1, 3), torch.ones(2, 3))


def test_an_empty_result_compares_without_reducing_over_nothing():
    """`.max()` of an empty tensor raises; an empty result is still comparable."""
    empty = torch.empty(0)

    assert _compare(empty, empty) == 0.0


def test_framework_comparisons_do_not_publish_op_evidence():
    """The real report hook must not attribute these synthetic results to Abs."""
    from tests.conftest import pytest_runtest_call

    item = SimpleNamespace(user_properties=[])
    hook = pytest_runtest_call(item)
    next(hook)
    assert _compare(torch.ones(1), torch.ones(1)) == 0.0
    with pytest.raises(StopIteration):
        next(hook)
    assert item.user_properties == []


def test_a_reference_that_skipped_every_output_records_no_comparison():
    """`max_abs_err` is the report's evidence that a comparison ran.

    A reference returns None in an output's place when it cannot produce that
    output. Doing so for every output leaves check() green having compared
    nothing, which the report must not read as the op verified.
    """
    assert _compare((None,), (torch.ones(2),)) is None


def test_a_reference_that_checked_one_output_records_the_comparison():
    """The same call shape, with one output compared, does leave evidence."""
    assert _compare((None, torch.ones(2)), (torch.zeros(2), torch.ones(2))) == 0.0


def test_a_failed_comparison_leaves_no_evidence(check_result):
    """A test may catch the AssertionError; the op is still not established."""
    with pytest.raises(AssertionError):
        _compare((torch.zeros(1),), (torch.full((1,), 999.0),))

    assert check_result.max_abs_err is None


@pytest.mark.parametrize("outcome", ["mismatch", "shape", "arity", "unchecked"])
def test_a_later_check_cannot_inherit_another_ops_comparison(check_result, outcome):
    """A caught failure or unchecked result must not reuse the first op's evidence."""
    assert _compare(torch.ones(1), torch.ones(1)) == 0.0
    reference, returned = {
        "mismatch": (torch.zeros(1), torch.ones(1)),
        "shape": (torch.ones(1, 3), torch.ones(2, 3)),
        "arity": ((torch.ones(1), torch.ones(1)), (torch.ones(1),)),
        "unchecked": ((None,), (torch.ones(1),)),
    }[outcome]
    expected = nullcontext() if outcome == "unchecked" else pytest.raises(AssertionError)
    with expected:
        TestBase.check(_Reference(reference), NegFwdOp(), runs=lambda: returned)

    assert check_result.op_name == "NegFwdOp"
    assert check_result.op_module == NegFwdOp.__module__
    assert check_result.max_abs_err is None


@pytest.mark.parametrize("stage", ["reference", "execution"])
def test_an_execution_failure_replaces_the_previous_attribution(check_result, stage):
    """Failures before comparison belong to the new op and establish no values."""
    assert _compare(torch.ones(1), torch.ones(1)) == 0.0

    def fail(*_):
        raise RuntimeError("computation failed")

    class Reference(_Reference):
        def ref_program(self, *_):
            return fail() if stage == "reference" else super().ref_program()

    with pytest.raises(RuntimeError, match="computation failed"):
        TestBase.check(Reference(torch.ones(1)), NegFwdOp(), runs=fail)

    assert check_result.op_name == "NegFwdOp"
    assert check_result.max_abs_err is None


def test_reference_oom_skips_without_comparison_evidence(check_result):
    """An infeasible oracle must produce a skip, not a successful comparison."""

    class Reference(_Reference):
        def ref_program(self, *_):
            raise torch.OutOfMemoryError("out of memory")

    with pytest.raises(pytest.skip.Exception, match="reference ran out of memory"):
        TestBase.check(Reference(None), AbsFwdOp())

    assert check_result.max_abs_err is None


def test_non_op_is_refused_before_running_the_reference():
    """A raw callable belongs in runs= and cannot own an op's report row."""

    class Reference(_Reference):
        def ref_program(self, *_):
            pytest.fail("the reference must not run for an invalid owner")

    with pytest.raises(AssertionError, match="pass what to execute as runs="):
        TestBase.check(Reference(None), lambda: torch.ones(1))
