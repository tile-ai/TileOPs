"""Tests for what TestBase.check() establishes, and what it must refuse to.

A helper that reports a pass where it compared nothing is worse than no helper:
the op's row then carries a result that stands for nothing.
"""

import pytest
import torch

from tests.test_base import TestBase, _check_result, allclose_compare
from tileops.ops.elementwise import AbsFwdOp

pytestmark = pytest.mark.smoke


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
    _check_result.max_abs_err = None
    TestBase.check(_Reference(reference), AbsFwdOp(), torch.zeros(1), runs=lambda *_: returned)
    return getattr(_check_result, "max_abs_err", None)


def test_a_wrong_shape_carrying_right_values_is_not_close():
    """Broadcasting the two first accepted (1, 3) where the reference is (2, 3)."""
    with pytest.raises(AssertionError):
        allclose_compare(torch.ones(1, 3), torch.ones(2, 3))


def test_an_empty_result_compares_without_reducing_over_nothing():
    """`.max()` of an empty tensor raises; an empty result is still comparable."""
    empty = torch.empty(0)

    assert _compare(empty, empty) == 0.0


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


def test_a_failed_comparison_leaves_no_evidence():
    """A test may catch the AssertionError; the op is still not established."""
    with pytest.raises(AssertionError):
        _compare((torch.zeros(1),), (torch.full((1,), 999.0),))

    assert getattr(_check_result, "max_abs_err", None) is None
