"""The smoke-tier rules collection holds every parametrized ops test to: at least one smoke
case, smoke cases first, smoke cases untuned and not xfail, one smoke case per dtype, and no
full case that differs from a smoke case only by dtype."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.conftest import _freeze_value, pytest_collection_modifyitems

pytestmark = pytest.mark.smoke


def _item(
    index: int,
    *markers: str,
    tune: bool | None = False,
    dtype: torch.dtype | None = None,
    path: str = "tests/ops/test_foo.py",
) -> SimpleNamespace:
    """The ``index``-th case of ``<path>::test_op``, carrying *markers*."""
    params = {k: v for k, v in (("dtype", dtype), ("tune", tune)) if v is not None}
    return SimpleNamespace(
        nodeid=f"{path}::test_op[{index}]",
        path=Path(path),
        name=f"test_op[{index}]",
        originalname="test_op",
        iter_markers=lambda name=None: iter(()),
        user_properties=[],
        get_closest_marker=lambda name: True if name in markers else None,
        callspec=SimpleNamespace(params=params) if params else None,
    )


def test_set_with_non_comparable_values_is_sorted_stably():
    """Signatures built from set values sort stably across non-comparable types."""
    frozen = _freeze_value({torch.float16, None, 1})
    assert frozen == tuple(sorted((torch.float16, None, 1), key=str))


def test_several_smoke_cases_at_the_front_pass():
    pytest_collection_modifyitems([_item(0, "smoke"), _item(1, "smoke"), _item(2, "full")])


def test_a_test_with_no_smoke_case_fails():
    with pytest.raises(pytest.UsageError, match="smoke"):
        pytest_collection_modifyitems([_item(0, "full"), _item(1, "full")])


@pytest.mark.parametrize(
    "items",
    [
        pytest.param(lambda: [_item(0, "full"), _item(1, "smoke")], id="smoke-after-full"),
        pytest.param(
            lambda: [_item(0, "smoke"), _item(1, "full"), _item(2, "smoke")], id="smoke-split"
        ),
    ],
)
def test_smoke_cases_out_of_position_fail(items):
    with pytest.raises(pytest.UsageError, match="smoke"):
        pytest_collection_modifyitems(items())


def test_an_xfail_case_before_the_smoke_cases_is_ignored_for_ordering():
    items = [_item(0, "full", "xfail"), _item(1, "smoke"), _item(2, "full")]
    pytest_collection_modifyitems(items)


@pytest.mark.parametrize(
    "items",
    [
        pytest.param(lambda: [_item(0, "smoke", tune=True), _item(1, "full")], id="first"),
        pytest.param(
            lambda: [_item(0, "smoke"), _item(1, "smoke", tune=True), _item(2, "full")],
            id="not-first",
        ),
    ],
)
def test_every_smoke_case_is_untuned(items):
    with pytest.raises(pytest.UsageError, match="tune=False"):
        pytest_collection_modifyitems(items())


@pytest.mark.parametrize(
    "items, also",
    [
        pytest.param(
            lambda: [_item(0, "smoke", "xfail"), _item(1, "full")],
            "at least one smoke case",
            id="no-valid-smoke-case-left",
        ),
        pytest.param(
            lambda: [_item(0, "smoke", "xfail"), _item(1, "smoke"), _item(2, "full")],
            None,
            id="valid-smoke-case-at-front",
        ),
        pytest.param(
            lambda: [_item(0, "smoke", "xfail", tune=None), _item(1, "smoke", "xfail", tune=None)],
            None,
            id="no-tune-param",
        ),
    ],
)
def test_an_xfail_smoke_case_fails_without_an_ordering_error(items, also):
    with pytest.raises(pytest.UsageError) as exc_info:
        pytest_collection_modifyitems(items())
    message = str(exc_info.value)
    assert "must not be xfail" in message
    assert also is None or also in message
    assert "must appear as the first" not in message


def test_one_smoke_case_per_dtype_and_a_tuned_full_case_pass():
    items = [
        _item(0, "smoke", dtype=torch.float16),
        _item(1, "smoke", dtype=torch.bfloat16),
        _item(2, "full", dtype=torch.float16, tune=True),
    ]
    pytest_collection_modifyitems(items)


def test_a_dtype_whose_only_case_is_full_fails_both_dtype_rules():
    items = [_item(0, "smoke", dtype=torch.float16), _item(1, "full", dtype=torch.bfloat16)]
    with pytest.raises(pytest.UsageError) as exc_info:
        pytest_collection_modifyitems(items)
    assert "each dtype must have at least one smoke case" in str(exc_info.value)
    assert "must not differ from a smoke case only by dtype" in str(exc_info.value)


def test_a_nightly_model_test_passes():
    pytest_collection_modifyitems([_item(0, "nightly", path="tests/models/test_foo.py")])


def test_a_model_test_below_nightly_fails():
    with pytest.raises(pytest.UsageError, match="a test under tests/models is nightly"):
        pytest_collection_modifyitems([_item(0, "full", path="tests/models/test_foo.py")])
