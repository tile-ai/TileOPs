"""Tests for smoke-param tier validation in conftest.py.

Verifies:
- AC-1: Zero smoke params still fail.
- AC-2: Multiple smoke params pass validation.
- AC-3: Smoke params must appear as first N non-xfail cases; ordering
         violation raises pytest.UsageError.
- AC-4: Existing single-smoke fixtures pass unchanged (backward compat).
- AC-5: tune=False and no-xfail constraints apply to every smoke case.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.conftest import pytest_collection_modifyitems

# ---------------------------------------------------------------------------
# Helpers to build mock pytest.Items
# ---------------------------------------------------------------------------


def _make_item(
    *,
    name: str = "test_op",
    originalname: str = "test_op",
    path: str = "tests/ops/test_foo.py",
    markers: list[str] | None = None,
    tune: bool | None = None,
) -> MagicMock:
    """Build a lightweight mock pytest.Item for tier validation tests."""
    markers = markers or []
    item = MagicMock(spec=["nodeid", "path", "name", "originalname",
                           "get_closest_marker", "callspec"])
    item.nodeid = f"{path}::{name}"
    item.path = Path(path)
    item.name = name
    item.originalname = originalname

    marker_set = set(markers)

    def _get_closest_marker(marker_name: str):
        if marker_name in marker_set:
            return True  # truthy sentinel
        return None

    item.get_closest_marker = _get_closest_marker

    params: dict[str, object] = {}
    if tune is not None:
        params["tune"] = tune

    if params:
        item.callspec = SimpleNamespace(params=params)
    else:
        item.callspec = None

    return item


# ===================================================================
# AC-1: Zero smoke params still fail
# ===================================================================


@pytest.mark.full
class TestZeroSmokeFails:
    """At least one smoke param is required per ops test function."""

    def test_zero_smoke_raises(self):
        items = [
            _make_item(markers=["full"], tune=False),
            _make_item(markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="smoke"):
            pytest_collection_modifyitems(items)


# ===================================================================
# AC-2: Multiple smoke params pass validation
# ===================================================================


@pytest.mark.full
class TestMultiSmokePasses:
    """Tests with >1 smoke params must pass tier validation."""

    def test_two_smoke_params_pass(self):
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=False),
            _make_item(name="test_op[2]", markers=["full"], tune=False),
        ]
        # Should not raise
        pytest_collection_modifyitems(items)

    def test_three_smoke_params_pass(self):
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=False),
            _make_item(name="test_op[2]", markers=["smoke"], tune=False),
            _make_item(name="test_op[3]", markers=["full"], tune=False),
        ]
        # Should not raise
        pytest_collection_modifyitems(items)


# ===================================================================
# AC-3: Smoke params must appear as the first N non-xfail cases
# ===================================================================


@pytest.mark.full
class TestSmokeOrdering:
    """Smoke cases must be contiguous at the front of non-xfail items."""

    def test_smoke_after_full_raises(self):
        """A smoke param appearing after a non-xfail full param is invalid."""
        items = [
            _make_item(name="test_op[0]", markers=["full"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="smoke"):
            pytest_collection_modifyitems(items)

    def test_smoke_gap_raises(self):
        """Smoke params with a full param in between are invalid."""
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=False),
            _make_item(name="test_op[1]", markers=["full"], tune=False),
            _make_item(name="test_op[2]", markers=["smoke"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="smoke"):
            pytest_collection_modifyitems(items)

    def test_xfail_before_smoke_ok(self):
        """xfail items before smoke are ignored for ordering purposes."""
        items = [
            _make_item(name="test_op[0]", markers=["full", "xfail"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=False),
            _make_item(name="test_op[2]", markers=["full"], tune=False),
        ]
        # Should not raise -- the xfail item is excluded from ordering check
        pytest_collection_modifyitems(items)

    def test_multi_smoke_correct_order_passes(self):
        """Multiple smoke at the front, then full -- valid."""
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=False),
            _make_item(name="test_op[2]", markers=["full"], tune=False),
            _make_item(name="test_op[3]", markers=["full"], tune=False),
        ]
        pytest_collection_modifyitems(items)


# ===================================================================
# AC-4: Backward compatibility -- single smoke still works
# ===================================================================


@pytest.mark.full
class TestSingleSmokeBackwardCompat:
    """Existing single-smoke test fixtures must pass unchanged."""

    def test_single_smoke_first(self):
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=False),
            _make_item(name="test_op[1]", markers=["full"], tune=False),
            _make_item(name="test_op[2]", markers=["full"], tune=True),
        ]
        # Should not raise
        pytest_collection_modifyitems(items)


# ===================================================================
# AC-5: tune=False and no-xfail constraints for every smoke case
# ===================================================================


@pytest.mark.full
class TestSmokeConstraints:
    """Every smoke case must have tune=False and must not be xfail."""

    def test_smoke_tune_true_raises(self):
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=True),
            _make_item(name="test_op[1]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="tune=False"):
            pytest_collection_modifyitems(items)

    def test_smoke_xfail_raises(self):
        items = [
            _make_item(name="test_op[0]", markers=["smoke", "xfail"], tune=False),
            _make_item(name="test_op[1]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="xfail"):
            pytest_collection_modifyitems(items)

    def test_second_smoke_tune_true_raises(self):
        """The tune=False constraint applies to ALL smoke cases, not just first."""
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=True),
            _make_item(name="test_op[2]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="tune=False"):
            pytest_collection_modifyitems(items)

    def test_second_smoke_xfail_raises(self):
        """The no-xfail constraint applies to ALL smoke cases."""
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke", "xfail"], tune=False),
            _make_item(name="test_op[2]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="xfail"):
            pytest_collection_modifyitems(items)

    def test_smoke_xfail_no_tune_raises(self):
        """Regression: smoke+xfail without a tune param must still be rejected."""
        items = [
            _make_item(name="test_op[0]", markers=["smoke", "xfail"], tune=None),
            _make_item(name="test_op[1]", markers=["smoke", "xfail"], tune=None),
        ]
        with pytest.raises(pytest.UsageError, match="xfail"):
            pytest_collection_modifyitems(items)

    def test_smoke_xfail_then_full_only_xfail_error(self):
        """Regression: smoke+xfail followed by full should only raise xfail error,
        not a spurious ordering error."""
        items = [
            _make_item(name="test_op[0]", markers=["smoke", "xfail"], tune=False),
            _make_item(name="test_op[1]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="xfail") as exc_info:
            pytest_collection_modifyitems(items)
        # Must also report missing smoke (no valid smoke cases), but NOT ordering
        assert "must not be xfail" in str(exc_info.value)
        assert "at least one smoke case" in str(exc_info.value)
        assert "must appear as the first" not in str(exc_info.value)

    def test_smoke_xfail_valid_smoke_full_only_xfail_error(self):
        """Regression: [smoke+xfail, smoke, full] should raise xfail error but
        not a spurious ordering error -- valid smoke is correctly at front."""
        items = [
            _make_item(name="test_op[0]", markers=["smoke", "xfail"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=False),
            _make_item(name="test_op[2]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="xfail") as exc_info:
            pytest_collection_modifyitems(items)
        # Only the xfail rejection should fire; ordering is fine
        assert "must not be xfail" in str(exc_info.value)
        assert "must appear as the first" not in str(exc_info.value)


@pytest.mark.full
class TestNonRuntimeOpsFileExemption:
    """Explicitly exempted non-runtime ops files may be full-only."""

    def test_exempt_ops_file_may_have_zero_smoke(self):
        items = [
            _make_item(
                name="test_compile[0]",
                originalname="test_compile",
                path="tests/ops/test_elementwise_compile.py",
                markers=["full"],
                tune=False,
            ),
            _make_item(
                name="test_compile[1]",
                originalname="test_compile",
                path="tests/ops/test_elementwise_compile.py",
                markers=["full"],
                tune=True,
            ),
        ]

        pytest_collection_modifyitems(items)
