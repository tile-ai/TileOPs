"""Tests for smoke-param tier validation in conftest.py.

Verifies:
- Zero smoke params still fail.
- Multiple smoke params pass validation.
- Smoke params must appear as first N non-xfail cases; ordering
  violation raises pytest.UsageError.
- tune=False and no-xfail constraints apply to every smoke case.
- Per-dtype smoke coverage and the full-not-dtype-only rule.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.conftest import _freeze_value, pytest_collection_modifyitems

# Helpers to build mock pytest.Items


def _make_item(
    *,
    name: str = "test_op",
    originalname: str = "test_op",
    path: str = "tests/ops/test_foo.py",
    markers: list[str] | None = None,
    dtype: object | None = None,
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
    if dtype is not None:
        params["dtype"] = dtype
    if tune is not None:
        params["tune"] = tune

    if params:
        item.callspec = SimpleNamespace(params=params)
    else:
        item.callspec = None

    return item


@pytest.mark.full
class TestFreezeValue:
    """Signatures built from set values must sort stably across non-comparable types."""

    def test_set_with_non_comparable_values_is_sorted_stably(self):
        frozen = _freeze_value({torch.float16, None, 1})
        assert frozen == tuple(sorted((torch.float16, None, 1), key=str))


@pytest.mark.full
class TestSmokeCount:
    """At least one smoke param is required per ops test function."""

    def test_zero_smoke_raises(self):
        items = [
            _make_item(markers=["full"], tune=False),
            _make_item(markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="smoke"):
            pytest_collection_modifyitems(items)

    def test_multiple_smoke_params_pass(self):
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=False),
            _make_item(name="test_op[2]", markers=["full"], tune=False),
        ]
        # Should not raise
        pytest_collection_modifyitems(items)


@pytest.mark.full
class TestSmokeOrdering:
    """Smoke cases must be contiguous at the front of non-xfail items."""

    def test_smoke_out_of_position_raises(self):
        """Smoke after full, or smoke split by a full param, is invalid."""
        item_sets = [
            # A smoke param appearing after a non-xfail full param.
            [
                _make_item(name="test_op[0]", markers=["full"], tune=False),
                _make_item(name="test_op[1]", markers=["smoke"], tune=False),
            ],
            # Smoke params with a full param in between.
            [
                _make_item(name="test_op[0]", markers=["smoke"], tune=False),
                _make_item(name="test_op[1]", markers=["full"], tune=False),
                _make_item(name="test_op[2]", markers=["smoke"], tune=False),
            ],
        ]
        for items in item_sets:
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


@pytest.mark.full
class TestSmokeConstraints:
    """Every smoke case must have tune=False and must not be xfail."""

    def test_smoke_tune_true_raises(self):
        """The tune=False constraint applies to ALL smoke cases, not just first."""
        item_sets = [
            [
                _make_item(name="test_op[0]", markers=["smoke"], tune=True),
                _make_item(name="test_op[1]", markers=["full"], tune=False),
            ],
            [
                _make_item(name="test_op[0]", markers=["smoke"], tune=False),
                _make_item(name="test_op[1]", markers=["smoke"], tune=True),
                _make_item(name="test_op[2]", markers=["full"], tune=False),
            ],
        ]
        for items in item_sets:
            with pytest.raises(pytest.UsageError, match="tune=False"):
                pytest_collection_modifyitems(items)

    def test_smoke_xfail_raises(self):
        """The no-xfail constraint applies to ALL smoke cases, with or
        without a tune param."""
        item_sets = [
            [
                _make_item(name="test_op[0]", markers=["smoke", "xfail"], tune=False),
                _make_item(name="test_op[1]", markers=["full"], tune=False),
            ],
            [
                _make_item(name="test_op[0]", markers=["smoke"], tune=False),
                _make_item(name="test_op[1]", markers=["smoke", "xfail"], tune=False),
                _make_item(name="test_op[2]", markers=["full"], tune=False),
            ],
            [
                _make_item(name="test_op[0]", markers=["smoke", "xfail"], tune=None),
                _make_item(name="test_op[1]", markers=["smoke", "xfail"], tune=None),
            ],
        ]
        for items in item_sets:
            with pytest.raises(pytest.UsageError, match="xfail"):
                pytest_collection_modifyitems(items)

    def test_smoke_xfail_reports_only_xfail_error(self):
        """A smoke+xfail case must raise the xfail rejection without a
        spurious ordering error, both when it leaves the group with no
        valid smoke case and when a valid smoke case sits at the front."""
        # No valid smoke case remains: xfail + missing-smoke errors only.
        items = [
            _make_item(name="test_op[0]", markers=["smoke", "xfail"], tune=False),
            _make_item(name="test_op[1]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="xfail") as exc_info:
            pytest_collection_modifyitems(items)
        assert "must not be xfail" in str(exc_info.value)
        assert "at least one smoke case" in str(exc_info.value)
        assert "must appear as the first" not in str(exc_info.value)

        # Valid smoke correctly at front: only the xfail rejection fires.
        items = [
            _make_item(name="test_op[0]", markers=["smoke", "xfail"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=False),
            _make_item(name="test_op[2]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="xfail") as exc_info:
            pytest_collection_modifyitems(items)
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


@pytest.mark.smoke
class TestPerDtypeSmokeCoverage:
    """Every dtype present in parametrized ops tests must have a smoke case."""

    def test_each_dtype_has_smoke_case(self):
        items = [
            _make_item(
                name="test_op[fp16]",
                markers=["smoke"],
                dtype=torch.float16,
                tune=False,
            ),
            _make_item(
                name="test_op[bf16]",
                markers=["smoke"],
                dtype=torch.bfloat16,
                tune=False,
            ),
            _make_item(
                name="test_op[fp32]",
                markers=["smoke"],
                dtype=torch.float32,
                tune=False,
            ),
            _make_item(
                name="test_op[fp16-full]",
                markers=["full"],
                dtype=torch.float16,
                tune=True,
            ),
        ]
        pytest_collection_modifyitems(items)

    def test_missing_dtype_smoke_raises(self):
        items = [
            _make_item(
                name="test_op[fp16]",
                markers=["smoke"],
                dtype=torch.float16,
                tune=False,
            ),
            _make_item(
                name="test_op[bf16]",
                markers=["full"],
                dtype=torch.bfloat16,
                tune=False,
            ),
        ]
        with pytest.raises(pytest.UsageError, match="each dtype must have at least one smoke"):
            pytest_collection_modifyitems(items)


@pytest.mark.smoke
class TestFullNotDtypeOnly:
    """A full case cannot duplicate a smoke case except for dtype."""

    def test_full_with_same_signature_except_dtype_raises(self):
        items = [
            _make_item(
                name="test_op[fp16]",
                markers=["smoke"],
                dtype=torch.float16,
                tune=False,
            ),
            _make_item(
                name="test_op[bf16]",
                markers=["full"],
                dtype=torch.bfloat16,
                tune=False,
            ),
        ]
        with pytest.raises(
            pytest.UsageError,
            match="must not differ from a smoke case only by dtype",
        ):
            pytest_collection_modifyitems(items)

    def test_full_with_distinct_non_dtype_params_passes(self):
        items = [
            _make_item(
                name="test_op[fp16-typical]",
                markers=["smoke"],
                dtype=torch.float16,
                tune=False,
            ),
            _make_item(
                name="test_op[bf16-typical]",
                markers=["smoke"],
                dtype=torch.bfloat16,
                tune=False,
            ),
            _make_item(
                name="test_op[fp16-tuned]",
                markers=["full"],
                dtype=torch.float16,
                tune=True,
            ),
        ]
        pytest_collection_modifyitems(items)
