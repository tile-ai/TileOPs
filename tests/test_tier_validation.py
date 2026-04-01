"""Tests for tier validation in tests/conftest.py.

Covers two layers of behavior:
- generic per-test tier constraints that still apply across tests/ops
- canonical smoke normalization for public Ops exported from tileops/ops/__init__.py
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tests.conftest as tier_conftest
from tests.conftest import _normalize_public_op_smoke, pytest_collection_modifyitems

pytestmark = pytest.mark.full


@pytest.fixture
def disable_public_op_smoke_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tier_conftest,
        "_normalize_public_op_smoke",
        lambda items, tier_errors: None,
    )


@pytest.fixture
def stub_public_op_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tier_conftest, "_load_public_op_names", lambda: ["ExampleOp"])
    monkeypatch.setattr(
        tier_conftest,
        "PUBLIC_OP_SMOKE_TARGETS",
        {"ExampleOp": ("tests/ops/test_example.py", "test_example_op")},
    )


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
    autotune: bool | None = None,
) -> MagicMock:
    """Build a lightweight mock pytest.Item for tier validation tests."""
    markers = markers or []
    item = MagicMock(
        spec=[
            "nodeid",
            "path",
            "name",
            "originalname",
            "get_closest_marker",
            "callspec",
            "own_markers",
            "add_marker",
            "parent",
        ]
    )
    item.nodeid = f"{path}::{name}"
    item.path = Path(path)
    item.name = name
    item.originalname = originalname

    item.own_markers = [SimpleNamespace(name=marker_name) for marker_name in markers]
    item.parent = SimpleNamespace(
        own_markers=[],
        parent=SimpleNamespace(own_markers=[], parent=None),
    )

    def _get_closest_marker(marker_name: str):
        for marker in item.own_markers:
            if marker.name == marker_name:
                return marker
        return None

    def _add_marker(marker):
        marker_name = getattr(getattr(marker, "mark", None), "name", None)
        if marker_name is None:
            marker_name = getattr(marker, "name", None)
        if marker_name is None:
            raise AssertionError(f"unsupported marker object: {marker!r}")
        item.own_markers.append(SimpleNamespace(name=marker_name))

    item.get_closest_marker = _get_closest_marker
    item.add_marker = _add_marker

    params: dict[str, bool] = {}
    if tune is not None:
        params["tune"] = tune
    if autotune is not None:
        params["autotune"] = autotune
    item.callspec = SimpleNamespace(params=params) if params else None
    return item


class TestGenericOpsTierValidation:
    def test_zero_smoke_is_allowed_when_not_a_canonical_target(
        self, disable_public_op_smoke_normalization: None
    ):
        items = [
            _make_item(markers=["full"], tune=False),
            _make_item(markers=["full"], tune=False),
        ]
        pytest_collection_modifyitems(items)

    def test_smoke_after_full_does_not_raise_without_ordering_rule(
        self, disable_public_op_smoke_normalization: None
    ):
        items = [
            _make_item(name="test_op[0]", markers=["full"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=False),
        ]
        pytest_collection_modifyitems(items)

    def test_smoke_xfail_is_allowed_for_noncanonical_items(
        self, disable_public_op_smoke_normalization: None
    ):
        items = [
            _make_item(name="test_op[0]", markers=["smoke", "xfail"], tune=False),
            _make_item(name="test_op[1]", markers=["full"], tune=False),
        ]
        pytest_collection_modifyitems(items)

    def test_smoke_tune_true_still_raises(
        self, disable_public_op_smoke_normalization: None
    ):
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=True),
            _make_item(name="test_op[1]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="tune=False"):
            pytest_collection_modifyitems(items)

    def test_second_smoke_tune_true_still_raises(
        self, disable_public_op_smoke_normalization: None
    ):
        items = [
            _make_item(name="test_op[0]", markers=["smoke"], tune=False),
            _make_item(name="test_op[1]", markers=["smoke"], tune=True),
            _make_item(name="test_op[2]", markers=["full"], tune=False),
        ]
        with pytest.raises(pytest.UsageError, match="tune=False"):
            pytest_collection_modifyitems(items)


class TestCanonicalPublicOpNormalization:
    def test_selects_single_autotune_false_candidate(
        self, stub_public_op_mapping: None
    ):
        items = [
            _make_item(
                name="test_example_op[param0]",
                originalname="test_example_op",
                path="tests/ops/test_example.py",
                markers=["full"],
                autotune=True,
            ),
            _make_item(
                name="test_example_op[param1]",
                originalname="test_example_op",
                path="tests/ops/test_example.py",
                markers=["full"],
                autotune=False,
            ),
            _make_item(
                name="test_other_op[param0]",
                originalname="test_other_op",
                path="tests/ops/test_other.py",
                markers=["smoke"],
                tune=False,
            ),
        ]
        tier_errors: list[str] = []

        _normalize_public_op_smoke(items, tier_errors)

        assert tier_errors == []
        assert items[0].get_closest_marker("smoke") is None
        assert items[0].get_closest_marker("full") is not None
        assert items[1].get_closest_marker("smoke") is not None
        assert items[2].get_closest_marker("smoke") is None
        assert items[2].get_closest_marker("full") is not None

    def test_missing_candidate_reports_error(self, stub_public_op_mapping: None):
        items = [
            _make_item(
                name="test_other_op[param0]",
                originalname="test_other_op",
                path="tests/ops/test_other.py",
                markers=["full"],
                tune=False,
            )
        ]
        tier_errors: list[str] = []

        _normalize_public_op_smoke(items, tier_errors)

        assert any("expected at least one non-xfail item" in error for error in tier_errors)
        assert any("expected exactly one smoke case, found 0" in error for error in tier_errors)

    def test_all_autotuned_candidates_report_error(self, stub_public_op_mapping: None):
        items = [
            _make_item(
                name="test_example_op[param0]",
                originalname="test_example_op",
                path="tests/ops/test_example.py",
                markers=["full"],
                autotune=True,
            ),
            _make_item(
                name="test_example_op[param1]",
                originalname="test_example_op",
                path="tests/ops/test_example.py",
                markers=["full"],
                tune=True,
            ),
        ]
        tier_errors: list[str] = []

        _normalize_public_op_smoke(items, tier_errors)

        assert any("use autotune/tune=True" in error for error in tier_errors)
        assert any("expected exactly one smoke case, found 0" in error for error in tier_errors)
