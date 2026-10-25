"""Unit tests for the compatibility rules of ``scripts/ci/check_api_compat.py``."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "check_api_compat.py"
_spec = importlib.util.spec_from_file_location("check_api_compat", _SCRIPT)
check_api_compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_api_compat)

pytestmark = pytest.mark.smoke


def _fn(*params):
    return {"kind": "function", "params": [list(p) for p in params]}


X = ("x", "POSITIONAL_OR_KEYWORD", None)
EPS = ("eps", "KEYWORD_ONLY", "1e-06")


def test_an_added_name_or_defaulted_parameter_is_compatible():
    old = {"tileops.norm.f": _fn(X)}
    new = {"tileops.norm.f": _fn(X, EPS), "tileops.norm.g": _fn(X)}
    assert check_api_compat.breaks(old, new) == []


@pytest.mark.parametrize(
    "new, reason",
    [
        ({}, "removed"),
        ({"tileops.norm.f": {"kind": "class", "params": []}}, "is now a class"),
        ({"tileops.norm.f": _fn(("y", "POSITIONAL_OR_KEYWORD", None), EPS)}, "`x` removed"),
        ({"tileops.norm.f": _fn(X, ("eps", "KEYWORD_ONLY", "1e-05"))}, "default changed"),
        ({"tileops.norm.f": _fn(X, ("eps", "POSITIONAL_OR_KEYWORD", "1e-06"))}, "changed from"),
        ({"tileops.norm.f": _fn(X, EPS, ("w", "KEYWORD_ONLY", None))}, "has no default"),
    ],
)
def test_a_removal_or_changed_parameter_breaks(new, reason):
    old = {"tileops.norm.f": _fn(X, EPS)}
    found = check_api_compat.breaks(old, new)
    assert any(reason in line for line in found), found


def test_reordered_positional_parameters_break():
    a, b = ("a", "POSITIONAL_OR_KEYWORD", None), ("b", "POSITIONAL_OR_KEYWORD", None)
    found = check_api_compat.breaks({"f": _fn(a, b)}, {"f": _fn(b, a)})
    assert found == ["f: positional order changed from ['a', 'b'] to ['b', 'a']"]
