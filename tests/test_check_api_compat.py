"""Unit tests for the compatibility rules of ``scripts/ci/check_api_compat.py``."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "check_api_compat.py"
_spec = importlib.util.spec_from_file_location("check_api_compat", _SCRIPT)
check_api_compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_api_compat)

pytestmark = pytest.mark.smoke


def _f(*params):
    return {"f": {"kind": "function", "params": [list(p) for p in params]}}


def _p(name, kind="POSITIONAL_OR_KEYWORD", default=None):
    return (name, kind, default)


A, B = _p("a"), _p("b")
EPS = _p("eps", "KEYWORD_ONLY", "1e-06")


@pytest.mark.parametrize(
    "old, new",
    [
        (_f(A), _f(A, EPS)),
        (_f(A), _f(A, _p("b", default="0"))),
        (_f(A), _f(_p("a", default="0"))),
        (_f(_p("a", "POSITIONAL_ONLY")), _f(_p("x", "POSITIONAL_ONLY"))),
        (_f(_p("args", "VAR_POSITIONAL")), _f(_p("rest", "VAR_POSITIONAL"))),
        (_f(_p("kw", "VAR_KEYWORD")), _f(_p("options", "VAR_KEYWORD"))),
        (_f(A, _p("b", "KEYWORD_ONLY")), _f(A, B)),
    ],
    ids=[
        "new-keyword-with-default",
        "new-trailing-positional-with-default",
        "required-gains-default",
        "positional-only-renamed",
        "var-positional-renamed",
        "var-keyword-renamed",
        "keyword-only-widened",
    ],
)
def test_a_change_every_old_call_survives_passes(old, new):
    assert check_api_compat.breaks(old, new) == []


@pytest.mark.parametrize(
    "old, new, reason",
    [
        (_f(A), {}, "removed"),
        (_f(A), {"f": {"kind": "class", "params": []}}, "is now a class"),
        (_f(A, B), _f(_p("x", default="0"), A, B), "slot 0 was `a`"),
        (_f(A, B), _f(B, A), "slot 0 was `a`"),
        (_f(A, EPS), _f(A), "keyword parameter `eps` removed"),
        (_f(A, EPS), _f(A, _p("eps", "KEYWORD_ONLY", "1e-05")), "default changed"),
        (_f(A, _p("b", default="0")), _f(A, B), "is required"),
        (_f(A), _f(A, _p("w", "KEYWORD_ONLY")), "is required"),
        (_f(A, _p("args", "VAR_POSITIONAL")), _f(A), "`*args` removed"),
        (_f(A, B), _f(A), "slot 1) removed"),
        (_f(A, EPS), _f(A, _p("eps", "POSITIONAL_ONLY", "1e-06")), "keyword parameter `eps`"),
    ],
    ids=[
        "name-removed",
        "kind-changed",
        "positional-inserted-before",
        "positional-reordered",
        "keyword-removed",
        "default-changed",
        "optional-became-required",
        "new-required-keyword",
        "var-positional-removed",
        "positional-removed",
        "keyword-narrowed-to-positional-only",
    ],
)
def test_a_change_that_breaks_an_old_call_is_reported(old, new, reason):
    found = check_api_compat.breaks(old, new)
    assert any(reason in line for line in found), found


def test_a_property_losing_its_setter_breaks():
    old = {"C.p": {"kind": "settable property"}}
    assert check_api_compat.breaks(old, {"C.p": {"kind": "property"}}) == [
        "C.p: was a settable property, is now a property"
    ]


def test_a_required_parameter_made_keyword_only_breaks_only_positional_calls():
    found = check_api_compat.breaks(_f(A), _f(_p("a", "KEYWORD_ONLY")))
    assert found == ["f: positional parameter `a` (slot 0) removed"]


def test_a_dataclass_field_is_a_member():
    import dataclasses

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Facts:
        arch: int

    Facts.__module__ = "tileops._probe"
    assert "Facts.arch" in check_api_compat._class_members("Facts", Facts)
