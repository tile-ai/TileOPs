"""A class documenting a constructor it inherits.

Ruff's DOC102 compares each docstring against the signature written beside it,
so it cannot see a class whose ``Args:`` describe an ``__init__`` defined in a
base — the shape that let `RoundFwdOp` document a removed `dtype` argument
while every static check passed. Resolving the signature needs the class.
"""

import inspect
import re

import pytest

import tileops.ops.elementwise as ew

_SECTION = re.compile(r"^\s*(Args|Returns|Raises|Example|Examples|Attributes|Note|Notes|Yields):\s*$")
_ARG_KEY = re.compile(r"^(\s*)(\*{0,2}\w+)\s*(?:\([^)]*\))?:\s")


def _documented_args(doc: str) -> list[str]:
    keys: list[str] = []
    indent: int | None = None
    in_args = False
    for line in doc.splitlines():
        section = _SECTION.match(line)
        if section:
            in_args = section.group(1) == "Args"
            indent = None
            continue
        if not in_args or not line.strip():
            continue
        m = _ARG_KEY.match(line)
        if not m:
            continue
        if indent is None:
            indent = len(m.group(1))
        if len(m.group(1)) == indent:
            keys.append(m.group(2).lstrip("*"))
    return keys


def _inheriting_op_classes() -> dict:
    """Op classes with a docstring of their own and no ``__init__`` of their own."""
    found = {}
    for name in dir(ew):
        cls = getattr(ew, name)
        if not inspect.isclass(cls) or name.startswith("_"):
            continue
        if "__init__" in vars(cls) or not cls.__dict__.get("__doc__"):
            continue
        if getattr(cls, "_op_name", None) is None:
            continue
        found[name] = cls
    return found


_CASES = _inheriting_op_classes()


@pytest.mark.smoke
@pytest.mark.parametrize("name", sorted(_CASES))
def test_inherited_ctor_args_exist(name):
    cls = _CASES[name]
    sig = inspect.signature(cls.__init__)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.skip("**kwargs may legitimately document forwarded names")
    params = set(sig.parameters) - {"self", "cls"}
    stale = [k for k in _documented_args(cls.__doc__) if k not in params]
    assert not stale, f"{name} documents non-existent parameter(s): {stale}"


@pytest.mark.smoke
def test_sweep_still_sees_inheriting_classes():
    assert len(_CASES) >= 20, f"only {len(_CASES)} found; the filter broke"
    assert "RoundFwdOp" in _CASES
