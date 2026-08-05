"""Every documented ``Args:`` key must be a real parameter.

Nothing else compares a docstring against the signature it describes: ruff's
docstring rules are off, and the manifest validator checks the signature
against the spec, not against prose. A parameter that moves out of a
constructor therefore leaves its ``Args:`` entry behind silently.

Signatures are resolved with ``inspect``, not from the source text, so a class
documenting a constructor it inherits is checked against the constructor it
actually got.
"""

import importlib
import inspect
import pathlib
import pkgutil
import re

import pytest

import tileops.ops

_SECTION = re.compile(r"^\s*(Args|Returns|Raises|Example|Examples|Attributes|Note|Notes|Yields):\s*$")
_ARG_KEY = re.compile(r"^(\s*)(\*{0,2}\w+)\s*(?:\([^)]*\))?:\s")
_OPS_DIR = pathlib.Path(tileops.ops.__file__).parent


def _documented_args(doc: str) -> list[str]:
    """Return the ``Args:`` keys of a Google-style docstring."""
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
        if len(m.group(1)) == indent:  # deeper lines describe the previous key
            keys.append(m.group(2).lstrip("*"))
    return keys


def _accepted_params(obj) -> set[str] | None:
    """Return the parameters ``obj``'s docstring is allowed to document."""
    target = obj.__init__ if inspect.isclass(obj) else obj
    if target is object.__init__:  # no constructor anywhere in the MRO
        return None
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    names = set()
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return None  # forwarded names are legitimate to document
        names.add(name)
    return names - {"self", "cls"}


def _own_docstring(obj) -> str | None:
    """Return the docstring defined on ``obj`` itself, never an inherited one."""
    if inspect.isclass(obj):
        return obj.__dict__.get("__doc__")
    return getattr(obj, "__doc__", None)


def _cases() -> list[tuple[str, list[str], set[str]]]:
    modules = []
    for info in pkgutil.walk_packages([str(_OPS_DIR)], prefix="tileops.ops."):
        modules.append(importlib.import_module(info.name))

    seen: set[str] = set()
    out: list[tuple[str, list[str], set[str]]] = []
    for mod in modules:
        for name, obj in vars(mod).items():
            if name.startswith("__"):
                continue
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            if getattr(obj, "__module__", None) != mod.__name__:
                continue  # re-export; checked where it is defined
            targets = [obj]
            if inspect.isclass(obj):
                targets += [v for v in vars(obj).values() if inspect.isfunction(v)]
            for target in targets:
                doc = _own_docstring(target)
                if not doc:
                    continue
                documented = _documented_args(doc)
                if not documented:
                    continue
                params = _accepted_params(target)
                if params is None:
                    continue
                label = f"{mod.__name__}.{getattr(target, '__qualname__', name)}"
                if label in seen:
                    continue
                seen.add(label)
                out.append((label, documented, params))
    return sorted(out)


_CASES = _cases()


@pytest.mark.smoke
def test_guard_covers_inherited_constructors() -> None:
    """A class documenting a constructor it inherits must still be checked."""
    from tileops.ops.elementwise import RoundFwdOp

    assert "__init__" not in vars(RoundFwdOp), "pick another class that inherits its ctor"
    assert any(label.endswith("RoundFwdOp") for label, _, _ in _CASES)
    params = _accepted_params(RoundFwdOp)
    assert params is not None and "dtype" not in params


@pytest.mark.smoke
@pytest.mark.parametrize(("label", "documented", "params"), _CASES, ids=[c[0] for c in _CASES])
def test_documented_args_exist(label: str, documented: list[str], params: set[str]) -> None:
    stale = [k for k in documented if k not in params]
    assert not stale, f"{label} documents non-existent parameter(s): {stale}"
