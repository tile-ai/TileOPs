"""Every documented ``Args:`` key must be a real parameter.

Nothing else compares a docstring against the signature it describes: ruff's
docstring rules are off, and the manifest validator checks the signature
against the spec, not against prose. A parameter that moves out of a
constructor therefore leaves its ``Args:`` entry behind silently.
"""

import ast
import pathlib
import re

import pytest

_OPS_ROOT = pathlib.Path(__file__).resolve().parents[2] / "tileops" / "ops"
_SECTION = re.compile(r"^\s*(Args|Returns|Raises|Example|Examples|Attributes|Note|Notes|Yields):\s*$")
_ARG_KEY = re.compile(r"^(\s*)(\*{0,2}\w+)\s*(?:\([^)]*\))?:\s")


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


def _accepted_params(node: ast.AST) -> set[str] | None:
    """Return the parameter names the docstring on ``node`` describes."""
    if isinstance(node, ast.ClassDef):  # a class docstring documents __init__
        fn = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if fn is None:
            return None
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        fn = node
    else:
        return None
    args = fn.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        return None  # **kwargs may legitimately document forwarded names
    return names - {"self", "cls"}


def _cases() -> list[tuple[str, str, list[str], set[str]]]:
    out = []
    for path in sorted(_OPS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            params = _accepted_params(node)
            if params is None:
                continue
            documented = _documented_args(doc)
            if documented:
                rel = path.relative_to(_OPS_ROOT.parents[1])
                out.append((str(rel), node.name, documented, params))
    return out


_CASES = _cases()


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("rel_path", "name", "documented", "params"),
    _CASES,
    ids=[f"{c[0]}::{c[1]}" for c in _CASES],
)
def test_documented_args_exist(rel_path: str, name: str, documented: list[str], params: set[str]) -> None:
    stale = [k for k in documented if k not in params]
    assert not stale, f"{rel_path}::{name} documents non-existent parameter(s): {stale}"
