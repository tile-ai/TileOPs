#!/usr/bin/env python3
"""Flag op docstrings the API reference cannot render or that leave the call undocumented.

For each class under ``src/tileops/_ops/``: each ``__init__`` and ``forward`` it defines has
a docstring without the reStructuredText the site shows literally (``.. math::``,
``:func:`` roles, ``>>>``), and a class that defines ``__init__`` has no ``Args:`` in its
own docstring, since parameters belong on ``__init__`` (docs/development.md).

In an op class, one that defines ``forward`` or whose name ends in ``Op`` or ``OpBase``,
that docstring's ``Args:`` names exactly the method's parameters, ``self`` and ``*args`` /
``**kwargs`` aside, and a ``forward`` that returns a value has ``Returns:``. A method a
class does not define is documented where it is defined.

Default scan: ``src/tileops/_ops/``.
"""

import ast
import re
import sys
from pathlib import Path

from common import files, relative, report

SCANNED = ("src/tileops/_ops",)

RST = re.compile(r"\.\. (?:math|note|warning)::|:(?:func|meth|class|mod):`|>>> ")
CLASS_ARGS = re.compile(r"^\s*Args:\s*$", re.M)
SECTION = re.compile(r"^(\s*)(\w+(?: \w+)?):\s*$")
ENTRY = re.compile(r"^(\s*)\**(\w+)\s*(?:\([^)]*\))?\s*:")


def _sections(doc: str) -> dict[str, list[str]]:
    """Each Google-style section of *doc* by its header, as its body lines."""
    found: dict[str, list[str]] = {}
    current = None
    for line in doc.splitlines():
        header = SECTION.match(line)
        if header and not header.group(1):
            current = found.setdefault(header.group(2), [])
        elif current is not None:
            current.append(line)
    return found


def _documented(body: list[str]) -> list[str]:
    """The parameter names an ``Args:`` body documents: its least-indented entries."""
    entries = [(len(m.group(1)), m.group(2)) for line in body if (m := ENTRY.match(line))]
    indent = min((i for i, _ in entries), default=0)
    return [name for i, name in entries if i == indent]


def _parameters(method: ast.FunctionDef) -> list[str]:
    """*method*'s named parameters, ``self`` aside."""
    a = method.args
    return [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)][1:]


def _returns_value(method: ast.FunctionDef) -> bool:
    """Whether *method* returns something other than None."""
    if method.returns is not None:
        return not (isinstance(method.returns, ast.Constant) and method.returns.value is None)
    return any(
        isinstance(n, ast.Return)
        and n.value is not None
        and not (isinstance(n.value, ast.Constant) and n.value.value is None)
        for n in ast.walk(method)
    )


def _call_problems(where: str, method: ast.FunctionDef, doc: str) -> list[str]:
    """What *doc* leaves out of *method*'s call: its parameters, and for ``forward`` its
    result."""
    sections = _sections(doc)
    problems = []
    parameters = _parameters(method)
    if parameters or "Args" in sections:
        documented = _documented(sections.get("Args", []))
        if documented != parameters:
            missing = [p for p in parameters if p not in documented]
            extra = [p for p in documented if p not in parameters]
            detail = "; ".join(
                part
                for part in (
                    f"missing {missing}" if missing else "",
                    f"not parameters {extra}" if extra else "",
                    "out of order" if not missing and not extra else "",
                )
                if part
            )
            problems.append(f"{where} Args: does not match the signature ({detail})")
    if method.name == "forward" and _returns_value(method) and "Returns" not in sections:
        problems.append(f"{where} returns a value and has no Returns:")
    return problems


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:  # ruff reports these; do not double up
        return [f"{path}: cannot parse ({exc})"]

    rel = relative(path)

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
        if not ({"__init__", "forward"} & methods.keys()):
            continue
        op_class = "forward" in methods or node.name.endswith(("Op", "OpBase"))

        for name in ("__init__", "forward"):
            method = methods.get(name)
            if method is None:
                continue
            doc = ast.get_docstring(method)
            if doc is None:
                problems.append(
                    f"{rel}:{method.lineno}: {node.name}.{name} has no docstring; "
                    f"the API page renders an empty entry for it"
                )
            else:
                if RST.search(doc):
                    problems.append(
                        f"{rel}:{method.lineno}: {node.name}.{name} carries "
                        f"reStructuredText; docstrings render as Markdown"
                    )
                if op_class:
                    where = f"{rel}:{method.lineno}: {node.name}.{name}"
                    problems += _call_problems(where, method, doc)

        if "__init__" in methods and CLASS_ARGS.search(ast.get_docstring(node) or ""):
            problems.append(
                f"{rel}:{node.lineno}: {node.name} documents parameters in its class "
                f"docstring; they belong on __init__"
            )
    return problems


def main(argv: list[str]) -> int:
    paths = files(argv, SCANNED, (".py",))
    return report(e for p in paths if p.exists() for e in check_file(p))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
