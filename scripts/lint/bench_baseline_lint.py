#!/usr/bin/env python3
"""Flag a benchmark comparison that times only tileops.

A comparison passes when it names a tag that does not start with ``tileops``, through
``compare({...})`` or ``record(..., tag=...)``. Tags this cannot read (``**`` unpacking, a
computed key, a helper's return value) pass.

Exempt: ``FIXME(staged-rollout)`` in the function, before the ``compare`` it waives, or
anywhere in a function that only calls ``record``. Default scan: ``benchmarks/``.
"""

import ast
import sys
from pathlib import Path

from common import files, relative, report

SCANNED = ("benchmarks",)

OURS = "tileops"  # tags start with it: "tileops", "tileops-nopad-3wg", f"tileops_{variant}"
MARKER = "FIXME(staged-rollout)"


def _theirs(tags: set[str]) -> set[str]:
    """The tags naming somebody else's implementation."""
    return {t for t in tags if not t.startswith(OURS)}


def _literal_keys(node: ast.AST) -> set[str]:
    """String keys of a dict display."""
    if not isinstance(node, ast.Dict):
        return set()
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _unreadable(node: ast.AST) -> bool:
    """Whether a dict display holds a ``**`` unpacking or a computed key."""
    return isinstance(node, ast.Dict) and any(
        k is None or not (isinstance(k, ast.Constant) and isinstance(k.value, str))
        for k in node.keys
    )


def _tags_of(name: str, func: ast.AST, before: int) -> set[str] | None:
    """Tags assigned to *name* above line *before*, or ``None`` if unreadable."""
    tags: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or getattr(node, "lineno", 0) > before:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                if _unreadable(node.value):
                    return None
                tags |= _literal_keys(node.value)
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
            ):
                if not (
                    isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str)
                ):
                    return None  # a computed key
                tags.add(target.slice.value)
    return tags


def _recorded_tags(func: ast.AST) -> tuple[set[str], bool]:
    """Literal tags a ``record(..., tag=...)`` names in *func*, and whether any is opaque."""
    tags: set[str] = set()
    opaque = False
    for node in ast.walk(func):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "record"
        ):
            continue
        for kw in node.keywords:
            if kw.arg != "tag":
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                tags.add(kw.value.value)
            else:
                opaque = True
    return tags, opaque


def unbaselined_comparisons(text: str) -> list[int]:
    """Line of every comparison that names no baseline and carries no marker."""
    tree = ast.parse(text)
    marked = [i for i, line in enumerate(text.split("\n"), start=1) if MARKER in line]
    findings = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        compares = [
            n
            for n in ast.walk(func)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "compare"
        ]
        if not compares:
            tags, opaque = _recorded_tags(func)
            if (
                tags
                and not opaque
                and not _theirs(tags)
                and not any(func.lineno <= m <= func.end_lineno for m in marked)
            ):
                findings.append(func.lineno)
        for node in ast.walk(func):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "compare"
                and node.args
            ):
                continue
            if any(func.lineno <= m <= node.lineno for m in marked):
                continue  # the row states why it has none
            arg = node.args[0]
            if isinstance(arg, ast.Dict):
                tags = None if _unreadable(arg) else _literal_keys(arg)
            elif isinstance(arg, ast.Name):
                tags = _tags_of(arg.id, func, node.lineno)
            else:
                continue  # a call or comprehension builds it
            if tags and not _theirs(tags):
                findings.append(node.lineno)
    return findings


def findings(path: Path) -> list[str]:
    """One finding per comparison in *path* that names no baseline and carries no marker."""
    try:
        lines = unbaselined_comparisons(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    return [
        f"{relative(path)}:{lineno}: comparison names no implementation but ours — add one, "
        f"or say why the row has none with a {MARKER} block above it"
        for lineno in lines
    ]


def main(argv: list[str]) -> int:
    return report(e for p in files(argv, SCANNED, (".py",)) for e in findings(p))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
