#!/usr/bin/env python3
"""Lint benchmark files for a comparison that names no baseline.

A benchmark timing only tileops reports a number with nothing to read it against,
and the rendered page shows that row with an empty comparison. A row that has no
baseline yet says why with ``FIXME(staged-rollout)`` above the comparison.

Reads the tags a call site passes, not which survive at runtime: a baseline behind
an optional import still counts, because dropping its row where the library is
missing is what the bench is meant to do. A dict built by ``**`` unpacking or
returned by a helper is decided elsewhere and left alone.

Usage: ``bench_baseline_lint.py [FILE ...]``. With no arguments, scans
``benchmarks/``. Exits 1 for a comparison with no baseline and no marker.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks"

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
    """Whether a dict display holds a key this cannot name.

    ``ast.Dict`` writes ``**expr`` as a ``None`` key; a computed key is an
    expression. Either way the tags come from somewhere else.
    """
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
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                tags.add(target.slice.value)
    return tags


def _recorded_tags(func: ast.AST) -> tuple[set[str], bool]:
    """Literal tags a ``record(..., tag=...)`` names in *func*, and whether any is opaque.

    The benchmarks that time one implementation at a time reach the report this
    way instead of through ``compare``.
    """
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


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] if argv else sorted(BENCH_DIR.rglob("*.py"))
    failed = False
    for path in paths:
        try:
            findings = unbaselined_comparisons(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for lineno in findings:
            failed = True
            print(
                f"{path}:{lineno}: comparison names no implementation but ours. Add one, "
                f"or say why the row has none with a {MARKER} block above it.",
                file=sys.stderr,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
