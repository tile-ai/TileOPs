#!/usr/bin/env python3
"""Lint tests and workloads for a CUDA device written outside ``cuda_only`` scope.

A test or workload places its tensors on ``workloads.device.run_device()``, the device the run
names with ``--tileops-device``. A test that needs CUDA whatever the target carries
``pytest.mark.cuda_only`` and may write ``"cuda"``; anywhere else the literal pins the suite to
CUDA without saying so.

Flags the string ``"cuda"`` or ``"cuda:<n>"``, an f-string starting with ``cuda:`` and an
argument-less ``.cuda()`` call, unless it sits inside a function or class decorated with
``pytest.mark.cuda_only``, a ``pytest.param(..., marks=...)`` carrying it, or a module whose
``pytestmark`` carries it. ``workloads/device.py`` holds the default and ``tests/conftest.py``
resolves the option, so both are exempt.

Usage: ``device_literal_lint.py [FILE ...]``. With no arguments, scans ``tests/`` and
``workloads/``. Exits 1 on a finding.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED = ("tests", "workloads")
EXEMPT = {"workloads/device.py", "tests/conftest.py"}


def _carries_mark(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Attribute) and n.attr == "cuda_only" for n in ast.walk(node))


def _literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value == "cuda" or node.value.startswith("cuda:")
    if isinstance(node, ast.JoinedStr) and node.values:
        head = node.values[0]
        return isinstance(head, ast.Constant) and str(head.value).startswith("cuda:")
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cuda"
        and not node.args
        and not node.keywords
    )


def _module_marked(tree: ast.Module) -> bool:
    return any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets)
        and _carries_mark(n.value)
        for n in tree.body
    )


def findings(source: str) -> list[int]:
    """Line numbers of CUDA device literals outside ``cuda_only`` scope."""
    tree = ast.parse(source)
    if _module_marked(tree):
        return []
    lines: list[int] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and any(
            _carries_mark(d) for d in node.decorator_list
        ):
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "param"
            and any(k.arg == "marks" and _carries_mark(k.value) for k in node.keywords)
        ):
            return
        if _literal(node):
            lines.append(node.lineno)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return sorted(set(lines))


def _files(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a).resolve() for a in argv]
    return [p for d in SCANNED for p in sorted((REPO_ROOT / d).rglob("*.py"))]


def main(argv: list[str]) -> int:
    failed = False
    for path in _files(argv):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if not rel.startswith(tuple(f"{d}/" for d in SCANNED)) or rel in EXEMPT:
            continue
        for line in findings(path.read_text()):
            failed = True
            print(
                f"{rel}:{line}: CUDA device literal; use workloads.device.run_device(), "
                "or mark the test pytest.mark.cuda_only"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
