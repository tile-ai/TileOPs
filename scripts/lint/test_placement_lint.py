#!/usr/bin/env python3
"""Lint ``tests/ops/`` for tests that reach a kernel some way other than through its op.

A test under ``tests/ops/`` guards what an op call promises, and the op's dispatch is the
only path a caller has to a kernel (``docs/design/testing.md`` § Tests). Two forms leave
that path:

- A call that constructs a kernel: any call whose callee is named ``...Kernel``.
- A function that calls ``compare_outputs`` and reaches no op. A function reaches an op
  when it names an op class (``...Op``, or a string holding that name), a variable for
  one (``op``, ``op_cls``, ``fwd_op``), or calls a function of the same module that does.

A kernel test belongs where ``testing.md`` puts it, and a check of a reference against
itself belongs with the workload.

Usage: ``test_placement_lint.py [--base REF] [FILE ...]``. With ``--base``, a kernel call
is reported when its line is added since the merge base of REF and HEAD, and a
``compare_outputs`` caller when any of its lines is (``lint_diff.py``); the hook passes
``--base origin/main``. Without it, every finding is. With no files, scans
``tests/ops/``. Exits 1 on a finding and 2 when the base has no merge base with HEAD.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

from lint_diff import NoMergeBaseError, added_lines, relative

REPO_ROOT = Path(__file__).resolve().parents[2]

_KERNEL = re.compile(r"\w*Kernel")
_OP = re.compile(r"(?:[A-Z]\w*)?Op|(?:\w*_)?op(?:_\w+)?")
_OP_CLASS = re.compile(r"[A-Z]\w*Op")
# What both the search for op-reaching helpers and the findings walk read as a function.
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _names_an_op(function: ast.AST) -> bool:
    for node in ast.walk(function):
        # `op.forward` and `self.op` name it on either side of the dot; a parameter, by its arg.
        identifier = (
            node.attr
            if isinstance(node, ast.Attribute)
            else node.id
            if isinstance(node, ast.Name)
            else node.arg
            if isinstance(node, ast.arg)
            else None
        )
        if identifier and _OP.fullmatch(identifier):
            return True
        # An op class named by a string, as a parametrized case or a getattr does.
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _OP_CLASS.fullmatch(node.value)
        ):
            return True
    return False


def _reaching(tree: ast.Module) -> set[str]:
    """The module's functions that reach an op, directly or through each other."""
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, _FUNCTIONS) and node.col_offset == 0
    }
    reach = {name for name, node in functions.items() if _names_an_op(node)}
    grew = True
    while grew:
        grew = False
        for name, node in functions.items():
            if name in reach:
                continue
            called = {_name(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}
            if called & reach:
                reach.add(name)
                grew = True
    return reach


def _span(function: ast.AST) -> range:
    first = min([function.lineno, *(d.lineno for d in function.decorator_list)])
    return range(first, function.end_lineno + 1)


def findings(path: Path, added: set[int] | None = None) -> list[str]:
    """Placement findings in *path* on the lines in *added* (None: all)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    reaching = _reaching(tree)

    def touched(lines: range) -> bool:
        return added is None or not added.isdisjoint(lines)

    out = []
    owner: dict[int, str] = {}
    for function in (n for n in ast.walk(tree) if isinstance(n, _FUNCTIONS)):
        for node in ast.walk(function):
            owner.setdefault(id(node), function.name)
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        compare = [c for c in calls if _name(c.func) == "compare_outputs"]
        if (
            compare
            and touched(_span(function))
            and not _names_an_op(function)
            and not {_name(c.func) for c in calls} & reaching
        ):
            out.append(
                (
                    compare[0].lineno,
                    f"{path}:{compare[0].lineno}: {function.name} calls compare_outputs "
                    "without calling an op — a test under tests/ops/ checks what an op returns",
                )
            )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _name(node.func)
        if callee and _KERNEL.fullmatch(callee) and touched(range(node.lineno, node.lineno + 1)):
            where = owner.get(id(node), "<module>")
            out.append(
                (
                    node.lineno,
                    f"{path}:{node.lineno}: {where} constructs {callee} directly — a test "
                    "under tests/ops/ reaches the kernel through the op",
                )
            )
    return [message for _, message in sorted(out)]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="report only what the lines added since it touch")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    paths = [p for p in args.files if p.suffix == ".py"] or sorted(
        (REPO_ROOT / "tests" / "ops").rglob("*.py")
    )
    added: dict[str, set[int] | None] = {}
    if args.base:
        try:
            added = added_lines(paths, args.base, REPO_ROOT)
        except NoMergeBaseError as exc:
            print(f"test_placement_lint: {exc}", file=sys.stderr)
            return 2
    errors = []
    for path in paths:
        lines = added.get(relative(path, REPO_ROOT)) if args.base else None
        if args.base and lines == set():
            continue
        errors += findings(path, lines)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
