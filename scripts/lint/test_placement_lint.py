#!/usr/bin/env python3
"""Lint ``tests/ops/`` for tests that reach a kernel some way other than through its op.

A test under ``tests/ops/`` guards what an op call promises, and the op's dispatch is the
only path a caller has to a kernel (``docs/design/testing.md`` § Tests). Two forms leave
that path:

- A call that constructs a kernel: any call whose callee is named ``...Kernel``.
- A function that calls ``compare_outputs`` and reaches no op. A function reaches an op
  when it invokes an op instance, invokes an instance it constructs from an ``...Op``
  class or from a class ``getattr`` reads by name off a module the file imports from
  ``tileops``, or calls a function of the same module that does.

A kernel test belongs where ``testing.md`` puts it, and a check of a reference against
itself belongs with the workload. An exceptional line carries
``lint: test-placement allow: <reason>`` on that same line.

Usage: ``test_placement_lint.py [--base REF] [FILE ...]``. With ``--base``, a kernel call
is reported when its line is added since the merge base of REF and HEAD, and a
``compare_outputs`` caller when any of its lines is (``lint_diff.py``); the hook passes
``--base auto`` to find the TileOPs upstream, overridable with ``TILEOPS_LINT_BASE``.
Without it, every finding is. With no files, scans
``tests/ops/``. Exits 1 on a finding and 2 when the base has no merge base with HEAD.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

from comment_lines import allowed_lines
from lint_diff import BASE_ENV, NoMergeBaseError, added_lines, relative, resolve_base

REPO_ROOT = Path(__file__).resolve().parents[2]

_KERNEL = re.compile(r"\w*Kernel")
_OP_CLASS = re.compile(r"[A-Z]\w*Op")
_OP_VALUE = re.compile(r"(?:fwd_|bwd_)?op(?:_cls)?")
# What both the search for op-reaching helpers and the findings walk read as a function.
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _binding_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef):
    """Nodes whose bindings belong to *function*, excluding nested scopes."""
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        yield node
        for child in reversed(list(ast.iter_child_nodes(node))):
            if isinstance(child, (*_FUNCTIONS, ast.ClassDef, ast.Lambda)):
                continue
            pending.append(child)


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return prefix and f"{prefix}.{node.attr}"
    return None


def _is_tileops(module: str) -> bool:
    return module == "tileops" or module.startswith("tileops.")


def _op_modules(tree: ast.Module) -> set[str]:
    """Names and dotted paths that hold a TileOPs module everywhere *tree* binds them.

    A TileOPs module is ``tileops`` or any module below it. A name qualifies when every
    binding of it in the file is an import of a TileOPs module, or an assignment whose every
    ``if``/``else`` branch is a qualifying name.
    """
    bindings: dict[str, list[ast.AST | bool]] = {}

    def bind(names, value: ast.AST | bool) -> None:
        for name in names:
            bindings.setdefault(name, []).append(value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bind([alias.asname or alias.name], _is_tileops(alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module if not node.level else None
            for alias in node.names:
                bind([alias.asname or alias.name], bool(module) and _is_tileops(module))
        elif isinstance(node, ast.Assign):
            bind({name for target in node.targets for name in _targets(target)}, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(_targets(node.target), node.value)
        elif isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
            bind([node.name], False)
        elif isinstance(node, ast.arg):
            bind([node.arg], False)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            bind(_targets(node.target), False)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bind(_targets(node.optional_vars), False)

    def qualifies(value: ast.AST | bool, modules: set[str]) -> bool:
        if isinstance(value, ast.IfExp):
            return qualifies(value.body, modules) and qualifies(value.orelse, modules)
        return value if isinstance(value, bool) else _dotted(value) in modules

    modules: set[str] = set()
    grew = True
    while grew:
        grew = False
        for name, values in bindings.items():
            if name not in modules and all(qualifies(v, modules) for v in values):
                modules.add(name)
                grew = True
    return modules


def _getattr_names_op(call: ast.AST, modules: set[str]) -> bool:
    """Whether *call* is ``getattr`` that yields an op class.

    It does when its attribute is a string constant naming an ``...Op`` class, or when its
    object is a TileOPs module in *modules* and its attribute is ``name`` or ``op_name``.
    """
    return (
        isinstance(call, ast.Call)
        and _name(call.func) == "getattr"
        and len(call.args) >= 2
        and (
            isinstance(call.args[1], ast.Name)
            and call.args[1].id in {"name", "op_name"}
            and _dotted(call.args[0]) in modules
            or isinstance(call.args[1], ast.Constant)
            and isinstance(call.args[1].value, str)
            and bool(_OP_CLASS.fullmatch(call.args[1].value))
        )
    )


def _op_factory(
    expr: ast.AST, factories: set[str], modules: set[str], op_classes: set[str] | None = None
) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    callee = _name(expr.func)
    if callee and (
        _OP_CLASS.fullmatch(callee)
        or _OP_VALUE.fullmatch(callee)
        or op_classes is not None
        and callee in op_classes
    ):
        return True
    if callee in factories:
        return True
    return isinstance(expr.func, ast.Call) and _getattr_names_op(expr.func, modules)


def _contains_op_factory(
    expr: ast.AST, factories: set[str], modules: set[str], op_classes: set[str] | None = None
) -> bool:
    if _op_factory(expr, factories, modules, op_classes):
        return True
    if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        elements = expr.elts
    elif isinstance(expr, ast.Dict):
        elements = expr.values
    elif isinstance(expr, ast.IfExp):
        elements = [expr.body, expr.orelse]
    else:
        return False
    return any(
        _contains_op_factory(element, factories, modules, op_classes) for element in elements
    )


def _targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return {name for element in node.elts for name in _targets(element)}
    return set()


def _op_factories(functions: dict[str, ast.AST], modules: set[str]) -> set[str]:
    """Functions that return an op instance, directly or through another factory."""
    factories = set()
    grew = True
    while grew:
        grew = False
        for name, function in functions.items():
            if name in factories:
                continue
            if any(
                isinstance(node, ast.Return)
                and node.value is not None
                and _contains_op_factory(node.value, factories, modules)
                for node in _binding_nodes(function)
            ):
                factories.add(name)
                grew = True
    return factories


def _calls_an_op(
    function: ast.FunctionDef | ast.AsyncFunctionDef, factories: set[str], modules: set[str]
) -> bool:
    binding_nodes = list(_binding_nodes(function))
    call_nodes = list(ast.walk(function))
    non_dispatch_calls = {
        "compare_outputs",
        "getattr",
        "hasattr",
        "id",
        "isinstance",
        "len",
        "print",
        "repr",
        "str",
        "type",
    }
    op_classes = set()
    for node in binding_nodes:
        if isinstance(node, ast.Assign) and _getattr_names_op(node.value, modules):
            op_classes |= {name for target in node.targets for name in _targets(target)}
        elif isinstance(node, ast.AnnAssign) and _getattr_names_op(node.value, modules):
            op_classes |= _targets(node.target)
    op_values = {
        arg.arg
        for args in (function.args.posonlyargs, function.args.args, function.args.kwonlyargs)
        for arg in args
        if _OP_VALUE.fullmatch(arg.arg)
    }
    op_values |= {
        node.func.id
        for node in call_nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and _OP_VALUE.fullmatch(node.func.id)
    }
    if function.args.vararg and _OP_VALUE.fullmatch(function.args.vararg.arg):
        op_values.add(function.args.vararg.arg)
    if function.args.kwarg and _OP_VALUE.fullmatch(function.args.kwarg.arg):
        op_values.add(function.args.kwarg.arg)
    grew = True
    while grew:
        grew = False
        for node in binding_nodes:
            if isinstance(node, ast.Assign) and _contains_op_factory(
                node.value, factories, modules, op_classes
            ):
                targets = {name for target in node.targets for name in _targets(target)}
            elif isinstance(node, ast.AnnAssign) and _contains_op_factory(
                node.value, factories, modules, op_classes
            ):
                targets = _targets(node.target)
            else:
                continue
            if not targets <= op_values:
                op_values |= targets
                grew = True
    for node in call_nodes:
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in op_values:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"forward", "__call__"}:
            if isinstance(node.func.value, ast.Name) and node.func.value.id in op_values:
                return True
            if _op_factory(node.func.value, factories, modules, op_classes):
                return True
        if isinstance(node.func, ast.Call) and _op_factory(
            node.func, factories, modules, op_classes
        ):
            return True
        if _name(node.func) not in non_dispatch_calls and any(
            isinstance(arg, ast.Name) and arg.id in op_values
            for arg in (*node.args, *(keyword.value for keyword in node.keywords))
        ):
            return True
    return False


def _reaching(tree: ast.Module) -> tuple[set[str], set[str], set[str]]:
    """The module's functions that reach an op, directly or through each other.

    Returns:
        Those functions, the module's op factories, and its TileOPs modules.
    """
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, _FUNCTIONS) and node.col_offset == 0
    }
    modules = _op_modules(tree)
    factories = _op_factories(functions, modules)
    reach = {name for name, node in functions.items() if _calls_an_op(node, factories, modules)}
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
    return reach, factories, modules


def _span(function: ast.AST) -> range:
    first = min([function.lineno, *(d.lineno for d in function.decorator_list)])
    return range(first, function.end_lineno + 1)


def findings(path: Path, added: set[int] | None = None) -> list[str]:
    """Placement findings in *path* on the lines in *added* (None: all)."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    reaching, factories, modules = _reaching(tree)
    allowed = allowed_lines(source, "test-placement")

    def touched(lines: range) -> bool:
        return added is None or not added.isdisjoint(lines)

    out = []
    owner: dict[int, str] = {}
    for function in (n for n in ast.walk(tree) if isinstance(n, _FUNCTIONS)):
        for node in ast.walk(function):
            owner.setdefault(id(node), function.name)
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        compare = [
            c for c in calls if _name(c.func) == "compare_outputs" and c.lineno not in allowed
        ]
        if (
            compare
            and touched(_span(function))
            and not _calls_an_op(function, factories, modules)
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
        if (
            callee
            and _KERNEL.fullmatch(callee)
            and node.lineno not in allowed
            and touched(range(node.lineno, node.lineno + 1))
        ):
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
            base = resolve_base(args.base, REPO_ROOT)
            if base is None:
                print(
                    "test_placement_lint: skipping diff-aware check: no fetched "
                    f"tile-ai/TileOPs main remote; set {BASE_ENV} to a base ref",
                    file=sys.stderr,
                )
                return 0
            added = added_lines(paths, base, REPO_ROOT)
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
