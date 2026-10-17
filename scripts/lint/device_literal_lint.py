#!/usr/bin/env python3
"""Flag a CUDA device written into a test or workload outside CUDA-only scope.

Tensors go on ``workloads.device.run_device()``. Flags ``"cuda"``, a string or f-string
starting ``cuda:``, and an argument-less ``.cuda()``, unless a function, class,
``pytest.param`` or module ``pytestmark`` carries ``pytest.mark.cuda_only`` or an
architecture mark (``pytest.mark.sm90`` and the like), which needs CUDA as well. Also flags
``torch.cuda.is_available()`` outside a ``conftest.py``; gate on
``workloads.device.run_device_available()``.

The ``type="cuda"`` keyword of an imported ``types.SimpleNamespace`` constructor
is host metadata. Only that literal is exempt; other arguments remain checked.

Exempt: ``workloads/device.py``. Default scan: ``tests/``, ``workloads/``.
"""

import ast
import re
import sys
from pathlib import Path

from common import files, relative, report

SCANNED = ("tests", "workloads")
# An architecture mark names a CUDA compute capability, so it needs CUDA as cuda_only does.
_ARCH_MARK = re.compile(r"sm\d+")
EXEMPT = {"workloads/device.py"}


def _carries_mark(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Attribute) and (n.attr == "cuda_only" or _ARCH_MARK.fullmatch(n.attr))
        for n in ast.walk(node)
    )


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


def _cuda_availability(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "is_available"
        and ast.unparse(node.func.value) == "torch.cuda"
    )


def availability_findings(source: str) -> list[int]:
    """Line numbers of ``torch.cuda.is_available()`` calls."""
    return sorted({n.lineno for n in ast.walk(ast.parse(source)) if _cuda_availability(n)})


def _module_marked(tree: ast.Module) -> bool:
    return any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets)
        and _carries_mark(n.value)
        for n in tree.body
    )


def findings(source: str) -> list[int]:
    """CUDA literals outside CUDA-only scope or an imported namespace's plain type field."""
    tree = ast.parse(source)
    if _module_marked(tree):
        return []
    lines: list[int] = []
    namespace_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "types" and not node.level
        for alias in node.names
        if alias.name == "SimpleNamespace"
    }
    types_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "types"
    }

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
        if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name)
            and node.func.id in namespace_names
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "SimpleNamespace"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in types_names
        ):
            for child in ast.iter_child_nodes(node):
                if (
                    isinstance(child, ast.keyword)
                    and child.arg == "type"
                    and isinstance(child.value, ast.Constant)
                    and child.value.value == "cuda"
                ):
                    continue
                visit(child)
            return
        if _literal(node):
            lines.append(node.lineno)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return sorted(set(lines))


def file_findings(path: Path) -> list[str]:
    """Both rules over the file at *path*; none outside the scanned trees or in an exempt file."""
    rel = relative(path)
    if not rel.startswith(tuple(f"{d}/" for d in SCANNED)) or rel in EXEMPT:
        return []
    source = path.read_text()
    out = [
        f"{rel}:{line}: CUDA device literal — use workloads.device.run_device(), "
        "or mark the test pytest.mark.cuda_only"
        for line in findings(source)
    ]
    if path.name != "conftest.py":
        out += [
            f"{rel}:{line}: torch.cuda.is_available() gate — "
            "use workloads.device.run_device_available()"
            for line in availability_findings(source)
        ]
    return out


def main(argv: list[str]) -> int:
    return report(e for p in files(argv, SCANNED, (".py",)) for e in file_findings(p))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
