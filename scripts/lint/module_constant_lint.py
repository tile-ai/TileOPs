#!/usr/bin/env python3
"""Lint module-level constants in ``src/tileops/`` against the naming rule.

Two checks, both on a name bound once at module scope to a fixed value:

- It is ``UPPER_SNAKE``. A lowercase name reads as state a reader may rebind.
- It carries a leading underscore and stays out of ``__all__`` unless another file
  references it. A public name invites an edit from outside the module that decides it.

A name any function declares ``global`` is mutable state, not a constant, and is skipped.

Usage: ``module_constant_lint.py``. Takes no arguments; the second check reads the whole
tree. Exits 1 on a finding.
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED = "src/tileops"
# Where a reference makes a constant shared rather than module-local.
REFERENCED_BY = ("src", "tests", "benchmarks", "workloads", "scripts", "docs")


def _is_fixed_value(node: ast.expr) -> bool:
    """Whether *node* spells a value out, rather than computing one.

    A dotted name counts: an enumerator such as ``tilelang.PassConfigKey.TL_ENABLE_FAST_MATH``
    is as fixed as a literal. A call is not, so a logger or a device object is left alone.
    """
    if isinstance(node, (ast.Constant, ast.Attribute)):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_fixed_value(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            k is not None and _is_fixed_value(k) and _is_fixed_value(v)
            for k, v in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.UnaryOp):
        return _is_fixed_value(node.operand)
    return False


def _module_constants(tree: ast.Module) -> list[tuple[str, int]]:
    rebound = {
        name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names
    }
    bindings: dict[str, int] = {}
    seen: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name) or target.id.startswith("__"):
                continue
            if target.id in seen or target.id in rebound:
                bindings.pop(target.id, None)
                continue
            seen.add(target.id)
            if node.value is not None and _is_fixed_value(node.value):
                bindings[target.id] = target.lineno
    return sorted(bindings.items())


def _exported(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            return {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    return set()


def _other_files_mentioning(name: str, own: Path) -> bool:
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    for root in REFERENCED_BY:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if path == own or "__pycache__" in path.parts:
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                return True
    return False


def main(argv: list[str]) -> int:
    if argv:
        print("module_constant_lint.py takes no arguments", file=sys.stderr)
        return 2
    errors = []
    for path in sorted((REPO_ROOT / SCANNED).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        exported = _exported(tree)
        rel = path.relative_to(REPO_ROOT)
        for name, lineno in _module_constants(tree):
            if name.lstrip("_") != name.lstrip("_").upper():
                errors.append(f"{rel}:{lineno}: module constant {name} is not UPPER_SNAKE")
                continue
            if name.startswith("_"):
                continue
            if not _other_files_mentioning(name, path):
                where = " and leave __all__" if name in exported else ""
                errors.append(
                    f"{rel}:{lineno}: module constant {name} is read by no other file; "
                    f"give it a leading underscore{where}"
                )
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
