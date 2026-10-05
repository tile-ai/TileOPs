#!/usr/bin/env python3
"""Lint the spelling of module-level constants in ``src/tileops/``.

A name bound once at module scope to a value written out in the source is ``UPPER_SNAKE``,
with or without a leading underscore. A lowercase name reads as state a reader may rebind,
and a short one hides what it holds: ``_pc`` and ``_cf`` in ``gqa/dense.py`` were byte-identical
copies of ``_PASS_CONFIGS`` and ``_COMPILE_FLAGS`` defined 25 lines above them.

Written out means a literal, a dotted name, or a tuple, list, set or dict of those. A value a
call produces is not, so a logger or a device object is left alone. A name any function
declares ``global``, binds twice, or annotates ``TypeAlias`` is not a constant either.

Usage: ``module_constant_lint.py [FILE ...]``. With no arguments, scans ``src/tileops/``.
Exits 1 on a finding.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED = "src/tileops"


def _is_written_out(node: ast.expr) -> bool:
    """Whether *node* spells its value out, rather than computing one."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Attribute):
        # A dotted name such as ``tilelang.PassConfigKey.TL_ENABLE_FAST_MATH``, but not an
        # attribute of whatever a call returned.
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_written_out(e.value if isinstance(e, ast.Starred) else e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            k is not None and _is_written_out(k) and _is_written_out(v)
            for k, v in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.UnaryOp):
        return _is_written_out(node.operand)
    if isinstance(node, ast.JoinedStr):
        return all(isinstance(part, ast.Constant) for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return all(
            isinstance(side, ast.Constant) and isinstance(side.value, str)
            for side in (node.left, node.right)
        )
    return False


def _is_type_alias(node: ast.stmt) -> bool:
    if not isinstance(node, ast.AnnAssign) or node.annotation is None:
        return False
    annotation = node.annotation
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "TypeAlias"
    return isinstance(annotation, ast.Name) and annotation.id == "TypeAlias"


def _module_statements(body: list[ast.stmt]):
    """Every statement that binds at module scope, through the blocks that open no scope."""
    nesting = (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)
    for node in body:
        yield node
        if isinstance(node, nesting):
            nested = list(node.body) + list(getattr(node, "orelse", []))
            nested += list(getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                nested += list(handler.body)
            yield from _module_statements(nested)


def _bound_names(target: ast.expr) -> list[ast.Name]:
    """Every name a target binds, through tuple, list and starred targets."""
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _bound_names(element)]
    return []


def _candidates(tree: ast.Module) -> list[tuple[str, int]]:
    """Names bound exactly once at module scope to a value written out in the source."""
    rebound = {
        name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names
    }
    bindings: dict[str, int] = {}
    for node in _module_statements(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound.add(node.name)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rebound.update(a.asname or a.name.split(".")[0] for a in node.names)
            continue
        if isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor)):
            rebound.update(name.id for name in _bound_names(node.target))
            continue
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if _is_type_alias(node):
            rebound.update(name.id for t in targets for name in _bound_names(t))
            continue
        for target in targets:
            # A destructuring target takes the matching element of a written-out sequence;
            # against anything else, every name it binds is state rather than a constant.
            if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
                node.value, (ast.Tuple, ast.List)
            ):
                pairs = list(zip(target.elts, node.value.elts, strict=False))
            else:
                pairs = [(target, node.value)]
            for name_node, value in pairs:
                names = _bound_names(name_node)
                if len(names) != 1 or names[0] is not name_node or not _is_written_out(value):
                    rebound.update(name.id for name in names)
                    continue
                if name_node.id in bindings:
                    rebound.add(name_node.id)
                else:
                    bindings[name_node.id] = name_node.lineno
    return sorted((n, line) for n, line in bindings.items() if n not in rebound)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def findings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    return [
        f"{rel}:{line}: module constant {name} is not UPPER_SNAKE"
        for name, line in _candidates(tree)
        if not _is_dunder(name) and name.lstrip("_") != name.lstrip("_").upper()
    ]


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted((REPO_ROOT / SCANNED).rglob("*.py"))
    errors = [
        e for p in paths if p.suffix == ".py" and "__pycache__" not in p.parts for e in findings(p)
    ]
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
