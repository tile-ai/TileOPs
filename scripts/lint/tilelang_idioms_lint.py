#!/usr/bin/env python3
"""Lint TileLang source for idioms that compile but do the wrong thing.

Each rule below is a form the compiler accepts, so nothing downstream reports it:

- ``T.Buffer`` as a TIR parameter annotation. TileLang deprecated it for
  ``T.Tensor``; the old form still parses and gives a kernel whose shape and dtype
  the newer passes do not read.
- ``T.reinterpret`` written dtype first. The signature is
  ``T.reinterpret(value, dtype)``. Swapped, the dtype string is reinterpreted as a
  value and the intended value is read as a type name.
- A literal cast to a narrow float (``T.cast(1.0, "float16")``). The constant is
  rounded at build time, so a value the surrounding math needed in fp32 silently
  loses its low bits. Reference ``x.dtype``, or compute wider and cast at the
  boundary. A narrow *integer* cast is not this: a uint8 mask compared against 0
  or 1 loses nothing.
- A ``@tilelang.jit`` builder closing over a value that is not a scalar. The
  autotuner folds free variables into its cache key and accepts only ``int``,
  ``float``, ``str``, ``bool`` and ``None``. Assignments and parameter
  annotations classify enclosing bindings.
- A file-level lint suppression (``# ruff: noqa``, ``# flake8: noqa``). It hides
  every future finding in the file, not the one being waived.

Usage: ``tilelang_idioms_lint.py [FILE ...]``. With no arguments, scans the
source trees that carry TileLang code (``src/tileops/``, ``tests/``,
``benchmarks/``, ``workloads/``). Exits 1 when any rule fires.
"""

import argparse
import ast
import io
import re
import symtable
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TREES = ("src/tileops", "tests", "benchmarks", "workloads")

# Narrow floats only — a literal in one of these is rounded where it is written.
# An integer cast is not this: a uint8 mask compared against 0 or 1 loses nothing.
_NARROW_FLOAT = re.compile(r"^(float16|bfloat16|float8[a-z0-9_]*)$")

# Where the dtype sits: `T.reinterpret(value, dtype)` and `T.cast(value, dtype)`
# take it second, `T.Cast(dtype, value)` first. The value takes the other slot.
_DTYPE_POS = {"reinterpret": 1, "cast": 1, "Cast": 0}

_FILE_LEVEL_NOQA = re.compile(r"^#\s*(ruff|flake8)\s*:\s*noqa")
_DTYPE_NAME = re.compile(r"^(u?int[0-9]+|b?float[0-9]+|float8[a-z0-9_]*|bool|handle)$")


def _attr_path(node: ast.AST) -> str | None:
    """Dotted name of an attribute chain, e.g. ``T.reinterpret``; None otherwise."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_dtype_expr(node: ast.AST) -> bool:
    """Whether the node names a dtype rather than carrying a value.

    A bare string (``"int32"``) or a bare attribute (``T.uint16``) names one.
    ``T.uint64(0)`` is a call, so it is a value and not this.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(_DTYPE_NAME.match(node.value))
    path = _attr_path(node) if isinstance(node, ast.Attribute) else None
    return bool(path and _DTYPE_NAME.match(path.rsplit(".", 1)[-1]))


def _is_numeric_literal(node: ast.AST) -> bool:
    """Whether the node is a number written in place, sign included."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _narrow_float(node: ast.AST) -> str | None:
    """The narrow float this node names, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if _NARROW_FLOAT.match(node.value) else None
    path = _attr_path(node) if isinstance(node, ast.Attribute) else None
    if path and _NARROW_FLOAT.match(tail := path.rsplit(".", 1)[-1]):
        return tail
    return None


def _tilelang_names(tree: ast.Module) -> tuple[set[str], dict[str, str]]:
    """What this file binds to `tilelang.language`.

    Returns the module aliases (conventionally ``T``) and, for
    ``from tilelang.language import cast``, the bare names mapped to the member
    they stand for. Binding is read file-wide: rebinding one of these names to
    another module inside a function is not tracked, and would be reported here.
    """
    aliases, bare = set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname or a.name for a in node.names if a.name == "tilelang.language"}
        elif isinstance(node, ast.ImportFrom):
            if node.module == "tilelang":
                aliases |= {a.asname or a.name for a in node.names if a.name == "language"}
            elif node.module == "tilelang.language":
                bare |= {a.asname or a.name: a.name for a in node.names}
    return aliases or {"T"}, bare


def _file_level_noqa_lines(text: str) -> list[int]:
    """Lines carrying a file-level suppression, read from comments only.

    The same text inside a docstring suppresses nothing, and trailing a statement
    (``x = 1  # ruff: noqa``) it waives that line rather than the file — a textual
    scan would report both.
    """
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return [
            tok.start[0]
            for tok in tokens
            if tok.type == tokenize.COMMENT
            and tok.line.lstrip().startswith("#")
            and _FILE_LEVEL_NOQA.match(tok.string)
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


def _member(node: ast.AST, aliases: set[str], bare: dict[str, str]) -> str | None:
    """The `tilelang.language` member this expression names, or None."""
    if isinstance(node, ast.Name):
        return bare.get(node.id)
    path = _attr_path(node)
    if not path or "." not in path:
        return None
    base, _, attr = path.rpartition(".")
    return attr if base in aliases else None


def _arg(call: ast.Call, pos: int, name: str) -> ast.AST | None:
    """The argument at *pos*, or the one passed as *name*."""
    if len(call.args) > pos:
        return call.args[pos]
    return next((k.value for k in call.keywords if k.arg == name), None)


_NONSCALAR_KINDS = {
    ast.List: "list",
    ast.ListComp: "list",
    ast.Dict: "dict",
    ast.DictComp: "dict",
    ast.Set: "set",
    ast.SetComp: "set",
    ast.Tuple: "tuple",
    ast.GeneratorExp: "generator",
    ast.Lambda: "function",
}

_SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

_Func = ast.FunctionDef | ast.AsyncFunctionDef


def _jit_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """What this file binds to ``tilelang.jit``: module aliases, then bare names.

    The package re-exports ``jit`` over its own submodule, so ``import tilelang.jit as
    tj`` binds the decorator and the same import unaliased binds the package.
    """
    aliases, bare = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "tilelang":
                    aliases.add(a.asname or a.name)
                elif a.name == "tilelang.jit":
                    bare.add(a.asname) if a.asname else aliases.add("tilelang")
        elif isinstance(node, ast.ImportFrom) and node.module == "tilelang":
            bare |= {a.asname or a.name for a in node.names if a.name == "jit"}
    return aliases, bare


def _is_jit_builder(func: _Func, aliases: set[str], bare: set[str]) -> bool:
    """Whether ``tilelang.jit`` decorates this function, called or bare."""
    for dec in func.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(node, ast.Name) and node.id in bare:
            return True
        path = _attr_path(node)
        if path:
            base, _, attr = path.rpartition(".")
            if attr == "jit" and base in aliases:
                return True
    return False


def _nested_functions(tree: ast.Module) -> list[tuple[_Func, tuple[_Func, ...]]]:
    """Each function defined inside another, with every function enclosing it.

    A cell comes from any enclosing scope, so the chain is ordered innermost first — the
    order in which a name resolves.
    """
    pairs, stack = [], [(tree, ())]
    while stack:
        node, outer = stack.pop()
        for child in ast.iter_child_nodes(node):
            nested = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if nested and outer:
                pairs.append((child, outer))
            stack.append((child, (child, *outer) if nested else outer))
    return pairs


def _own_scope(func: _Func) -> Iterator[ast.AST]:
    """Nodes belonging to this function's own scope, not to one nested inside it."""
    stack: list[ast.AST] = list(func.body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPE):
            stack.extend(ast.iter_child_nodes(node))


def _function_tables(top: symtable.SymbolTable) -> dict[tuple[str, int], symtable.SymbolTable]:
    """Every function scope in the file, keyed by its name and ``def`` line."""
    tables: dict[tuple[str, int], symtable.SymbolTable] = {}
    stack = [top]
    while stack:
        table = stack.pop()
        if table.get_type() == "function":
            tables.setdefault((table.get_name(), table.get_lineno()), table)
        stack.extend(table.get_children())
    return tables


_SCALAR_ANNOTATIONS = frozenset({"int", "float", "str", "bool", "None", "NoneType"})


def _string_annotation_kind(annotation: str) -> str | None:
    """Classify a quoted annotation."""
    try:
        return _annotation_kind(ast.parse(annotation, mode="eval").body)
    except SyntaxError:
        return None if annotation in _SCALAR_ANNOTATIONS else annotation


def _annotation_kind(annotation: ast.AST | None) -> str | None:
    """What non-scalar this annotation provably names, or None."""
    if annotation is None:
        return None
    if isinstance(annotation, ast.Constant):
        if not isinstance(annotation.value, str):
            return None
        return _string_annotation_kind(annotation.value)
    if isinstance(annotation, ast.Name):
        return None if annotation.id in _SCALAR_ANNOTATIONS else annotation.id
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        left = _annotation_kind(annotation.left)
        right = _annotation_kind(annotation.right)
        return left or right
    if isinstance(annotation, ast.Subscript):
        head = annotation.value
        name = head.id if isinstance(head, ast.Name) else getattr(head, "attr", "")
        if name in ("Optional", "Union"):
            arms = annotation.slice
            elts = arms.elts if isinstance(arms, ast.Tuple) else [arms]
            for arm in elts:
                kind = _annotation_kind(arm)
                if kind:
                    return kind
            return None
        if name == "Literal":
            values = annotation.slice
            elts = values.elts if isinstance(values, ast.Tuple) else [values]
            for elt in elts:
                if not isinstance(elt, ast.Constant) or not isinstance(
                    elt.value, (int, float, str, bool, type(None))
                ):
                    return ast.unparse(annotation)
            return None
        return ast.unparse(annotation)
    if isinstance(annotation, ast.Attribute):
        return _attr_path(annotation)
    return None


def _nonscalar_kind(value: ast.AST, classes: set[str]) -> str | None:
    """What kind of non-scalar this expression provably builds, or None.

    One-sided on purpose: an expression this cannot classify is left alone, so a
    factory call returning an object passes.
    """
    for node_type, kind in _NONSCALAR_KINDS.items():
        if isinstance(value, node_type):
            return kind
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name in classes:
            return f"{name} instance"
    return None


def _nonscalar_closures(path: Path, text: str, tree: ast.Module) -> list[str]:
    """Free variables of a jit builder that provably hold something other than a scalar."""
    try:
        tables = _function_tables(symtable.symtable(text, str(path), "exec"))
    except (SyntaxError, ValueError):
        return []

    aliases, bare = _jit_names(tree)
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    out = []

    for func, outers in _nested_functions(tree):
        table = tables.get((func.name, func.lineno))
        if table is None or not _is_jit_builder(func, aliases, bare):
            continue
        free = {sym.get_name() for sym in table.get_symbols() if sym.is_free()}
        # Search enclosing scopes from nearest to farthest; shadowed names stop there.
        for outer in outers:
            args = outer.args
            params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            params += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
            last: dict[str, tuple[int, str | None]] = {
                arg.arg: (outer.lineno, _annotation_kind(arg.annotation))
                for arg in params
                if arg.arg in free
            }
            bound = set(last)
            # The final assignment in a scope determines the captured cell value.
            for node in _own_scope(outer):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if not isinstance(target, ast.Name) or target.id not in free:
                        continue
                    bound.add(target.id)
                    if node.lineno >= last.get(target.id, (-1, None))[0]:
                        last[target.id] = (node.lineno, _nonscalar_kind(node.value, classes))
            out += [
                f"{path}:{lineno}: {func.name} closes over `{name}` ({kind}) — a jit "
                "builder's free variables enter the autotune cache key, which takes "
                "only int, float, str, bool and None"
                for name, (lineno, kind) in last.items()
                if kind
            ]
            free -= bound
    return sorted(out)


def check(path: Path) -> list[str]:
    """Violations in one file, each rendered as ``path:line: message``."""
    raw = path.read_bytes()
    try:
        # Python reads a BOM and a `coding:` line before parsing; decoding without
        # them leaves a stray \ufeff that turns every later rule into a no-op.
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{path}: source could not be decoded ({exc}) — nothing here could read it"]
    out = []

    for line in _file_level_noqa_lines(text):
        out.append(f"{path}:{line}: file-level lint suppression — waive the one finding inline")

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out  # check-ast reports it; nothing here to add

    out += _nonscalar_closures(path, text, tree)

    aliases, bare = _tilelang_names(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _member(node, aliases, bare) == "Buffer":
            out.append(f"{path}:{node.lineno}: T.Buffer is deprecated — use T.Tensor(shape, dtype)")

        if not isinstance(node, ast.Call):
            continue
        member = _member(node.func, aliases, bare)
        if member not in _DTYPE_POS:
            continue

        dtype_pos = _DTYPE_POS[member]
        dtype = _arg(node, dtype_pos, "dtype")
        value = _arg(node, 1 - dtype_pos, "value")

        # A dtype sitting in the value slot is the swapped call.
        if member == "reinterpret" and value is not None and _is_dtype_expr(value):
            out.append(f"{path}:{node.lineno}: T.reinterpret takes the value first, then the dtype")
            continue

        if (
            dtype is not None
            and value is not None
            and _is_numeric_literal(value)
            and (dt := _narrow_float(dtype))
        ):
            out.append(
                f"{path}:{node.lineno}: literal cast to {dt} rounds at build time — "
                "reference x.dtype, or compute wider and cast at the boundary"
            )

    return out


def _targets(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args if a.endswith(".py")]
    return sorted(p for tree in DEFAULT_TREES for p in (REPO_ROOT / tree).rglob("*.py"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    files = _targets(parser.parse_args().files)

    violations = [v for path in files if path.is_file() for v in check(path)]
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
