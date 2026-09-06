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
import sys
import tokenize
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
