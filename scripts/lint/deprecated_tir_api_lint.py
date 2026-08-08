#!/usr/bin/env python3
"""Lint shipped Python for deprecated TIR spellings.

Both forms below still parse, so nothing fails until a toolchain bump drops
them; a check is what keeps them from drifting back in. Flags:

- ``T.Buffer`` as a parameter type. The supported spelling is
  ``T.Tensor(shape, dtype)``; the signature is identical, so correcting a site
  is a rename.
- ``T.reinterpret`` called with a string literal first, the deprecated
  dtype-first order. The supported spelling puts the value first:
  ``T.reinterpret(value, dtype)``.

Detection works on the parsed syntax tree, not on the raw text, which is what
makes it indifferent to how the source is laid out: whitespace or newlines
around the dot, a space before the call parenthesis, and prose naming a
deprecated form in a comment, string, or this very docstring are all handled by
construction rather than by a pattern per spelling. A file that will not parse
falls back to a text scan rather than going unchecked.

Usage: ``deprecated_tir_api_lint.py [FILE ...]``. With no arguments, scans the
default shipped-source trees excluding ``tileops/manifest/``. Exits 1 when any
violation is found.
"""

import ast
import re
import sys

from _common import run

_NAMESPACE = "T"
_BUFFER_MESSAGE = "deprecated T.Buffer; use T.Tensor(shape, dtype)"
_REINTERPRET_MESSAGE = "deprecated dtype-first T.reinterpret; use T.reinterpret(value, dtype)"

# Only for sources that will not parse, where there is no tree to inspect.
_FALLBACK_PATTERNS = (
    (re.compile(r"\bT\s*\.\s*Buffer\b"), _BUFFER_MESSAGE),
    (re.compile(r"\bT\s*\.\s*reinterpret\s*\(\s*[A-Za-z]*[\"']"), _REINTERPRET_MESSAGE),
)


def _is_namespace_attribute(node: ast.AST, attr: str) -> bool:
    """True when ``node`` is the attribute ``T.<attr>``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == _NAMESPACE
    )


def _is_string_expression(node: ast.AST) -> bool:
    """True for a literal string or an f-string, the dtype-shaped arguments.

    A concatenation or a call that happens to produce a string is not one:
    ``T.reinterpret("abc" + str(x), "f16")`` passes a value first and is
    correct.
    """
    return isinstance(node, ast.JoinedStr) or (
        isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


def _lint_tree(tree: ast.AST) -> list[tuple[int, str]]:
    findings = []
    for node in ast.walk(tree):
        if _is_namespace_attribute(node, "Buffer"):
            findings.append((node.lineno, _BUFFER_MESSAGE))
        elif (
            isinstance(node, ast.Call)
            and _is_namespace_attribute(node.func, "reinterpret")
            and node.args
            and _is_string_expression(node.args[0])
        ):
            findings.append((node.lineno, _REINTERPRET_MESSAGE))
    return sorted(findings)


def _lint_text_fallback(text: str) -> list[tuple[int, str]]:
    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, message in _FALLBACK_PATTERNS:
            if pattern.search(line):
                findings.append((lineno, message))
    return findings


def lint_text(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, message)`` pairs for every violation in ``text``."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _lint_text_fallback(text)
    return _lint_tree(tree)


def main(argv: list[str] | None = None) -> int:
    return run(lint_text, __doc__, argv, suffixes=(".py",))


if __name__ == "__main__":
    sys.exit(main())
