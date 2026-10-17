"""The comment and docstring lines of a Python source, for the lints that read prose.

A comment is a ``#`` token. A docstring is the string that opens a module, class or function,
every line it spans. Code is never returned.
"""

import ast
import io
import tokenize
from typing import NamedTuple

from common import allow_marker


class ProseLine(NamedTuple):
    line: int
    text: str
    kind: str  # "comment" or "docstring"


def allowed_lines(source: str, rule: str) -> set[int]:
    """Lines carrying ``lint: <rule> allow: <reason>`` in their prose."""
    marker = allow_marker(rule)
    return {prose.line for prose in prose_lines(source) if marker.search(prose.text)}


def prose_lines(source: str) -> list[ProseLine]:
    """Every comment and docstring line of *source*, in line order.

    A source that does not tokenize or parse yields what was read before the error;
    ``check-ast`` reports the error itself.
    """
    out = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                out.append(ProseLine(token.start[0], token.string, "comment"))
    except (tokenize.TokenError, SyntaxError):
        pass
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sorted(out)
    physical = source.splitlines()
    scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, scopes) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            for number in range(first.lineno, first.end_lineno + 1):
                out.append(ProseLine(number, physical[number - 1], "docstring"))
    return sorted(out)
