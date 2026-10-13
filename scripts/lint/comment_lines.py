"""The comment and docstring lines of a Python source, for the lints that read prose.

A comment is a ``#`` token, so a ``#`` inside a string is not one. A docstring is the string
literal that opens a module, class or function body; every physical line it spans is
returned, delimiters included. Code is never returned, so a name such as ``phase`` or
``measured_ms`` in an expression is invisible to a rule run over these lines.
"""

import ast
import io
import tokenize
from typing import NamedTuple


class ProseLine(NamedTuple):
    line: int
    text: str
    kind: str  # "comment" or "docstring"


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
