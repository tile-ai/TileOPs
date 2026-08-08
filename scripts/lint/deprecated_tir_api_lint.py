#!/usr/bin/env python3
"""Lint shipped Python for deprecated TIR spellings.

Both forms below still parse, so nothing fails until a toolchain bump drops
them; a check is what keeps them from drifting back in. Flags, in code only:

- ``T.Buffer`` as a parameter type. The supported spelling is
  ``T.Tensor(shape, dtype)``; the signature is identical, so correcting a site
  is a rename.
- ``T.reinterpret`` called with a string literal first, the deprecated
  dtype-first order. The supported spelling puts the value first:
  ``T.reinterpret(value, dtype)``.

Comments and string literals are skipped, so prose naming a deprecated form —
this docstring included — is not a violation, while real usage anywhere in the
tree still is. A file that will not tokenize falls back to a raw line scan
rather than going unchecked.

Usage: ``deprecated_tir_api_lint.py [FILE ...]``. With no arguments, scans the
default shipped-source trees excluding ``tileops/manifest/``. Exits 1 when any
violation is found.
"""

import io
import re
import sys
import tokenize

from _common import run

# ``T . Buffer`` is legal Python, so tolerate whitespace around the dot.
_DOT = r"\s*\.\s*"
_PATTERNS = (
    (
        re.compile(rf"\bT{_DOT}Buffer\b"),
        "deprecated T.Buffer; use T.Tensor(shape, dtype)",
    ),
    (
        # A string literal as the first argument means the dtype was passed
        # first; the optional letters cover prefixes such as f, r and b.
        re.compile(rf"\bT{_DOT}reinterpret\(\s*[A-Za-z]*[\"']"),
        "deprecated dtype-first T.reinterpret; use T.reinterpret(value, dtype)",
    ),
)

# Python 3.12 splits f-strings into start/middle/end tokens; the literal text
# arrives as FSTRING_MIDDLE rather than STRING. Older versions emit one STRING.
_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)


def _blank_comments_and_strings(text: str) -> dict[int, str] | None:
    """Return each line with comment and string-literal content blanked out.

    Quote characters survive so the dtype-first ``T.reinterpret("f16", x)``
    shape stays recognizable; only what is written *inside* a literal — where
    a deprecated spelling is prose, not usage — is erased. Blanking rather
    than deleting keeps column positions intact. Returns None when the source
    will not tokenize.
    """
    lines = dict(enumerate(text.splitlines(), start=1))
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError, ValueError):
        return None
    blankable = {tokenize.COMMENT, tokenize.STRING}
    if _FSTRING_MIDDLE is not None:
        blankable.add(_FSTRING_MIDDLE)
    for token in tokens:
        if token.type not in blankable:
            continue
        # Only a whole STRING token carries its own quotes; f-string literal
        # text does not, so blanking it wholesale is right.
        keep_quotes = token.type == tokenize.STRING
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        for row in range(start_row, end_row + 1):
            line = lines.get(row, "")
            begin = start_col if row == start_row else 0
            finish = end_col if row == end_row else len(line)
            span = line[begin:finish]
            blanked = "".join(c if keep_quotes and c in "\"'" else " " for c in span)
            lines[row] = line[:begin] + blanked + line[finish:]
    return lines


def lint_text(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, message)`` pairs for every violation in ``text``."""
    scanned = _blank_comments_and_strings(text)
    if scanned is None:  # untokenizable: scan raw so the file is not skipped
        scanned = dict(enumerate(text.splitlines(), start=1))
    findings = []
    for lineno in sorted(scanned):
        for pattern, message in _PATTERNS:
            if pattern.search(scanned[lineno]):
                findings.append((lineno, message))
    return findings


def main(argv: list[str] | None = None) -> int:
    return run(lint_text, __doc__, argv, suffixes=(".py",))


if __name__ == "__main__":
    sys.exit(main())
