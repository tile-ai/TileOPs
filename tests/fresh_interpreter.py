"""Evaluate an expression in a fresh Python interpreter.

A check on what importing does has to start from a process that has imported nothing.
The child's stdout is not the result: a package may print while it loads, as a device
runtime that finds no device does. The result travels on one marked line instead.
"""

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_MARK = "TILEOPS_PROBE_RESULT="


def probe(setup: str, expr: str, *, env: "dict[str, str] | None" = None) -> str:
    """Run *setup* in a fresh interpreter, then return ``str(expr)`` as the child saw it.

    Args:
        setup: Statements to run first, from the repository root.
        expr: The expression whose value is returned.
        env: The child's environment; ``None`` inherits this one.

    Returns:
        ``str`` of *expr*'s value. Every other line on the child's stdout is ignored.
    """
    # The newline ends whatever a package left unterminated, so the mark starts its line.
    source = f"{setup}\nprint('\\n' + {_MARK!r} + str({expr}))"
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    found = [line[len(_MARK) :] for line in result.stdout.splitlines() if line.startswith(_MARK)]
    assert len(found) == 1, f"expected one {_MARK} line on stdout:\n{result.stdout}"
    return found[0]
