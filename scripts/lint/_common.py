"""Shared file selection and CLI plumbing for the shipped-source linters.

A linter supplies only ``lint_text(text) -> [(lineno, message)]``. Keeping the
scanned roots and the excluded subtree here is what stops two linters from
drifting into different scopes. ``from _common import run`` resolves because
running a script puts its directory on ``sys.path``.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

DEFAULT_ROOTS = ("tileops", "tests", "benchmarks", "scripts")
DEFAULT_EXCLUDE = "tileops/manifest"

Finding = tuple[int, str]


def default_files(
    roots: tuple[str, ...] = DEFAULT_ROOTS, exclude: str = DEFAULT_EXCLUDE
) -> list[Path]:
    """Return every file under ``roots``, minus the ``exclude`` subtree."""
    files = []
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for path in sorted(root_path.rglob("*")):
            if not path.is_file():
                continue
            if path.as_posix().startswith(exclude + "/"):
                continue
            files.append(path)
    return files


def run(
    lint_text: Callable[[str], list[Finding]],
    description: str,
    argv=None,
    suffixes: tuple[str, ...] | None = None,
) -> int:
    """Lint the named files, or the default trees, and return the exit code.

    ``suffixes`` narrows the scan to those file extensions, for checks that
    only make sense against one language.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.files or default_files():
        if suffixes is not None and path.suffix not in suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not shipped text source
        for lineno, message in lint_text(text):
            print(f"{path}:{lineno}: {message}")
            exit_code = 1
    return exit_code
