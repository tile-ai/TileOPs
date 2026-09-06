"""Unit tests for ``scripts/lint/tilelang_idioms_lint.py``.

Each case runs the linter as a subprocess on a tmp file built from an inline
fixture string — never on the live repo. The accepted forms matter as much as the
rejected ones: each rule has a neighbouring form it must leave alone.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_SCRIPT = REPO_ROOT / "scripts" / "lint" / "tilelang_idioms_lint.py"

# Concatenated so this file stays clean under the hook it tests.
FILE_NOQA = "# ruff" + ": noqa"


def run_lint(tmp_path: Path, content: str) -> subprocess.CompletedProcess:
    target = tmp_path / "fixture.py"
    target.write_text(content)
    return subprocess.run(
        [sys.executable, str(LINT_SCRIPT), str(target)],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        ("def f(x: T.Buffer((4,), 'float16')): pass\n", "T.Buffer is deprecated"),
        ("y = T.reinterpret('int32', x)\n", "value first"),
        ("y = T.reinterpret(T.uint16, x)\n", "value first"),
        ("y = T.reinterpret('bfloat16', x)\n", "value first"),
        ("y = T.reinterpret(value='int32', dtype=x)\n", "value first"),
        # A positional value with the dtype by keyword is still a literal cast.
        ("y = T.reinterpret(1.0, dtype='float16')\n", "rounds at build time"),
        ("y = T.cast(1.0, 'float16')\n", "rounds at build time"),
        ("y = T.cast(-1.0, T.bfloat16)\n", "rounds at build time"),
        ("y = T.cast(1.0, 'float8_e4m3fn')\n", "rounds at build time"),
        ("y = T.cast(1.0, dtype='float16')\n", "rounds at build time"),
        # T.Cast takes the dtype first, so the literal is the second argument.
        ("y = T.Cast('float16', 1.0)\n", "rounds at build time"),
        # An alias other than T still names tilelang.language.
        ("import tilelang.language as tl\ny = tl.reinterpret('int32', x)\n", "value first"),
        ("from tilelang import language as tl\ny = tl.cast(1.0, 'float16')\n", "rounds"),
        (f"{FILE_NOQA}\n", "file-level lint suppression"),
        # A bare import names the same member.
        (
            "from tilelang.language import reinterpret\ny = reinterpret('int32', x)\n",
            "value first",
        ),
    ],
)
def test_rejected(tmp_path, source, expected):
    result = run_lint(tmp_path, source)
    assert result.returncode == 1
    assert expected in result.stdout


@pytest.mark.parametrize(
    "source",
    [
        "def f(x: T.Tensor((4,), 'float16')): pass\n",
        "y = T.reinterpret(x, 'int32')\n",
        # The value is a call, not a dtype name.
        "y = T.reinterpret(T.uint64(0), dtype='handle')\n",
        # A uint8 mask compared against 0 or 1 loses nothing.
        "y = T.cast(0, 'uint8')\n",
        # The dtype comes from a variable, so nothing is rounded where it is written.
        "y = T.cast(0, dtype)\n",
        "y = T.cast(x, 'float16')\n",
        # T.Cast's own order: dtype first, value second — the form alibi.py uses.
        "y = T.Cast(dtype, -slope * abs_dist)\n",
        # An unrelated module's cast is not a TileLang call.
        "import numpy as np\ny = np.cast(1.0, 'float16')\n",
        "x = 1  # noqa: F841\n",
        # The same text inside a docstring suppresses nothing.
        f'DOC = """\n{FILE_NOQA}\n"""\n',
        # Trailing a statement it waives that line, not the file.
        f"x = 1  {FILE_NOQA}\n",
    ],
)
def test_accepted(tmp_path, source):
    result = run_lint(tmp_path, source)
    assert result.returncode == 0, result.stdout


def test_bom_does_not_hide_the_rules(tmp_path):
    """A byte-order mark is valid Python; decoding past it must not skip the parse."""
    target = tmp_path / "fixture.py"
    target.write_bytes(b"\xef\xbb\xbf" + b"y = T.cast(1.0, 'float16')\n")
    result = subprocess.run(
        [sys.executable, str(LINT_SCRIPT), str(target)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "rounds at build time" in result.stdout


def test_undecodable_source_is_reported(tmp_path):
    """A file no reader can decode fails loudly rather than passing silently.

    The bad byte sits past the first two lines, which is all `detect_encoding`
    reads — so this reaches the decode itself rather than stopping short of it.
    """
    target = tmp_path / "fixture.py"
    target.write_bytes(b"x = 1\ny = 2\nz = '\xff'\ny = T.cast(1.0, 'float16')\n")
    result = subprocess.run(
        [sys.executable, str(LINT_SCRIPT), str(target)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "could not be decoded" in result.stdout


def test_default_scan_reaches_the_source_tree():
    """A no-argument run must actually visit files, not pass by scanning nothing."""
    sys.path.insert(0, str(LINT_SCRIPT.parent))
    try:
        import tilelang_idioms_lint as lint
    finally:
        sys.path.pop(0)

    targets = lint._targets([])
    assert REPO_ROOT / "src" / "tileops" / "kernels" / "rope.py" in targets


def test_repo_is_clean():
    result = subprocess.run(
        [sys.executable, str(LINT_SCRIPT)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stdout
