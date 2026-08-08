"""Unit tests for ``scripts/lint/shipped_refs_lint.py``.

Each case runs the linter as a subprocess on a tmp file built from an inline
fixture string — never on the live repo. Violating tokens are assembled by
concatenation so this test file itself stays clean under the hook.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_SCRIPT = REPO_ROOT / "scripts" / "lint" / "shipped_refs_lint.py"

# Concatenated so the literals never appear verbatim in this file.
ISSUE_REF = "#" + "12345"  # 5 digits: not a CSS hex-color length
AC_LABEL = "AC" + "-3"
ROUND_REVIEW = "round" + "-2 review"
FOLLOW_UP = "Follow" + "-up: #" + "12345"


def run_lint(tmp_path: Path, content: str) -> subprocess.CompletedProcess:
    target = tmp_path / "fixture.py"
    target.write_text(content)
    return subprocess.run(
        [sys.executable, str(LINT_SCRIPT), str(target)],
        capture_output=True,
        text=True,
    )


def test_issue_number_reference_fails(tmp_path):
    result = run_lint(tmp_path, f"# fixed in {ISSUE_REF}\n")
    assert result.returncode == 1
    assert "fixture.py" in result.stdout


def test_ac_label_fails(tmp_path):
    result = run_lint(tmp_path, f"# satisfies {AC_LABEL}\n")
    assert result.returncode == 1


def test_round_review_fails(tmp_path):
    result = run_lint(tmp_path, f"# addressed in {ROUND_REVIEW}\n")
    assert result.returncode == 1


def test_follow_up_fails(tmp_path):
    result = run_lint(tmp_path, f"# {FOLLOW_UP}\n")
    assert result.returncode == 1


def test_clean_file_passes(tmp_path):
    result = run_lint(tmp_path, "def add(a, b):\n    return a + b\n")
    assert result.returncode == 0


def test_hex_color_literals_pass(tmp_path):
    content = (
        '_INK = "#191a16"\n'
        '_BLUE = "#112233"\n'
        '_SHORT = "#fff"\n'
        '_RGBA = "#1234"\n'
        '_ALPHA = "#11223344"\n'
        "css = 'color:#e8e1d1;background:#191a16'\n"
    )
    result = run_lint(tmp_path, content)
    assert result.returncode == 0, result.stdout


def test_multiple_files_reports_each(tmp_path):
    clean = tmp_path / "clean.py"
    clean.write_text("x = 1\n")
    dirty = tmp_path / "dirty.py"
    dirty.write_text(f"# see {ISSUE_REF}\n")
    result = subprocess.run(
        [sys.executable, str(LINT_SCRIPT), str(clean), str(dirty)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "dirty.py" in result.stdout
    assert "clean.py" not in result.stdout
