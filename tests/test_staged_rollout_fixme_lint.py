"""Unit tests for ``scripts/lint/staged_rollout_fixme_lint.py``.

Each case runs the linter as a subprocess on a tmp file built from an inline
fixture string — never on the live repo. The marker token is assembled by
concatenation so this test file itself stays clean under the hook.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_SCRIPT = REPO_ROOT / "scripts" / "lint" / "staged_rollout_fixme_lint.py"

# Concatenated so the marker never appears verbatim in this file.
MARKER = "FIXME" + "(staged-rollout)"

CONFORMING = f"""\
# {MARKER}: contract stub raises instead of being abstract.
#
# Broken invariant: the base class does not force implementations.
# Why: making it abstract breaks every op that has not migrated yet;
#     migration happens one op at a time.
# Cleanup: once every op implements the method, mark it abstract and
#     delete this marker.
def stub():
    raise NotImplementedError
"""

MISSING_WHY = f"""\
# {MARKER}: contract stub raises instead of being abstract.
#
# Broken invariant: the base class does not force implementations.
# Cleanup: mark it abstract and delete this marker.
def stub():
    raise NotImplementedError
"""

NO_BLOCK = f"""\
# {MARKER}: bare marker with no block at all.
def stub():
    raise NotImplementedError
"""

PR_NUMBER_IN_CLEANUP = f"""\
# {MARKER}: contract stub raises instead of being abstract.
#
# Broken invariant: the base class does not force implementations.
# Why: migration happens one op at a time.
# Cleanup: remove after {"#" + "1234"} merges.
def stub():
    raise NotImplementedError
"""

OUT_OF_ORDER = f"""\
# {MARKER}: contract stub raises instead of being abstract.
#
# Why: migration happens one op at a time.
# Broken invariant: the base class does not force implementations.
# Cleanup: mark it abstract and delete this marker.
def stub():
    raise NotImplementedError
"""


def run_lint(tmp_path: Path, content: str) -> subprocess.CompletedProcess:
    target = tmp_path / "fixture.py"
    target.write_text(content)
    return subprocess.run(
        [sys.executable, str(LINT_SCRIPT), str(target)],
        capture_output=True,
        text=True,
    )


def test_conforming_block_passes(tmp_path):
    result = run_lint(tmp_path, CONFORMING)
    assert result.returncode == 0, result.stdout


def test_missing_section_fails(tmp_path):
    result = run_lint(tmp_path, MISSING_WHY)
    assert result.returncode == 1
    assert "Why:" in result.stdout


def test_bare_marker_fails(tmp_path):
    result = run_lint(tmp_path, NO_BLOCK)
    assert result.returncode == 1


def test_pr_number_in_cleanup_fails(tmp_path):
    result = run_lint(tmp_path, PR_NUMBER_IN_CLEANUP)
    assert result.returncode == 1
    assert "Cleanup" in result.stdout


def test_out_of_order_sections_fail(tmp_path):
    result = run_lint(tmp_path, OUT_OF_ORDER)
    assert result.returncode == 1


def test_file_without_marker_passes(tmp_path):
    result = run_lint(tmp_path, "def add(a, b):\n    return a + b\n")
    assert result.returncode == 0
