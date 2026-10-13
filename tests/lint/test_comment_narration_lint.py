"""Unit tests for ``scripts/lint/comment_narration_lint.py``."""

from pathlib import Path

import comment_narration_lint as lint
import pytest

pytestmark = pytest.mark.smoke


def _findings(tmp_path: Path, source: str, added: set[int] | None = None) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return lint.findings(path, added)


FLAGGED = [
    pytest.param("# Tried a wider tile first.\n", id="tried"),
    pytest.param("# Measured 9% slower.\n", id="measured"),
    pytest.param("# Step 2: reduce the partials.\n", id="step-n"),
    pytest.param("# This used to spill.\n", id="used-to"),
    pytest.param("# Tiles are read as before.\n", id="as-before"),
    pytest.param("# Unlike the earlier version, this pads.\n", id="earlier-version"),
    pytest.param("# The previous version padded.\n", id="previous-version"),
    pytest.param("# Phase 1 reads every tile.\n", id="phase-n"),
    pytest.param("# Swizzles the grid as DeepGEMM does.\n", id="as-library-does"),
    pytest.param('def f():\n    """Faster than the earlier version."""\n', id="docstring"),
]

ACCEPTED = [
    pytest.param("phase = 1\nmeasured_ms = 2.0\n", id="identifiers"),
    pytest.param('status = "tried"\n', id="string-that-is-not-a-docstring"),
    pytest.param("# The buffer is used to stage tiles.\n", id="passive-use"),
    pytest.param("# The barrier flips its phase every round.\n", id="phase-without-a-number"),
    pytest.param("# A retried call returns the same tile.\n", id="inside-a-longer-word"),
]


@pytest.mark.parametrize("source", FLAGGED)
def test_narration_in_a_comment_or_docstring_is_reported(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source), source


@pytest.mark.parametrize("source", ACCEPTED)
def test_what_is_not_narration_in_prose_is_left_alone(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == [], source


def test_only_the_added_lines_are_read(tmp_path: Path) -> None:
    source = "# Tried 64 first.\nx = 1\n# Tried 128 next.\n"

    (found,) = _findings(tmp_path, source, added={3})

    assert ":3:" in found


def test_the_message_names_the_file_the_line_and_the_phrase(tmp_path: Path) -> None:
    (found,) = _findings(tmp_path, "x = 1\n# Step 3: store.\n")

    assert found.startswith(f"{tmp_path / 'probe.py'}:2: comment narrates history ('Step 3')")
