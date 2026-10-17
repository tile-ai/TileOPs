"""Unit tests for ``scripts/lint/comment_narration_lint.py``."""

from pathlib import Path

import comment_narration_lint as lint
import pytest

pytestmark = pytest.mark.smoke


def _findings(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return lint.findings(path)


FLAGGED = [
    pytest.param("# Tried a wider tile first.\n", id="tried"),
    pytest.param("# We measured 9% slower.\n", id="we-measured"),
    pytest.param("# I have measured at 4096 rows that 64 wins.\n", id="i-have-measured-at"),
    pytest.param("# After measuring on H200, 64 rows won.\n", id="after-measuring-on"),
    pytest.param("# This used to spill.\n", id="used-to"),
    pytest.param("# Tiles are read as before.\n", id="as-before"),
    pytest.param("# Unlike the earlier version, this pads.\n", id="earlier-version"),
    pytest.param("# The previous version padded.\n", id="previous-version"),
    pytest.param('def f():\n    """Faster than the earlier version."""\n', id="docstring"),
    pytest.param(
        'def f():\n    """Contract.\n\n    Tried 64 first.\n    """\n', id="docstring-line"
    ),
]

ACCEPTED = [
    pytest.param("phase = 1\nmeasured_ms = 2.0\n", id="identifiers"),
    pytest.param('status = "tried"\n', id="string-that-is-not-a-docstring"),
    pytest.param("# The buffer is used to stage tiles.\n", id="passive-use"),
    pytest.param(
        "# The stride used to select key-value pairs is contiguous.\n", id="used-to-select"
    ),
    pytest.param("# Set only once every entry point has been tried.\n", id="tried-state"),
    pytest.param("# A retried call returns the same tile.\n", id="inside-a-longer-word"),
    # Where a tuned value was measured is a contract code-style.md asks for.
    pytest.param("# Measured on H200: 64 rows is 3% faster.\n", id="measured-on"),
    pytest.param(
        'def f():\n    """Pick the tile.\n\n    Measured on H200 over M 512-8192.\n    """\n',
        id="measured-in-a-docstring",
    ),
]


@pytest.mark.parametrize("source", FLAGGED)
def test_narration_in_a_comment_or_docstring_is_reported(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source), source


@pytest.mark.parametrize("source", ACCEPTED)
def test_what_is_not_narration_in_prose_is_left_alone(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == [], source


def test_the_message_names_the_file_the_line_and_the_phrase(tmp_path: Path) -> None:
    (found,) = _findings(tmp_path, "x = 1\n# This used to spill.\n")

    assert found.startswith(f"{tmp_path / 'probe.py'}:2: comment narrates history ('This used to')")
    assert found.endswith('or add "# lint: narration allow: <reason>" on the line')


def test_a_line_allowance_with_a_reason_suppresses_the_finding(tmp_path: Path) -> None:
    source = "# This used to be the public name.  # lint: narration allow: compatibility\n"

    assert _findings(tmp_path, source) == []
