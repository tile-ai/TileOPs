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
    pytest.param("# We measured 9% slower.\n", id="we-measured"),
    pytest.param("# We measured on H200 that 64 wins.\n", id="we-measured-on"),
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
    pytest.param(
        "# The kernel used to reduce rows is chosen by the selector.\n",
        id="kernel-used-to-reduce",
    ),
    pytest.param("# Set only once every entry point has been tried.\n", id="tried-state"),
    pytest.param("# Uses measured long-context defaults.\n", id="measured-default"),
    pytest.param("# This row stays below its measured crossover.\n", id="measured-crossover"),
    pytest.param(
        'def f():\n    """The calibration measured on *device_name*\'s board."""\n',
        id="measured-on-participle",
    ),
    pytest.param("# The rotation itself is measured.\n", id="benchmark-vocabulary"),
    pytest.param("# Threshold measured at 4096 rows on H200.\n", id="threshold-measured-at"),
    pytest.param("# Measured at 1.4 GHz: 64 rows is 3% faster.\n", id="measured-at-a-clock"),
    pytest.param("# Step 2 reduces the partials.\n", id="algorithm-step"),
    pytest.param("# Phase 1 reads every tile.\n", id="algorithm-phase"),
    pytest.param("# Swizzles the grid as DeepGEMM does.\n", id="compatibility-source"),
    pytest.param("# The barrier flips its phase every round.\n", id="phase-without-a-number"),
    pytest.param("# A retried call returns the same tile.\n", id="inside-a-longer-word"),
]


# Where a threshold was measured is the reason code-style.md asks for beside a tuned value.
THRESHOLD_EVIDENCE = [
    pytest.param(
        "# What selects a body.\n"
        "# Measured on H200 under the benchmark's cold L2, each body at its best config (us):\n"
        "# - Held row: (32768, 8192) float16 248.7 against 255.6 on the fragment body.\n",
        id="comment-block-opening-a-config-table",
    ),
    pytest.param(
        "def select(call):\n"
        "    # Measured on H200: the fused shape wins from about three quarters of\n"
        "    # the SMs covered and loses badly below half.\n"
        "    return call.sequences * call.value_heads\n",
        id="comment-beside-a-dispatch-threshold",
    ),
    pytest.param(
        "def warp_row_vector(n):\n"
        '    """Elements of the vector a lane reads, for a row one warp holds.\n\n'
        "    A warp guards only the vectors past the row. Measured on H200 at\n"
        "    64 to 1024 columns, where it led or tied the fragment body at every dtype.\n"
        '    """\n',
        id="docstring-sentence-after-a-period",
    ),
    pytest.param(
        "def pick_tile(m, n, k):\n"
        '    """Pick the ping-pong or coop2 tile.\n\n'
        "    It is taken where that width fills the waves at least as well as the coop2\n"
        "    width. Measured on H200 over M 512-8192, N 1536-24576, K 256-7168: up to 64\n"
        "    K steps that comparison decides.\n"
        '    """\n',
        id="docstring-measurement-range",
    ),
]


@pytest.mark.parametrize("source", THRESHOLD_EVIDENCE)
def test_where_a_threshold_was_measured_is_left_alone(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == [], source


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
    (found,) = _findings(tmp_path, "x = 1\n# This used to spill.\n")

    assert found.startswith(f"{tmp_path / 'probe.py'}:2: comment narrates history ('This used to')")
    assert found.endswith('or add "# lint: narration allow: <reason>" on the line')


def test_a_line_allowance_with_a_reason_suppresses_the_finding(tmp_path: Path) -> None:
    source = "# This used to be the public name.  # lint: narration allow: compatibility\n"

    assert _findings(tmp_path, source) == []


def test_a_bare_line_allowance_does_not_suppress_the_finding(tmp_path: Path) -> None:
    assert _findings(tmp_path, "# This used to spill.  # lint: narration allow:\n")
