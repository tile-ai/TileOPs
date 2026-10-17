"""Unit tests for ``scripts/lint/comment_code_ref_lint.py``."""

from pathlib import Path

import comment_code_ref_lint as lint
import pytest

pytestmark = pytest.mark.smoke


def _findings(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return lint.findings(path)


FLAGGED = [
    pytest.param("# Checked in kernel.py:120.\n", id="file-colon-line"),
    pytest.param("# See kernel.py#L120.\n", id="file-hash-line"),
    pytest.param("# See tests/ops/test_gemm.py:L752.\n", id="path-colon-l-line"),
    pytest.param("# kernel.cuh, line 40, sets the bound.\n", id="file-comma-line"),
    pytest.param("# kernel.cuh line 40 sets the bound.\n", id="file-then-line"),
    pytest.param("# The guard at line 120 of kernel.py enforces it.\n", id="line-of-file"),
    pytest.param("# Lines 3-4 in config.yaml hold the header.\n", id="line-range-in-file"),
    pytest.param("# see foo.py:12\n", id="lowercase-see"),
    pytest.param('def f():\n    """Mirrors template.py:88."""\n', id="docstring"),
]

ACCEPTED = [
    pytest.param("# L1 cache has 64-byte line 128\n", id="cache-line-number"),
    pytest.param("# The tiles read twice go first, while L2 holds them.\n", id="cache-level"),
    pytest.param("# One cache line 128 bytes wide.\n", id="cache-line-size"),
    pytest.param("# A 128-byte line holds one row.\n", id="byte-line"),
    pytest.param("# The guard at line 120 enforces it.\n", id="line-without-a-file"),
    pytest.param("# Lines 3-4 hold the header.\n", id="line-range-without-a-file"),
    pytest.param("# kernel.py reads L2 twice.\n", id="file-then-cache-level"),
    pytest.param("# :func:`check_tiles` enforces it.\n", id="sphinx-role"),
    pytest.param("# _check_tiles enforces it.\n", id="symbol"),
    pytest.param('where = "kernel.py:120"\n', id="string-that-is-not-a-docstring"),
]


@pytest.mark.parametrize("source", FLAGGED)
def test_a_code_location_in_a_comment_or_docstring_is_reported(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source), source


@pytest.mark.parametrize("source", ACCEPTED)
def test_what_is_not_a_code_location_is_left_alone(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == [], source


def test_the_message_names_the_file_the_line_and_the_citation(tmp_path: Path) -> None:
    (found,) = _findings(tmp_path, "x = 1\n# See ops.py:12.\n")

    assert found.startswith(
        f"{tmp_path / 'probe.py'}:2: comment cites a code location ('ops.py:12')"
    )
    assert found.endswith('or add "# lint: code-ref allow: <reason>" on the line')


def test_a_line_allowance_with_a_reason_suppresses_the_finding(tmp_path: Path) -> None:
    source = "# See spec.md#L12.  # lint: code-ref allow: external-spec wording\n"

    assert _findings(tmp_path, source) == []
