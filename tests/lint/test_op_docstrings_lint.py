"""Unit tests for ``scripts/lint/op_docstrings_lint.py``."""

from pathlib import Path

import op_docstrings_lint as lint
import pytest

pytestmark = pytest.mark.smoke

DOCUMENTED = '''
class GemmFwdOp:
    """Matrix multiply."""

    def __init__(self, m):
        """Build the op.

        Args:
            m: Rows.
        """

    def forward(self, a):
        """Multiply."""
'''


def _problems(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    return lint.check_file(path)


def test_a_documented_op_is_left_alone(tmp_path: Path) -> None:
    assert _problems(tmp_path, DOCUMENTED) == []


@pytest.mark.parametrize(
    "old, new, expected",
    [
        pytest.param(
            '        """Multiply."""\n', "        pass\n", "has no docstring", id="missing"
        ),
        pytest.param('"""Multiply."""', '""":func:`mm`."""', "reStructuredText", id="rst"),
        pytest.param(
            '"""Matrix multiply."""',
            '"""Matrix multiply.\n\n    Args:\n        m: Rows.\n    """',
            "in its class docstring",
            id="class-args",
        ),
    ],
)
def test_a_docstring_the_site_cannot_render_is_reported(
    tmp_path: Path, old: str, new: str, expected: str
) -> None:
    (found,) = _problems(tmp_path, DOCUMENTED.replace(old, new))

    assert expected in found
