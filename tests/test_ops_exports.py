"""Guard the public export list of ``tileops.ops``.

``__all__`` is ordered by op family, not alphabetically, so a name left out of it
no longer stands out as a gap in a sorted list. These checks catch that.
"""

import ast
from pathlib import Path

import pytest

import tileops.ops as ops

INIT = Path(ops.__file__)


def _imported_names() -> set[str]:
    """Every name the package's ``__init__`` binds through an import."""
    tree = ast.parse(INIT.read_text())
    return {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


@pytest.mark.smoke
def test_all_covers_every_import():
    assert _imported_names() == set(ops.__all__)


@pytest.mark.smoke
def test_all_resolves():
    unresolved = [name for name in ops.__all__ if not hasattr(ops, name)]
    assert unresolved == []


@pytest.mark.smoke
def test_all_has_no_duplicates():
    assert len(ops.__all__) == len(set(ops.__all__))
