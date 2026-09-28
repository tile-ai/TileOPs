"""Locate the native sources under ``src/csrc`` that kernels compile in."""

from pathlib import Path

__all__ = ["csrc_path"]

_PACKAGE = Path(__file__).resolve().parent
# A wheel installs ``src/csrc`` as ``tileops/csrc``; a source tree or an editable
# install reads it in place, beside the package.
_ROOTS = (_PACKAGE / "csrc", _PACKAGE.parent / "csrc")


def csrc_path(rel: str) -> str:
    """Return the absolute path of ``rel``, a path relative to ``src/csrc``."""
    for root in _ROOTS:
        path = root / rel
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"{rel} is in neither {_ROOTS[0]} nor {_ROOTS[1]}")
