"""Locate the C++/CUDA sources under ``tileops/csrc`` that kernels compile in."""

from pathlib import Path

__all__ = ["csrc_path"]

_CSRC = Path(__file__).resolve().parent / "csrc"


def csrc_path(rel: str) -> str:
    """Return the absolute path of ``rel``, a path relative to ``tileops/csrc``."""
    return str(_CSRC / rel)
