"""Locate the C++/CUDA sources under ``tileops/csrc`` that kernels compile in."""

import hashlib
from pathlib import Path

__all__ = ["csrc_include", "csrc_path"]

_CSRC = Path(__file__).resolve().parent / "csrc"


def csrc_path(name: str) -> str:
    """Return the absolute path of ``name`` under ``tileops/csrc``."""
    return str(_CSRC / name)


def _csrc_digest() -> str:
    """Hash the name and content of every file under ``tileops/csrc``."""
    digest = hashlib.sha256()
    for path in sorted(p for p in _CSRC.rglob("*") if p.is_file()):
        digest.update(path.relative_to(_CSRC).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def csrc_include(*names: str) -> list[str]:
    """Return the nvcc flags that pre-include ``names`` from ``tileops/csrc``.

    TileLang keys its kernel and binary caches on the flag strings, not on the
    files they name, so an edited header would otherwise reuse a binary built
    from the old one. The trailing ``-D`` define carries a hash of every csrc
    file; no source reads it, but it changes the flags whenever a header does.
    An edit takes effect in the next process: kernel builders keep the flags
    they were built with for the life of one.

    Args:
        names: Header file names under ``tileops/csrc``, included in order.

    Returns:
        ``["-include", <path>, ..., "-DTILEOPS_CSRC_SHA256=<hash>"]``.
    """
    flags = [flag for name in names for flag in ("-include", csrc_path(name))]
    return [*flags, f"-DTILEOPS_CSRC_SHA256={_csrc_digest()}"]
