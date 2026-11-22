"""Locate the C++/CUDA sources under ``tileops/csrc`` that kernels compile in."""

import hashlib
import re
from pathlib import Path

__all__ = ["csrc_include", "csrc_path"]

_CSRC = Path(__file__).resolve().parent / "csrc"


def csrc_path(name: str) -> str:
    """Return the absolute path of ``name`` under ``tileops/csrc``."""
    return str(_CSRC / name)


def _csrc_digest(names: tuple[str, ...]) -> str:
    """Hash the name and content of ``names`` and of every csrc file they reach through
    quoted includes.

    A quoted include, the form one csrc header uses to pull in another, resolves against the
    including file's directory. Files outside ``tileops/csrc`` and system headers are not
    followed.
    """
    quoted_include = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
    root = _CSRC.resolve()
    seen: set[Path] = set()
    pending = [_CSRC / name for name in names]
    while pending:
        path = pending.pop().resolve()
        if path in seen or not path.is_file() or not path.is_relative_to(root):
            continue
        seen.add(path)
        for target in quoted_include.findall(path.read_text(errors="replace")):
            pending.append(path.parent / target)
    digest = hashlib.sha256()
    for path in sorted(seen):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def csrc_include(*names: str) -> list[str]:
    """Return the nvcc flags that pre-include ``names`` from ``tileops/csrc``.

    TileLang keys its kernel and binary caches on the flag strings, not on the
    files they name, so an edited header would otherwise reuse a binary built
    from the old one. The trailing ``-D`` define carries a hash of ``names`` and
    of every csrc header they include; no source reads it, but it changes the
    flags whenever one of those headers does. An edit to a header the kernel
    does not include leaves the flags, and so the cached binary, unchanged.
    An edit takes effect in the next process: kernel builders keep the flags
    they were built with for the life of one.

    Args:
        names: Header file names under ``tileops/csrc``, included in order.

    Returns:
        ``["-include", <path>, ..., "-DTILEOPS_CSRC_SHA256=<hash>"]``.
    """
    flags = [flag for name in names for flag in ("-include", csrc_path(name))]
    return [*flags, f"-DTILEOPS_CSRC_SHA256={_csrc_digest(names)}"]
