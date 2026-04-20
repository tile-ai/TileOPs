#!/usr/bin/env python3
"""Compute a stable 16-char venv cache key for the CI trusted-venv scheme.

The CI workflow caches the installed virtualenv under a path derived from
this hash, so the hash must change **only** when the installed package set
would differ. Hashing the full pyproject.toml is too coarse — every tweak to
``[tool.pytest]``, ``[tool.ruff]``, or similar style/tooling sections
invalidates the cache and forces every fork PR to rebuild the venv from
scratch, even though the resolved wheel set is identical.

Narrow the hash to the sections that actually drive dependency resolution:

* ``[project]`` — dependencies, requires-python, name/version.
* ``[project.optional-dependencies]`` — the ``[dev]`` extras we install in CI.
* ``[build-system]`` — build backend and its pinned requires.

Fall back to hashing the full file if tomllib parsing fails, so a syntax
error never silently collapses all venvs to the same key. The fallback
still produces a 16-char hex digest, preserving the ``tileops_ci_venv_*``
path format downstream.

Usage:
    python scripts/ci_venv_hash.py [path/to/pyproject.toml]
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import tomllib

HASH_LEN = 16


def _narrow_hash(data: dict) -> str:
    """Hash only the dependency-relevant sections of a parsed pyproject."""
    relevant = {
        "project": data.get("project", {}),
        "project.optional-dependencies": data.get("project", {}).get(
            "optional-dependencies", {}
        ),
        "build-system": data.get("build-system", {}),
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:HASH_LEN]


def _full_file_hash(raw: bytes) -> str:
    """Fallback: hash the entire pyproject.toml as bytes."""
    return hashlib.sha256(raw).hexdigest()[:HASH_LEN]


def compute(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        # Fail open: any parse error collapses to a full-file hash so the
        # venv path still has the correct 16-hex format. Emit a warning to
        # stderr so operators can spot the degraded path in CI logs.
        print(
            f"ci_venv_hash: tomllib parse failed ({exc}); falling back to full-file hash",
            file=sys.stderr,
        )
        return _full_file_hash(raw)
    return _narrow_hash(data)


def main(argv: list[str]) -> int:
    path = pathlib.Path(argv[1]) if len(argv) > 1 else pathlib.Path("pyproject.toml")
    if not path.exists():
        print(f"ci_venv_hash: {path} does not exist", file=sys.stderr)
        return 1
    print(compute(path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
