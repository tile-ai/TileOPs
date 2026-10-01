"""Programmatic access to the ops manifest.

The manifest is split across one or more YAML files per op family in this
package's ``spec/`` directory. Most families use a single file, but large
families (e.g., ``elementwise``) are sharded across multiple files. At load
time, all files are merged into a single ``ops`` dict. A family file is
``spec/<family>.yaml`` or ``spec/<family>_<shard>.yaml`` and holds entries of
that one family; ``spec/types.yaml`` holds the ADTs. A duplicate op name
across files, or a file that breaks the naming rule, raises `ValueError`.

Public entry points:

- `load_workloads` — return the workloads list for an op.
- `load_manifest` — return the full merged ``ops`` dict.
- `manifest_files` — list the YAML files contributing to the manifest.
- `load_adts` — return the algebraic data types shared through ``types.yaml``.
"""

from __future__ import annotations

import functools
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

import yaml

__all__ = [
    "load_adts",
    "load_manifest",
    "load_workloads",
    "manifest_files",
    "manifest_key",
    "try_load_entry",
    "types_document",
]

_PACKAGE = "tileops.manifest"
_SPEC_DIR = "spec"
_TYPES_FILE = "types.yaml"


def manifest_files() -> list[Traversable]:
    """Return the YAML files contributing to the merged manifest, sorted by name."""
    root = resources.files(_PACKAGE) / _SPEC_DIR
    return sorted(
        (
            p
            for p in root.iterdir()
            if p.is_file() and p.name.endswith(".yaml") and p.name != _TYPES_FILE
        ),
        key=lambda p: p.name,
    )


def _check_family_file(file_name: str, ops: dict[str, Any]) -> None:
    """Raise `ValueError` unless *ops* is non-empty and every entry's family names the file.

    A family file is ``<family>.yaml`` or ``<family>_<shard>.yaml``.
    """
    if not ops:
        raise ValueError(f"manifest file {file_name} holds no op entries")
    stem = file_name.removesuffix(".yaml")
    families = set()
    for name, entry in ops.items():
        family = entry.get("family") if isinstance(entry, dict) else None
        if not isinstance(family, str):
            raise ValueError(f"op {name!r} in {file_name} has no string `family`")
        if stem != family and not stem.startswith(family + "_"):
            raise ValueError(
                f"op {name!r} in {file_name} has family {family!r}; it belongs in "
                f"{family}.yaml or {family}_<shard>.yaml"
            )
        families.add(family)
    if len(families) > 1:
        raise ValueError(f"manifest file {file_name} mixes families {sorted(families)}")


@functools.lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    """Load and cache the merged ``ops`` mapping. Called once per process."""
    merged: dict[str, Any] = {}
    origin: dict[str, str] = {}
    for path in manifest_files():
        text = path.read_text(encoding="utf-8")
        ops = yaml.safe_load(text) or {}
        if not isinstance(ops, dict):
            raise ValueError(
                f"manifest file {path.name} must contain a top-level mapping of "
                f"op name -> entry, got {type(ops).__name__}"
            )
        _check_family_file(path.name, ops)
        for name, entry in ops.items():
            if name in merged:
                raise ValueError(
                    f"duplicate op {name!r} in {path.name} (already defined in {origin[name]})"
                )
            merged[name] = entry
            origin[name] = path.name
    return merged


def types_document() -> object:
    """The parsed ``types.yaml``, whatever its shape, or None when the file is absent."""
    path = resources.files(_PACKAGE) / _SPEC_DIR / _TYPES_FILE
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None


@functools.lru_cache(maxsize=1)
def load_adts() -> dict[str, Any]:
    """Return the ADTs of ``types.yaml`` that ``check_adts`` accepts; empty when the file is absent."""
    from tileops.manifest.plan import check_adts

    data = types_document()
    return check_adts(data.get("adts", {}) if isinstance(data, dict) else {})[0]


def try_load_entry(op_name: str) -> dict[str, Any] | None:
    """Return the manifest entry for *op_name*, or None if unavailable.

    Every failure — missing key, unreadable manifest — downgrades to None so
    an incomplete manifest never blocks op-class construction at import time.
    """
    try:
        ops = load_manifest()
    except Exception:
        return None
    entry = ops.get(op_name)
    return entry if isinstance(entry, dict) else None


def manifest_key(op: "str | type") -> str:
    """Return the manifest key naming *op*, which may be an Op class or that key.

    A manifest key is its op class's name, so a caller that holds the class does
    not have to repeat the name as a string.
    """
    return op.__name__ if isinstance(op, type) else op


def load_workloads(op: "str | type") -> list[dict[str, Any]]:
    """Return the workloads list for *op*, an Op class or its manifest key.

    ```python linenums="1"
    workloads = load_workloads(RMSNormFwdOp)
    workloads[0]["label"]
    ```
    """
    op_name = manifest_key(op)
    ops = load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in ops manifest")
    return ops[op_name]["workloads"]
