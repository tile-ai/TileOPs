"""Manifest loader — reads YAML spec files and merges them into one dict.

This is a standalone re-implementation that does not depend on TileOPs.
One or more YAML files live in this package directory; each maps op names
to spec entries (signature, workloads, roofline, source).
"""

from __future__ import annotations

import functools
from importlib import resources
from typing import Any

import yaml

_PACKAGE = "manifest"


def manifest_files() -> list:
    root = resources.files(_PACKAGE)
    return sorted(
        (p for p in root.iterdir() if p.is_file() and p.name.endswith(".yaml")),
        key=lambda p: p.name,
    )


@functools.lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in manifest_files():
        text = path.read_text(encoding="utf-8")
        ops = yaml.safe_load(text) or {}
        for name, entry in ops.items():
            if name in merged:
                raise ValueError(f"duplicate op {name!r} in {path.name}")
            merged[name] = entry
    return merged


def load_workloads(op_name: str) -> list[dict[str, Any]]:
    ops = load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in manifest")
    return ops[op_name]["workloads"]


__all__ = ["load_manifest", "load_workloads", "manifest_files"]
