"""Programmatic access to ops_manifest.yaml.

Public entry points:

- :func:`load_workloads` — return the workloads list for an op.
"""

from __future__ import annotations

import functools
from importlib import resources
from typing import Any

import yaml

# Load ops_manifest.yaml from package data via importlib.resources.
_MANIFEST_REF = resources.files("tileops").joinpath("ops_manifest.yaml")


@functools.lru_cache(maxsize=1)
def _load_manifest() -> dict[str, Any]:
    """Load and cache the manifest. Called once per process."""
    text = _MANIFEST_REF.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data["ops"]


def load_workloads(op_name: str) -> list[dict[str, Any]]:
    """Return the workloads list for *op_name*.

    *op_name* must be the canonical PascalCase manifest key
    (e.g. ``RMSNormFwdOp``).

    >>> workloads = load_workloads("RMSNormFwdOp")
    >>> workloads[0]
    {'x_shape': [2048, 4096], 'dtypes': ['float16', 'bfloat16'], 'label': 'llama-3.1-8b-prefill'}
    """
    ops = _load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in ops_manifest.yaml")
    return ops[op_name]["workloads"]

__all__ = [
    "_load_manifest",
    "load_workloads",
]
