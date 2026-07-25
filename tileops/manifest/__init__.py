"""Programmatic access to the ops manifest.

The manifest is split across one or more YAML files per op family in this
package directory. Most families use a single file, but large families
(e.g., ``elementwise``) are sharded across multiple files. At load time,
all files are merged into a single ``ops`` dict; duplicate op names
across files raise :class:`ValueError`.

Public entry points:

- :func:`load_workloads` — return the workloads list for an op.
- :func:`load_manifest` — return the full merged ``ops`` dict.
- :func:`manifest_files` — list the YAML files contributing to the manifest.
"""

from __future__ import annotations

import functools
from importlib import resources
from typing import Any

import yaml

_PACKAGE = "tileops.manifest"


def manifest_files() -> list:
    """Return the YAML files contributing to the merged manifest, sorted by name.

    Each element is an ``importlib.resources`` ``Traversable`` (file-like
    handle with ``read_text``); typed as ``list`` for Python 3.10
    compatibility, since ``importlib.resources.abc`` was added in 3.11.
    """
    root = resources.files(_PACKAGE)
    return sorted(
        (p for p in root.iterdir() if p.is_file() and p.name.endswith(".yaml")),
        key=lambda p: p.name,
    )


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
        for name, entry in ops.items():
            if name in merged:
                raise ValueError(
                    f"duplicate op {name!r} in {path.name} (already defined in {origin[name]})"
                )
            merged[name] = entry
            origin[name] = path.name
    return merged


def load_workloads(op_name: str) -> list[dict[str, Any]]:
    """Return the workloads list for *op_name*.

    *op_name* must be the canonical PascalCase manifest key
    (e.g. ``RMSNormFwdOp``).

    >>> workloads = load_workloads("RMSNormFwdOp")
    >>> workloads[0]
    {'x_shape': [2048, 4096], 'dtypes': ['float16', 'bfloat16'], 'label': 'llama-3.1-8b-prefill'}
    """
    ops = load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in ops manifest")
    return ops[op_name]["workloads"]


# Workload keys reserved for benchmark parametrization: ``dtypes`` expands
# a workload across dtypes, ``label`` names the generated test id.
WORKLOAD_RESERVED_KEYS: frozenset[str] = frozenset({"dtypes", "label"})


def single_input_workload_contract(
    signature: dict[str, Any],
) -> tuple[str, frozenset[str]] | None:
    """Return ``(shape_key, allowed_workload_keys)`` for a signature with
    exactly one tensor input; ``None`` for any other input arity.

    The shape key is ``{input}_shape``; allowed keys are the shape key,
    :data:`WORKLOAD_RESERVED_KEYS`, and the declared ``params`` names.
    """
    sig = signature if isinstance(signature, dict) else {}
    inputs = sig.get("inputs")
    if not isinstance(inputs, dict) or len(inputs) != 1:
        return None
    (input_name,) = inputs
    if not isinstance(input_name, str):
        return None
    shape_key = f"{input_name}_shape"
    params = sig.get("params")
    param_names = (
        frozenset(k for k in params if isinstance(k, str))
        if isinstance(params, dict) else frozenset()
    )
    allowed = param_names | WORKLOAD_RESERVED_KEYS | {shape_key}
    return shape_key, allowed


__all__ = [
    "WORKLOAD_RESERVED_KEYS",
    "load_manifest",
    "load_workloads",
    "manifest_files",
    "single_input_workload_contract",
]
