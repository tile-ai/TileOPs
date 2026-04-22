"""Programmatic access to ops_manifest.yaml.

Public entry points:

- :func:`load_workloads` — return the workloads list for an op.
- :func:`eval_roofline` / :func:`resolve_roofline_vars` /
  :func:`has_roofline_vars` — legacy roofline evaluator surface,
  re-exported from :mod:`tileops.manifest_legacy_roofline` for
  backward compatibility. Scheduled for removal once per-op codegen
  emits each Op's own ``eval_roofline()``. Do not add new callers.
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


# Re-export the legacy roofline-evaluator surface for backward
# compatibility. The implementation now lives in
# ``tileops.manifest_legacy_roofline`` so the migration boundary is
# visible to readers and the final deletion is surgical. The re-export
# is below ``_load_manifest`` so the submodule's ``from .manifest
# import _load_manifest`` resolves during its own module-load.
from .manifest_legacy_roofline import (  # noqa: E402
    _safe_eval,
    eval_roofline,
    has_roofline_vars,
    resolve_roofline_vars,
)

__all__ = [
    "_load_manifest",
    "_safe_eval",
    "eval_roofline",
    "has_roofline_vars",
    "load_workloads",
    "resolve_roofline_vars",
]
