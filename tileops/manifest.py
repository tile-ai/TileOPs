"""Programmatic access to ops_manifest.yaml.

Provides two core functions:
- load_workloads(op_name) — returns workload dicts for an op
- eval_roofline(op_name, **variables) — evaluates flops/bytes expressions
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "ops_manifest.yaml"


@functools.lru_cache(maxsize=1)
def _load_manifest() -> dict[str, Any]:
    """Load and cache the manifest. Called once per process."""
    with open(_MANIFEST_PATH) as f:
        data = yaml.safe_load(f)
    return data["ops"]


def load_workloads(op_name: str) -> list[dict[str, Any]]:
    """Return the workloads list for *op_name*.

    >>> workloads = load_workloads("rmsnorm_fwd")
    >>> workloads[0]
    {'x_shape': [2048, 4096], 'dtypes': ['float16', 'bfloat16'], 'label': 'llama-3.1-8b-prefill'}
    """
    ops = _load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in ops_manifest.yaml")
    return ops[op_name]["workloads"]


def eval_roofline(op_name: str, **variables: float) -> tuple[float, float]:
    """Evaluate roofline expressions for *op_name* with given variable bindings.

    Returns (flops, bytes).

    >>> flops, mem_bytes = eval_roofline("rmsnorm_fwd", M=2048, N=4096, elem_bytes=2)
    """
    ops = _load_manifest()
    if op_name not in ops:
        raise KeyError(f"op '{op_name}' not found in ops_manifest.yaml")

    roofline = ops[op_name]["roofline"]

    if "func" in roofline:
        raise NotImplementedError(
            f"func-mode roofline not yet supported (op={op_name})"
        )

    # Evaluate inline expressions in a restricted namespace.
    ns: dict[str, Any] = {"__builtins__": {}}
    ns.update(variables)

    flops_expr = roofline["flops"]
    bytes_expr = roofline["bytes"]

    try:
        flops = float(eval(flops_expr, ns))  # noqa: S307
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate roofline flops for '{op_name}': "
            f"{flops_expr!r} with {variables}"
        ) from exc

    try:
        mem_bytes = float(eval(bytes_expr, ns))  # noqa: S307
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate roofline bytes for '{op_name}': "
            f"{bytes_expr!r} with {variables}"
        ) from exc

    return flops, mem_bytes
