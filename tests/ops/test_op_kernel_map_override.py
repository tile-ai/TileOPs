"""Runtime invariant: ``kernel_map=`` ctor override is forwarded.

Strict-parity gate C5 (``scripts/validate_manifest.py``) statically
verifies that ``__init__`` declares ``kernel_map`` (Slot S12) and that
its body calls ``self.dispatch_kernel(...)`` (Slot S13). C5 does NOT
verify that the ``kernel_map`` parameter is actually forwarded into the
``dispatch_kernel(...)`` call — an op could accept ``kernel_map=`` and
silently drop it. This file's runtime sentinel covers exactly that gap
across the elementwise bypass-fix sites the dispatch-routing change in
PR #1368 touched.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch

import tileops.ops.elementwise as elementwise_mod

pytestmark = pytest.mark.smoke

_FLOAT_DTYPE = torch.float16


def _make_sentinel(default_kernel_cls: type) -> type:
    """Return a marker subclass of the default kernel class.

    Subclassing keeps any kernel-side ``SUPPORTED_DTYPES`` /
    ``supported_archs`` filters intact.
    """

    class SentinelKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        """Marker subclass; identical behavior, distinct identity."""

    SentinelKernel.__name__ = f"Sentinel_{default_kernel_cls.__name__}"
    return SentinelKernel


def _prelu_kwargs() -> dict[str, Any]:
    return {"shape": (2, 4), "dtype": _FLOAT_DTYPE, "num_channels": 4}


def _where_kwargs() -> dict[str, Any]:
    return {
        "condition": (4,),
        "input": (4,),
        "other": (4,),
        "dtype": _FLOAT_DTYPE,
    }


def _masked_fill_kwargs() -> dict[str, Any]:
    return {
        "input": (4,),
        "mask": (4,),
        "value": (),
        "dtype": _FLOAT_DTYPE,
    }


def _masked_fill_scalar_kwargs() -> dict[str, Any]:
    return {
        "input": (4,),
        "mask": (4,),
        "value": 0.0,
        "dtype": _FLOAT_DTYPE,
    }


def _alibi_kwargs() -> dict[str, Any]:
    return {"seq_len": 8, "num_heads": 4, "dtype": _FLOAT_DTYPE}


def _sinusoidal_kwargs() -> dict[str, Any]:
    return {"seq_len": 8, "d_model": 16, "dtype": _FLOAT_DTYPE}


_TOUCHED_OPS_KWARGS: dict[str, Callable[[], dict[str, Any]]] = {
    "PreluFwdOp": _prelu_kwargs,
    "WhereFwdOp": _where_kwargs,
    "MaskedFillFwdOp": _masked_fill_kwargs,
    "MaskedFillScalarFwdOp": _masked_fill_scalar_kwargs,
    "AlibiFwdOp": _alibi_kwargs,
    "SinusoidalFwdOp": _sinusoidal_kwargs,
}


@pytest.mark.parametrize(
    ("op_name", "kwargs_factory"),
    list(_TOUCHED_OPS_KWARGS.items()),
    ids=list(_TOUCHED_OPS_KWARGS.keys()),
)
def test_touched_op_honors_kernel_map_override(
    op_name: str, kwargs_factory: Callable[[], dict[str, Any]],
) -> None:
    """Constructing with ``kernel_map={name: Sentinel}`` must install the
    sentinel onto ``self.kernel_map[name]``. Catches an op that accepts
    ``kernel_map=`` (passes C5) but drops it before ``dispatch_kernel``."""
    op_cls = getattr(elementwise_mod, op_name)
    baseline = op_cls(**kwargs_factory())
    overrides: dict[str, type] = {
        key: _make_sentinel(default_kernel_cls)
        for key, default_kernel_cls in baseline.default_kernel_map.items()
    }
    overridden = op_cls(**kwargs_factory(), kernel_map=overrides)
    for key, sentinel_cls in overrides.items():
        assert overridden.kernel_map[key] is sentinel_cls, (
            f"{op_name}: kernel_map override for {key!r} was dropped; "
            f"got {overridden.kernel_map[key]!r}"
        )
