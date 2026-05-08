"""L1 signature parity coverage for ``elementwise_binary`` ops.

Drives expectations from ``tileops/manifest/elementwise_binary.yaml`` so
future drift between manifest and code surfaces immediately. Covers:

- ``forward`` second-arg name matches ``signature.inputs[1].name`` — guards
  the ``BinaryOp.__init_subclass__`` ``_other_name`` rebinding for ops like
  ``PowFwdOp`` (``exponent``) and ``LerpFwdOp`` (``end``).
- Every manifest-declared input name appears as a ``forward`` parameter.
- Every manifest-declared param appears in ``__init__`` with the same
  default value.
- CPU-side construction smoke for one workload per op (no autotune).

Behavior tests (bidirectional broadcast) remain CUDA-only and live at the
end of the file.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
import yaml

import tileops.ops.elementwise as elementwise_mod

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "tileops" / "manifest" / "elementwise_binary.yaml"
)


def _load_manifest() -> Dict[str, Dict[str, Any]]:
    with _MANIFEST_PATH.open() as f:
        return yaml.safe_load(f)


_MANIFEST = _load_manifest()
_OP_NAMES = sorted(_MANIFEST.keys())


@pytest.mark.smoke
def test_manifest_covers_all_24_ops() -> None:
    """Sanity: the manifest file lists 24 elementwise_binary ops."""
    assert len(_OP_NAMES) == 24, (
        f"Expected 24 elementwise_binary manifest entries, got {len(_OP_NAMES)}"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", _OP_NAMES)
def test_op_class_exists(op_name: str) -> None:
    """Every manifest entry has a matching class in ``tileops.ops.elementwise``."""
    spec = _MANIFEST[op_name]
    if spec.get("parity_opt_out"):
        pytest.skip(f"{op_name} parity_opt_out: {spec['parity_opt_out']}")
    assert hasattr(elementwise_mod, op_name), (
        f"{op_name} missing from tileops.ops.elementwise"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", _OP_NAMES)
def test_forward_input_names_match_manifest(op_name: str) -> None:
    """``forward`` parameter names match ``signature.inputs`` keys.

    The second argument is the load-bearing case: ops like ``PowFwdOp``
    rebind ``other`` to ``exponent`` via ``BinaryOp.__init_subclass__``.
    Removing that rebinding flips this test red.
    """
    spec = _MANIFEST[op_name]
    if spec.get("parity_opt_out"):
        pytest.skip(f"{op_name} parity_opt_out: {spec['parity_opt_out']}")
    cls = getattr(elementwise_mod, op_name)
    inputs = spec["signature"].get("inputs", {}) or {}
    manifest_input_names: List[str] = list(inputs.keys())
    assert manifest_input_names, f"{op_name} manifest declares no inputs"

    fwd_sig = inspect.signature(cls.forward)
    fwd_params = [n for n in fwd_sig.parameters if n != "self"]
    n = len(manifest_input_names)
    assert fwd_params[:n] == manifest_input_names, (
        f"{op_name}: forward params={fwd_params[:n]!r} but manifest "
        f"signature.inputs={manifest_input_names!r}"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", _OP_NAMES)
def test_forward_second_arg_matches_manifest(op_name: str) -> None:
    """Dedicated guard for the ``_other_name`` rebinding contract.

    For ops where ``signature.inputs[1].name != "other"`` (``Pow`` ->
    ``exponent``, ``Lerp`` -> ``end``, ``Prelu`` -> ``weight``,
    ``MaskedFill`` -> ``mask``), the second positional ``forward`` arg
    must carry the manifest name. Removing the ``__init_subclass__``
    rebinding in ``BinaryOp`` reverts ``forward`` to the ``other`` name
    and trips this test for ``Pow`` / ``Lerp``.
    """
    spec = _MANIFEST[op_name]
    if spec.get("parity_opt_out"):
        pytest.skip(f"{op_name} parity_opt_out: {spec['parity_opt_out']}")
    cls = getattr(elementwise_mod, op_name)
    inputs = spec["signature"].get("inputs", {}) or {}
    manifest_input_names = list(inputs.keys())
    if len(manifest_input_names) < 2:
        pytest.skip(f"{op_name} declares fewer than 2 inputs")
    expected = manifest_input_names[1]

    fwd_sig = inspect.signature(cls.forward)
    fwd_params = [n for n in fwd_sig.parameters if n != "self"]
    assert len(fwd_params) >= 2, (
        f"{op_name}: forward has fewer than two non-self params: {fwd_params}"
    )
    assert fwd_params[1] == expected, (
        f"{op_name}: forward second arg {fwd_params[1]!r} != "
        f"manifest signature.inputs[1] {expected!r}"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", _OP_NAMES)
def test_init_includes_manifest_params(op_name: str) -> None:
    """Every manifest ``signature.params`` entry shows up in ``__init__``.

    Defaults must agree (``alpha=1``, ``rounding_mode=None``,
    ``weight=0.5``, ``value=0.0``). Removing one of these silently is the
    drift this test catches.
    """
    spec = _MANIFEST[op_name]
    if spec.get("parity_opt_out"):
        pytest.skip(f"{op_name} parity_opt_out: {spec['parity_opt_out']}")
    params = spec["signature"].get("params") or {}
    if not params:
        pytest.skip(f"{op_name} declares no manifest params")

    cls = getattr(elementwise_mod, op_name)
    init_sig = inspect.signature(cls.__init__)
    init_params = init_sig.parameters
    for pname, pspec in params.items():
        assert pname in init_params, (
            f"{op_name}: manifest param {pname!r} missing from __init__ "
            f"({list(init_params.keys())})"
        )
        manifest_default = pspec.get("default", inspect.Parameter.empty)
        actual_default = init_params[pname].default
        # Compare leniently for numeric defaults: 0 == 0.0, 1 == 1.0.
        if manifest_default is not inspect.Parameter.empty:
            assert actual_default == manifest_default, (
                f"{op_name}.{pname}: __init__ default {actual_default!r} "
                f"!= manifest default {manifest_default!r}"
            )


# Construction smoke ---------------------------------------------------------
#
# Drives the right __init__ kwargs from manifest workload zero. Pure-Python
# instantiation; no CUDA tensors, no autotune. Some ops compile a TileLang
# kernel at construction (Prelu, MaskedFill); that is a one-shot JIT compile,
# not autotune, and runs on CPU CI.

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _pick_dtype(workload: Dict[str, Any]) -> torch.dtype:
    dtypes = workload.get("dtypes") or ["float16"]
    for d in dtypes:
        if d in _DTYPE_MAP:
            return _DTYPE_MAP[d]
    pytest.skip(f"no torch dtype mapping for workload dtypes={dtypes!r}")


def _binary_ctor_kwargs(
    spec: Dict[str, Any], workload: Dict[str, Any], dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build ``(args, kwargs)`` for a ``BinaryOp``-shaped ctor.

    Uses ``input_shape`` for ``a_shape`` and a broadcast-compatible
    ``b_shape`` (or the workload's second-input shape if declared).
    """
    inputs = spec["signature"].get("inputs", {}) or {}
    in_keys = list(inputs.keys())
    a_shape = tuple(workload.get("input_shape", (8, 8)))
    second_key = in_keys[1] if len(in_keys) > 1 else None
    b_shape_key = f"{second_key}_shape" if second_key else None
    b_shape = (
        tuple(workload[b_shape_key])
        if b_shape_key and b_shape_key in workload
        else a_shape
    )
    return {"a_shape": a_shape, "b_shape": b_shape, "dtype": dtype}


def _ctor_for(op_name: str, spec: Dict[str, Any]) -> Tuple[type, Dict[str, Any]]:
    """Resolve ``(cls, kwargs)`` for the manifest's first workload."""
    cls = getattr(elementwise_mod, op_name)
    workloads = spec.get("workloads") or []
    if not workloads:
        pytest.skip(f"{op_name} has no workloads to drive ctor smoke")
    workload = workloads[0]
    dtype = _pick_dtype(workload)

    # Special-shape ops (non-BinaryOp templates): manifest input set diverges
    # from the standard (input, other) pair, so dispatch on op name. Each
    # branch reads the workload directly — no hidden defaults.
    if op_name == "PreluFwdOp":
        shape = tuple(workload["input_shape"])
        weight_shape = tuple(workload["weight_shape"])
        num_channels = weight_shape[0] if weight_shape else 1
        return cls, {"shape": shape, "dtype": dtype, "num_channels": num_channels}
    if op_name == "MaskedFillFwdOp":
        return cls, {
            "input": tuple(workload["input_shape"]),
            "mask": tuple(workload["mask_shape"]),
            "value": tuple(workload.get("value_shape", ())),
            "dtype": dtype,
        }
    if op_name == "MaskedFillScalarFwdOp":
        return cls, {
            "input": tuple(workload["input_shape"]),
            "mask": tuple(workload.get("mask_shape", workload["input_shape"])),
            "dtype": dtype,
        }

    # Standard BinaryOp shape: (a_shape, b_shape, dtype).
    kwargs = _binary_ctor_kwargs(spec, workload, dtype)
    return cls, kwargs


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", _OP_NAMES)
def test_construction_smoke(op_name: str) -> None:
    """Each op constructs from manifest workload zero on CPU."""
    spec = _MANIFEST[op_name]
    if spec.get("parity_opt_out"):
        pytest.skip(f"{op_name} parity_opt_out: {spec['parity_opt_out']}")
    cls, kwargs = _ctor_for(op_name, spec)
    op = cls(**kwargs)
    assert op is not None
    # Every op exposes the dtype it was built with.
    assert getattr(op, "dtype", None) == kwargs["dtype"], (
        f"{op_name}: instance.dtype != ctor dtype"
    )


# Behavior smoke (CUDA-only) -------------------------------------------------

# (op_name, dtype, gen_a, gen_b, ref_fn) — float-kernel paths.
_BROADCAST_FLOAT_OPS = [
    ("AddFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a + b),
    ("SubFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a - b),
    ("MulFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a * b),
    ("DivFwdOp",
     torch.float16,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda a, b: a / b),
    ("RemainderFwdOp",
     torch.float16,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda a, b: torch.remainder(a, b)),
    ("PowFwdOp",
     torch.float16,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.5,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") * 2.0,
     lambda a, b: torch.pow(a, b)),
    ("FloorDivideFwdOp",
     torch.float16,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda s, d: torch.rand(*s, dtype=d, device="cuda") + 0.1,
     lambda a, b: torch.floor_divide(a, b)),
    ("LerpFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: torch.lerp(a, b, 0.5)),
    ("MaximumFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: torch.maximum(a, b)),
    ("MinimumFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: torch.minimum(a, b)),
    ("EqFwdOp",
     torch.float16,
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda a, b: a == b),
    ("NeFwdOp",
     torch.float16,
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda a, b: a != b),
    ("GtFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a > b),
    ("LtFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a < b),
    ("GeFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a >= b),
    ("LeFwdOp",
     torch.float16,
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda s, d: torch.randn(*s, dtype=d, device="cuda"),
     lambda a, b: a <= b),
    ("LogicalAndFwdOp",
     torch.float16,
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda a, b: torch.logical_and(a, b)),
    ("LogicalOrFwdOp",
     torch.float16,
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda s, d: (torch.randn(*s, dtype=d, device="cuda") > 0).to(d),
     lambda a, b: torch.logical_or(a, b)),
]

_BROADCAST_INT_OPS = [
    ("BitwiseAndFwdOp",
     torch.int32,
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda a, b: torch.bitwise_and(a, b)),
    ("BitwiseOrFwdOp",
     torch.int32,
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda a, b: torch.bitwise_or(a, b)),
    ("BitwiseXorFwdOp",
     torch.int32,
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda s, d: torch.randint(-1000, 1000, s, dtype=d, device="cuda"),
     lambda a, b: torch.bitwise_xor(a, b)),
]


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_name, dtype, gen_a, gen_b, ref_fn",
    _BROADCAST_FLOAT_OPS + _BROADCAST_INT_OPS,
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_binary_op_bidirectional_broadcast(
    op_name: str, dtype: torch.dtype, gen_a, gen_b, ref_fn,
) -> None:
    """Bidirectional broadcast: (3,1) x (1,4) -> (3,4)."""
    cls = getattr(elementwise_mod, op_name)
    a_shape = (3, 1)
    b_shape = (1, 4)
    a = gen_a(a_shape, dtype)
    b = gen_b(b_shape, dtype)
    op = cls(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    out = op(a, b)
    ref = ref_fn(a, b)
    assert tuple(out.shape) == (3, 4), (
        f"{op_name}: expected output shape (3, 4), got {tuple(out.shape)}"
    )
    if out.dtype.is_floating_point:
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(out, ref.to(out.dtype))


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
