"""L1 signature parity coverage for ``elementwise_binary`` ops.

Drives expectations from ``tileops/manifest/elementwise_binary.yaml`` so
future drift between manifest and code surfaces immediately. Covers:

- ``forward`` second-arg name matches ``signature.inputs[1].name`` — guards
  the ``BinaryOp.__init_subclass__`` ``_other_name`` rebinding for ops like
  ``PowFwdOp`` (``exponent``) and ``LerpFwdOp`` (``end``).
- Every manifest-declared input name appears as a ``forward`` parameter.
- Every manifest-declared param appears in ``__init__`` with the same
  default value.
- Construction-only smoke per op using test-owned shapes/dtypes (no
  manifest workload reads, no autotune); runs wherever the test suite
  runs (GPU runner, since ``BinaryOp.__init__`` resolves an SM version
  via ``torch.cuda.get_device_capability``).

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
def test_manifest_entries_resolve_to_classes() -> None:
    """Every manifest op resolves to an exported class.

    Structural check that scales as the family grows: rather than pinning
    a specific count, assert (a) every manifest entry has a matching class
    on ``tileops.ops.elementwise`` and (b) the family has at least the
    initial 24 ops, so an accidental mass-deletion still fails loud.
    """
    missing = [name for name in _OP_NAMES if not hasattr(elementwise_mod, name)]
    assert not missing, (
        f"Manifest entries missing from tileops.ops.elementwise: {missing}"
    )
    assert len(_OP_NAMES) >= 24, (
        f"elementwise_binary manifest shrank below initial floor: "
        f"got {len(_OP_NAMES)} entries"
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
# Pure-Python instantiation with test-owned shapes/dtypes — no manifest
# workload reads. No CUDA tensors, no autotune. Some ops compile a TileLang
# kernel at construction (Prelu, MaskedFill); that is a one-shot JIT compile,
# not autotune, and runs on CPU CI.
#
# Manifest reads remain legitimate for signature/parity assertions elsewhere
# in this file. Per .claude/domain-rules/testing-budget.md, fixtures must
# not be generated from tileops/manifest/ workloads.

_SMOKE_SHAPE: Tuple[int, int] = (8, 8)


def _pick_smoke_dtype(cls: type) -> torch.dtype:
    """Pick a test-owned smoke dtype.

    Default to ``float32``. For kernels that restrict dtypes (e.g. bitwise
    ops accept ints only), fall back to the first supported integer dtype
    from ``cls.kernel_cls.SUPPORTED_DTYPES``. The decision is taken in
    test code from a curated preference list, not from manifest data.
    """
    sup = getattr(getattr(cls, "kernel_cls", None), "SUPPORTED_DTYPES", None)
    if sup and torch.float32 not in sup:
        for pref in (torch.int32, torch.int64, torch.int16, torch.int8):
            if pref in sup:
                return pref
        return sup[0]
    return torch.float32


def _ctor_for(op_name: str) -> Tuple[type, Dict[str, Any]]:
    """Resolve ``(cls, kwargs)`` from test-owned smoke constants."""
    cls = getattr(elementwise_mod, op_name)
    dtype = _pick_smoke_dtype(cls)
    shape = _SMOKE_SHAPE

    # Special-shape ops (non-BinaryOp templates) take divergent kwargs.
    if op_name == "PreluFwdOp":
        num_channels = shape[1] if len(shape) > 1 else 1
        return cls, {"shape": shape, "dtype": dtype, "num_channels": num_channels}
    if op_name == "MaskedFillFwdOp":
        return cls, {
            "input": shape,
            "mask": shape,
            "value": (),
            "dtype": dtype,
        }
    if op_name == "MaskedFillScalarFwdOp":
        return cls, {
            "input": shape,
            "mask": shape,
            "dtype": dtype,
        }

    # Standard BinaryOp shape: (a_shape, b_shape, dtype).
    return cls, {"a_shape": shape, "b_shape": shape, "dtype": dtype}


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", _OP_NAMES)
def test_construction_smoke(op_name: str) -> None:
    """Each op constructs from test-owned smoke shapes/dtypes on CPU."""
    spec = _MANIFEST[op_name]
    if spec.get("parity_opt_out"):
        pytest.skip(f"{op_name} parity_opt_out: {spec['parity_opt_out']}")
    cls, kwargs = _ctor_for(op_name)
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
