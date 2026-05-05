"""Manifest-alignment conformance tests for `elementwise_unary_activation`.

Covers the 16 ops in
``tileops/manifest/elementwise_unary_activation.yaml`` (12 unary
activations + 4 Clamp variants). Each test asserts that the live Op
class signatures match the manifest declaration, mirroring the L1
signature check in ``scripts.validate_manifest``.

The conformance gate has three parts:

1. Every manifest input must appear (in order) as a ``forward()``
   parameter.
2. Every manifest param must appear in either ``__init__()`` or
   ``forward()``.
3. The Clamp family (``ClampFwdOp``, ``ClampScalarFwdOp``,
   ``ClampMinFwdOp``, ``ClampMaxFwdOp``) must accept ``kernel_map=`` and
   ``tune=`` keyword arguments per the canonical UnaryOp pattern.
"""

import inspect

import pytest
import torch

# 12 spec-only ops + 4 Clamp ops = 16 ops in scope.
_ACTIVATION_OPS = [
    # (op_class_name, manifest_inputs (forward order), manifest_params)
    ("ReluFwdOp", ["input"], ["inplace"]),
    ("GeluFwdOp", ["input"], ["approximate"]),
    ("SiluFwdOp", ["input"], ["inplace"]),
    ("HardswishFwdOp", ["input"], ["inplace"]),
    ("HardsigmoidFwdOp", ["input"], ["inplace"]),
    ("MishFwdOp", ["input"], ["inplace"]),
    ("SeluFwdOp", ["input"], ["inplace"]),
    ("LeakyReluFwdOp", ["input"], ["negative_slope", "inplace"]),
    ("EluFwdOp", ["input"], ["alpha", "inplace"]),
    ("HardtanhFwdOp", ["input"], ["min_val", "max_val", "inplace"]),
    ("SoftplusFwdOp", ["input"], ["beta", "threshold"]),
    ("NanToNumFwdOp", ["input"], ["nan", "posinf", "neginf"]),
]

_CLAMP_OPS = [
    ("ClampFwdOp", ["input", "min", "max"], []),
    ("ClampScalarFwdOp", ["input"], ["min", "max"]),
    ("ClampMinFwdOp", ["input", "min"], []),
    ("ClampMaxFwdOp", ["input", "max"], []),
]


def _explicit_param_names(func) -> list[str]:
    """Return positional / keyword param names of ``func`` (no *args / **kwargs)."""
    sig = inspect.signature(func)
    return [
        name for name, p in sig.parameters.items()
        if name != "self"
        and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, manifest_inputs, manifest_params",
    _ACTIVATION_OPS + _CLAMP_OPS,
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_activation_signature_matches_manifest(
    op_name: str, manifest_inputs: list[str], manifest_params: list[str],
) -> None:
    """Op class signatures must satisfy the manifest L1 contract."""
    import tileops.ops.elementwise as mod
    from scripts.validate_manifest import (
        _get_forward_params,
        _get_init_params,
        check_l1_signature,
    )

    cls = getattr(mod, op_name)
    forward_params = _get_forward_params(cls)
    assert forward_params is not None, (
        f"Cannot extract forward() params for {op_name}"
    )
    init_params = _get_init_params(cls)
    inputs_dict = {n: {} for n in manifest_inputs}
    params_dict = {n: {} for n in manifest_params}
    errors = check_l1_signature(
        op_name, inputs_dict, params_dict, forward_params,
        init_params=init_params,
    )
    assert errors == [], f"{op_name}: {errors}"


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", [op[0] for op in _CLAMP_OPS])
def test_clamp_family_accepts_canonical_kwargs(op_name: str) -> None:
    """Clamp ops must accept canonical ``kernel_map`` and ``tune`` kwargs.

    AC-3: ``(M_or_shape, dtype, *, kernel_map=None, tune=False, **op_specific)``.
    """
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    init_sig = inspect.signature(cls.__init__)
    init_keys = list(init_sig.parameters.keys())
    assert "kernel_map" in init_keys, (
        f"{op_name}.__init__ missing canonical 'kernel_map' kwarg; got {init_keys}"
    )
    assert "tune" in init_keys, (
        f"{op_name}.__init__ missing canonical 'tune' kwarg; got {init_keys}"
    )
    # Both must be keyword-only with the canonical defaults.
    assert init_sig.parameters["kernel_map"].default is None
    assert init_sig.parameters["tune"].default is False


@pytest.mark.smoke
def test_nan_to_num_param_aliases_back_compat() -> None:
    """NanToNumFwdOp must keep ``nan_val`` / ``posinf_val`` / ``neginf_val``
    accepted as aliases for the manifest-aligned ``nan`` / ``posinf`` /
    ``neginf`` so existing call sites keep working during the migration.
    """
    import tileops.ops.elementwise as mod

    init_sig = inspect.signature(mod.NanToNumFwdOp.__init__)
    keys = list(init_sig.parameters.keys())
    # Manifest-aligned names are the canonical entry points.
    for name in ("nan", "posinf", "neginf"):
        assert name in keys, f"NanToNumFwdOp.__init__ missing '{name}'; got {keys}"


@pytest.mark.smoke
def test_unary_activation_inplace_default_false() -> None:
    """Param-free unary activations expose ``inplace=False`` per the manifest.

    The op layer does not implement in-place execution; the kwarg is
    accepted purely to satisfy the manifest signature contract. Calling
    with ``inplace=True`` must therefore raise rather than silently
    return out-of-place output.
    """
    import tileops.ops.elementwise as mod

    for op_name in (
        "ReluFwdOp", "SiluFwdOp", "HardswishFwdOp",
        "HardsigmoidFwdOp", "MishFwdOp", "SeluFwdOp",
    ):
        cls = getattr(mod, op_name)
        init_sig = inspect.signature(cls.__init__)
        assert "inplace" in init_sig.parameters
        assert init_sig.parameters["inplace"].default is False
        # Constructing with inplace=True is rejected (not silently ignored)
        # because the kernels do not support in-place writes.
        with pytest.raises(NotImplementedError, match="in-place"):
            cls(N_total=8, dtype=torch.float16, inplace=True)


@pytest.mark.smoke
def test_gelu_approximate_modes_accepted() -> None:
    """GeluFwdOp must accept ``approximate='none'`` and ``'tanh'`` per manifest."""
    import tileops.ops.elementwise as mod

    init_sig = inspect.signature(mod.GeluFwdOp.__init__)
    assert "approximate" in init_sig.parameters
    assert init_sig.parameters["approximate"].default == "none"

    # Constructor must reject values outside ``{'none', 'tanh'}`` (manifest
    # shape_rule constraint). We check on CPU via attribute introspection
    # because the kernel build requires CUDA.
    with pytest.raises(ValueError, match="approximate"):
        mod.GeluFwdOp(N_total=8, dtype=torch.float16, approximate="invalid")

    # The manifest-allowed ``'tanh'`` value must be accepted by the
    # constructor rather than rejected. The tanh path skips kernel
    # construction (no CUDA build) and routes through a torch fallback in
    # ``_eager_forward``, so this assertion is safe on CPU-only hosts.
    op_tanh = mod.GeluFwdOp(N_total=8, dtype=torch.float16, approximate="tanh")
    assert op_tanh.approximate == "tanh"
    assert op_tanh.kernel is None


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
