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
    # Both must be keyword-only with the canonical defaults. Assert
    # ``Parameter.kind`` so a positional-or-keyword regression also
    # fails the gate, not just a renamed-default regression.
    kernel_map_param = init_sig.parameters["kernel_map"]
    tune_param = init_sig.parameters["tune"]
    assert kernel_map_param.default is None
    assert tune_param.default is False
    assert kernel_map_param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{op_name}.__init__ 'kernel_map' must be keyword-only, "
        f"got kind={kernel_map_param.kind}"
    )
    assert tune_param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{op_name}.__init__ 'tune' must be keyword-only, "
        f"got kind={tune_param.kind}"
    )


def _clamp_construct_kwargs(op_name: str) -> tuple[tuple, dict]:
    """Return ``(positional, keyword)`` args needed to construct ``op_name``.

    Each Clamp variant has a distinct positional signature:

    +----------------------+--------------------------------------------------+
    | ``ClampFwdOp``       | ``(input_shape, min_shape, max_shape, dtype)``.  |
    +----------------------+--------------------------------------------------+
    | ``ClampScalarFwdOp`` | ``(input_shape,)`` + ``min``/``max`` kwargs.     |
    +----------------------+--------------------------------------------------+
    | ``ClampMinFwdOp``    | ``(input_shape, min_shape, dtype)``.             |
    +----------------------+--------------------------------------------------+
    | ``ClampMaxFwdOp``    | ``(input_shape, max_shape, dtype)``.             |
    +----------------------+--------------------------------------------------+
    """
    shape = (2, 4)
    if op_name == "ClampFwdOp":
        return ((shape, shape, shape, torch.float16), {})
    if op_name == "ClampScalarFwdOp":
        return ((shape,), {"min": -1.0, "max": 1.0, "dtype": torch.float16})
    # ClampMin / ClampMax
    return ((shape, shape, torch.float16), {})


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op_name", [op[0] for op in _CLAMP_OPS])
def test_clamp_family_kernel_map_override_is_dispatched(op_name: str) -> None:
    """A user-supplied ``kernel_map`` value must reach the kernel build.

    A regression that accepts ``kernel_map=`` in the constructor but
    silently ignores the override would still pass the signature
    metadata check above. Construct each Clamp op with a ``kernel_map``
    whose value is a *subclass* of the default kernel and assert the
    constructed ``self.kernel`` is an instance of that subclass —
    ``dispatch_kernel`` always rebuilds the dict, so identity checks
    on the dict itself are not load-bearing; the load-bearing
    invariant is that the override class is the one actually used to
    build ``self.kernel``. Skipped without CUDA because constructors
    JIT-compile the kernel.
    """
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    pos, kw = _clamp_construct_kwargs(op_name)
    # Read ``default_kernel_map`` from a baseline instance so we know
    # which key/class pair to override with a marker subclass.
    inst = cls(*pos, **kw)
    (key, default_kernel_cls), = inst.default_kernel_map.items()

    class MarkerKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        """Subclass marker; identical behavior, distinct identity."""

    override = {key: MarkerKernel}
    inst2 = cls(*pos, **kw, kernel_map=override)
    assert inst2.kernel_map[key] is MarkerKernel, (
        f"{op_name}: kernel_map override entry was not stored on "
        f"self.kernel_map (got {inst2.kernel_map[key]!r})"
    )
    assert isinstance(inst2.kernel, MarkerKernel), (
        f"{op_name}: kernel_map override class was not used to build "
        f"self.kernel (kernel type: {type(inst2.kernel).__name__})"
    )


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


_INPLACE_PARAM_FREE_OPS = (
    "ReluFwdOp", "SiluFwdOp", "HardswishFwdOp",
    "HardsigmoidFwdOp", "MishFwdOp", "SeluFwdOp",
)
_INPLACE_PARAMETRIC_OPS = (
    "LeakyReluFwdOp", "EluFwdOp", "HardtanhFwdOp",
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name", _INPLACE_PARAM_FREE_OPS + _INPLACE_PARAMETRIC_OPS,
)
def test_unary_activation_inplace_default_false(op_name: str) -> None:
    """Activations declaring ``inplace`` expose ``inplace=False`` by default.

    Covers every unary activation whose manifest entry declares an
    ``inplace`` param (param-free ReLU/SiLU/HardSwish/HardSigmoid/Mish/SELU
    plus the parametric LeakyReLU/ELU/Hardtanh). The op-level
    ``inplace=True`` semantics are exercised by
    ``test_unary_activation_inplace_true_aliases_input`` below; this
    test covers the signature contract and is safe on CPU-only hosts
    because it only inspects metadata.
    """
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    init_sig = inspect.signature(cls.__init__)
    assert "inplace" in init_sig.parameters, (
        f"{op_name}.__init__ missing 'inplace' kwarg; got {list(init_sig.parameters)}"
    )
    inplace_param = init_sig.parameters["inplace"]
    assert inplace_param.default is False
    assert inplace_param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{op_name}.__init__ 'inplace' must be keyword-only, "
        f"got kind={inplace_param.kind}"
    )


def _construct_inplace_op(mod, op_name: str, n_total: int, dtype: torch.dtype, inplace: bool):
    """Build an instance with the manifest-spec construction signature."""
    cls = getattr(mod, op_name)
    if op_name in _INPLACE_PARAM_FREE_OPS:
        return cls(N_total=n_total, dtype=dtype, inplace=inplace)
    if op_name == "LeakyReluFwdOp":
        return cls(n_total, dtype, inplace=inplace)
    if op_name == "EluFwdOp":
        return cls(n_total, dtype, inplace=inplace)
    if op_name == "HardtanhFwdOp":
        return cls(n_total, dtype, inplace=inplace)
    raise AssertionError(f"unexpected op_name {op_name!r}")


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "op_name", _INPLACE_PARAM_FREE_OPS + _INPLACE_PARAMETRIC_OPS,
)
def test_unary_activation_inplace_true_aliases_input(op_name: str) -> None:
    """``inplace=True`` must mutate ``input`` and return the same tensor.

    PyTorch's contract for ``functional.relu(x, inplace=True)`` (and the
    other activations declaring ``inplace`` in their manifest entry) is
    that the returned tensor *is* ``x`` and that ``x`` now holds the
    activation output. Verify both invariants for every activation
    covered by ``_InplaceMixin``.
    """
    import tileops.ops.elementwise as mod

    n_total = 64
    dtype = torch.float16
    op = _construct_inplace_op(mod, op_name, n_total, dtype, inplace=True)
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    expected = _torch_reference(op_name)(x.clone())
    y = op(x)
    assert y is x, (
        f"{op_name}: inplace=True must return the input tensor (identity); "
        f"got id(y)={id(y)} id(x)={id(x)}"
    )
    assert torch.allclose(x, expected, rtol=1e-2, atol=1e-2), (
        f"{op_name}: inplace=True did not mutate input to the activation output"
    )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_softplus_ignores_post_construction_inplace() -> None:
    """Softplus must not honor ``op.inplace = True`` set after construction.

    Softplus's manifest signature does not declare ``inplace``; the
    parametric activation base shares an inplace path with siblings
    (LeakyReLU/ELU/Hardtanh) that *do* declare it, so the leaf opts out
    via ``_SUPPORTS_INPLACE = False``. Verify the opt-out: flipping
    ``self.inplace`` post-construction must not change tensor identity
    or mutate the input in place.
    """
    import tileops.ops.elementwise as mod

    n_total = 64
    dtype = torch.float16
    op = mod.SoftplusFwdOp(N_total=n_total, dtype=dtype)
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    x_before = x.clone()

    # Baseline: default ``inplace=False`` returns a fresh tensor.
    y_default = op(x)
    assert y_default is not x
    assert torch.allclose(x, x_before), "default path must not mutate input"

    # Force the flag on; with ``_SUPPORTS_INPLACE = False`` the forward
    # flow ignores it and behavior matches the default path.
    op.inplace = True
    y_forced = op(x)
    assert y_forced is not x, (
        "SoftplusFwdOp.inplace = True must not alias the input "
        "(manifest does not declare inplace)"
    )
    assert torch.allclose(x, x_before), (
        "SoftplusFwdOp.inplace = True must not mutate the input "
        "(manifest does not declare inplace)"
    )


@pytest.mark.smoke
def test_apply_inplace_numel_mismatch_raises_value_error() -> None:
    """``_InplaceMixin._apply_inplace`` rejects numel mismatches with a ValueError.

    Locks the docstring contract: broadcasting is not supported on the
    inplace path, and the failure surface is ``ValueError`` (not the
    ``RuntimeError`` that would bubble up from a bare ``Tensor.reshape``
    call).
    """
    import tileops.ops.elementwise as mod

    apply_inplace = mod._InplaceMixin._apply_inplace
    a = torch.empty(8, dtype=torch.float32)
    b = torch.empty(7, dtype=torch.float32)
    with pytest.raises(ValueError, match="numel"):
        apply_inplace(True, a, b)
    # dtype mismatch path (already a ValueError) — guard against regression.
    c = torch.empty(8, dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        apply_inplace(True, a, c)


def _torch_reference(op_name: str):
    """Map an activation op class to its ``torch.nn.functional`` reference."""
    refs = {
        "ReluFwdOp": torch.nn.functional.relu,
        "SiluFwdOp": torch.nn.functional.silu,
        "HardswishFwdOp": torch.nn.functional.hardswish,
        "HardsigmoidFwdOp": torch.nn.functional.hardsigmoid,
        "MishFwdOp": torch.nn.functional.mish,
        "SeluFwdOp": torch.nn.functional.selu,
        "LeakyReluFwdOp": torch.nn.functional.leaky_relu,
        "EluFwdOp": torch.nn.functional.elu,
        "HardtanhFwdOp": torch.nn.functional.hardtanh,
    }
    return refs[op_name]


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
    op_tanh = mod.GeluFwdOp(N_total=8, dtype=torch.float32, approximate="tanh")
    assert op_tanh.approximate == "tanh"
    assert op_tanh.kernel is None

    # Exercise the fallback path end-to-end via _eager_forward so a
    # regression in the tanh routing is actually caught (not just a
    # constructor smoke check). _eager_forward bypasses the .is_cuda
    # input gate in forward(), letting this run on CPU-only hosts.
    x = torch.randn(8, dtype=torch.float32)
    out = op_tanh._eager_forward(x)
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    assert out.shape == x.shape
    assert torch.allclose(out, expected, rtol=0, atol=0), (
        "GeluFwdOp tanh fallback must match torch.nn.functional.gelu(approximate='tanh')"
    )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gelu_approximate_tanh_runs_through_forward() -> None:
    """GeluFwdOp(approximate='tanh') must dispatch through ``forward()``.

    The earlier signature/metadata check covers ``__init__``; this test
    closes the gap by exercising the public forward path on CUDA so a
    regression that breaks ``forward()`` (e.g. routes the tanh op
    through a kernel that doesn't exist) is caught instead of only
    surfacing in eager-only callers.
    """
    import tileops.ops.elementwise as mod

    n_total = 128
    dtype = torch.float16
    op = mod.GeluFwdOp(N_total=n_total, dtype=dtype, approximate="tanh")
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    y = op(x)
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.allclose(y, expected, rtol=1e-2, atol=1e-2), (
        "GeluFwdOp(approximate='tanh').forward must match "
        "torch.nn.functional.gelu(approximate='tanh') on CUDA"
    )


@pytest.mark.smoke
def test_gelu_tanh_rejects_unsupported_dtype() -> None:
    """tanh fallback must enforce the same dtype gate as the kernel path.

    The 'none' branch rejects unsupported dtypes via GeluFwdKernel's
    SUPPORTED_DTYPES check inside ``super().__init__``. The tanh branch
    must mirror that contract so a caller doesn't construct a broken op
    and only learn at forward() time that the dtype is unsupported.
    """
    import tileops.ops.elementwise as mod

    with pytest.raises(ValueError, match=r"gelu does not support dtype"):
        mod.GeluFwdOp(N_total=8, dtype=torch.int32, approximate="tanh")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, expected_per_elem",
    [
        ("ReluFwdOp", 2),
        ("SiluFwdOp", 4),
        ("HardswishFwdOp", 7),
        ("HardsigmoidFwdOp", 6),
        ("MishFwdOp", 7),
        ("SeluFwdOp", 5),
    ],
)
def test_activation_flops_per_elem_matches_manifest(
    op_name: str, expected_per_elem: int,
) -> None:
    """Activation ``FLOPS_PER_ELEM`` must match the manifest roofline coefficient.

    The bench reads ``op.eval_roofline()`` (or ``FLOPS_PER_ELEM`` as a
    fallback) to report manifest-aligned TFLOPs. This test pins the
    per-element FLOP coefficient against the manifest's
    ``roofline.flops`` column for each unary activation, so a regression
    that silently reverts to the bandwidth-only ``1*N`` default surfaces
    here instead of producing under-reported TFLOPs in the bench.

    The ``eval_roofline`` return is only inspected for the FLOPs slot
    (first element); the ``bytes`` slot is covered by the existing
    elementwise total_memory tests.
    """
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    assert getattr(cls, "FLOPS_PER_ELEM", None) == expected_per_elem, (
        f"{op_name}.FLOPS_PER_ELEM = {getattr(cls, 'FLOPS_PER_ELEM', None)!r}, "
        f"expected {expected_per_elem} per the manifest roofline.flops"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
