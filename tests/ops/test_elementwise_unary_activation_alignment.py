"""Behavior tests for the ``elementwise_unary_activation`` family.

The manifest L1 signature contract is enforced by
``scripts/validate_manifest.py`` for every op family; these tests
exercise activation-specific *behavior* — ``inplace=True`` aliasing
identity, ``approximate`` validation, kernel_map override
dispatch, and end-to-end correctness against the PyTorch reference.
"""

import pytest
import torch

_INPLACE_PARAM_FREE_OPS = (
    "ReluFwdOp", "SiluFwdOp", "HardswishFwdOp",
    "HardsigmoidFwdOp", "MishFwdOp", "SeluFwdOp",
)
_INPLACE_PARAMETRIC_OPS = (
    "LeakyReluFwdOp", "EluFwdOp", "HardtanhFwdOp",
)

_CLAMP_OPS = ("ClampFwdOp", "ClampScalarFwdOp", "ClampMinFwdOp", "ClampMaxFwdOp")


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


def _construct_inplace_op(mod, op_name: str, n_total: int, dtype: torch.dtype, inplace: bool):
    """Build an instance with the manifest-spec construction signature."""
    cls = getattr(mod, op_name)
    if op_name in _INPLACE_PARAM_FREE_OPS:
        return cls(N_total=n_total, dtype=dtype, inplace=inplace)
    return cls(n_total, dtype, inplace=inplace)


def _clamp_construct_kwargs(op_name: str) -> tuple[tuple, dict]:
    """Return ``(positional, keyword)`` args needed to construct ``op_name``."""
    shape = (2, 4)
    if op_name == "ClampFwdOp":
        return ((shape, shape, shape, torch.float16), {})
    if op_name == "ClampScalarFwdOp":
        return ((shape,), {"min": -1.0, "max": 1.0, "dtype": torch.float16})
    return ((shape, shape, torch.float16), {})


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op_name", _CLAMP_OPS)
def test_clamp_family_kernel_map_override_is_dispatched(op_name: str) -> None:
    """A user-supplied ``kernel_map`` value must reach the kernel build.

    Construct each Clamp op with a ``kernel_map`` whose value is a
    *subclass* of the default kernel and assert the constructed
    ``self.kernel`` is an instance of that subclass — the load-bearing
    invariant is that the override class is the one actually used to
    build ``self.kernel``.
    """
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    pos, kw = _clamp_construct_kwargs(op_name)
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
def test_nan_to_num_canonical_kwarg_names() -> None:
    """NanToNumFwdOp accepts the manifest-aligned names end-to-end."""
    import tileops.ops.elementwise as mod

    op = mod.NanToNumFwdOp(
        N_total=8, dtype=torch.float16, nan=0.0, posinf=1.0, neginf=-1.0,
    )
    assert op.nan == 0.0
    assert op.posinf == 1.0
    assert op.neginf == -1.0


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
    activation output.
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
@pytest.mark.parametrize(
    "op_name", _INPLACE_PARAM_FREE_OPS + _INPLACE_PARAMETRIC_OPS,
)
def test_unary_activation_inplace_false_returns_fresh_tensor(op_name: str) -> None:
    """Default ``inplace=False`` must not alias or mutate the input."""
    import tileops.ops.elementwise as mod

    n_total = 64
    dtype = torch.float16
    op = _construct_inplace_op(mod, op_name, n_total, dtype, inplace=False)
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    x_before = x.clone()
    y = op(x)
    assert y is not x, f"{op_name}: inplace=False must return a fresh tensor"
    assert torch.equal(x, x_before), (
        f"{op_name}: inplace=False must not mutate the input tensor"
    )


@pytest.mark.smoke
def test_gelu_approximate_validation() -> None:
    """GeluFwdOp must accept ``approximate='none'`` and reject invalid values.

    ``approximate='tanh'`` is a manifest-allowed value but no fused
    tanh-GELU unary kernel is implemented yet (see follow-up issue
    referenced in the ``NotImplementedError`` message).
    """
    import tileops.ops.elementwise as mod

    with pytest.raises(ValueError, match="approximate"):
        mod.GeluFwdOp(N_total=8, dtype=torch.float16, approximate="invalid")
    with pytest.raises(NotImplementedError, match=r"tanh"):
        mod.GeluFwdOp(N_total=8, dtype=torch.float16, approximate="tanh")


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gelu_approximate_none_runs_through_forward() -> None:
    """GeluFwdOp(approximate='none') must dispatch through ``forward()``."""
    import tileops.ops.elementwise as mod

    n_total = 128
    dtype = torch.float16
    op = mod.GeluFwdOp(N_total=n_total, dtype=dtype)
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    y = op(x)
    expected = torch.nn.functional.gelu(x, approximate="none")
    assert y.shape == x.shape
    assert torch.allclose(y, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
