"""Behavior tests for the unary activation family: ``inplace`` aliasing, ``approximate``
validation and registered-implementation dispatch."""

import pytest
import torch

from tileops.backend import BUILTIN
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type
from workloads.device import run_device, run_device_available
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs

_INPLACE_PARAM_FREE_OPS = (
    "ReLUFwdOp",
    "SiLUFwdOp",
    "HardswishFwdOp",
    "HardsigmoidFwdOp",
    "MishFwdOp",
    "SELUFwdOp",
)
_INPLACE_PARAMETRIC_OPS = (
    "LeakyReLUFwdOp",
    "ELUFwdOp",
    "HardtanhFwdOp",
)

# Each form of ClampFwdOp's bounds: the kernel key serving it, its interface, and the bounds.
_CLAMP_FORMS = {
    "tensor": ("clamp_tensor", "tensor_bounds", lambda x: (torch.zeros_like(x), None)),
    "number": ("clamp", "number_bounds", lambda x: (-1.0, 1.0)),
}


_TORCH_REFERENCES = {
    "ReLUFwdOp": torch.nn.functional.relu,
    "SiLUFwdOp": torch.nn.functional.silu,
    "HardswishFwdOp": torch.nn.functional.hardswish,
    "HardsigmoidFwdOp": torch.nn.functional.hardsigmoid,
    "MishFwdOp": torch.nn.functional.mish,
    "SELUFwdOp": torch.nn.functional.selu,
    "LeakyReLUFwdOp": torch.nn.functional.leaky_relu,
    "ELUFwdOp": torch.nn.functional.elu,
    "HardtanhFwdOp": torch.nn.functional.hardtanh,
}


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("form", sorted(_CLAMP_FORMS))
def test_clamp_family_registered_implementation_is_dispatched(form: str) -> None:
    """A registered subclass preferred over the in-tree kernel is the class that builds each
    form of ClampFwdOp's bounds."""
    import tileops._ops.elementwise as mod

    key, interface, bounds = _CLAMP_FORMS[form]
    default_kernel_cls = mod.ClampFwdOp.kernel_types[key]

    class MarkerKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        """Subclass marker; identical behavior, distinct identity."""

        general = False
        preferred_over = frozenset({key})

    state = registry.snapshot()
    try:
        register_kernel_type("ClampFwdOp", f"{key}_marker", MarkerKernel)
        inst = mod.ClampFwdOp(target=BUILTIN)
    finally:
        registry.restore(state)
    x = torch.randn(2, 4, device=run_device(), dtype=torch.float16)
    inst(x, *bounds(x))
    ((built,),) = [tuple(inst._entries_built(interface).values())]
    assert isinstance(built, MarkerKernel), (
        f"ClampFwdOp {form}: the registered class was not used to build the "
        f"kernel (kernel type: {type(built).__name__})"
    )


@pytest.mark.smoke
def test_nan_to_num_canonical_kwarg_names() -> None:
    """NanToNumFwdOp accepts the manifest-aligned names end-to-end."""
    import tileops._ops.elementwise as mod

    op = mod.NanToNumFwdOp(nan=0.0, posinf=1.0, neginf=-1.0)
    assert op.nan == 0.0
    assert op.posinf == 1.0
    assert op.neginf == -1.0


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    "op_name",
    _INPLACE_PARAM_FREE_OPS + _INPLACE_PARAMETRIC_OPS,
)
def test_unary_activation_inplace_true_aliases_input(op_name: str) -> None:
    """``inplace=True`` writes the activation into ``input`` and returns ``input``."""
    import tileops._ops.elementwise as mod

    op = getattr(mod, op_name)(inplace=True)
    x = torch.randn(64, dtype=torch.float16, device=run_device())
    expected = _TORCH_REFERENCES[op_name](x.clone())
    y = op(x)
    assert y is x
    compare_outputs(x, expected, ElementwiseWorkload(op_name, (x,)).verification(x))


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    "op_name",
    _INPLACE_PARAM_FREE_OPS + _INPLACE_PARAMETRIC_OPS,
)
def test_unary_activation_inplace_false_returns_fresh_tensor(op_name: str) -> None:
    """Default ``inplace=False`` neither aliases nor mutates the input."""
    import tileops._ops.elementwise as mod

    op = getattr(mod, op_name)(inplace=False)
    x = torch.randn(64, dtype=torch.float16, device=run_device())
    x_before = x.clone()
    y = op(x)
    assert y is not x, f"{op_name}: inplace=False must return a fresh tensor"
    assert torch.equal(x, x_before), f"{op_name}: inplace=False must not mutate the input tensor"


@pytest.mark.smoke
def test_gelu_approximate_validation() -> None:
    """GELUFwdOp rejects ``approximate`` values outside the manifest set."""
    import tileops._ops.elementwise as mod

    with pytest.raises(ValueError, match="approximate"):
        mod.GELUFwdOp(approximate="invalid")
