"""Behavior tests for the ``elementwise_unary_activation`` family.

The manifest L1 signature contract is enforced by
``scripts/validate_manifest.py`` for every op family; these tests
exercise activation-specific *behavior* — ``inplace=True`` aliasing
identity, ``approximate`` validation, registered-implementation dispatch
dispatch, and end-to-end correctness against the PyTorch reference.
"""

import pytest
import torch

from tileops.backend import BUILTIN, register_kernel_type, registry
from tileops.ops.elementwise._base import ELEMENTWISE
from workloads.device import run_device, run_device_available
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs

_INPLACE_PARAM_FREE_OPS = (
    "ReluFwdOp",
    "SiluFwdOp",
    "HardswishFwdOp",
    "HardsigmoidFwdOp",
    "MishFwdOp",
    "SeluFwdOp",
)
_INPLACE_PARAMETRIC_OPS = (
    "LeakyReluFwdOp",
    "EluFwdOp",
    "HardtanhFwdOp",
)

_CLAMP_OPS = ("ClampTensorFwdOp", "ClampScalarFwdOp")


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


def _construct_inplace_op(mod, op_name: str, n_total: int, inplace: bool):
    """Build an instance with the manifest-spec construction signature."""
    cls = getattr(mod, op_name)
    return cls(inplace=inplace)


def _clamp_construct_kwargs(op_name: str) -> dict:
    """The manifest params each Clamp op is constructed with; shapes are not among them."""
    if op_name == "ClampScalarFwdOp":
        return {"min": -1.0, "max": 1.0}
    return {}


@pytest.mark.in_tree_kernels
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="CUDA required")
@pytest.mark.parametrize("op_name", _CLAMP_OPS)
def test_clamp_family_registered_implementation_is_dispatched(op_name: str) -> None:
    """A registered implementation preferred over the in-tree one must reach the kernel build.

    Register, for each Clamp op, a *subclass* of its in-tree kernel under a new key and
    assert the kernel built for a dtype is an instance of that subclass — the load-bearing
    invariant is that the registered class is the one actually used to build it.
    """
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    kw = _clamp_construct_kwargs(op_name)
    ((key, default_kernel_cls),) = cls(**kw).kernel_types.items()

    class MarkerKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        """Subclass marker; identical behavior, distinct identity."""

        general = False
        preferred_over = frozenset({key})

    state = registry.snapshot()
    try:
        register_kernel_type(op_name, f"{key}_marker", MarkerKernel)
        inst = cls(**kw, target=BUILTIN)
    finally:
        registry.restore(state)
    x = torch.randn(2, 4, device="cuda", dtype=torch.float16)
    bound = torch.zeros_like(x)
    inst(x, bound) if op_name == "ClampTensorFwdOp" else inst(x)
    ((built,),) = [tuple(inst.built_entries(ELEMENTWISE).values())]
    assert isinstance(built, MarkerKernel), (
        f"{op_name}: the registered class was not used to build the "
        f"kernel (kernel type: {type(built).__name__})"
    )


@pytest.mark.smoke
def test_nan_to_num_canonical_kwarg_names() -> None:
    """NanToNumFwdOp accepts the manifest-aligned names end-to-end."""
    import tileops.ops.elementwise as mod

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
    """``inplace=True`` must mutate ``input`` and return the same tensor.

    PyTorch's contract for ``functional.relu(x, inplace=True)`` (and the
    other activations declaring ``inplace`` in their manifest entry) is
    that the returned tensor *is* ``x`` and that ``x`` now holds the
    activation output.
    """
    import tileops.ops.elementwise as mod

    n_total = 64
    dtype = torch.float16
    op = _construct_inplace_op(mod, op_name, n_total, inplace=True)
    x = torch.randn(n_total, dtype=dtype, device=run_device())
    expected = _torch_reference(op_name)(x.clone())
    y = op(x)
    assert y is x, (
        f"{op_name}: inplace=True must return the input tensor (identity); "
        f"got id(y)={id(y)} id(x)={id(x)}"
    )
    compare_outputs(x, expected, ElementwiseWorkload(op_name, (x,)).verification(x))


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    "op_name",
    _INPLACE_PARAM_FREE_OPS + _INPLACE_PARAMETRIC_OPS,
)
def test_unary_activation_inplace_false_returns_fresh_tensor(op_name: str) -> None:
    """Default ``inplace=False`` must not alias or mutate the input."""
    import tileops.ops.elementwise as mod

    n_total = 64
    dtype = torch.float16
    op = _construct_inplace_op(mod, op_name, n_total, inplace=False)
    x = torch.randn(n_total, dtype=dtype, device=run_device())
    x_before = x.clone()
    y = op(x)
    assert y is not x, f"{op_name}: inplace=False must return a fresh tensor"
    assert torch.equal(x, x_before), f"{op_name}: inplace=False must not mutate the input tensor"


@pytest.mark.smoke
def test_gelu_approximate_validation() -> None:
    """GeluFwdOp must reject ``approximate`` values outside the manifest set."""
    import tileops.ops.elementwise as mod

    with pytest.raises(ValueError, match="approximate"):
        mod.GeluFwdOp(approximate="invalid")


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_approximate_runs_through_forward(approximate: str) -> None:
    """Both ``approximate='none'`` and ``'tanh'`` must dispatch end-to-end.

    Each mode is checked against ``torch.nn.functional.gelu`` with the
    matching ``approximate`` argument so the kernel selection is
    observable from the op layer.
    """
    import tileops.ops.elementwise as mod

    n_total = 128
    dtype = torch.float16
    op = mod.GeluFwdOp(approximate=approximate)
    x = torch.randn(n_total, dtype=dtype, device=run_device())
    y = op(x)
    expected = torch.nn.functional.gelu(x, approximate=approximate)
    assert y.shape == x.shape
    compare_outputs(
        y, expected, ElementwiseWorkload("GeluFwdOp", (x,), approximate=approximate).verification(x)
    )


# Frozen ``__init__`` signatures for every unary activation Op. Tests, benches
# and codegen read them, so pulling shared ``__init__`` / ``forward`` /
# ``forward`` logic into a base or mixin must keep them byte-identical.
