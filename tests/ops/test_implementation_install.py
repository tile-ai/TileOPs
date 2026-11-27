"""Installing an op's implementations at construction resolves classes only and probes no
device; an implementation that cannot run is refused when a kernel is first selected.
"""

import pytest
import torch

from tileops._ops.elementwise import AlibiFwdOp, SinusoidalFwdOp
from tileops._ops.elementwise._base import ELEMENTWISE
from tileops._utils import forget_device_properties, get_sm_version
from tileops.backend import BUILTIN, OpNotAvailableError
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type
from workloads.device import run_device, run_device_available
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs

pytestmark = [
    pytest.mark.in_tree_kernels,
    pytest.mark.skipif(
        not run_device_available(),
        reason="implementation install tests build kernels on the current device",
    ),
]


@pytest.fixture(autouse=True)
def isolated_registry():
    """Implementations a test registers do not outlive it."""
    state = registry.snapshot()
    yield
    registry.restore(state)


@pytest.mark.smoke
def test_construction_succeeds_where_the_device_cannot_be_queried() -> None:
    """An op constructs where every device query raises."""
    import tileops._ops.elementwise as mod
    from tileops.gemm import GEMMFwdOp

    def unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no device to query")

    # The properties are cached per device, so a probe that already succeeded
    # would never reach the failing one.
    forget_device_properties()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.cuda, "get_device_capability", unavailable)
            patch.setattr(torch.cuda, "get_device_name", unavailable)
            patch.setattr(torch.cuda, "is_available", lambda: False)
            mod.ReLUFwdOp()
            GEMMFwdOp()
    finally:
        # After the patches are undone, so no answer read under them stays cached.
        forget_device_properties()


@pytest.mark.smoke
def test_auto_discovered_incompatible_kernel_is_refused_at_first_call() -> None:
    """An op whose every implementation excludes the device constructs, and its first kernel
    selection is refused."""
    from tileops._kernels.gemm import GEMMCall
    from tileops.gemm import GEMMFwdOp

    incompatible_archs = [a for a in (70, 75, 80, 86, 89, 90, 100) if a != get_sm_version()]

    class AutoDiscoveredIncompatibleOp(GEMMFwdOp):
        # Inside tileops: a class defined elsewhere cannot subclass an op of tileops.
        __module__ = "tileops._test_stub"
        kernel_types = {
            key: type(
                f"Incompatible{cls.__name__}", (cls,), {"supported_archs": incompatible_archs}
            )
            for key, cls in GEMMFwdOp.kernel_types.items()
        }

    op = AutoDiscoveredIncompatibleOp()

    with pytest.raises(OpNotAvailableError, match="no implementation serves this call"):
        op.kernel_for(
            "gemm", lambda: GEMMCall(m=128, n=128, k=128, dtype=torch.float16, trans_b=True)
        )


@pytest.mark.smoke
def test_a_kernel_declaring_no_supported_archs_runs_anywhere() -> None:
    """``supported_archs=None`` means no restriction: a forward runs through such a kernel."""
    import tileops._ops.elementwise as mod

    cls = mod.ReLUFwdOp
    ((key, default_kernel_cls),) = cls.kernel_types.items()

    class UnrestrictedKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = None
        preferred_over = frozenset({key})

    register_kernel_type("ReLUFwdOp", "relu_unrestricted", UnrestrictedKernel)
    op = cls(target=BUILTIN)
    x = torch.randn(8, device=run_device(), dtype=torch.float16)

    y = op(x)
    ((built,),) = [tuple(op._entries_built(ELEMENTWISE).values())]
    assert isinstance(built, UnrestrictedKernel), "the added implementation is what got built"
    compare_outputs(y, torch.relu(x), ElementwiseWorkload(type(op).__name__, ()).verification())


# Which implementation serves an element type is the key's own applicability, and the
# storage it computes in is its own ``entry_for``; the op passes the semantic dtype.


@pytest.mark.smoke
def test_a_bool_call_takes_the_key_preferred_over_the_general_one():
    """The bool sibling wins the bool call and builds in the storage it names."""
    from tileops._kernels.elementwise import BitwiseAndBoolStorageFwdKernel
    from tileops._ops.elementwise import BitwiseAndFwdOp

    class NativeBoolAnd(BitwiseAndBoolStorageFwdKernel):
        # Preferred over the bool key it stands in for, and so over the general one too.
        preferred_over = frozenset({"bitwise_and_bool"})

        @classmethod
        def entry_for(cls, call):
            return call, lambda: cls(call.a_shape, call.b_shape, call.dtype)

        def __init__(self, a_shape, b_shape, dtype, config=None):
            self.ctor_dtype = dtype

        def forward(self, a, b):
            return a & b

    register_kernel_type("BitwiseAndFwdOp", "native_bool_and", NativeBoolAnd)
    op = BitwiseAndFwdOp(target=BUILTIN)
    x = torch.tensor([True, False] * 32, device=run_device())

    compare_outputs(op(x, ~x), x & ~x, ElementwiseWorkload(type(op).__name__, ()).verification())
    ((built,),) = [tuple(op._entries_built(ELEMENTWISE).values())]
    assert isinstance(built, NativeBoolAnd)
    assert built.ctor_dtype == torch.bool, "the op imposed a storage dtype"


@pytest.mark.smoke
def test_an_integral_call_takes_the_key_that_states_it_serves_integers():
    """The float program and the integral answer are two keys with disjoint regions."""
    from tileops._ops.elementwise import FloorFwdOp

    op = FloorFwdOp(target=BUILTIN)
    ints = torch.arange(1, 65, device=run_device(), dtype=torch.int32)
    floats = torch.randn(64, device=run_device(), dtype=torch.float32)

    compare_outputs(op(ints), ints, ElementwiseWorkload(type(op).__name__, ()).verification())
    compare_outputs(
        op(floats), torch.floor(floats), ElementwiseWorkload(type(op).__name__, ()).verification()
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls,kwargs",
    [
        (AlibiFwdOp, {"seq_len": 8, "num_heads": 4}),
        (SinusoidalFwdOp, {"seq_len": 8, "d_model": 8}),
    ],
    ids=["AlibiFwdOp", "SinusoidalFwdOp"],
)
def test_a_generative_op_delivers_the_dtype_it_declared(op_cls, kwargs):
    """Whatever storage the implementation computes in, the op returns ``out_dtype``."""
    assert op_cls(out_dtype=torch.float16, **kwargs)().dtype == torch.float16
