"""Tests for installing an op's implementations at construction.

Installing them resolves classes only. It does not probe the device, so an op
constructs wherever it is imported and a target that cannot run it is refused
when a kernel is first selected — not at construction, where most ops do not
yet know which device they will run on.
"""

import pytest
import torch

from tileops.backend import BUILTIN, register_implementation, registry
from tileops.ops.elementwise._base import ELEMENTWISE
from tileops.utils import forget_device_properties, get_sm_version
from workloads.device import run_device_available
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


def _make_incompatible_arch_list() -> list[int]:
    """Return a ``supported_archs`` list that excludes the current device."""
    current = get_sm_version()
    candidates = [70, 75, 80, 86, 89, 90, 100]
    incompatible = [a for a in candidates if a != current]
    assert incompatible, "no incompatible arch candidate available"
    return incompatible


def _gemm_call():
    """A dense GEMM call every in-tree implementation would otherwise serve."""
    from tileops.kernels.gemm import GemmCall

    return GemmCall(m=128, n=128, k=128, dtype=torch.float16, trans_b=True)


@pytest.mark.smoke
def test_construction_succeeds_where_the_device_cannot_be_queried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An op constructs on a machine that cannot answer what the device is.

    Most ops do not yet know where they will run, and on hardware other than the
    one being asked about the query is not merely wrong but unavailable. Driven
    by making the probe raise rather than by naming who may call it, so importing
    it under another name or probing from elsewhere fails this too.
    """
    import tileops.ops.elementwise as mod
    from tileops.ops import GemmFwdOp

    def unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no device to query")

    # The properties are cached per device, so a probe that already succeeded
    # would never reach the failing one.
    forget_device_properties()
    monkeypatch.setattr(torch.cuda, "get_device_capability", unavailable)
    monkeypatch.setattr(torch.cuda, "get_device_name", unavailable)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        mod.ReluFwdOp()
        GemmFwdOp()
    finally:
        forget_device_properties()


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_auto_discovered_incompatible_kernel_is_refused_at_first_call() -> None:
    """The auto-discovery path is refused at the same point, the same way.

    Every implementation of the interface is made incompatible: one that is still
    available serves the call, which is what makes the interface's general
    implementation a fallback rather than a sibling of the one that went away.
    """
    from tileops.ops import GemmFwdOp

    incompatible_archs = _make_incompatible_arch_list()

    class AutoDiscoveredIncompatibleOp(GemmFwdOp):
        kernel_types = {
            key: type(
                f"Incompatible{cls.__name__}", (cls,), {"supported_archs": incompatible_archs}
            )
            for key, cls in GemmFwdOp.kernel_types.items()
        }

    op = AutoDiscoveredIncompatibleOp()

    with pytest.raises(ValueError, match="no implementation serves this call"):
        op.kernel_for("gemm", _gemm_call())


@pytest.mark.cuda_only
@pytest.mark.skipif(not run_device_available(), reason="CUDA required")
@pytest.mark.smoke
def test_a_kernel_declaring_no_supported_archs_runs_anywhere() -> None:
    """``supported_archs=None`` means no restriction, and the op runs.

    The base ``Kernel`` declares it ``Optional[list[int]]`` defaulting to
    ``None``. Anything testing membership against it would raise ``TypeError``
    instead of admitting the call, so this drives a forward through such a
    kernel rather than inspecting where it was installed.
    """
    import tileops.ops.elementwise as mod

    cls = mod.ReluFwdOp
    ((key, default_kernel_cls),) = cls.kernel_types.items()

    class UnrestrictedKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = None
        preferred_over = frozenset({key})

    register_implementation("ReluFwdOp", "relu_unrestricted", UnrestrictedKernel)
    op = cls(target=BUILTIN)
    x = torch.randn(8, device="cuda", dtype=torch.float16)

    y = op(x)
    ((built,),) = [tuple(op.built_kernels(ELEMENTWISE).values())]
    assert isinstance(built, UnrestrictedKernel), "the added implementation is what got built"
    compare_outputs(y, torch.relu(x), ElementwiseWorkload(type(op).__name__, ()).verification())


# A slot holds one entry per specialization; an enumeration that misses one
# silently tunes nothing.


# Which implementation serves an element type is the key's own applicability, and the
# storage it computes in is its own ``entry_for``. The op passes the semantic dtype and
# names neither.


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_a_bool_call_takes_the_key_preferred_over_the_general_one():
    """The bool sibling wins the bool call and builds in the storage it names."""
    from tileops.kernels.elementwise import BitwiseAndBoolStorageFwdKernel
    from tileops.ops.elementwise import BitwiseAndFwdOp

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

    register_implementation("BitwiseAndFwdOp", "native_bool_and", NativeBoolAnd)
    op = BitwiseAndFwdOp(target=BUILTIN)
    x = torch.tensor([True, False] * 32, device="cuda")

    compare_outputs(op(x, ~x), x & ~x, ElementwiseWorkload(type(op).__name__, ()).verification())
    ((built,),) = [tuple(op.built_kernels(ELEMENTWISE).values())]
    assert isinstance(built, NativeBoolAnd)
    assert built.ctor_dtype == torch.bool, "the op imposed a storage dtype"


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_an_integral_call_takes_the_key_that_states_it_serves_integers():
    """The float program and the integral answer are two keys with disjoint regions."""
    from tileops.ops.elementwise import FloorFwdOp

    op = FloorFwdOp(target=BUILTIN)
    ints = torch.arange(1, 65, device="cuda", dtype=torch.int32)
    floats = torch.randn(64, device="cuda", dtype=torch.float32)

    compare_outputs(op(ints), ints, ElementwiseWorkload(type(op).__name__, ()).verification())
    compare_outputs(
        op(floats), torch.floor(floats), ElementwiseWorkload(type(op).__name__, ()).verification()
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name,kwargs",
    [
        ("AlibiFwdOp", {"seq_len": 8, "num_heads": 4}),
        ("SinusoidalFwdOp", {"seq_len": 8, "d_model": 8}),
    ],
)
def test_a_generative_op_delivers_the_dtype_it_declared(op_name, kwargs):
    """Whatever storage the implementation computes in, the op returns ``out_dtype``."""
    import tileops.ops.elementwise as ew

    op = getattr(ew, op_name)(out_dtype=torch.float16, **kwargs)
    assert op().dtype == torch.float16
