"""Tests for the shared ``_install_kernel_map`` validation path.

A user-supplied ``kernel_map`` and the auto-discovered ``default_kernel_map``
must traverse the same validate-and-install path in the Op base, so that
architecture-compatibility checks fire identically regardless of provenance.
"""


import pytest
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="install_kernel_map tests query CUDA arch via get_sm_version()",
)


def _make_incompatible_arch_list() -> list[int]:
    """Return a ``supported_archs`` list that excludes the current device."""
    current = get_sm_version()
    candidates = [70, 75, 80, 86, 89, 90, 100]
    incompatible = [a for a in candidates if a != current]
    assert incompatible, "no incompatible arch candidate available"
    return incompatible


@pytest.mark.smoke
def test_install_kernel_map_user_supplied_incompatible_raises_valueerror() -> None:
    """User-supplied incompatible kernel raises ``ValueError`` (auto-discovery class)."""
    import tileops.ops.elementwise as mod

    cls = mod.ReluFwdOp
    inst = cls(N_total=8)
    (key, default_kernel_cls), = inst.default_kernel_map.items()
    incompatible_archs = _make_incompatible_arch_list()

    class IncompatibleKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = incompatible_archs

    with pytest.raises(ValueError, match="not supported on architecture"):
        cls(N_total=8, kernel_map={key: IncompatibleKernel})


@pytest.mark.smoke
def test_install_kernel_map_auto_discovery_incompatible_raises_same_class() -> None:
    """Auto-discovery path raises the same ``ValueError`` on identical input.

    Build an Op subclass whose ``default_kernel_map`` already points at an
    arch-incompatible kernel; constructing it must raise the same class
    that the user-supplied path produces.
    """
    import tileops.ops.elementwise as mod

    base_cls = mod.ReluFwdOp
    base_inst = base_cls(N_total=8)
    (key, default_kernel_cls), = base_inst.default_kernel_map.items()
    incompatible_archs = _make_incompatible_arch_list()

    class IncompatibleKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = incompatible_archs

    class AutoDiscoveredIncompatibleOp(base_cls):  # type: ignore[misc, valid-type]
        @property
        def default_kernel_map(self) -> dict[str, Kernel]:
            return {key: IncompatibleKernel}

    with pytest.raises(ValueError, match="not supported on architecture"):
        AutoDiscoveredIncompatibleOp(N_total=8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.smoke
def test_install_kernel_map_compatible_override_forward_bit_identical() -> None:
    """A compatible user-supplied override yields bit-identical forward output.

    Build an op with the default kernel and the same op with a marker
    subclass override (same kernel logic, distinct identity). Both must
    produce the exact same forward output on identical input.
    """
    import tileops.ops.elementwise as mod

    cls = mod.ReluFwdOp
    n_total = 128
    dtype = torch.float16

    baseline = cls(N_total=n_total)
    (key, default_kernel_cls), = baseline.default_kernel_map.items()

    class MarkerKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        """Subclass marker; identical behavior, distinct identity."""

    overridden = cls(
        N_total=n_total, kernel_map={key: MarkerKernel},
    )
    assert isinstance(overridden._entry(torch.float16).kernel, MarkerKernel)

    torch.manual_seed(0)
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    y_baseline = baseline(x.clone())
    y_overridden = overridden(x.clone())
    assert torch.equal(y_baseline, y_overridden), (
        "compatible kernel_map override must yield bit-identical forward output"
    )


@pytest.mark.smoke
def test_install_kernel_map_supported_archs_none() -> None:
    """A kernel with ``supported_archs=None`` installs without raising.

    The base ``Kernel`` class declares ``supported_archs: Optional[list[int]]``
    defaulting to ``None``. The validate-and-install path must treat ``None``
    as "no arch restriction" rather than attempting ``in None``, which would
    raise ``TypeError``.
    """
    import tileops.ops.elementwise as mod

    cls = mod.ReluFwdOp
    inst = cls(N_total=8)
    (key, default_kernel_cls), = inst.default_kernel_map.items()

    class UnrestrictedKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = None

    overridden = cls(N_total=8, kernel_map={key: UnrestrictedKernel})
    assert overridden.kernel_map[key] is UnrestrictedKernel


@pytest.mark.smoke
def test_install_kernel_map_is_private_helper_only() -> None:
    """Refactor exposes ``_install_kernel_map`` only — no new public API.

    Guards AC: the shared path is private (leading underscore); the public
    surface is unchanged (``dispatch_kernel``, ``kernel_map``, ``tune``).
    """
    from tileops.ops.op_base import Op

    assert hasattr(Op, "_install_kernel_map")
    assert hasattr(Op, "dispatch_kernel")
    public_names = [n for n in vars(Op) if not n.startswith("_")]
    assert "install_kernel_map" not in public_names


# ``Op.autotune`` walks whatever the op used as its kernel cache. The cache
# value is a bare kernel for some families and a record for others, so a
# traversal that only descends containers silently tunes nothing.


@pytest.mark.smoke
def test_autotune_reaches_kernels_held_in_a_record():
    """A dataclass-valued cache must not hide its kernels from autotune."""
    import dataclasses

    from tileops.ops.op_base import _iter_kernels

    class _FakeKernel(Kernel):
        def __init__(self):
            pass

        def forward(self, *args, **kwargs):
            raise AssertionError("never called")

    @dataclasses.dataclass(frozen=True)
    class _Entry:
        kernel: object
        compute_dtype: torch.dtype

    k = _FakeKernel()
    assert _iter_kernels({torch.float16: _Entry(k, torch.float16)}) == [k]
    assert _iter_kernels(_Entry(None, torch.float16)) == []


@pytest.mark.smoke
def test_autotune_reaches_elementwise_entries():
    """The elementwise cache is record-valued; every built kernel must be seen."""
    from tileops.ops.elementwise import AbsFwdOp
    from tileops.ops.op_base import _iter_kernels

    op = AbsFwdOp(N_total=256)
    for dtype in (torch.float16, torch.float32):
        op(torch.randn(256, device="cuda", dtype=dtype))

    found = _iter_kernels(op._entries)
    assert len(found) == 2, f"autotune would see {len(found)} of 2 built kernels"


# A backend answers which implementation serves a dtype, and in what storage.
# The op passes the semantic dtype and names neither, so a backend that handles
# bool natively is served by its own implementation rather than being handed a
# uint8 construction argument the op chose for it.


@pytest.mark.smoke
def test_native_bool_backend_is_constructed_with_bool():
    """An override declaring no bool substitute gets the semantic dtype."""
    from tileops.kernels.elementwise import BitwiseAndFwdKernel
    from tileops.ops.elementwise import BitwiseAndFwdOp

    class NativeBoolAnd(BitwiseAndFwdKernel):
        SUPPORTED_DTYPES = (torch.bool,)
        BOOL_IMPL = None  # this backend needs no uint8 detour

        def __init__(self, a_shape, b_shape, dtype, config=None, tune=False):
            self.ctor_dtype = dtype

        def forward(self, a, b):
            return a & b

    op = BitwiseAndFwdOp((64,), (64,), kernel_map={"bitwise_and": NativeBoolAnd})
    x = torch.tensor([True, False] * 32, device="cuda")

    torch.testing.assert_close(op(x, ~x), x & ~x)
    entry = op._entries[torch.bool]
    assert isinstance(entry.kernel, NativeBoolAnd)
    assert entry.kernel.ctor_dtype == torch.bool, "the op imposed a storage dtype"
    assert sorted(op.kernel_map) == ["bitwise_and"], "a second slot survives"


@pytest.mark.smoke
def test_default_backend_still_routes_bool_through_uint8():
    """The shipped kernels declare the substitution, so bool keeps working."""
    from tileops.kernels.elementwise import (
        BitwiseAndBoolStorageFwdKernel,
        BitwiseAndFwdKernel,
    )

    assert BitwiseAndFwdKernel.specialize(torch.bool) == (
        BitwiseAndBoolStorageFwdKernel, torch.uint8,
    )
    assert BitwiseAndFwdKernel.specialize(torch.int32) == (
        BitwiseAndFwdKernel, torch.int32,
    )


@pytest.mark.smoke
def test_integer_fallback_yields_to_a_backend_that_serves_integers():
    """The op-level integer handler is a fallback, not a decision.

    The shipped kernels are float-only, so integers are answered by the op. A
    backend declaring integer support must be used instead — intercepting before
    asking would discard the override silently, and the caller would never learn
    that the kernel they supplied was ignored.
    """
    from tileops.kernels.elementwise import FloorFwdKernel
    from tileops.ops.elementwise import FloorFwdOp

    x = torch.arange(1, 65, device="cuda", dtype=torch.int32)

    shipped = FloorFwdOp(N_total=64)
    torch.testing.assert_close(shipped(x), x)
    assert shipped._entries[torch.int32].kernel is None, "float-only kernel was used"

    class NativeIntFloor(FloorFwdKernel):
        SUPPORTED_DTYPES = (torch.int32, torch.float32)

        def __init__(self, N_total, dtype, config=None, tune=False):
            self.dtype = dtype

        def forward(self, x):
            return x.clone()

    op = FloorFwdOp(N_total=64, kernel_map={"floor": NativeIntFloor})
    torch.testing.assert_close(op(x), x)
    entry = op._entries[torch.int32]
    assert isinstance(entry.kernel, NativeIntFloor), "the override was bypassed"
    assert entry.kernel.dtype == torch.int32


# Rounds of review kept surfacing the same shape: an op deciding something on the
# backend's behalf instead of asking it. The instances were removable one at a
# time; the pattern was not, until stated as a contract every builder must meet.
#
# A syntactic check ("no `_build_entry` may index `kernel_map`") is not that
# contract: it cannot see a helper that indexes it elsewhere, an alias, a
# construction in `__init__`, or a `_build_kernel_instance` override that ignores
# the class it was handed. Injecting a backend and observing what gets built
# does, wherever the construction happens.


class _ProbeKernel:
    """Stands in for whatever the injected backend says should be built."""

    instances: list = []
    SUPPORTED_DTYPES = None  # the probe accepts whatever it is handed
    supported_archs = None

    def __init__(self, *args, **kwargs):
        type(self).instances.append(self)
        self.ctor_args = args
        self.ctor_kwargs = kwargs
        self.ctor_dtype = next((a for a in args if isinstance(a, torch.dtype)), None)

    def __call__(self, *args, **kwargs):
        raise AssertionError("the probe is not meant to run")


def _probe_backend(probe_dtype: torch.dtype):
    """A backend whose ``specialize`` answers with the probe and *probe_dtype*."""

    class ProbeBackend:
        SUPPORTED_DTYPES = None
        supported_archs = None  # installable on any arch

        @classmethod
        def specialize(cls, dtype):
            return _ProbeKernel, probe_dtype

    return ProbeBackend


# One entry per distinct builder shape: (op class, ctor kwargs, slots, entry args).
_BUILDER_SHAPES = [
    ("AbsFwdOp", {"N_total": 64}, ["abs"], ()),
    ("ReciprocalFwdOp", {"N_total": 64}, ["reciprocal"], ()),
    ("FloorFwdOp", {"N_total": 64}, ["floor"], ()),
    ("EluFwdOp", {"N_total": 64, "alpha": 1.0}, ["elu"], ()),
    ("AddFwdOp", {"a_shape": (64,), "b_shape": (64,)}, ["add"], ()),
    ("EqFwdOp", {"a_shape": (64,), "b_shape": (64,)}, ["eq"], ()),
    ("BitwiseAndFwdOp", {"a_shape": (64,), "b_shape": (64,)}, ["bitwise_and"], ()),
    ("LogicalNotFwdOp", {"N_total": 64}, ["logical_not"], ()),
    ("LerpTensorFwdOp", {"input": (64,), "end": (64,), "weight": (64,)}, ["lerp_tensor"], ()),
    ("SiluAndMulFwdOp", {"M": 16, "N": 8}, ["silu_and_mul"], (16, 8)),
    ("WhereFwdOp", {"condition": (64,), "input": (64,), "other": (64,)}, ["where"], ()),
    ("PreluFwdOp", {"shape": (4, 8), "num_channels": 8}, ["prelu"], ()),
    ("NanToNumFwdOp", {"N_total": 64}, ["nan_to_num"], ()),
    ("ClampFwdOp", {"input": (64,), "min": (64,), "max": (64,)}, ["clamp_tensor"], ()),
    ("ClampScalarFwdOp", {"input": (64,), "min": 0.0}, ["clamp"], ()),
    ("MaskedFillScalarFwdOp", {"input": (64,), "mask": (64,), "value": 1.0}, ["masked_fill"], ()),
    ("MaskedFillFwdOp", {"input": (64,), "mask": (64,), "value": ()},
     ["masked_fill_tensor_value"], ()),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_name", "kwargs", "slots", "entry_args"),
    _BUILDER_SHAPES,
    ids=[c[0] for c in _BUILDER_SHAPES],
)
def test_builder_constructs_what_the_backend_specialized(op_name, kwargs, slots, entry_args):
    """Whatever `specialize` returns is what gets built, and nothing else."""
    import tileops.ops.elementwise as ew

    probe_dtype = torch.bfloat16
    backend = _probe_backend(probe_dtype)
    op = getattr(ew, op_name)(kernel_map={slot: backend for slot in slots}, **kwargs)

    _ProbeKernel.instances = []
    entry = op._build_entry(torch.float16, *entry_args)

    assert len(_ProbeKernel.instances) == 1, (
        f"{op_name} built {len(_ProbeKernel.instances)} kernels; the backend's "
        "answer was ignored or something else was constructed too"
    )
    assert entry.kernel is _ProbeKernel.instances[0]
    assert entry.kernel.ctor_dtype == probe_dtype, (
        f"{op_name} constructed with {entry.kernel.ctor_dtype}, not the dtype "
        "the backend asked for"
    )
    assert entry.compute_dtype == probe_dtype


@pytest.mark.smoke
@pytest.mark.parametrize("op_name,kwargs", [
    ("AlibiFwdOp", {"seq_len": 8, "num_heads": 4}),
    ("SinusoidalFwdOp", {"seq_len": 8, "d_model": 8}),
])
def test_generative_op_also_defers_to_the_backend(op_name, kwargs):
    """A dtype supplied as a parameter is still the backend's to specialize on."""
    import tileops.ops.elementwise as ew

    probe_dtype = torch.bfloat16
    slot = {"AlibiFwdOp": "alibi", "SinusoidalFwdOp": "sinusoidal"}[op_name]
    _ProbeKernel.instances = []
    op = getattr(ew, op_name)(
        dtype=torch.float16, kernel_map={slot: _probe_backend(probe_dtype)}, **kwargs,
    )

    assert len(_ProbeKernel.instances) == 1
    assert op.kernel is _ProbeKernel.instances[0]
    assert op.kernel.ctor_dtype == probe_dtype
