"""Tests for the shared ``_install_kernel_map`` path.

Installing the kernel map resolves classes only. It does not probe the device,
so an op constructs wherever it is imported and a target that cannot run it is
refused when a kernel is first selected — not at construction, where most ops
do not yet know which device they will run on.
"""


import pytest
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import forget_device_properties, get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="kernel-map install tests build kernels on the current device",
)


def _make_incompatible_arch_list() -> list[int]:
    """Return a ``supported_archs`` list that excludes the current device."""
    current = get_sm_version()
    candidates = [70, 75, 80, 86, 89, 90, 100]
    incompatible = [a for a in candidates if a != current]
    assert incompatible, "no incompatible arch candidate available"
    return incompatible


def _decode_op(**kwargs: object):
    from tileops.ops import GroupedQueryAttentionDecodeWithKVCacheFwdOp

    defaults = {"batch": 1, "heads": 32, "heads_kv": 4, "seqlen_kv": 8192, "dim": 128}
    defaults.update(kwargs)
    return GroupedQueryAttentionDecodeWithKVCacheFwdOp(**defaults)


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

    def unavailable(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no device to query")

    # The properties are cached per device, so a probe that already succeeded
    # would never reach the failing one.
    forget_device_properties()
    monkeypatch.setattr(torch.cuda, "get_device_capability", unavailable)
    monkeypatch.setattr(torch.cuda, "get_device_name", unavailable)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        mod.ReluFwdOp(N_total=8)
        _decode_op()
    finally:
        forget_device_properties()


@pytest.mark.smoke
def test_user_supplied_incompatible_kernel_is_refused_at_first_call() -> None:
    """An override that cannot run here is named, not silently passed over.

    The override is the reason the call was made; falling back to the stock
    kernel would report a result the caller believes came from theirs.
    """
    from tileops.kernels.attention import GQADecodeBs1Kernel, GQADecodeKernel

    incompatible_archs = _make_incompatible_arch_list()

    class IncompatibleBs1(GQADecodeBs1Kernel):
        supported_archs = incompatible_archs

    class IncompatibleGeneral(GQADecodeKernel):
        supported_archs = incompatible_archs

    op = _decode_op(kernel_map={
        "gqa_decode_bs1_kernel": IncompatibleBs1,
        "gqa_decode_kernel": IncompatibleGeneral,
    })

    with pytest.raises(ValueError, match="the kernel supplied for"):
        op._get_kernel(torch.float16)


@pytest.mark.smoke
def test_auto_discovered_incompatible_kernel_is_refused_at_first_call() -> None:
    """The auto-discovery path is refused at the same point, the same way."""
    from tileops.kernels.attention import GQADecodeBs1Kernel, GQADecodeKernel
    from tileops.ops import GroupedQueryAttentionDecodeWithKVCacheFwdOp

    incompatible_archs = _make_incompatible_arch_list()

    class IncompatibleBs1(GQADecodeBs1Kernel):
        supported_archs = incompatible_archs

    class IncompatibleGeneral(GQADecodeKernel):
        supported_archs = incompatible_archs

    class AutoDiscoveredIncompatibleOp(GroupedQueryAttentionDecodeWithKVCacheFwdOp):
        @property
        def default_kernel_map(self) -> dict[str, Kernel]:
            return {
                "gqa_decode_bs1_kernel": IncompatibleBs1,
                "gqa_decode_kernel": IncompatibleGeneral,
            }

    op = AutoDiscoveredIncompatibleOp(
        batch=1, heads=32, heads_kv=4, seqlen_kv=8192, dim=128)

    with pytest.raises(ValueError, match="no implementation serves this call"):
        op._get_kernel(torch.float16)


@pytest.mark.smoke
def test_single_implementation_slot_is_refused_at_first_build() -> None:
    """A slot with one implementation reports the same class as a slot with several.

    Nothing selects here — there is no second candidate to pass over — so the
    refusal comes from the kernel as it is built. It must still be a
    ``ValueError``, or a caller would need two excepts for one condition.
    """
    import tileops.ops.elementwise as mod

    (key, default_kernel_cls), = mod.ReluFwdOp(N_total=8).default_kernel_map.items()

    class IncompatibleKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = _make_incompatible_arch_list()

    op = mod.ReluFwdOp(N_total=8, kernel_map={key: IncompatibleKernel})

    with pytest.raises(ValueError, match="is built for architectures"):
        op(torch.randn(8, device="cuda", dtype=torch.float16))


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
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
    (key, default_kernel_cls), = cls(N_total=8).default_kernel_map.items()

    class UnrestrictedKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = None

    op = cls(N_total=8, kernel_map={key: UnrestrictedKernel})
    x = torch.randn(8, device="cuda", dtype=torch.float16)

    torch.testing.assert_close(op(x), torch.relu(x))


# A slot entry is a bare kernel for some families and a record for others, so
# an enumeration that does not descend into the record silently tunes nothing.


@pytest.mark.smoke
def test_autotune_reaches_elementwise_entries():
    """The elementwise slot is record-valued; every built kernel must be seen."""
    from tileops.ops.elementwise import AbsFwdOp

    op = AbsFwdOp(N_total=256)
    for dtype in (torch.float16, torch.float32):
        op(torch.randn(256, device="cuda", dtype=dtype))

    found = list(op.iter_kernels())
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
    entry = op.built_kernels(op._op_name)[torch.bool]
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
    assert shipped.built_kernels(shipped._op_name)[torch.int32].kernel is None, "float-only kernel was used"

    class NativeIntFloor(FloorFwdKernel):
        SUPPORTED_DTYPES = (torch.int32, torch.float32)

        def __init__(self, N_total, dtype, config=None, tune=False):
            self.dtype = dtype

        def forward(self, x):
            return x.clone()

    op = FloorFwdOp(N_total=64, kernel_map={"floor": NativeIntFloor})
    torch.testing.assert_close(op(x), x)
    entry = op.built_kernels(op._op_name)[torch.int32]
    assert isinstance(entry.kernel, NativeIntFloor), "the override was bypassed"
    assert entry.kernel.dtype == torch.int32


class _ProbeKernel:
    """Stands in for whatever the injected backend says should be built."""

    instances: list = []
    SUPPORTED_DTYPES = None  # the probe accepts whatever it is handed
    supported_archs = None

    def __init__(self, *args, **kwargs):
        type(self).instances.append(self)
        self.ctor_dtype = next((a for a in args if isinstance(a, torch.dtype)), None)


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
    ("LerpFwdOp", {"a_shape": (64,), "b_shape": (64,), "weight": 0.5}, ["lerp"], ()),
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

    # Whatever storage the backend computed in, the op delivers what it declared.
    shipped = getattr(ew, op_name)(dtype=torch.float16, **kwargs)
    assert shipped().dtype == torch.float16
