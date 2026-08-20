"""The seam between an op and a target's kernels.

What a third-party backend gets, observed from the op side: its builder is called with the
manifest's inputs and params, its kernel is memoized under the input signature, and
everything the op layer does for every target — validation, contiguity, output shape — still
happens. Uses a fake target, so no vendor hardware is involved.
"""

import pytest
import torch

import tileops.ops.op_base as op_base
from tileops.backend import BUILTIN, OpNotAvailableError, TensorSpec, registry
from tileops.kernels.attention.call_spec import AttentionCall
from tileops.ops import GroupedQueryAttentionPrefillDenseFwdOp, MultiHeadAttentionFwdOp
from tileops.ops.convolution import Conv2dFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from tileops.ops.pool import MaxPool2dFwdOp

pytestmark = pytest.mark.smoke

DTYPE = torch.float16
NORMALIZED_SHAPE = (256,)


def test_attention_call_non_cuda_device_never_falls_back_to_current_cuda():
    """Kernel-side platform discovery follows the manifest input device exactly."""
    call = AttentionCall.from_device(torch.device("cpu"), dtype=torch.float16)

    assert call.arch == 0
    assert call.h200 is False


@pytest.fixture(autouse=True)
def isolated_registry():
    """Each test starts with an empty registry and no backend discovery."""
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    yield
    registry.restore(state)


class _Recorder:
    """A target that records how it was asked and returns a kernel of its own."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def build_kernel(self, *inputs, **params):
        self.calls.append((inputs, params))
        result = self.result

        def kernel(x, weight):
            assert x.is_contiguous() and weight.is_contiguous()
            return torch.full_like(x, 7) if result is None else result

        return kernel


def _register(recorder, target="acme", op="RMSNormFwdOp", claims=True):
    registry.register_detector(target, lambda device: claims)
    registry.register_kernel_builder(op, target, recorder.build_kernel)


def _stub_op(**kwargs):
    """An op of the kind that still takes a kernel without handing over its tensors."""

    class StubOp(RMSNormFwdOp):
        def forward(self, x, weight):
            return self.get_or_build_kernel("stub", key=x.dtype, build=lambda: None)

    StubOp.__name__ = "StubOp"
    return StubOp(normalized_shape=NORMALIZED_SHAPE, **kwargs)


def _inputs(rows=4, shape=NORMALIZED_SHAPE, dtype=DTYPE, device="cpu"):
    x = torch.randn(rows, *shape, dtype=dtype, device=device)
    weight = torch.randn(*shape, dtype=dtype, device=device)
    return x, weight


def test_dense_gqa_target_preserves_omitted_optional_inputs():
    builds = []
    runs = []

    def build_kernel(*specs, **params):
        builds.append((specs, params))

        def kernel(*inputs):
            assert all(tensor.is_contiguous() for tensor in inputs)
            runs.append(inputs)
            return torch.empty_like(inputs[0], dtype=params["dtype"])

        return kernel

    registry.register_kernel_builder(
        "GroupedQueryAttentionPrefillDenseFwdOp", "gqa_fake", build_kernel
    )
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        dtype=torch.float16,
        target="gqa_fake",
    )
    q = torch.randn(2, 3, 4, 16, dtype=torch.float16)[..., ::2]
    k = torch.randn(2, 3, 2, 8, dtype=torch.float16)
    v = torch.randn_like(k)
    assert not q.is_contiguous(), "the test must exercise boundary normalization"

    output = op(q, k, v)
    op(-q, k + 1, v - 1)

    assert output.shape == q.shape
    assert len(builds) == 1, "one input signature builds one external callable"
    specs, params = builds[0]
    assert specs == tuple(TensorSpec.of(tensor) for tensor in runs[0])
    assert params == {
        "is_causal": True,
        "sm_scale": 8**-0.5,
        "softcap": 0.0,
        "window_size_left": -1,
        "window_size_right": -1,
        "dtype": torch.float16,
        "pos_encoding_mode": "none",
        "rotary_dim": None,
        "rope_layout": "neox",
        "rope_base": 10000.0,
    }
    assert tuple(params) == op.__manifest_param_names__
    assert len(runs[0]) == 3
    assert len(runs) == 2, "one callable serves different contents of one signature"
    assert not torch.equal(runs[0][0], runs[1][0])
    assert runs[0][0].shape == q.shape, "external Dense ABI stays BSHD"


def test_dense_gqa_target_gets_rope_as_constructor_configuration():
    builds = []
    runs = []

    def build_kernel(*specs, **params):
        builds.append((specs, params))

        def kernel(*inputs):
            runs.append(inputs)
            return torch.empty_like(inputs[0])

        return kernel

    registry.register_kernel_builder(
        "GroupedQueryAttentionPrefillDenseFwdOp", "gqa_fake", build_kernel
    )
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        pos_encoding_mode="rope",
        rotary_dim=4,
        rope_layout="interleaved",
        rope_base=500000.0,
        target="gqa_fake",
    )
    q = torch.randn(1, 3, 4, 8, dtype=torch.float16)
    k = torch.randn(1, 3, 2, 8, dtype=torch.float16)
    v = torch.randn_like(k)

    output = op(q, k, v)

    assert output.shape == q.shape
    assert len(builds) == 1
    specs, params = builds[0]
    assert len(specs) == 3, "RoPE tables and omitted scales are not runtime inputs"
    assert len(runs[0]) == 3
    assert params["pos_encoding_mode"] == "rope"
    assert params["rotary_dim"] == 4
    assert params["rope_layout"] == "interleaved"
    assert params["rope_base"] == 500000.0
    assert params["sm_scale"] == 8**-0.5
    assert params["dtype"] == torch.float16
    assert specs[0].dtype == torch.float16


def test_mha_wrapper_defers_external_replacement_to_dense_delegate():
    builds = []

    def build_kernel(*specs, **params):
        builds.append((specs, params))

        def kernel(q, k, v):
            return torch.empty_like(q, dtype=params["dtype"])

        return kernel

    registry.register_detector("gqa_fake", lambda device: device.type == "cpu")
    registry.register_kernel_builder(
        "GroupedQueryAttentionPrefillDenseFwdOp", "gqa_fake", build_kernel
    )

    op = MultiHeadAttentionFwdOp(1, 4, 3, 8, is_causal=False)
    q = torch.randn(1, 3, 4, 8, dtype=torch.float16)
    output = op(q, torch.randn_like(q), torch.randn_like(q))

    assert output.shape == q.shape
    assert op._settled_target is BUILTIN
    assert op._gqa_op._settled_target == "gqa_fake"
    assert len(builds) == 1
    _, params = builds[0]
    assert params["sm_scale"] == 8**-0.5
    assert params["dtype"] == torch.float16


def test_dense_gqa_external_builder_gets_defaults_resolved_per_signature():
    params_seen = []

    def build_kernel(*specs, **params):
        params_seen.append(params)

        def kernel(q, k, v):
            return torch.empty_like(q, dtype=params["dtype"])

        return kernel

    registry.register_detector("gqa_fake", lambda device: device.type == "cpu")
    registry.register_kernel_builder(
        "GroupedQueryAttentionPrefillDenseFwdOp", "gqa_fake", build_kernel
    )
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=False,
        pos_encoding_mode="rope",
        target="gqa_fake",
    )

    for dim in (8, 16):
        q = torch.randn(1, 3, 4, dim, dtype=torch.float16)
        k = torch.randn(1, 3, 2, dim, dtype=torch.float16)
        op(q, k, torch.randn_like(k))

    assert [params["sm_scale"] for params in params_seen] == [8**-0.5, 16**-0.5]
    assert [params["rotary_dim"] for params in params_seen] == [8, 16]
    assert [params["dtype"] for params in params_seen] == [torch.float16, torch.float16]


def test_dense_gqa_external_signature_cache_is_bounded_lru(monkeypatch):
    monkeypatch.setattr(op_base, "_EXTERNAL_KERNEL_SIGNATURE_CACHE_SIZE", 2)
    builds = []

    def build_kernel(*specs, **params):
        builds.append(specs)

        def kernel(q, k, v):
            return torch.empty_like(q)

        return kernel

    registry.register_detector("gqa_fake", lambda device: device.type == "cpu")
    registry.register_kernel_builder(
        "GroupedQueryAttentionPrefillDenseFwdOp", "gqa_fake", build_kernel
    )
    op = GroupedQueryAttentionPrefillDenseFwdOp(is_causal=False, target="gqa_fake")

    for dim in range(8, 11):
        q = torch.randn(1, 2, 4, dim, dtype=torch.float16)
        k = torch.randn(1, 2, 2, dim, dtype=torch.float16)
        op(q, k, torch.randn_like(k))

    assert len(builds) == 3
    assert len(op.built_kernels("gqa_prefill_dense")) == 2

    # The first signature was least recently used and must build again.
    q = torch.randn(1, 2, 4, 8, dtype=torch.float16)
    k = torch.randn(1, 2, 2, 8, dtype=torch.float16)
    op(q, k, torch.randn_like(k))
    assert len(builds) == 4


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
def test_dense_gqa_target_preserves_present_optional_inputs():
    builds = []
    runs = []

    def build_kernel(*specs, **params):
        builds.append((specs, params))

        def kernel(*inputs):
            runs.append(inputs)
            return torch.empty_like(inputs[0], dtype=params["dtype"])

        return kernel

    registry.register_kernel_builder(
        "GroupedQueryAttentionPrefillDenseFwdOp", "gqa_fake", build_kernel
    )
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        is_causal=False,
        dtype=torch.float16,
        target="gqa_fake",
    )
    q = torch.empty((1, 3, 8, 128), dtype=torch.float8_e4m3fn)
    k = torch.empty((1, 3, 2, 128), dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    scales = torch.ones((1, 2), dtype=torch.float32)

    output = op(q, k, v, scales, scales, scales)

    assert output.dtype == torch.float16
    assert len(builds) == 1
    specs, _ = builds[0]
    assert len(specs) == 6
    assert len(runs[0]) == 6
    assert specs == tuple(TensorSpec.of(tensor) for tensor in runs[0])


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch fp8 is unavailable")
def test_dense_gqa_rejects_fp8_without_output_dtype_before_external_builder():
    builds = []

    def build_kernel(*specs, **params):
        builds.append((specs, params))
        raise AssertionError("an invalid public call must not reach the target")

    registry.register_kernel_builder(
        "GroupedQueryAttentionPrefillDenseFwdOp", "gqa_fake", build_kernel
    )
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        target="gqa_fake",
    )
    q = torch.empty((1, 3, 8, 128), dtype=torch.float8_e4m3fn)
    k = torch.empty((1, 3, 2, 128), dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)

    with pytest.raises(ValueError, match="dtype must select a float16 or bfloat16"):
        op(q, k, v)

    assert builds == []


# --------------------------------------------------------------------------------------
# What the backend is asked, and what it gets back
# --------------------------------------------------------------------------------------


def test_a_target_takes_over_the_op_and_is_asked_with_the_manifest_signature():
    recorder = _Recorder()
    _register(recorder)
    x, weight = _inputs()

    out = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)(x, weight)

    ((inputs, params),) = recorder.calls
    assert inputs == (TensorSpec.of(x), TensorSpec.of(weight)), "signature.inputs order"
    # eps is optional; whether it was passed or defaulted, a backend gets the number.
    assert params == {"normalized_shape": NORMALIZED_SHAPE, "eps": 1e-6}
    assert torch.equal(out, torch.full_like(x, 7)), "the target's kernel produced the result"

    RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE, eps=1e-5)(x, weight)
    assert recorder.calls[1][1]["eps"] == 1e-5


def test_the_op_layer_still_does_its_half():
    """A backend writes kernels, not ops: validation and normalization are not its job."""
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)

    with pytest.raises(ValueError, match="Expected x trailing shape"):
        op(torch.randn(4, 999, dtype=DTYPE), torch.randn(*NORMALIZED_SHAPE, dtype=DTYPE))
    with pytest.raises(ValueError, match="same_as"):
        op(
            torch.randn(4, *NORMALIZED_SHAPE, dtype=DTYPE),
            torch.randn(*NORMALIZED_SHAPE, dtype=torch.bfloat16),
        )
    assert recorder.calls == [], "a rejected call never reaches the backend"


def test_a_non_contiguous_input_reaches_the_kernel_contiguous():
    recorder = _Recorder()
    _register(recorder)
    x, weight = _inputs(rows=8)
    strided = x[::2]
    assert not strided.is_contiguous()

    RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)(strided, weight)

    ((inputs, _),) = recorder.calls
    assert inputs[0].shape == (4, *NORMALIZED_SHAPE)  # the kernel asserts contiguity itself


# --------------------------------------------------------------------------------------
# How the result is remembered
# --------------------------------------------------------------------------------------


def test_the_same_input_signature_is_built_once():
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)
    x, weight = _inputs()

    op(x, weight)
    op(torch.randn_like(x), torch.randn_like(weight))

    assert len(recorder.calls) == 1, "same dtypes and shapes, so the same kernel"


@pytest.mark.parametrize(
    ("second", "why"),
    [
        (dict(rows=8), "a different shape may need a different kernel"),
        (dict(dtype=torch.bfloat16), "a different dtype certainly does"),
        # A second real device, not meta: meta inputs dispatch to the op's fake, which
        # returns before a kernel is ever asked for.
        (dict(device="cuda"), "a kernel may hold resources allocated on one device"),
    ],
    ids=["shape", "dtype", "device"],
)
def test_a_different_input_signature_asks_again(second, why):
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)

    op(*_inputs())
    op(*_inputs(**second))

    assert len(recorder.calls) == 2, why


def test_the_target_is_settled_once_and_kept():
    """The kernels this instance holds belong to that target."""
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)
    op(*_inputs())

    assert op._settled_target == "acme"
    registry.default_target = BUILTIN  # would mean "in-tree" for a fresh instance
    op(*_inputs())
    assert len(recorder.calls) == 1, "an instance that has built kernels is not re-aimed"


# --------------------------------------------------------------------------------------
# When a target cannot serve the call
# --------------------------------------------------------------------------------------


def test_a_target_without_this_op_raises_and_names_the_ones_that_have_it():
    """No fall back: the in-tree kernels do not run on another target's devices."""
    recorder = _Recorder()
    _register(recorder, target="has_it")
    registry.register_detector("claims_device", lambda device: True)
    registry.DETECTORS.pop("has_it")  # only the op-less target claims the device

    with pytest.raises(OpNotAvailableError, match=r"claims_device.*has_it.*no fall back"):
        RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)(*_inputs())


def test_an_op_that_has_not_handed_over_its_tensors_says_so():
    """The op layer's own gap, reported as such rather than as a backend's."""
    recorder = _Recorder()
    _register(recorder, op="StubOp")

    with pytest.raises(OpNotAvailableError, match="not wired to external targets yet"):
        _stub_op()(*_inputs())


def test_builtin_keeps_the_in_tree_kernels_even_when_a_target_claims_the_device():
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE, target=BUILTIN)

    assert op._builder is None or op._builder is not recorder.build_kernel
    with pytest.raises(ValueError, match="is a CUDA kernel"):
        op(*_inputs())
    assert recorder.calls == [], "BUILTIN went to the in-tree implementation"


def test_a_call_with_no_tensor_leaves_the_question_open():
    """Nothing was probed, so nothing is remembered: the next call decides."""
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)

    op._resolve_builder((), {})
    assert op._builder is not None and op._settled_target is None

    op(*_inputs())
    assert op._settled_target == "acme", "the first call with a tensor decides"


def test_a_build_that_fails_pins_nothing():
    """The next call resolves again rather than being stuck on a target that could not."""
    attempts = []

    def build_kernel(*inputs, **params):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("vendor compiler unhappy")
        return lambda x, weight: torch.full_like(x, 3)

    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("RMSNormFwdOp", "acme", build_kernel)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)

    with pytest.raises(RuntimeError, match="vendor compiler unhappy"):
        op(*_inputs())
    assert op._settled_target is None, "a failed build settles no target"

    out = op(*_inputs())
    assert torch.equal(out, torch.full_like(out, 3)), "asking again tries again"
    assert op._settled_target == "acme"


def test_a_builder_must_return_something_callable():
    """One of the rules this boundary owes, checked where it is crossed."""
    recorder = _Recorder()
    recorder.build_kernel = lambda *inputs, **params: "not a kernel"
    _register(recorder)

    with pytest.raises(OpNotAvailableError, match="not callable"):
        RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)(*_inputs())


def test_params_are_this_ops_manifest_params_and_not_an_inherited_set():
    """A subclass with no manifest entry of its own hands a backend nothing."""

    class Untyped(RMSNormFwdOp):
        pass

    assert Untyped.__manifest_param_names__ == ()
    assert RMSNormFwdOp.__manifest_param_names__ == ("normalized_shape", "eps")


def test_a_call_that_fails_validation_pins_nothing():
    """One invalid call must not aim the instance for good.

    The first tensor's device picks the target, so a mixed-device call would otherwise send
    every later call where that one pointed.
    """
    recorder = _Recorder()
    registry.register_detector("cpu_target", lambda device: device.type == "cpu")
    registry.register_kernel_builder("RMSNormFwdOp", "cpu_target", recorder.build_kernel)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)

    with pytest.raises(ValueError):
        op(
            torch.randn(4, *NORMALIZED_SHAPE, dtype=DTYPE),
            torch.randn(*NORMALIZED_SHAPE, dtype=torch.bfloat16),
        )

    assert op._settled_target is None and not op.built_kernels("rms_norm")
    op(*_inputs())
    assert op._settled_target == "cpu_target", "the first call that worked decides"
    assert len(recorder.calls) == 1


@pytest.mark.usefixtures("isolated_dynamo")
def test_the_first_compiled_call_obeys_the_target_it_picked():
    """Settling only in a traced ``__call__`` gives the in-tree kernel's numbers, once."""
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)
    x, weight = _inputs()

    output = torch.compile(op, fullgraph=True)(x, weight)

    assert torch.equal(output, torch.full_like(x, 7)), "the in-tree kernel ran instead"
    assert len(recorder.calls) == 1


@pytest.mark.usefixtures("isolated_dynamo")
def test_a_compiled_call_whose_build_fails_pins_nothing():
    """``__call__``'s handler does not run when the failure comes out of a compiled graph."""
    attempts = []

    def build_kernel(*inputs, **params):
        attempts.append(1)
        return "not a kernel" if len(attempts) == 1 else (lambda x, w: torch.full_like(x, 7))

    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("RMSNormFwdOp", "acme", build_kernel)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)

    with pytest.raises(OpNotAvailableError, match="not callable"):
        torch.compile(op, fullgraph=True)(*_inputs())

    x, weight = _inputs()
    assert torch.equal(op(x, weight), torch.full_like(x, 7)), "asking again tries again"


def test_a_call_without_tensors_still_honours_an_explicit_target():
    """A named target needs no device, so handing over no tensors is no reason to fall back."""
    _register(_Recorder(), op="StubOp")

    with pytest.raises(OpNotAvailableError, match="not wired to external targets yet"):
        _stub_op(target="acme").forward(*_inputs())


def test_a_settled_instance_is_bound_to_that_target_s_devices():
    """One instance, one target. A kernel handed a foreign tensor is what says so."""
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE, target=BUILTIN)
    x = torch.randn(4, *NORMALIZED_SHAPE, dtype=DTYPE, device="cuda")
    weight = torch.randn(*NORMALIZED_SHAPE, dtype=DTYPE, device="cuda")
    op(x, weight)

    with pytest.raises(ValueError, match="is a CUDA kernel"):
        op(*_inputs())  # same signature, CPU tensors


# --------------------------------------------------------------------------------------
# Two optional inputs at the seam: ClampFwdOp's min and max
# --------------------------------------------------------------------------------------


class _ClampRecorder:
    """A target for ClampFwdOp; its kernel takes whatever the op hands over."""

    def __init__(self):
        self.calls = []
        self.kernel_calls = 0

    def build_kernel(self, *inputs, **params):
        self.calls.append((inputs, params))

        def kernel(input, min=None, max=None):
            assert input.is_contiguous()
            self.kernel_calls += 1
            return torch.full_like(input, 7)

        return kernel


def _clamp_inputs(rows=4, cols=8, dtype=DTYPE, device="cpu"):
    make = lambda: torch.randn(rows, cols, dtype=dtype, device=device)  # noqa: E731
    return make(), make(), make()


def test_an_absent_optional_input_keeps_its_slot():
    """The slot says which input is missing; how many slots there are cannot."""
    recorder = _ClampRecorder()
    _register(recorder, op="ClampFwdOp")
    from tileops.ops.elementwise import ClampFwdOp

    input, lower, _ = _clamp_inputs()
    ClampFwdOp()(input, lower, None)

    ((inputs, params),) = recorder.calls
    assert inputs == (TensorSpec.of(input), TensorSpec.of(lower), None)
    assert params == {}, "ClampFwdOp declares no manifest params"


def test_the_two_one_sided_clamps_are_two_kernels():
    """Both hand over two tensors of one shape; only the slot tells them apart."""
    recorder = _ClampRecorder()
    _register(recorder, op="ClampFwdOp")
    from tileops.ops.elementwise import ClampFwdOp

    op = ClampFwdOp()
    input, lower, upper = _clamp_inputs()

    op(input, lower, None)
    op(input, None, upper)
    op(input, lower, None)

    assert len(recorder.calls) == 2, "a lower bound and an upper bound are not one kernel"
    assert recorder.calls[0][0][1] is not None and recorder.calls[0][0][2] is None
    assert recorder.calls[1][0][1] is None and recorder.calls[1][0][2] is not None


def test_a_clamp_with_neither_bound_never_reaches_the_backend():
    recorder = _ClampRecorder()
    _register(recorder, op="ClampFwdOp")
    from tileops.ops.elementwise import ClampFwdOp

    input, _, _ = _clamp_inputs()
    with pytest.raises(ValueError, match="at least one of"):
        ClampFwdOp()(input)
    assert recorder.calls == [], "the op layer's checks run for every target"


# --------------------------------------------------------------------------------------
# An elementwise op whose shape is learned from the call
# --------------------------------------------------------------------------------------


class _ReluRecorder:
    def __init__(self):
        self.calls = []

    def build_kernel(self, *inputs, **params):
        self.calls.append((inputs, params))
        return lambda x: torch.full_like(x, 7)


def test_an_elementwise_op_hands_over_the_manifest_shape():
    """Not the flat view the in-tree kernel wants: that is the kernel's own business."""
    recorder = _ReluRecorder()
    _register(recorder, op="ReluFwdOp")
    from tileops.ops.elementwise import ReluFwdOp

    x = torch.randn(4, 8, 16, dtype=DTYPE)
    out = ReluFwdOp()(x)

    ((inputs, params),) = recorder.calls
    assert inputs == (TensorSpec.of(x),), "the shape the manifest declares, not (512,)"
    assert params == {"inplace": False}
    assert torch.equal(out, torch.full_like(x, 7))


def test_an_elementwise_op_without_a_builder_for_this_target_raises():
    recorder = _ReluRecorder()
    _register(recorder, op="ReluFwdOp")
    from tileops.ops.elementwise import SiluFwdOp

    with pytest.raises(OpNotAvailableError, match="registers no kernel builder"):
        SiluFwdOp()(torch.randn(4, 8, dtype=DTYPE))


# --------------------------------------------------------------------------------------
# An optional input at the seam: Conv2dFwdOp's bias
# --------------------------------------------------------------------------------------


class _ConvRecorder:
    """A target for Conv2dFwdOp; its kernel takes whatever the op hands over."""

    def __init__(self):
        self.calls = []

    def build_kernel(self, *inputs, **params):
        self.calls.append((inputs, params))

        def kernel(x, weight, bias=None):
            assert x.is_contiguous() and weight.is_contiguous()
            return torch.zeros(x.shape[0], weight.shape[0], x.shape[2], x.shape[3], dtype=x.dtype)

        return kernel


def _conv_inputs(bias=False):
    x = torch.randn(1, 8, 8, 8, dtype=DTYPE)
    weight = torch.randn(4, 8, 3, 3, dtype=DTYPE)
    return x, weight, (torch.randn(4, dtype=DTYPE) if bias else None)


def test_a_missing_optional_input_keeps_its_place_in_the_hand_over():
    """Presence is what the backend reads, and it reads it off the argument.

    One argument per ``signature.inputs`` entry: a bias this call did not pass is ``None``
    there. Dropping the argument would leave the count to say what is missing, which it
    cannot do for an op with two optional inputs.
    """
    recorder = _ConvRecorder()
    _register(recorder, op="Conv2dFwdOp")
    x, weight, _ = _conv_inputs()

    Conv2dFwdOp(padding=1)(x, weight)

    ((inputs, params),) = recorder.calls
    assert inputs == (TensorSpec.of(x), TensorSpec.of(weight), None), "signature.inputs order"
    assert params == {"stride": (1, 1), "padding": 1, "dilation": (1, 1), "groups": 1}


def test_a_bias_that_is_passed_reaches_the_backend_as_a_third_spec():
    recorder = _ConvRecorder()
    _register(recorder, op="Conv2dFwdOp")
    x, weight, bias = _conv_inputs(bias=True)

    Conv2dFwdOp(padding=1)(x, weight, bias)

    ((inputs, _),) = recorder.calls
    assert inputs == (TensorSpec.of(x), TensorSpec.of(weight), TensorSpec.of(bias))


def test_the_two_sides_of_an_optional_input_are_two_kernels():
    """Bias presence changes what a kernel is built for, so it is part of the signature."""
    recorder = _ConvRecorder()
    _register(recorder, op="Conv2dFwdOp")
    op = Conv2dFwdOp(padding=1)
    x, weight, bias = _conv_inputs(bias=True)

    op(x, weight)
    op(x, weight, bias)
    op(x, weight)

    assert len(recorder.calls) == 2


def test_a_rejected_conv_call_never_reaches_the_backend():
    recorder = _ConvRecorder()
    _register(recorder, op="Conv2dFwdOp")
    x, weight, _ = _conv_inputs()

    with pytest.raises(ValueError, match="bias shape"):
        Conv2dFwdOp(padding=1)(x, weight, torch.randn(999, dtype=DTYPE))
    assert recorder.calls == [], "the op layer's checks run for every target"


# --------------------------------------------------------------------------------------
# An explicit target: MaxPool2dFwdOp
# --------------------------------------------------------------------------------------


class _PoolRecorder:
    """A target for MaxPool2dFwdOp; its kernel takes the one input the op hands over."""

    def __init__(self):
        self.calls = []

    def build_kernel(self, *inputs, **params):
        self.calls.append((inputs, params))

        def kernel(x):
            assert x.is_contiguous()
            return torch.full((x.shape[0], x.shape[1], 4, 4), 7, dtype=x.dtype)

        return kernel


def test_an_explicit_target_serves_a_pool_op_no_detector_claims_the_device():
    """``target=`` is the override, so it routes with nothing claiming the device."""
    recorder = _PoolRecorder()
    _register(recorder, op="MaxPool2dFwdOp", claims=False)
    x = torch.randn(1, 4, 8, 8, dtype=DTYPE)

    out = MaxPool2dFwdOp(kernel_size=2, target="acme")(x)

    ((inputs, params),) = recorder.calls
    assert inputs == (TensorSpec.of(x),), "signature.inputs order"
    assert params == {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "ceil_mode": False,
    }
    assert torch.equal(out, torch.full_like(out, 7)), "the target's kernel produced the result"
