"""The seam between an op and a target's kernels.

What a third-party backend gets, observed from the op side: its builder is called with the
manifest's inputs and params, its kernel is memoized under the input signature, and
everything the op layer does for every target — validation, contiguity, output shape — still
happens. Uses a fake target, so no vendor hardware is involved.
"""

import pytest
import torch

from tests.test_base import served_in_tree
from tileops.backend import BUILTIN, OpNotAvailableError, TensorSpec, registry
from tileops.ops.convolution import Conv2dFwdOp
from tileops.ops.norm.instance_norm import InstanceNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from tileops.ops.pool import MaxPool2dFwdOp

pytestmark = pytest.mark.smoke

DTYPE = torch.float16
NORMALIZED_SHAPE = (256,)


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
    """An op whose in-tree kernel is a no-op, so the in-tree path runs on any device."""

    class StubOp(RMSNormFwdOp):
        def _eager_forward(self, x, weight=None):
            self.kernel_for("stub", (), x.dtype)
            return torch.zeros_like(x)

        def entry_for(self, role, call):
            return call, lambda: None

    StubOp.__name__ = "StubOp"
    return StubOp(normalized_shape=NORMALIZED_SHAPE, **kwargs)


def _inputs(rows=4, shape=NORMALIZED_SHAPE, dtype=DTYPE, device="cpu"):
    x = torch.randn(rows, *shape, dtype=dtype, device=device)
    weight = torch.randn(*shape, dtype=dtype, device=device)
    return x, weight


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
    # A backend gets the manifest parameters as the caller set them, ``eps=None`` included.
    assert params == {"normalized_shape": NORMALIZED_SHAPE, "eps": None}
    assert torch.equal(out, torch.full_like(x, 7)), "the target's kernel produced the result"

    RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE, eps=1e-5)(x, weight)
    assert recorder.calls[1][1]["eps"] == 1e-5


def test_a_dtype_the_manifest_does_not_admit_never_reaches_the_backend():
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)

    with pytest.raises(ValueError, match="weight dtype differs"):
        op(
            torch.randn(4, *NORMALIZED_SHAPE, dtype=DTYPE),
            torch.randn(*NORMALIZED_SHAPE, dtype=torch.bfloat16),
        )
    assert recorder.calls == []


def test_tensors_on_two_devices_never_reach_the_backend():
    recorder = _Recorder()
    _register(recorder)
    x, weight = _inputs()

    with pytest.raises(ValueError, match="one device"):
        RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)(x, weight.cuda())
    assert recorder.calls == []


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


class _InstanceNormTarget:
    """A backend that owns its kernel class and does not derive from TileOPs' ``Kernel``."""

    class Kernel:
        def __init__(self, eps):
            self.eps = eps

        def __call__(self, x, running_mean, running_var, weight, bias):
            return torch.nn.functional.instance_norm(x, weight=weight, bias=bias, eps=self.eps)

    def __init__(self):
        self.built = []

    def build_kernel(self, *inputs, **params):
        self.built.append(self.Kernel(params["eps"]))
        return self.built[-1]


def _run_instance_norm(op, n=2, c=8):
    x = torch.randn(n, c, 4, 4)
    weight, bias = torch.randn(c), torch.randn(c)
    out = op(x, weight=weight, bias=bias)
    torch.testing.assert_close(
        out, torch.nn.functional.instance_norm(x, weight=weight, bias=bias, eps=op.eps)
    )


def test_a_targets_kernels_are_the_ops_entries():
    """Caching and enumeration hold for a kernel that is not a TileOPs ``Kernel``."""
    target = _InstanceNormTarget()
    _register(target, op="InstanceNormFwdOp")
    op = InstanceNormFwdOp()

    for n in (2, 2, 3):
        _run_instance_norm(op, n=n)

    assert len(target.built) == 2, "the same signature is built once"
    assert list(op.built_kernels("instance_norm").values()) == target.built
    assert list(op.iter_kernels()) == [], "nothing here for autotune"
    assert op.run_config() is None
    assert op.settled_target == "acme" and not served_in_tree(op)


def test_a_first_call_that_fails_in_the_targets_kernel_leaves_no_entry():
    """The unsettled instance must not keep showing what the failed call built."""

    class Failing(_InstanceNormTarget.Kernel):
        def __call__(self, *args):
            raise RuntimeError("device fault")

    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("InstanceNormFwdOp", "acme", lambda *i, **p: Failing(1e-5))
    op = InstanceNormFwdOp()

    with pytest.raises(RuntimeError, match="device fault"):
        _run_instance_norm(op)

    assert op.settled_target is None
    assert not op.built_kernels("instance_norm") and op.kernel is None


def test_one_callable_a_target_returns_for_two_signatures_is_two_entries():
    """A target may cache on its own; the op still holds one entry per signature."""
    shared = _InstanceNormTarget.Kernel(1e-5)
    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("InstanceNormFwdOp", "acme", lambda *i, **p: shared)
    op = InstanceNormFwdOp()

    _run_instance_norm(op, n=2)
    _run_instance_norm(op, n=3)

    assert list(op.built_kernels("instance_norm").values()) == [shared, shared]


@pytest.mark.parametrize("ask", ["constructor", "autotune_first", "autotune_after"])
def test_a_tuning_request_a_target_cannot_receive_warns_once(ask):
    """``tune`` does not cross ``build_kernel``, so each way of asking says so."""
    _register(_InstanceNormTarget(), op="InstanceNormFwdOp")
    op = InstanceNormFwdOp(tune=ask == "constructor")
    if ask == "autotune_first":
        op.autotune()  # nothing is settled yet; the first build is where it is dropped

    with pytest.warns(UserWarning, match="not passed tune") as caught:
        _run_instance_norm(op)
        _run_instance_norm(op, n=3)
        if ask == "autotune_after":
            op.autotune()
            op.autotune()

    assert len(caught) == 1


def test_an_in_tree_settling_reads_builtin_however_it_was_chosen():
    """``served_in_tree`` gates in-tree assertions, so detection must count too."""
    detected, pinned = _stub_op(), _stub_op(target=BUILTIN)
    assert detected.settled_target is None and pinned.settled_target is None

    for op in (detected, pinned):
        op(*_inputs())
        assert op.settled_target is BUILTIN and served_in_tree(op)


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


def test_builtin_keeps_the_in_tree_kernels_even_when_a_target_claims_the_device():
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE, target=BUILTIN)

    assert op._builder is None or op._builder is not recorder.build_kernel
    with pytest.raises(OpNotAvailableError, match="in-tree kernels do not run on cpu"):
        op(*_inputs())
    assert recorder.calls == [], "BUILTIN went to the in-tree implementation"


def test_a_replacement_kernel_runs_on_the_devices_it_declares():
    """Device support is the kernel class's statement, so a CPU replacement is not refused."""
    from tileops.kernels.kernel_base import Kernel

    class CpuRMSNorm(Kernel):
        devices = frozenset({"cpu"})

        def __init__(self, n, eps, dtype, tune=False):
            super().__init__()

        def forward(self, x, weight):
            return torch.full_like(x, 7)

    x, weight = _inputs()
    op = RMSNormFwdOp(NORMALIZED_SHAPE, kernel_map={"rms_norm": CpuRMSNorm}, target=BUILTIN)
    assert torch.equal(op(x, weight), torch.full_like(x, 7))


def test_a_call_without_tensors_is_refused_on_its_declared_device():
    """The ``device`` parameter decides the call device, and the in-tree kernels run on CUDA."""
    from tileops.ops.elementwise import AlibiFwdOp

    with pytest.raises(OpNotAvailableError, match="do not run on cpu"):
        AlibiFwdOp(seq_len=8, num_heads=4, device="cpu", target=BUILTIN)()


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


def test_a_settled_instance_is_bound_to_that_target_s_devices():
    """One instance, one target. A kernel asked for on a foreign device is refused."""
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE, target=BUILTIN)
    x = torch.randn(4, *NORMALIZED_SHAPE, dtype=DTYPE, device="cuda")
    weight = torch.randn(*NORMALIZED_SHAPE, dtype=DTYPE, device="cuda")
    op(x, weight)

    with pytest.raises(OpNotAvailableError, match="in-tree kernels do not run on cpu"):
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
    with pytest.raises(ValueError, match="ClampOut"):
        ClampFwdOp()(input)
    assert recorder.calls == []


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

    with pytest.raises(ValueError, match="C_out"):
        Conv2dFwdOp(padding=1)(x, weight, torch.randn(999, dtype=DTYPE))
    assert recorder.calls == []


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
    # The manifest parameters as the caller set them, defaults included.
    assert params == {
        "kernel_size": 2,
        "stride": None,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
    }
    assert torch.equal(out, torch.full_like(out, 7)), "the target's kernel produced the result"


# --------------------------------------------------------------------------------------
# Five inputs, two of them written: BatchNormFwdOp at the seam
# --------------------------------------------------------------------------------------


def test_a_five_input_op_hands_over_its_inputs_in_the_manifest_order():
    """Order is the only thing that tells a backend which tensor is which."""
    received = []

    def build_kernel(*inputs, **params):
        def kernel(x, running_mean, running_var, weight, bias):
            received.append((x, running_mean, running_var, weight, bias))
            return torch.full_like(x, 7)

        return kernel

    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("BatchNormFwdOp", "acme", build_kernel)
    from tileops.ops.norm.batch_norm import BatchNormFwdOp

    x = torch.randn(2, 4, 8, 8, dtype=DTYPE)
    channels = [torch.randn(4, dtype=torch.float32) for _ in range(4)]
    running_mean, running_var, weight, bias = channels

    out = BatchNormFwdOp()(x, running_mean, running_var, weight, bias)

    ((got_x, got_mean, got_var, got_weight, got_bias),) = received
    assert got_x.shape == x.shape
    for got, expected in (
        (got_mean, running_mean),
        (got_var, running_var),
        (got_weight, weight),
        (got_bias, bias),
    ):
        assert torch.equal(got, expected)
    assert torch.equal(out, torch.full_like(x, 7))


# --------------------------------------------------------------------------------------
# A reduction op at the seam: the declared rank, and the axes as a param
# --------------------------------------------------------------------------------------


class _ReduceRecorder:
    """A target for a reduction op, returning a result of the shape the op declares."""

    def __init__(self, out_shape):
        self.calls = []
        self._out_shape = out_shape

    def build_kernel(self, *inputs, **params):
        self.calls.append((inputs, params))
        (spec,) = inputs

        def kernel(x):
            assert x.is_contiguous(), "the op normalizes contiguity before handing over"
            return torch.full(self._out_shape, 7, dtype=spec.dtype, device=spec.device)

        return kernel


def test_a_reduction_op_hands_over_the_declared_rank():
    """Not the ``(M, N)`` rows the in-tree kernel wants: that permute is the kernel's."""
    recorder = _ReduceRecorder((4,))
    _register(recorder, op="SumFwdOp")
    from tileops.ops.reduction import SumFwdOp

    x = torch.randn(4, 8, 16, dtype=DTYPE)
    out = SumFwdOp(dim=[1, 2])(x)

    ((inputs, params),) = recorder.calls
    assert inputs == (TensorSpec.of(x),), "the rank the manifest declares, not (4, 128)"
    assert params == {"dim": [1, 2], "keepdim": False, "dtype": None}
    assert torch.equal(out, torch.full((4,), 7, dtype=DTYPE))


def test_a_non_contiguous_reduction_input_reaches_the_backend_contiguous():
    recorder = _ReduceRecorder((4,))
    _register(recorder, op="SumFwdOp")
    from tileops.ops.reduction import SumFwdOp

    SumFwdOp(dim=-1)(torch.randn(8, 4, dtype=DTYPE).t())

    assert len(recorder.calls) == 1  # the assertion that matters is inside the kernel


def test_a_reduction_call_naming_an_absent_axis_never_reaches_the_backend():
    recorder = _ReduceRecorder((4,))
    _register(recorder, op="SumFwdOp")
    from tileops.ops.reduction import SumFwdOp

    with pytest.raises(ValueError, match="shape_rules"):
        SumFwdOp(dim=5)(torch.randn(4, 8, dtype=DTYPE))

    assert recorder.calls == []


# --------------------------------------------------------------------------------------
# An op with no tensor input, and an op built from other ops
# --------------------------------------------------------------------------------------


def test_an_op_with_no_tensor_input_is_placed_by_its_device_param():
    from tileops.ops.elementwise import AlibiFwdOp

    calls = []

    def build_kernel(**params):
        calls.append(params)
        return lambda: torch.zeros(4, 8, 8)

    registry.register_detector("acme", lambda device: device.type == "cpu")
    registry.register_kernel_builder("AlibiFwdOp", "acme", build_kernel)

    AlibiFwdOp(seq_len=8, num_heads=4, device="cpu")()

    assert calls == [{"seq_len": 8, "num_heads": 4, "out_dtype": torch.float32, "device": "cpu"}]


def _mamba2():
    from tileops.ops.mamba.mamba2_fwd import Mamba2FwdOp

    x = torch.randn(1, 64, 2, 16, dtype=DTYPE)
    dt = torch.rand(1, 64, 2)
    a = -torch.rand(2)
    b = torch.randn(1, 64, 1, 8, dtype=DTYPE)
    return Mamba2FwdOp(chunk_size=32, target="acme"), (x, dt, a, b, b.clone())


def test_a_target_that_builds_a_composite_serves_it_whole():
    op, inputs = _mamba2()
    x, b = inputs[0], inputs[3]
    served = (x.float(), torch.zeros(x.shape[0], x.shape[2], x.shape[3], b.shape[3]))
    registry.register_detector("acme", lambda device: False)
    registry.register_kernel_builder(
        "Mamba2FwdOp", "acme", lambda *specs, **params: lambda *tensors: served
    )

    y, final_states = op(*inputs)
    assert y is served[0] and final_states is served[1]


def test_a_composite_without_a_builder_hands_each_sub_op_to_the_target():
    op, inputs = _mamba2()
    registry.register_detector("acme", lambda device: False)

    with pytest.raises(OpNotAvailableError, match="no kernel builder for DaCumsumFwdOp"):
        op(*inputs)


def test_an_output_buffer_is_held_to_the_output_s_dtype_and_shape():
    from tileops.ops.moe.contracts import ContiguousLayoutSpec
    from tileops.ops.moe.staged import MoeGroupedGemmFwdOp

    registry.register_detector("acme", lambda device: device.type == "cpu")
    registry.register_kernel_builder(
        "MoeGroupedGemmFwdOp", "acme", lambda *specs, **params: lambda a, b, meta, out=None: out
    )
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    a, b = torch.randn(32, 64, dtype=DTYPE), torch.randn(4, 16, 64, dtype=DTYPE)
    meta = torch.zeros(4, dtype=torch.int32)

    op(a, b, meta, out=torch.empty(32, 16, dtype=DTYPE))
    with pytest.raises(ValueError, match="out does not have the shape of output"):
        op(a, b, meta, out=torch.empty(32, 8, dtype=DTYPE))
    with pytest.raises(ValueError, match="out does not have the dtype of output"):
        op(a, b, meta, out=torch.empty(32, 16, dtype=torch.float32))


def test_an_input_this_call_does_not_write_reaches_the_kernel_contiguous():
    """An activation's input is written only when ``inplace`` is set."""
    from tileops.ops.elementwise import ReluFwdOp

    seen = []
    registry.register_detector("acme", lambda device: device.type == "cpu")
    registry.register_kernel_builder(
        "ReluFwdOp", "acme", lambda *specs, **params: lambda x: seen.append(x) or x.clone()
    )

    ReluFwdOp()(torch.randn(8, 8, dtype=DTYPE)[:, ::2])

    assert seen[0].is_contiguous()


def test_an_input_typed_after_an_output_follows_that_output_s_dtype():
    """``bias: same_as(d)``: the output's dtype is the op's to state, and it binds the input."""
    from tileops.ops.gemm.gemm import GemmFp8FwdOp

    registry.register_detector("acme", lambda device: device.type == "cpu")
    registry.register_kernel_builder(
        "GemmFp8FwdOp", "acme", lambda *specs, **params: lambda *tensors: torch.empty(16, 16)
    )
    fp8 = torch.float8_e4m3fn
    a, b = torch.empty(16, 32, dtype=fp8), torch.empty(16, 32, dtype=fp8)
    scale = torch.ones(1, 1)
    op = GemmFp8FwdOp(out_dtype=torch.bfloat16)

    op(a, b, scale, scale, torch.empty(16, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bias"):
        op(a, b, scale, scale, torch.empty(16, dtype=torch.float16))
