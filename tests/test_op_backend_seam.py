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
    """An op of the kind that still takes a kernel without handing over its tensors."""

    class StubOp(RMSNormFwdOp):
        def forward(self, x, weight):
            return self.kernel_for("stub", (), x.dtype)

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


def _run_instance_norm(op, n=2, c=8, call=None):
    x = torch.randn(n, c, 4, 4)
    weight, bias = torch.randn(c), torch.randn(c)
    out = (call or op)(x, weight=weight, bias=bias)
    torch.testing.assert_close(
        out, torch.nn.functional.instance_norm(x, weight=weight, bias=bias, eps=op.eps)
    )


class _FlakyTarget(_InstanceNormTarget):
    """Its kernels raise while ``fail`` is set, after building normally."""

    fail = False

    def build_kernel(self, *inputs, **params):
        target, kernel = self, super().build_kernel(*inputs, **params)

        def run(*args):
            if target.fail:
                raise RuntimeError("device fault")
            return kernel(*args)

        return run


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


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_a_first_call_that_fails_in_the_targets_kernel_leaves_no_entry(compiled, request):
    """The kernel runs after it is built, so the call's owner, not the build, must undo."""
    if compiled:
        request.getfixturevalue("isolated_dynamo")
    target = _FlakyTarget()
    _register(target, op="InstanceNormFwdOp")
    op = InstanceNormFwdOp()
    call = torch.compile(op, fullgraph=True) if compiled else op

    target.fail = True
    with pytest.raises(RuntimeError, match="device fault"):
        _run_instance_norm(op, call=call)

    assert op.settled_target is None
    assert not op.built_kernels("instance_norm") and op.kernel is None
    target.fail = False
    _run_instance_norm(op, call=call)
    assert op.settled_target == "acme", "asking again settles again"


def test_a_later_call_that_fails_keeps_the_settling():
    """Only the call that settled the instance may undo it."""
    target = _FlakyTarget()
    _register(target, op="InstanceNormFwdOp")
    op = InstanceNormFwdOp()
    _run_instance_norm(op)

    target.fail = True
    with pytest.raises(RuntimeError, match="device fault"):
        _run_instance_norm(op)

    assert op.settled_target == "acme" and len(op.built_kernels("instance_norm")) == 1


@pytest.mark.usefixtures("isolated_dynamo")
def test_a_failure_after_the_kernel_returns_undoes_the_settling():
    """The operator's own steps after ``_eager_forward`` are part of the call it owns."""
    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("ReluFwdOp", "acme", lambda *i, **p: lambda x: torch.zeros(3))
    from tileops.ops.elementwise import ReluFwdOp

    op = ReluFwdOp(inplace=True)
    with pytest.raises(RuntimeError, match="shape"):
        # Compiled, so the operator owns the call; the inplace copy cannot reshape the result.
        torch.compile(op, fullgraph=True)(torch.randn(4, 8, dtype=DTYPE))

    assert op.settled_target is None and not op.built_kernels(op._op_name)


@pytest.mark.usefixtures("isolated_dynamo")
def test_a_call_is_recorded_once_for_roofline(monkeypatch):
    """Eager, cold compiled and warm compiled calls each record once, and only in a block."""
    from tileops.ops.op_base import Op, _recording_roofline_calls

    recorded = []
    track = Op._track_roofline_call
    monkeypatch.setattr(
        Op, "_track_roofline_call", lambda self, a, k: (recorded.append(k), track(self, a, k))
    )
    _register(_InstanceNormTarget(), op="InstanceNormFwdOp")
    op = InstanceNormFwdOp()
    compiled = torch.compile(op, fullgraph=True)

    with _recording_roofline_calls():
        for call in (op, compiled, compiled):
            recorded.clear()
            _run_instance_norm(op, call=call)
            assert len(recorded) == 1
    assert op._roofline_call_tensors["x"] == ((2, 8, 4, 4), torch.float32)

    _run_instance_norm(op)
    assert op._roofline_call_tensors is None, "a call outside the block drops the record"


def _composite(delegate, reach):
    """An in-tree parent op whose forward reaches *delegate* as ``reach`` says."""

    class Parent(RMSNormFwdOp):
        def forward(self, x, weight):
            if self.reach == "call":
                _run_instance_norm(delegate)
            elif self.reach == "forward":
                _run_instance_norm(delegate, call=delegate.forward)
            return x

    Parent.__name__ = "Parent"
    parent = Parent(normalized_shape=NORMALIZED_SHAPE, target=BUILTIN)
    parent.reach = reach
    return parent


@pytest.mark.parametrize("parent_warm", [False, True], ids=["cold_parent", "warm_parent"])
@pytest.mark.parametrize("delegate_warm", [False, True], ids=["cold_delegate", "warm_delegate"])
def test_a_delegate_failure_undoes_only_what_the_failing_call_settled(parent_warm, delegate_warm):
    """A composite and its delegate each own their own settling."""
    target = _FlakyTarget()
    _register(target, op="InstanceNormFwdOp")
    delegate = InstanceNormFwdOp()
    if delegate_warm:
        _run_instance_norm(delegate)
    parent = _composite(delegate, reach="none")
    if parent_warm:
        parent(*_inputs())
    parent.reach = "call"

    target.fail = True
    with pytest.raises(RuntimeError, match="device fault"):
        parent(*_inputs())

    assert (parent.settled_target is BUILTIN) is parent_warm
    assert (delegate.settled_target == "acme") is delegate_warm


def test_a_delegate_reached_through_forward_is_owned_by_its_operator():
    """A composite may call a delegate's ``forward``; the delegate's operator owns that call."""
    target = _FlakyTarget()
    _register(target, op="InstanceNormFwdOp")
    delegate = InstanceNormFwdOp()
    parent = _composite(delegate, reach="forward")

    target.fail = True
    with pytest.raises(RuntimeError, match="device fault"):
        parent(*_inputs())

    assert delegate.settled_target is None and not delegate.built_kernels("instance_norm")


@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="RoPE runs a CUDA kernel")
def test_a_compiled_rope_call_checks_its_positions_inside_the_operator():
    """The range check is the call's last step, so a first call failing it settles nothing."""
    from tileops.ops.rope import RopeNeoxPositionIdsFwdOp

    op = RopeNeoxPositionIdsFwdOp(max_position=16)
    x = torch.randn(4, 2, 64, device="cuda", dtype=DTYPE)
    position_ids = torch.tensor([0, 1, 2, 99], device="cuda")

    with pytest.raises(ValueError, match="max_position"):
        torch.compile(op, fullgraph=True)(x, position_ids)

    assert op.settled_target is None


@pytest.mark.usefixtures("isolated_dynamo")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="RoPE validates a CUDA input")
def test_a_compiled_rope_call_is_served_by_the_target_its_operator_picks():
    """RoPE resolves its kernel inside the operator, so the target's builder serves it."""
    from tileops.ops.rope import RopeNeoxFwdOp

    fail = [True]

    def build_kernel(*inputs, **params):
        def kernel(x):
            if fail[0]:
                raise RuntimeError("device fault")
            return torch.full_like(x, 7)

        return kernel

    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("RopeNeoxFwdOp", "acme", build_kernel)
    op = RopeNeoxFwdOp()
    call = torch.compile(op, fullgraph=True)
    x = torch.randn(16, 64, device="cuda", dtype=DTYPE)

    with pytest.raises(RuntimeError, match="device fault"):
        call(x)
    assert op.settled_target is None and not op.built_kernels("rope_neox") and op.kernel is None

    fail[0] = False
    assert torch.equal(call(x), torch.full_like(x, 7))
    assert op.settled_target == "acme"


def test_a_keyword_only_call_settles_and_records_by_name():
    """``__call__`` hands the call's keywords to settling and recording."""
    from tileops.ops.op_base import _recording_roofline_calls

    _register(_Recorder())
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)
    x, weight = _inputs()
    with _recording_roofline_calls():
        op(x=x, weight=weight)

    assert op.settled_target == "acme"
    assert set(op._roofline_call_tensors) == {"x", "weight"}


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
    """The operator settles the target at run time, so the target's kernel runs, once."""
    recorder = _Recorder()
    _register(recorder)
    op = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)
    x, weight = _inputs()

    output = torch.compile(op, fullgraph=True)(x, weight)

    assert torch.equal(output, torch.full_like(x, 7)), "the in-tree kernel ran instead"
    assert len(recorder.calls) == 1


@pytest.mark.usefixtures("isolated_dynamo")
def test_a_compiled_call_whose_build_fails_pins_nothing():
    """A compiled graph reaches the op through its operator, which owns the call."""
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
    assert params == {"dim": [1, 2], "keepdim": False}
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

    with pytest.raises(IndexError):
        SumFwdOp(dim=5)(torch.randn(4, 8, dtype=DTYPE))

    assert recorder.calls == []


# --------------------------------------------------------------------------------------
# What a target's callable serves: the whole op, on the tensors it was described with
# --------------------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="validates CUDA inputs")


@requires_cuda
def test_a_target_serves_the_whole_backward_the_in_tree_path_runs_in_two_passes():
    """GQA backward is two kernels in-tree; a target returns the three gradients at once."""
    from tileops.ops.attention.gqa import GroupedQueryAttentionBwdOp

    seen = []

    def build_kernel(*inputs, **params):
        def kernel(*tensors):
            seen.append(tensors)
            q, k, v = tensors[:3]
            return torch.ones_like(q), torch.ones_like(k), torch.ones_like(v)

        return kernel

    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("GroupedQueryAttentionBwdOp", "acme", build_kernel)
    op = GroupedQueryAttentionBwdOp(batch=1, heads=4, heads_kv=2, seq_len=64, dim=64)
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=DTYPE)
    k, v = (torch.randn(1, 64, 2, 64, device="cuda", dtype=DTYPE) for _ in range(2))
    o, do = torch.randn_like(q), torch.randn_like(q)
    lse = torch.randn(1, 4, 64, device="cuda", dtype=torch.float32)

    dq, dk, dv = op(q, k, v, o, do, lse)

    (tensors,) = seen
    assert all(a is b for a, b in zip(tensors, (q, k, v, o, do, lse), strict=True))
    assert torch.equal(dq, torch.ones_like(q)) and torch.equal(dv, torch.ones_like(v))


@requires_cuda
def test_a_target_serving_an_autograd_op_carries_its_own_backward():
    """DeltaNet's in-tree entry attaches the in-tree backward; a target's output brings its own."""
    from tileops.ops.linear_attention.deltanet import DeltaNetAutogradOp

    class _Doubled(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            return v * 2

        @staticmethod
        def backward(ctx, grad):
            return grad * 2

    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder(
        "DeltaNetAutogradOp", "acme", lambda *i, **p: lambda q, k, v, beta: _Doubled.apply(v)
    )
    op = DeltaNetAutogradOp(chunk_size=64)
    q, k, v = (
        torch.randn(1, 2, 64, 32, device="cuda", dtype=DTYPE, requires_grad=True) for _ in range(3)
    )
    beta = torch.rand(1, 2, 64, device="cuda", dtype=DTYPE)

    op(q, k, v, beta).sum().backward()

    assert torch.equal(v.grad, torch.full_like(v, 2))


@requires_cuda
def test_a_value_check_a_target_makes_reaches_the_caller():
    """RoPE's in-tree entry checks positions after launching; a target's callable checks its own."""
    from tileops.ops.rope import RopeNeoxPositionIdsFwdOp

    def build_kernel(*inputs, **params):
        def kernel(x, position_ids):
            raise ValueError("position_ids must be in [0, max_position)")

        return kernel

    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("RopeNeoxPositionIdsFwdOp", "acme", build_kernel)
    op = RopeNeoxPositionIdsFwdOp(max_position=16)
    x = torch.randn(4, 2, 64, device="cuda", dtype=DTYPE)

    with pytest.raises(ValueError, match="max_position"):
        op(x, torch.tensor([0, 1, 2, 99], device="cuda", dtype=torch.int32))


def test_a_callers_out_buffer_is_handed_over_per_call_not_built_in():
    """``out`` reaches the callable as a keyword and never the builder's params."""
    from tileops.ops.moe import ContiguousLayoutSpec, MoeGroupedGemmFwdOp

    built, called = [], []

    def build_kernel(*inputs, **params):
        built.append(params)

        def kernel(a, b, layout_metadata, out=None):
            called.append(out)
            return None if out is None else out.fill_(7)

        return kernel

    registry.register_detector("acme", lambda device: True)
    registry.register_kernel_builder("MoeGroupedGemmFwdOp", "acme", build_kernel)
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    a = torch.randn(4, 8, dtype=torch.bfloat16)
    b = torch.randn(2, 6, 8, dtype=torch.bfloat16)
    ends = torch.tensor([2, 4], dtype=torch.int32)
    out = torch.empty(4, 6, dtype=torch.bfloat16)

    assert op(a, b, ends, out=out) is out

    assert "out" not in built[0]
    assert called == [out] and torch.equal(out, torch.full_like(out, 7))
