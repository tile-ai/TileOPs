"""The seam between an op and a target's kernels.

What a third-party backend gets, observed from the op side: its builder is called with the
manifest's inputs and params, its kernel is memoized under the input signature, and
everything the op layer does for every target — validation, contiguity, output shape — still
happens. Uses a fake target, so no vendor hardware is involved.
"""

import pytest
import torch

from tileops.backend import BUILTIN, OpNotAvailableError, TensorSpec, registry
from tileops.ops.norm.rms_norm import RMSNormFwdOp

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


def _inputs(rows=4, shape=NORMALIZED_SHAPE, dtype=DTYPE):
    x = torch.randn(rows, *shape, dtype=dtype)
    weight = torch.randn(*shape, dtype=dtype)
    return x, weight


# --------------------------------------------------------------------------------------
# What the backend is asked, and what it gets back
# --------------------------------------------------------------------------------------


def test_a_target_takes_over_the_op_and_is_asked_with_the_manifest_signature():
    recorder = _Recorder()
    _register(recorder)
    x, weight = _inputs()

    out = RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)(x, weight)

    (inputs, params), = recorder.calls
    assert inputs == (TensorSpec.of(x), TensorSpec.of(weight)), "signature.inputs order"
    # eps defaults to null in the manifest; the op settles it, so a backend gets a number.
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
        op(torch.randn(4, *NORMALIZED_SHAPE, dtype=DTYPE),
           torch.randn(*NORMALIZED_SHAPE, dtype=torch.bfloat16))
    assert recorder.calls == [], "a rejected call never reaches the backend"


def test_a_non_contiguous_input_reaches_the_kernel_contiguous():
    recorder = _Recorder()
    _register(recorder)
    x, weight = _inputs(rows=8)
    strided = x[::2]
    assert not strided.is_contiguous()

    RMSNormFwdOp(normalized_shape=NORMALIZED_SHAPE)(strided, weight)

    (inputs, _), = recorder.calls
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
    ],
    ids=["shape", "dtype"],
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

    class StubOp(RMSNormFwdOp):
        """Stands in for the ops that still call get_or_build_kernel without inputs."""

        def forward(self, x, weight):
            return self.get_or_build_kernel("stub", key=x.dtype, build=lambda: None)

    StubOp.__name__ = "StubOp"
    with pytest.raises(OpNotAvailableError, match="not wired to external targets yet"):
        StubOp(normalized_shape=NORMALIZED_SHAPE)(*_inputs())


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
        op(torch.randn(4, *NORMALIZED_SHAPE, dtype=DTYPE),
           torch.randn(*NORMALIZED_SHAPE, dtype=torch.bfloat16))

    assert op._settled_target is None and not op.built_kernels("rms_norm")
    op(*_inputs())
    assert op._settled_target == "cpu_target", "the first call that worked decides"
    assert len(recorder.calls) == 1
