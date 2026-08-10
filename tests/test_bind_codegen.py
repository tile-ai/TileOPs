"""Binding a forward call from the manifest, and dispatching the bound call to a target."""

import pytest
import torch

from tileops.backend import InputSpec, OpNotAvailableError, registry
from tileops.ops._bind_codegen import synthesize_bind_call
from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke

_SIG = {
    "inputs": {"x": {"dtype": "float16"}, "weight": {"dtype": "same_as(x)"}},
    "params": {"normalized_shape": {"type": "tuple[int, ...]"}, "eps": {"default": None}},
}


class _Bound:
    """Enough of an op to bind against: the params live on the instance, by manifest name."""

    _bind_call = synthesize_bind_call("FakeOp", _SIG)

    def __init__(self, normalized_shape=(4,), eps=None):
        self.normalized_shape = normalized_shape
        self.eps = eps


def test_tensors_come_back_in_manifest_order_and_params_off_the_instance():
    x, weight = torch.zeros(2, 4), torch.ones(4)
    tensors, params = _Bound()._bind_call(x, weight)

    assert tensors == (x, weight)
    assert params == {"normalized_shape": (4,), "eps": None}


def test_tensors_may_be_named():
    x, weight = torch.zeros(2, 4), torch.ones(4)
    assert _Bound()._bind_call(weight=weight, x=x)[0] == (x, weight)


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((), {}, "missing tensor inputs"),
        ((1, 2, 3), {}, "takes 2 tensor inputs"),
        ((1,), {"x": 2}, "two values"),
        ((1, 2), {"z": 3}, "no tensor input"),
    ],
    ids=["too few", "too many", "duplicate", "unknown name"],
)
def test_a_call_that_does_not_match_the_manifest_says_so(args, kwargs, message):
    with pytest.raises(TypeError, match=message):
        _Bound()._bind_call(*args, **kwargs)


def test_a_param_kept_privately_still_binds():
    """Several ops store a param as ``_name``; both spellings are one declaration."""
    op = _Bound()
    del op.eps
    op._eps = 1e-5
    _, params = op._bind_call(torch.zeros(2, 4), torch.ones(4))
    assert params["eps"] == 1e-5


def test_a_param_the_instance_does_not_hold_is_named():
    """The manifest param name is the attribute name; a mismatch is a bug worth pointing at."""
    op = _Bound()
    del op.eps
    with pytest.raises(AttributeError, match=r"manifest param 'eps'.*'_eps'"):
        op._bind_call(torch.zeros(2, 4), torch.ones(4))


def test_an_op_with_no_tensor_input_cannot_be_bound():
    """Nothing to bind, and no device to dispatch from either."""
    with pytest.raises(ValueError, match="signature.inputs is missing or empty"):
        synthesize_bind_call("NoTensors", {"inputs": {}})


class _Routed(Op):
    """A minimal op on the backend chain, so dispatch can be observed without a GPU."""

    OP_NAME = "FakeRoutedOp"
    _bind_call = synthesize_bind_call("FakeRoutedOp", _SIG)

    def __init__(self, target=None):
        self.normalized_shape = (4,)
        self.eps = None
        self.target = target

    def forward(self, x, weight):
        tensors, params = self._bind_call(x, weight)
        return self.backend_kernel(*tensors, **params)


@pytest.fixture
def registered():
    """Register a fake target for the routed op, and put the registry back afterwards."""
    state = registry.snapshot()
    built = []

    def get_kernel(*specs, **params):
        built.append((specs, params))
        return lambda *tensors: tensors[0]

    registry.register(op="FakeRoutedOp", target="fake", get_kernel=get_kernel)
    registry.register_detector(target="fake", detect=lambda device: device.type == "cpu")
    yield built
    registry.restore(state)


def test_the_backend_is_asked_with_descriptions_of_what_the_kernel_receives(registered):
    x, weight = torch.zeros(2, 4), torch.ones(4)
    _Routed()(x, weight)

    ((specs, params),) = registered
    assert specs == (InputSpec.of(x), InputSpec.of(weight))
    assert params == {"normalized_shape": (4,), "eps": None}


def test_a_repeat_call_on_the_same_shapes_does_not_ask_again(registered):
    op = _Routed()
    op(torch.zeros(2, 4), torch.ones(4))
    op(torch.ones(2, 4), torch.zeros(4))
    assert len(registered) == 1

    op(torch.zeros(3, 4), torch.ones(4))  # a shape it has not built for
    assert len(registered) == 2


def test_a_failed_first_call_pins_nothing(registered):
    """Until a build succeeds the op holds no target's work, so nothing is settled yet."""
    from tileops.backend import set_default_target

    registry.register_detector(target="broken", detect=lambda device: False)
    set_default_target("broken")
    op = _Routed()
    with pytest.raises(OpNotAvailableError):
        op(torch.zeros(2, 4), torch.ones(4))

    set_default_target("fake")
    op(torch.zeros(2, 4), torch.ones(4))
    assert len(registered) == 1


def test_the_target_is_settled_on_first_dispatch(registered):
    """Kernels already built belong to a target, so a later default must not re-aim them."""
    from tileops.backend import set_default_target

    op = _Routed()
    op(torch.zeros(2, 4), torch.ones(4))

    registry.register(op="FakeRoutedOp", target="other", get_kernel=lambda *a, **k: None)
    set_default_target("other")
    op(torch.zeros(3, 4), torch.ones(4))  # a fresh shape: builds again, same target
    assert len(registered) == 2


def test_an_explicit_target_is_used_as_named(registered):
    x, weight = torch.zeros(2, 4), torch.ones(4)
    _Routed(target="fake")(x, weight)
    assert len(registered) == 1


def test_an_explicit_target_that_does_not_serve_this_op_does_not_fall_back(registered):
    """Even with a detector that would have claimed the device."""
    with pytest.raises(OpNotAvailableError, match=r"'absent'.*this op: \['fake'\]"):
        _Routed(target="absent")(torch.zeros(2, 4), torch.ones(4))
