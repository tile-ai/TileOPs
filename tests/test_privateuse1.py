"""The PrivateUse1 test backend and its ``tileops_test`` target.

The backend keeps a tensor's bytes on its own device and copies its views. The target states
each call as the manifest instantiates it.
"""

import functools

import pytest
import torch

from tests.manifest_calls import implemented_ops, smallest_call
from tileops.backend import registered_targets
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest._expr import names, parse
from tileops.manifest._plan import entry_plan
from tileops.manifest._values import ADTValue
from workloads.device import run_device

pytestmark = pytest.mark.smoke

_ON_TEST_DEVICE = pytest.mark.skipif(
    torch.device(run_device()).type != "tileops_test",
    reason="tests the tileops_test backend, which this run does not use",
)
_ON_TEST_TARGET = pytest.mark.skipif(
    "tileops_test" not in registered_targets(),
    reason="tests the tileops_test target, which this run does not load",
)


@_ON_TEST_DEVICE
def test_a_tensor_grows_on_its_own_device_whichever_is_current(two_devices):
    first, second = two_devices
    x = torch.ones(3, device=second)

    with torch.accelerator.device_index(first.index):
        x.resize_(8)

    assert x.untyped_storage().device == second
    assert (x[:3] + 1).device == second


@_ON_TEST_DEVICE
@pytest.mark.parametrize(
    ("view", "expected"),
    [
        (lambda t: t.conj(), torch.tensor([1 - 2j])),
        (lambda t: torch._neg_view(t), torch.tensor([-1 - 2j])),
    ],
    ids=["conjugate", "negative"],
)
def test_a_lazy_view_reaches_the_host_resolved(view, expected):
    source = torch.tensor([1 + 2j], device=run_device())
    assert torch.equal(view(source).cpu(), expected)


@_ON_TEST_DEVICE
def test_a_copy_into_an_empty_tensor_checks_the_shapes():
    with pytest.raises(RuntimeError, match="must match the size"):
        torch.empty(0, device=run_device()).copy_(torch.ones(2))


@_ON_TEST_DEVICE
def test_a_device_generator_draws_what_the_host_generator_draws(two_devices):
    _, second = two_devices
    drawn = torch.randn(4, device=second, generator=torch.Generator(second).manual_seed(5))
    expected = torch.randn(4, generator=torch.Generator().manual_seed(5))
    assert drawn.device == second
    assert torch.equal(drawn.cpu(), expected)


@_ON_TEST_DEVICE
def test_a_convolution_and_its_gradients_match_the_host():
    host = [torch.randn(1, 2, 8), torch.randn(3, 2, 3), torch.randn(3)]
    on_device = [t.to(run_device()).requires_grad_() for t in host]
    for t in host:
        t.requires_grad_()
    expected = torch.nn.functional.conv1d(*host)
    got = torch.nn.functional.conv1d(*on_device)
    expected.sum().backward()
    got.sum().backward()
    assert torch.allclose(got.detach().cpu(), expected.detach())
    for t, u in zip(on_device, host, strict=True):
        assert torch.allclose(t.grad.cpu(), u.grad)


@_ON_TEST_DEVICE
def test_a_device_reports_free_and_total_memory():
    free, total = torch.accelerator.get_memory_info(0)
    assert 0 < free <= total


def _comparable(value: object) -> object:
    if isinstance(value, ADTValue):
        return (value.adt, value.kind, value.fields)
    if isinstance(value, (list, tuple)):
        return tuple(_comparable(v) for v in value)
    return value


@functools.cache
def _smallest_call(name: str) -> tuple:
    """The plan of op *name* and its manifest call with the smallest inputs."""
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    return plan, smallest_call(plan, load_workloads(name))


@_ON_TEST_TARGET
@pytest.mark.parametrize("name", implemented_ops())
def test_a_call_is_solved_from_its_tensors_as_the_manifest_row_states_it(name):
    """The indices and tensors the target solves are the row's, on the op's smallest call.

    An index only a generator reads is the row's choice of values, which the call holds.
    """
    from tests.privateuse1.target import _call

    plan, expected = _smallest_call(name)
    tensors = {}
    for t in plan.sig.inputs:
        spec = expected.specs[t]
        if spec is None:
            tensors[t] = expected.scalars.get(t)
        elif isinstance(spec, tuple):
            tensors[t] = [
                torch.empty(e.shape, dtype=getattr(torch, e.dtype), device="meta") for e in spec
            ]
        elif spec.values is not None:
            tensors[t] = torch.tensor(spec.values, dtype=getattr(torch, spec.dtype))
        else:
            tensors[t] = torch.empty(spec.shape, dtype=getattr(torch, spec.dtype), device="meta")
    inputs = {t: tensors[t] for t in plan.sig.inputs}
    params = expected.arguments()

    call = _call(name, inputs, params)

    generated = set().union(*(names(parse(t.values)) for t in plan.sig.inputs.values() if t.values))
    assert set(expected.indices) - set(call.indices) <= generated
    assert {n: _comparable(v) for n, v in call.indices.items()} == {
        n: _comparable(expected.indices[n]) for n in call.indices
    }
    assert call.tensors == expected.tensors


@_ON_TEST_TARGET
def test_a_call_whose_reference_reads_another_input_form_is_refused():
    """W4A8's reference reads row-major weights, its call the op's repack order."""
    from tests.conftest import UnservedError
    from tests.privateuse1.target import _Reference

    _, call = _smallest_call("GEMMW4A8FwdOp")
    tensors = call.materialize(run_device())
    kernel = _Reference("GEMMW4A8FwdOp", call.arguments())
    with pytest.raises(UnservedError, match="does not read the call's own inputs"):
        kernel(*(tensors[t] for t in call.signature.inputs))
