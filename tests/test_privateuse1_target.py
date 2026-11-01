"""The tileops_test target states each call as the manifest instantiates it."""

import functools
import math

import pytest
import torch

from tileops.backend import registered_targets
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest._expr import names, parse
from tileops.manifest._plan import entry_plan
from tileops.manifest._values import ADTValue
from tileops.manifest._workload import instantiate
from workloads.device import run_device

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        "tileops_test" not in registered_targets(),
        reason="tests the tileops_test target, which this run does not load",
    ),
]


def _comparable(value: object) -> object:
    if isinstance(value, ADTValue):
        return (value.adt, value.kind, value.fields)
    if isinstance(value, (list, tuple)):
        return tuple(_comparable(v) for v in value)
    return value


def _implemented() -> list[str]:
    return sorted(n for n, e in load_manifest().items() if e.get("status") == "implemented")


@functools.cache
def _smallest_call(name: str) -> tuple:
    """The plan of op *name* and its manifest call with the smallest inputs."""
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    calls = (
        instantiate(plan, row, case)
        for row in load_workloads(name)
        for case in row.get("dtype_cases") or [{}]
    )
    return plan, min(calls, key=lambda call: sum(math.prod(s) for s, _ in call.tensors.values()))


@pytest.mark.parametrize("name", _implemented())
def test_a_call_is_solved_from_its_tensors_as_the_manifest_row_states_it(name):
    """The indices and tensors the target solves are the row's, on the op's smallest call.

    An index only a generator reads is the row's choice of values, which the call holds.
    """
    from tests.privateuse1.target import _call

    plan, expected = _smallest_call(name)
    tensors = {}
    for t in plan.sig.call_tensors:
        spec = expected.specs[t]
        if spec is None:
            tensors[t] = None
        elif spec.values is not None:
            tensors[t] = torch.tensor(spec.values, dtype=getattr(torch, spec.dtype))
        else:
            # A construction-time tensor reaches the target with its values; an input's
            # values are not read.
            device = "cpu" if t in plan.sig.ctor_tensors else "meta"
            tensors[t] = torch.empty(spec.shape, dtype=getattr(torch, spec.dtype), device=device)
    inputs = {t: tensors[t] for t in plan.sig.inputs}
    params = expected.arguments(tensors)

    call, _ = _call(name, inputs, params)

    generated = set().union(
        *(names(parse(t.values)) for t in plan.sig.call_tensors.values() if t.values)
    )
    assert set(expected.indices) - set(call.indices) <= generated
    assert {n: _comparable(v) for n, v in call.indices.items()} == {
        n: _comparable(expected.indices[n]) for n in call.indices
    }
    assert call.tensors == expected.tensors


def test_a_call_whose_reference_reads_another_input_form_is_refused():
    """W4A8's reference reads row-major weights, its call the op's repack order."""
    from tests.conftest import UnservedError
    from tests.privateuse1.target import _Reference

    _, call = _smallest_call("GEMMW4A8FwdOp")
    tensors = call.materialize(run_device())
    kernel = _Reference("GEMMW4A8FwdOp", call.arguments(tensors))
    with pytest.raises(UnservedError, match="does not read the call's own inputs"):
        kernel(*(tensors[t] for t in call.signature.inputs))
