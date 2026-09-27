"""Every implemented op, served by a target for another kind of device.

What a backend is promised, checked on the whole manifest rather than on a sample: its
builder is described with the op's ``forward`` inputs, the kernel it returns is called
with exactly those tensors, and nothing on the way asks a CUDA device anything. A new op
joins by having a manifest entry; its call is the manifest call with the smallest inputs.
"""

import math

import pytest
import torch

from tileops.backend import TensorSpec, registry
from tileops.manifest import load_adts, load_manifest
from tileops.manifest.plan import entry_plan
from tileops.manifest.registry import op_class
from tileops.manifest.workload import instantiate

pytestmark = pytest.mark.smoke


def _from_call(cls: type, name: str, entry: dict) -> tuple:
    """The op, its ``forward`` arguments and the outputs its call declares, from the manifest
    call with the smallest inputs."""
    plan = entry_plan(name, entry, load_adts(), resolve=False)
    calls = [
        instantiate(plan, row, case)
        for row in entry["workloads"]
        for case in row.get("dtype_cases") or [{}]
    ]
    call = min(calls, key=lambda c: sum(math.prod(c.tensors[t][0]) for t in c.tensors))
    tensors = {
        n: None if spec is None else torch.empty(spec.shape, dtype=getattr(torch, spec.dtype))
        for n, spec in call.specs.items()
    }
    args = [tensors[n] for n in plan.sig.inputs]
    while args and args[-1] is None:
        args.pop()
    outputs = [tensors[o] for o in plan.sig.outputs]
    return cls(**call.arguments(tensors)), tuple(args), outputs


def _implemented() -> list[str]:
    return sorted(n for n, e in load_manifest().items() if e.get("status") == "implemented")


@pytest.fixture(autouse=True)
def isolated_registry():
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    registry.register_detector("elsewhere", lambda device: device.type == "cpu")
    registry.default_target = "elsewhere"
    yield
    registry.restore(state)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    """A host with no CUDA: any device query fails the call that makes it."""
    from tileops.utils import forget_device_properties

    def refuse(*args, **kwargs):
        raise AssertionError("queried a CUDA device")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in (
        "current_device",
        "get_device_capability",
        "get_device_properties",
        "get_device_name",
    ):
        monkeypatch.setattr(torch.cuda, name, refuse)
    forget_device_properties()
    yield
    forget_device_properties()


@pytest.mark.parametrize("name", _implemented())
def test_a_target_is_described_and_called_with_the_forward_inputs(name):
    entry = load_manifest()[name]
    cls = op_class(name, entry)
    op, args, declared_outputs = _from_call(cls, name, entry)
    declared = tuple(entry["signature"].get("inputs") or {})
    passed = args + (None,) * (len(declared) - len(args))
    described = tuple(None if t is None else TensorSpec.of(t) for t in passed)
    seen, returned = [], []

    def build_kernel(*specs, **params):
        assert specs == described
        assert params == op._manifest_params()

        def kernel(*tensors, **writes):
            seen.append(tensors)
            result = declared_outputs
            returned.append(
                None if not result else result[0] if len(result) == 1 else tuple(result)
            )
            return returned[-1]

        return kernel

    registry.register_kernel_builder(name, "elsewhere", build_kernel)

    result = op(*args)

    assert len(seen) == 1, "the kernel the target built runs the call"
    assert all(a is b for a, b in zip(seen[0], passed, strict=True)), (
        "called with what it was described"
    )
    outputs = tuple(entry["signature"]["outputs"])
    if all(o in declared for o in outputs):
        assert result is None, "an op that writes its caller's buffer returns nothing"
    elif isinstance(returned[0], tuple):
        assert all(a is b for a, b in zip(result, returned[0], strict=True))
    else:
        assert result is returned[0], "the op returns what the target's kernel returned"
    assert hasattr(cls, "_signature"), "a target is held to its signature"
    inputs = entry["signature"].get("inputs") or {}
    assert all((inputs[o] or {}).get("mutated") for o in outputs if o in inputs), (
        "an output passed in as an input is one the call writes"
    )
