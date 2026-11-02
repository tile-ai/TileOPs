"""The ``tileops_test`` target: every manifest op, computed by its reference on the device.

Registered through the public backend protocol, as an installed backend registers: a
detector for the test device type, that type's device facts, and one kernel builder per
manifest op. An op's kernel runs the reference of the op's call workload, built
from the manifest call its tensors and params state.
"""

import ast
import functools
import inspect

import torch

from tests.conftest import UnservedError
from tests.privateuse1 import DEVICE_TYPE
from tileops.backend import (
    DeviceFacts,
    register_detector,
    register_device_facts,
    register_kernel_builder,
)
from tileops.manifest import load_adts, load_manifest
from tileops.manifest._expr import parse, value_at
from tileops.manifest._kinds import split_union
from tileops.manifest._plan import EntryPlan, entry_plan
from tileops.manifest._signature import unification
from tileops.manifest._values import ADTValue, convert
from tileops.manifest._workload import Call, TensorSpec, _apply, _dtype_value, _point, _shape
from workloads.calls import call_workload

TARGET = DEVICE_TYPE


@functools.cache
def _plan(name: str) -> EntryPlan:
    return entry_plan(name, load_manifest()[name], load_adts(), resolve=False)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_names(value: object) -> object:
    """*value* with each dtype in it named, as the manifest names one."""
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_dtype_names(v) for v in value)
    return value


def _manifest_value(value: object, type_text: object, adts: dict) -> object:
    """A param value a constructor received, as manifest expressions read it: a dtype by
    its name, an ADT object as the literal of its constructor."""
    value = convert(_dtype_names(value), type_text, adts)
    for member in (m.strip() for m in split_union(str(type_text))):
        ctors = adts.get(member, {}).get("sum", {})
        kind = getattr(value, "kind", None)
        if kind in ctors and not isinstance(value, ADTValue):
            fields = ctors[kind].get("fields") or {}
            return ADTValue(
                member,
                {kind: {f: getattr(getattr(value, f), "value", getattr(value, f)) for f in fields}},
            )
    return value


def _call(name: str, inputs: dict, params: dict) -> tuple[Call, frozenset]:
    """The manifest call of op *name* on *inputs* with *params*, and the inputs it writes.

    The indices are solved from the shapes and dtypes of the call-time inputs and the
    construction-time tensors among *params*, as a workload row's are. A construction-time
    tensor's spec carries its values, which the call does not otherwise pass.
    """
    plan = _plan(name)
    sig = plan.sig
    tensors = {**{t: params.get(t) for t in sig.ctor_tensors}, **inputs}
    scope = {
        p: _manifest_value(params.get(p, decl.get("default")), decl.get("type"), sig.adts)
        for p, decl in sig.params.items()
    }
    present = {t for t, v in tensors.items() if v is not None}
    point = _point(sig, scope, present)
    b = plan.branch(point)
    for t in present:
        node = b.dtypes[t]
        if isinstance(node, ast.Name) and node.id in sig.forall:
            scope[node.id] = _dtype_name(tensors[t].dtype)
    nodes = {t: b.shapes[t] for t in present}
    shapes = {t: tuple(tensors[t].shape) for t in present}
    _apply(sig, unification(nodes, set(scope), b.lets), nodes, shapes, scope, b.lets)
    specs = {}
    for t in sig.call_tensors.values():
        tensor = tensors[t.name]
        values = t.values is not None or t.name in sig.ctor_tensors
        specs[t.name] = (
            None
            if tensor is None
            else TensorSpec(
                shapes[t.name],
                _dtype_name(tensor.dtype),
                tensor.tolist() if values else None,
                t.cpu,
            )
        )
    for t in sig.outputs.values():
        specs[t.name] = (
            TensorSpec(
                _shape(b.shapes[t.name], sig, scope, f"tensor {t.name!r} shape"),
                _dtype_value(b.dtypes[t.name], scope),
                None,
                t.cpu,
            )
            if t.name in b.shapes
            else None
        )
    seqs = {n for n, k in sig.forall.items() if k == "Seq[Int]"}
    indices = {
        n: v
        for n, v in scope.items()
        if n in sig.params or (n in sig.forall and n not in seqs) or n in b.lets
    }
    written = frozenset(
        t.name
        for t in sig.inputs.values()
        if t.name in present
        and (
            t.write_only
            or t.mutated is True
            or (isinstance(t.mutated, str) and value_at(parse(t.mutated), point) is True)
        )
    )
    return Call("", {p: scope[p] for p in sig.params}, specs, indices, sig), written


class _Reference:
    """One op's kernel: the reference of its call workload, on the call's device."""

    def __init__(self, name: str, params: dict):
        self.name, self.params = name, params

    def __call__(self, *inputs, **writes):
        name, sig = self.name, _plan(self.name).sig
        named = dict(zip(sig.inputs, inputs, strict=True))
        call, written = _call(name, named, self.params)
        try:
            workload = call_workload(name)(call)
        except Exception as exc:
            raise UnservedError(f"{name}: no workload states this call: {exc!r}") from exc
        if not workload.reference_takes_call_inputs:
            raise UnservedError(f"{name}: the reference does not read the call's own inputs")
        reference = workload.ref_program
        seed = self.params.get("seed")
        try:
            passed = list(inputs)
            while passed and passed[-1] is None and not _takes(reference, len(passed)):
                passed.pop()
            # A reference runs with gradients on, as the shared verifier runs it.
            with torch.enable_grad(), torch.random.fork_rng(devices=[], enabled=seed is not None):
                if seed is not None:
                    # The op's seed param seeds its draws, so a seeded call repeats its result.
                    torch.manual_seed(seed)
                expected = reference(*passed)
        except Exception as exc:
            raise UnservedError(
                f"{name}: the reference does not compute this call: {exc!r}"
            ) from exc
        values = list(expected) if isinstance(expected, (tuple, list)) else [expected]
        outputs = list(sig.outputs)
        aliased = {sig.outputs[o].alias for o in outputs}
        mutable = [
            t
            for t, decl in sig.inputs.items()
            if (decl.mutated or decl.write_only) and t not in aliased
        ]
        if len(values) not in (len(outputs), len(outputs) + len(mutable)):
            raise UnservedError(
                f"{name}: the reference returns {len(values)} values for outputs {outputs} "
                f"and mutable inputs {mutable}"
            )
        stated = dict(zip(outputs + mutable, values, strict=False))
        produced = [o for o in outputs if call.specs[o] is not None]
        for o in [*produced, *(t for t in mutable if t in written)]:
            shape = call.specs[o].shape
            value = stated.get(o)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise UnservedError(f"{name}: the reference does not state {o} of shape {shape}")
        device = self._device(inputs, writes)
        results = []
        for o in outputs:
            spec, decl = call.specs[o], sig.outputs[o]
            if spec is None:
                results.append(None)
                continue
            if decl.alias in written:
                result = named[decl.alias]
            elif decl.buffer and "out" in writes:
                result = writes["out"]
            else:
                result = torch.empty(
                    spec.shape,
                    dtype=getattr(torch, spec.dtype),
                    device="cpu" if spec.cpu else device,
                )
            results.append(result.copy_(stated[o].detach()))
        for t in mutable:
            if t in written:
                named[t].copy_(stated[t].detach())
        if not results:
            return None
        return results[0] if len(results) == 1 else tuple(results)

    def _device(self, inputs: tuple, writes: dict) -> torch.device:
        """The call device, by the rule the op's checks place a call by."""
        tensors = [t for t in (*inputs, *writes.values()) if t is not None]
        devices = {t.device for t in tensors if t.device.type != "cpu"} or {
            t.device for t in tensors
        }
        if devices:
            return next(iter(devices))
        if self.params.get("device") is not None:
            return torch.device(self.params["device"])
        return torch.device(DEVICE_TYPE, torch.accelerator.current_device_index())


def _takes(fn, count: int) -> bool:
    """Whether *fn* accepts *count* positional arguments."""
    try:
        inspect.signature(fn).bind(*range(count))
    except TypeError:
        return False
    return True


def _builder(name: str):
    def build_kernel(*specs, **params):
        return _Reference(name, params)

    return build_kernel


register_detector(TARGET, lambda device: device.type == DEVICE_TYPE)
# The host runs every kernel: no compute units or shared memory to select by.
register_device_facts(DEVICE_TYPE, lambda device: DeviceFacts(arch=0, sm_count=1, smem_budget=0))
for _name in load_manifest():
    register_kernel_builder(_name, TARGET, _builder(_name))
