"""Contracts every implemented entry keeps (docs/design/manifest.md § Workloads, § Effects,
§ Algebraic Data Types).

An op that cannot run on meta tensors keeps its completed-call contract in its family's
benchmark, which runs every manifest call.
"""

import inspect

import pytest
import torch

from tileops._ops._signature_codegen import CheckError
from tileops.backend import OpNotAvailableError
from tileops.manifest import load_adts, load_manifest
from tileops.manifest._plan import check_entry, entry_plan
from tileops.manifest._registry import op_class
from tileops.manifest._values import ADTValue, convert
from tileops.manifest._workload import instantiate

pytestmark = pytest.mark.smoke

_MANIFEST = load_manifest()
_IMPLEMENTED = sorted(n for n, e in _MANIFEST.items() if e.get("status") == "implemented")


def _calls(name: str):
    entry = _MANIFEST[name]
    plan = entry_plan(name, entry, load_adts())
    for row in entry["workloads"]:
        for case in row.get("dtype_cases") or [{}]:
            yield instantiate(plan, row, case)


@pytest.mark.parametrize("name", _IMPLEMENTED)
def test_every_manifest_call_completes_on_meta(name):
    assert check_entry(name, _MANIFEST[name], load_adts()) == ([], [])
    cls = op_class(name, _MANIFEST[name])
    for call in _calls(name):
        tensors = call.materialize("meta")
        arguments = call.arguments(tensors)
        if (
            "device" in _MANIFEST[name]["signature"].get("params", {})
            and arguments.get("device") is None
        ):
            # An op with no tensor input runs where its device param says.
            arguments["device"] = "meta"
        op = cls(**arguments)
        try:
            args, keywords = call.signature.forward_arguments(
                tensors[t] for t in call.signature.inputs
            )
            op(*args, **keywords)
        except OpNotAvailableError:
            pytest.skip(f"{name} cannot run on meta tensors")
        try:
            flops, moved = op.eval_roofline()
        except OpNotAvailableError:
            # A formula reading metadata values is priced in the op's benchmark.
            continue
        assert flops >= 0 and moved >= 0, call.case_id


_WRITERS = [
    n
    for n in _IMPLEMENTED
    if any(
        (d or {}).get("mutated") and t != "out"
        for t, d in (_MANIFEST[n]["signature"].get("inputs") or {}).items()
    )
]


@pytest.mark.parametrize("name", _WRITERS)
def test_an_op_writing_an_input_says_so_in_its_summary_line(name):
    summary = (inspect.getdoc(op_class(name, _MANIFEST[name])) or "").splitlines()[:1]
    assert summary and "in place" in summary[0], f"{name}: {summary}"


_SOFTMAX_SCALED = [
    n for n in _IMPLEMENTED if "sm_scale" in _MANIFEST[n]["signature"].get("params", {})
]


@pytest.mark.parametrize("sm_scale", [0.0, -0.125])
@pytest.mark.parametrize("name", _SOFTMAX_SCALED)
def test_a_softmax_scale_is_none_or_positive(name, sm_scale):
    call = next(_calls(name))
    arguments = {**call.arguments(call.materialize("meta")), "sm_scale": sm_scale}
    with pytest.raises(CheckError, match="sm_scale"):
        op_class(name, _MANIFEST[name])(**arguments)


def _offsets_inputs(name: str) -> list[str]:
    inputs = _MANIFEST[name]["signature"].get("inputs") or {}
    return [
        t
        for t, d in inputs.items()
        if any(r.startswith("prefix_offsets") for r in (d or {}).get("requires", []))
    ]


def _cpu_mirrors(name: str) -> list[str]:
    inputs = _MANIFEST[name]["signature"].get("inputs") or {}
    return [t for t in _offsets_inputs(name) if (inputs[t] or {}).get("device") == "cpu"]


def _corrupted(kind: str, offsets: torch.Tensor) -> torch.Tensor:
    """*offsets*, of at least three entries, broken as *kind* says."""
    bad = offsets.clone()
    if kind == "nonzero-start":
        bad[0] += 1
    elif kind == "decreasing":
        bad[1] = bad[2] + 1
    else:  # "mirror-differs": still valid offsets of the same tokens, but other ones.
        bad[1:-1] = bad[-1]
        assert not torch.equal(bad, offsets)
    return bad


@pytest.mark.parametrize(
    "name, kind",
    [
        (n, kind)
        for n in _IMPLEMENTED
        if _offsets_inputs(n)
        for kind in ("nonzero-start", "decreasing", "mirror-differs")
        if kind != "mirror-differs" or _cpu_mirrors(n)
    ],
)
def test_validate_inputs_rejects_offsets_that_do_not_span_the_tokens(name, kind):
    offsets = _offsets_inputs(name)
    call = next(
        c
        for c in _calls(name)
        if all(c.specs.get(t) is not None and c.specs[t].shape[0] >= 3 for t in offsets)
    )
    tensors = call.materialize("cpu")
    op = op_class(name, _MANIFEST[name])(**{**call.arguments(tensors), "validate_inputs": True})
    for corrupt in _cpu_mirrors(name) if kind == "mirror-differs" else offsets:
        changed = {**tensors, corrupt: _corrupted(kind, tensors[corrupt])}
        args, keywords = call.signature.forward_arguments(changed[t] for t in call.signature.inputs)
        with pytest.raises(ValueError, match=rf"^{corrupt}[\[ ]"):
            op(*args, **keywords)


@pytest.mark.parametrize("adt", sorted(load_adts()))
def test_each_adt_constructor_round_trips(adt):
    """A row's literal builds its constructor's object, which the construction check takes
    back with the literal's fields."""
    seen = set()
    for name in _MANIFEST:
        for call in _calls(name):
            objects = call.arguments(dict.fromkeys(call.signature.ctor_tensors))
            for p, literal in call.params.items():
                if not (isinstance(literal, ADTValue) and literal.adt == adt):
                    continue
                built = objects[p]
                assert convert(built, call.signature.params[p]["type"], load_adts()) is built
                for f in literal.fields:
                    value = getattr(built, f)
                    assert getattr(value, "value", value) == literal.fields[f], (name, p, f)
                seen.add(literal.kind)
    ctors = load_adts()[adt]["sum"]
    assert seen == set(ctors), f"no row builds {sorted(set(ctors) - seen)}"
