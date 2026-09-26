"""Contracts every implemented parametric entry keeps (docs/design/manifest.md § Workloads,
§ Algebraic Data Types).

An op that cannot run on meta tensors keeps its completed-call contract in its family's
benchmark, which runs every manifest call.
"""

import pytest

from tileops.backend import OpNotAvailableError
from tileops.manifest import load_adts, load_manifest
from tileops.manifest.plan import check_entry, entry_plan
from tileops.manifest.registry import op_class
from tileops.manifest.signature import is_legacy
from tileops.manifest.values import ADTValue, convert
from tileops.manifest.workload import instantiate

pytestmark = pytest.mark.smoke

_PARAMETRIC = {n: e for n, e in load_manifest().items() if not is_legacy(e)}
_IMPLEMENTED = sorted(n for n, e in _PARAMETRIC.items() if e.get("status") == "implemented")


def _calls(name: str):
    entry = _PARAMETRIC[name]
    plan = entry_plan(name, entry, load_adts())
    for row in entry["workloads"]:
        for case in row.get("dtype_cases") or [{}]:
            yield instantiate(plan, row, case)


@pytest.mark.parametrize("name", _IMPLEMENTED)
def test_every_manifest_call_completes_on_meta(name):
    assert check_entry(name, _PARAMETRIC[name], load_adts()) == ([], [])
    cls = op_class(name, _PARAMETRIC[name])
    for call in _calls(name):
        tensors = call.materialize("meta")
        op = cls(**call.arguments(tensors))
        try:
            op(*(tensors[t] for t in call.signature.inputs))
        except OpNotAvailableError:
            pytest.skip(f"{name} cannot run on meta tensors")
        flops, moved = op.eval_roofline()
        assert flops >= 0 and moved >= 0, call.case_id


@pytest.mark.parametrize("adt", sorted(load_adts()))
def test_each_adt_constructor_round_trips(adt):
    """A row's literal builds its constructor's object, which the construction check takes
    back with the literal's fields."""
    seen = set()
    for name in _PARAMETRIC:
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
