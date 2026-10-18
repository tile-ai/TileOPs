"""The trace tool's build state and output split, which need no GPU to check."""

import pytest
import tilelang.language as T

from tileops.trace import trace
from tileops.trace.passes import lookup_meta

pytestmark = pytest.mark.smoke

_MAX_EVENTS = 16


class _Compiled:
    """A compiled kernel as ``trace.run`` reads it: its ``prim_func`` and its outputs."""

    def __init__(self, prim_func, outputs):
        self.prim_func = prim_func
        self._outputs = outputs

    def __call__(self, *inputs):
        return self._outputs


def _kernel(group: str, flow: bool):
    @T.prim_func
    def main(a: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128), trace.group(group, lead=0), trace.range("r"):
            T.evaluate(0)
        if flow:
            trace.dag("r", "r")

    return main


def test_an_untraced_build_leaves_nothing_for_the_next_traced_one():
    for _ in range(2):
        trace.finalize(_kernel("other", flow=True), traced=False)
    lowered = trace.finalize(_kernel("mine", flow=False), traced=True, max_events=_MAX_EVENTS)

    maps = lookup_meta(_Compiled(lowered, None))
    assert maps["group_id_to_name"] == {0: "mine"}
    assert maps["flows"] == []
    slots = list(lowered.buffer_map.values())[-1]
    assert [int(d) for d in slots.shape] == [1, 1, (_MAX_EVENTS + 1) * 2]


def test_run_splits_slots_off_a_traced_build_with_tracing_off(monkeypatch):
    monkeypatch.setattr(trace, "_enabled", False)
    lowered = trace.finalize(_kernel("g", flow=False), traced=True, max_events=_MAX_EVENTS)
    assert trace.run(_Compiled(lowered, ("out", "slots")), (), stem="unused") == "out"

    stripped = trace.finalize(_kernel("g", flow=False), traced=False)
    assert trace.run(_Compiled(stripped, ("out", "x")), (), stem="unused") == ("out", "x")
