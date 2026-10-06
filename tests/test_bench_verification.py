"""The pytest and benchmark adapters must consume the same workload contract."""

from types import SimpleNamespace

import pytest
import torch

from workloads.numerics import Exact

pytestmark = pytest.mark.smoke


def _workload():
    return SimpleNamespace(
        gen_inputs=lambda: (torch.ones(1),),
        ref_program=lambda x: x,
        verification=lambda *inputs: Exact(atol=0, rtol=0),
        arguments=dict,
    )


@pytest.mark.parametrize("consumer", ["test", "benchmark"])
def test_both_consumers_reject_the_same_wrong_result(consumer, monkeypatch):
    from tileops.ops.elementwise import AbsFwdOp

    if consumer == "test":
        from tests.workload_test_base import TestBase

        with pytest.raises(AssertionError):
            TestBase.check(_workload(), AbsFwdOp(), torch.ones(1), runs=lambda x: x + 0.001)
    else:
        from benchmarks import api
        from benchmarks._cases import Entry

        monkeypatch.setattr(
            api, "bench_kernel", lambda *_a, **_k: pytest.fail("timed wrong output")
        )

        # The op's own row is right; the wrong implementation alone must be rejected.
        def right(op, case):
            return api.Implementation(run=lambda x: x + 0)

        case = api.Case("probe", None, Entry(lambda _call: _workload(), binder=right))
        with pytest.raises(AssertionError):
            api.Runner(AbsFwdOp(), case).compare({"wrong": lambda x: x + 0.001})
