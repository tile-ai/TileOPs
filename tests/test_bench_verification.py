"""The pytest and benchmark adapters must consume the same workload contract."""

from types import SimpleNamespace

import pytest
import torch

from workloads.numerics import Exact

pytestmark = pytest.mark.smoke


def _workload():
    return SimpleNamespace(
        ref_program=lambda x: x, verification=lambda *inputs: Exact(atol=0, rtol=0)
    )


@pytest.mark.parametrize("consumer", ["test", "benchmark"])
def test_both_consumers_reject_the_same_wrong_result(consumer, monkeypatch):
    if consumer == "test":
        from tests.workload_test_base import TestBase
        from tileops.ops.elementwise import AbsFwdOp

        with pytest.raises(AssertionError):
            TestBase.check(_workload(), AbsFwdOp(), torch.ones(1), runs=lambda x: x + 0.001)
    else:
        from benchmarks import benchmark_base

        monkeypatch.setattr(
            benchmark_base, "bench_kernel", lambda *_a, **_k: pytest.fail("timed wrong output")
        )
        with pytest.raises(AssertionError):
            benchmark_base.OpBenchmark(object(), _workload()).compare(
                {"wrong": lambda x: x + 0.001}, torch.ones(1)
            )


def test_benchmark_uses_one_workload_declaration_for_every_tag(monkeypatch):
    from benchmarks import benchmark_base

    monkeypatch.setattr(benchmark_base, "_verifying", True)
    calls = []
    workload = _workload()
    workload.verification = lambda *inputs: (calls.append(inputs), Exact())[1]
    benchmark_base.OpBenchmark(object(), workload).compare(
        {"one": lambda x: x, "two": lambda x: x}, torch.ones(1)
    )
    assert len(calls) == 1


@pytest.mark.parametrize("tag", ["tileops", "absent"])
def test_noncomparable_cannot_disable_op_verification_or_name_a_missing_tag(tag):
    from benchmarks.benchmark_base import OpBenchmark

    with pytest.raises(ValueError, match="external baseline"):
        OpBenchmark(object(), _workload()).compare(
            {"tileops": lambda x: x}, torch.ones(1), noncomparable={tag: "different semantics"}
        )
