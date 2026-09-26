"""Benchmarks for the three Native Sparse Attention (NSA) varlen passes.

Each pass is compared against the torch reference in ``workloads.attention.deepseek``;
the top-k pass returns block ids, so it is timed but not compared for closeness.
"""

import pytest

from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.attention import NSACmpVarlenFwdOp, NSATopkVarlenFwdOp, NSAVarlenFwdOp
from workloads.attention.deepseek import NsaCmpFwdCall, NsaFwdCall, NsaTopkCall


def _bench(op_cls, workload_cls, call) -> None:
    test = workload_cls(call)
    inputs = test.gen_inputs()
    op = op_cls(**test.arguments())
    bm = ManifestBenchmark(op, test)
    bm.compare({"tileops": op, "torch-ref": test.ref_program}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(NSACmpVarlenFwdOp))
def test_nsa_cmp_fwd_varlen_bench(call) -> None:
    _bench(NSACmpVarlenFwdOp, NsaCmpFwdCall, call)


@pytest.mark.parametrize("call", manifest_calls(NSATopkVarlenFwdOp))
def test_nsa_topk_varlen_bench(call) -> None:
    _bench(NSATopkVarlenFwdOp, NsaTopkCall, call)


@pytest.mark.parametrize("call", manifest_calls(NSAVarlenFwdOp))
def test_nsa_fwd_varlen_bench(call) -> None:
    _bench(NSAVarlenFwdOp, NsaFwdCall, call)
