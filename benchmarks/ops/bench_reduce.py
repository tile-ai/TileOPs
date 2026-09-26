"""Benchmarks for the 8 basic reduce ops.

Measures latency, TFLOPS, and DRAM bandwidth against PyTorch baselines. Each case is one
manifest call (src/tileops/manifest/); the roofline comes from ``op.eval_roofline()``.

Every row is timed against torch eager, the same reference through inductor, and
flag_gems' Triton reduction where one exists. amin has none: flag_gems 5.0.2
exposes amax but no amin, and ``min_dim`` also produces indices. A row passing ``dtype``
has no flag_gems tag either: its reductions take no output dtype.

The flag_gems callables cast nothing — like aten, its reductions accumulate in fp32
and return the storage dtype — so each tag is one kernel.
"""

from typing import Callable, Optional

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_dims,
    flaggems_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.reduction.reduce import (
    AmaxFwdOp,
    AminFwdOp,
    MeanFwdOp,
    ProdFwdOp,
    StdFwdOp,
    SumFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
)
from workloads.reduction import ProdCall, ReductionCall


def _bench(
    op_cls: type,
    workload: ReductionCall,
    baseline_fn: Callable,
    flaggems_fn: Optional[Callable] = None,
) -> None:
    """Check flag_gems against the reference, then time it and the op with torch."""
    inputs = workload.gen_inputs()
    op = op_cls(**workload.arguments())
    tolerance = reference_tolerance(inputs[0].dtype)
    if inputs[0].dtype == torch.float32:
        # Summation order moves the low bits of a long float32 reduction by more than the
        # elementwise float32 tolerance allows.
        tolerance = {"rtol": 1e-4, "atol": 1e-4}
    functors = {"tileops": op}
    if flaggems_fn is not None:
        assert_matches_reference(flaggems_fn, baseline_fn, *inputs, **tolerance)
        functors[FLAGGEMS_TAG] = flaggems_fn
    functors["torch"] = baseline_fn
    functors[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    ManifestBenchmark(op, workload).compare(functors, *inputs)


def _out_dtype(x: torch.Tensor, params: dict) -> torch.dtype:
    return getattr(torch, params["dtype"]) if params.get("dtype") else x.dtype


def _flaggems(name: str, params: dict, *extra, **kwargs) -> Optional[Callable]:
    """flag_gems' reduction over the row's axes, or ``None`` for a row passing ``dtype``."""
    if params.get("dtype"):
        return None
    fn = flaggems_op(name)
    dim = params["dim"]
    return lambda x: fn(x, flaggems_dims(dim), *extra, keepdim=params["keepdim"], **kwargs)


@pytest.mark.parametrize("call", manifest_calls(SumFwdOp))
def test_sum_bench(call) -> None:
    p = call.params

    def baseline_fn(x):
        return x.float().sum(dim=p["dim"], keepdim=p["keepdim"]).to(_out_dtype(x, p))

    _bench(SumFwdOp, ReductionCall(call), baseline_fn, _flaggems("sum_dim", p))


@pytest.mark.parametrize("call", manifest_calls(MeanFwdOp))
def test_mean_bench(call) -> None:
    p = call.params

    def baseline_fn(x):
        return x.float().mean(dim=p["dim"], keepdim=p["keepdim"]).to(_out_dtype(x, p))

    _bench(MeanFwdOp, ReductionCall(call), baseline_fn, _flaggems("mean_dim", p))


@pytest.mark.parametrize("call", manifest_calls(AmaxFwdOp))
def test_amax_bench(call) -> None:
    p = call.params

    def baseline_fn(x):
        return x.amax(dim=p["dim"], keepdim=p["keepdim"])

    _bench(AmaxFwdOp, ReductionCall(call), baseline_fn, _flaggems("amax", p))


@pytest.mark.parametrize("call", manifest_calls(AminFwdOp))
def test_amin_bench(call) -> None:
    p = call.params

    def baseline_fn(x):
        return x.amin(dim=p["dim"], keepdim=p["keepdim"])

    _bench(AminFwdOp, ReductionCall(call), baseline_fn)


@pytest.mark.parametrize("call", manifest_calls(ProdFwdOp))
def test_prod_bench(call) -> None:
    p = call.params

    def baseline_fn(x):
        return x.float().prod(dim=p["dim"], keepdim=p["keepdim"]).to(_out_dtype(x, p))

    flaggems_fn = None
    if not p.get("dtype"):
        flaggems_prod = flaggems_op("prod_dim")

        def flaggems_fn(x):
            return flaggems_prod(x, p["dim"], p["keepdim"])

    _bench(ProdFwdOp, ProdCall(call), baseline_fn, flaggems_fn)


def _correction(params: dict):
    return 1 if params["correction"] is None else params["correction"]


@pytest.mark.parametrize("call", manifest_calls(StdFwdOp))
def test_std_bench(call) -> None:
    p = call.params
    c = _correction(p)

    def baseline_fn(x):
        return x.float().std(dim=p["dim"], keepdim=p["keepdim"], correction=c).to(x.dtype)

    _bench(StdFwdOp, ReductionCall(call), baseline_fn, _flaggems("std", p, correction=c))


@pytest.mark.parametrize("call", manifest_calls(VarFwdOp))
def test_var_bench(call) -> None:
    p = call.params
    c = _correction(p)

    def baseline_fn(x):
        return x.float().var(dim=p["dim"], keepdim=p["keepdim"], correction=c).to(x.dtype)

    _bench(VarFwdOp, ReductionCall(call), baseline_fn, _flaggems("var_dim", p, correction=c))


@pytest.mark.parametrize("call", manifest_calls(VarMeanFwdOp))
def test_var_mean_bench(call) -> None:
    p = call.params
    c = _correction(p)

    def baseline_fn(x):
        v = x.float().var(dim=p["dim"], keepdim=p["keepdim"], correction=c).to(x.dtype)
        m = x.float().mean(dim=p["dim"], keepdim=p["keepdim"]).to(x.dtype)
        return (v, m)

    _bench(VarMeanFwdOp, ReductionCall(call), baseline_fn, _flaggems("var_mean", p, correction=c))
