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
    compiled_reference,
    flaggems_dims,
    flaggems_op,
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


def _bench(op_cls: type, workload: ReductionCall, flaggems_fn: Optional[Callable] = None) -> None:
    """Check flag_gems against the reference, then time it and the op with torch."""
    baseline_fn = workload.ref_program
    inputs = workload.gen_inputs()
    op = op_cls(**workload.arguments())
    functors = {"tileops": op}
    if flaggems_fn is not None:
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

    _bench(SumFwdOp, ReductionCall(call), _flaggems("sum_dim", p))


@pytest.mark.parametrize("call", manifest_calls(MeanFwdOp))
def test_mean_bench(call) -> None:
    p = call.params

    _bench(MeanFwdOp, ReductionCall(call), _flaggems("mean_dim", p))


@pytest.mark.parametrize("call", manifest_calls(AmaxFwdOp))
def test_amax_bench(call) -> None:
    p = call.params

    _bench(AmaxFwdOp, ReductionCall(call), _flaggems("amax", p))


@pytest.mark.parametrize("call", manifest_calls(AminFwdOp))
def test_amin_bench(call) -> None:
    _bench(AminFwdOp, ReductionCall(call))


@pytest.mark.parametrize("call", manifest_calls(ProdFwdOp))
def test_prod_bench(call) -> None:
    p = call.params

    flaggems_fn = None
    if not p.get("dtype"):
        flaggems_prod = flaggems_op("prod_dim")

        def flaggems_fn(x):
            return flaggems_prod(x, p["dim"], p["keepdim"])

    _bench(ProdFwdOp, ProdCall(call), flaggems_fn)


def _correction(params: dict):
    return 1 if params["correction"] is None else params["correction"]


@pytest.mark.parametrize("call", manifest_calls(StdFwdOp))
def test_std_bench(call) -> None:
    p = call.params
    c = _correction(p)

    _bench(StdFwdOp, ReductionCall(call), _flaggems("std", p, correction=c))


@pytest.mark.parametrize("call", manifest_calls(VarFwdOp))
def test_var_bench(call) -> None:
    p = call.params
    c = _correction(p)

    _bench(VarFwdOp, ReductionCall(call), _flaggems("var_dim", p, correction=c))


@pytest.mark.parametrize("call", manifest_calls(VarMeanFwdOp))
def test_var_mean_bench(call) -> None:
    p = call.params
    c = _correction(p)

    _bench(VarMeanFwdOp, ReductionCall(call), _flaggems("var_mean", p, correction=c))
