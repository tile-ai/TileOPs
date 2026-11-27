"""Benchmarks for the 8 basic reduce ops.

Every row is timed against torch eager, the same reference through inductor, and
flag_gems' Triton reduction where one exists. amin has none: flag_gems 5.0.2
exposes amax but no amin, and ``min_dim`` also produces indices. A row passing ``dtype``
has no flag_gems tag either: its reductions take no output dtype.

The flag_gems callables cast nothing — like aten, its reductions accumulate in fp32
and return the storage dtype — so each tag is one kernel.
"""

from collections.abc import Callable

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    compiled_reference,
    flaggems_dims,
    flaggems_op,
)
from tileops._ops.reduction.reduce import (
    AmaxFwdOp,
    AminFwdOp,
    MeanFwdOp,
    ProdFwdOp,
    StdFwdOp,
    SumFwdOp,
    VarFwdOp,
    VarMeanFwdOp,
)


def _bench(op_cls: type, case: bench.Case, flaggems_fn: Callable | None = None) -> None:
    """Check flag_gems against the reference, then time it and the op with torch."""
    baseline_fn = case.torch
    op = op_cls(**case.arguments)
    implementations = {}
    if flaggems_fn is not None:
        implementations[FLAGGEMS_TAG] = flaggems_fn
    implementations["torch"] = baseline_fn
    implementations[TORCH_COMPILE_TAG] = compiled_reference(baseline_fn)
    bench.Runner(op, case).compare(implementations)


def _flaggems(name: str, params: dict, *extra, **kwargs) -> Callable | None:
    """flag_gems' reduction over the row's axes, or ``None`` for a row passing ``dtype``."""
    if params.get("dtype"):
        return None
    fn = flaggems_op(name)
    dim = params["dim"]
    return lambda x: fn(x, flaggems_dims(dim), *extra, keepdim=params["keepdim"], **kwargs)


@pytest.mark.parametrize("case", bench.cases(SumFwdOp))
def test_sum_bench(case) -> None:
    _bench(SumFwdOp, case, _flaggems("sum_dim", case.params))


@pytest.mark.parametrize("case", bench.cases(MeanFwdOp))
def test_mean_bench(case) -> None:
    _bench(MeanFwdOp, case, _flaggems("mean_dim", case.params))


@pytest.mark.parametrize("case", bench.cases(AmaxFwdOp))
def test_amax_bench(case) -> None:
    _bench(AmaxFwdOp, case, _flaggems("amax", case.params))


@pytest.mark.parametrize("case", bench.cases(AminFwdOp))
def test_amin_bench(case) -> None:
    _bench(AminFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(ProdFwdOp))
def test_prod_bench(case) -> None:
    p = case.params

    flaggems_fn = None
    if not p.get("dtype"):
        flaggems_prod = flaggems_op("prod_dim")

        def flaggems_fn(x):
            return flaggems_prod(x, p["dim"], p["keepdim"])

    _bench(ProdFwdOp, case, flaggems_fn)


def _correction(params: dict):
    return 1 if params["correction"] is None else params["correction"]


@pytest.mark.parametrize("case", bench.cases(StdFwdOp))
def test_std_bench(case) -> None:
    _bench(StdFwdOp, case, _flaggems("std", case.params, correction=_correction(case.params)))


@pytest.mark.parametrize("case", bench.cases(VarFwdOp))
def test_var_bench(case) -> None:
    _bench(VarFwdOp, case, _flaggems("var_dim", case.params, correction=_correction(case.params)))


@pytest.mark.parametrize("case", bench.cases(VarMeanFwdOp))
def test_var_mean_bench(case) -> None:
    _bench(
        VarMeanFwdOp, case, _flaggems("var_mean", case.params, correction=_correction(case.params))
    )
