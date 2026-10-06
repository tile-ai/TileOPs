"""Benchmarks for the convolution op family (1d/2d/3d).

Workload shapes, channel counts, kernel sizes, strides, paddings, and dtypes
are loaded from the ops manifest (``src/tileops/manifest/spec/convolution.yaml``);
FLOP/byte counts come from each op's ``eval_roofline()``.

One ``test_*_bench`` per op, over the op's manifest calls; a row passes bias when its
``some`` names it.

Every row is timed against flag_gems' Triton convolutions, ``F.convNd`` eager
(which is cuDNN, so cuDNN takes no tag of its own), and that reference through
inductor.
"""

from typing import Callable, Optional

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import FLAGGEMS_TAG, TORCH_COMPILE_TAG, compiled_reference, flaggems_op
from tileops.ops import Conv1dFwdOp, Conv2dFwdOp, Conv3dFwdOp

# Bench-local: autotuning is benchmark infrastructure, not a workload property.
_TUNE = True

# flag_gems' aten-level convolutions, by spatial rank.
_FLAGGEMS_CONV = {1: "conv1d", 2: "conv2d", 3: "conv3d"}


def _spatial(value, rank: int) -> tuple:
    """A per-axis parameter as one value per spatial axis, as flag_gems takes it."""
    return tuple(value) if isinstance(value, (list, tuple)) else (value,) * rank


def _flaggems_conv_baseline(rank: int, workload) -> Callable:
    """Return flag_gems' convolution of this rank, bound to the workload's parameters.

    Same parameter names as ``F.convNd``, but the spatial extents go in as lists.
    """
    conv_fn = flaggems_op(_FLAGGEMS_CONV[rank])
    stride = list(_spatial(workload.stride, rank))
    padding = list(_spatial(workload.padding, rank))
    dilation = list(_spatial(workload.dilation, rank))

    def baseline_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return conv_fn(x, weight, bias, stride, padding, dilation, workload.groups)

    return baseline_fn


def _run_conv(op, case: bench.Case, *, rank: int, static_weight: bool = False) -> None:
    """Profile op against flag_gems and torch on the same inputs, recording all.

    flag_gems takes per-axis padding only, so a row padding by mode drops its tag.
    """
    workload = case.workload
    baseline = case.reference
    baselines = {"torch": baseline, TORCH_COMPILE_TAG: compiled_reference(baseline)}
    if not isinstance(workload.padding, str):
        flaggems = _flaggems_conv_baseline(rank, workload)
        baselines = {FLAGGEMS_TAG: flaggems, **baselines}
    if static_weight:
        baselines = {tag: _bind_static_weight(fn, case) for tag, fn in baselines.items()}
    bench.Runner(op, case).compare(baselines)


def _bind_static_weight(fn: Callable, case: bench.Case) -> bench.Implementation:
    """*fn* called on ``x`` alone, the weight and bias bound across calls."""
    x, weight, bias = case.inputs

    def run(x_i):
        return fn(x_i, weight, bias)

    return bench.Implementation(run=run, args=(x,))


@pytest.mark.parametrize("case", bench.cases(Conv1dFwdOp), ids=lambda case: case.id)
def test_conv1d_bench(case) -> None:
    op = Conv1dFwdOp(**case.arguments, tune=_TUNE)
    _run_conv(op, case, rank=1, static_weight=True)


@pytest.mark.parametrize("case", bench.cases(Conv2dFwdOp), ids=lambda case: case.id)
def test_conv2d_bench(case) -> None:
    op = Conv2dFwdOp(**case.arguments, tune=_TUNE)
    _run_conv(op, case, rank=2)


@pytest.mark.parametrize("case", bench.cases(Conv3dFwdOp), ids=lambda case: case.id)
def test_conv3d_bench(case) -> None:
    op = Conv3dFwdOp(**case.arguments, tune=_TUNE)
    _run_conv(op, case, rank=3)
