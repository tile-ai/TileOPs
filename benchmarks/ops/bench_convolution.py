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

from collections.abc import Callable

import pytest
import torch

from benchmarks import api as bench
from benchmarks._cases.convolution import static_weight as bind_static_weight
from benchmarks.baselines import FLAGGEMS_TAG, TORCH_COMPILE_TAG, compiled_reference, flaggems_op
from tileops._ops._spatial import per_axis_values
from tileops.convolution import Conv1dFwdOp, Conv2dFwdOp, Conv3dFwdOp


def _flaggems_conv_baseline(rank: int, workload) -> Callable:
    """Return flag_gems' convolution of this rank, bound to the workload's parameters.

    Same parameter names as ``F.convNd``, but the spatial extents go in as lists.
    """
    conv_fn = flaggems_op(f"conv{rank}d")
    stride = list(per_axis_values(workload.stride, rank))
    padding = list(per_axis_values(workload.padding, rank))
    dilation = list(per_axis_values(workload.dilation, rank))

    def baseline_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return conv_fn(x, weight, bias, stride, padding, dilation, workload.groups)

    return baseline_fn


def _run_conv(op, case: bench.Case, *, rank: int, static_weight: bool = False) -> None:
    """Profile op against flag_gems and torch on the same inputs, recording all.

    flag_gems takes per-axis padding only, so a row padding by mode drops its tag.
    """
    workload = case.workload
    baseline = case.torch
    baselines = {"torch": baseline, TORCH_COMPILE_TAG: compiled_reference(baseline)}
    if not isinstance(workload.padding, str):
        flaggems = _flaggems_conv_baseline(rank, workload)
        baselines = {FLAGGEMS_TAG: flaggems, **baselines}
    if static_weight:
        baselines = {tag: bind_static_weight(fn, case) for tag, fn in baselines.items()}
    bench.Runner(op, case).compare(baselines)


@pytest.mark.parametrize("case", bench.cases(Conv1dFwdOp), ids=lambda case: case.id)
def test_conv1d_bench(case) -> None:
    op = Conv1dFwdOp(**case.arguments)
    op.request_tune()
    _run_conv(op, case, rank=1, static_weight=True)


@pytest.mark.parametrize("case", bench.cases(Conv2dFwdOp), ids=lambda case: case.id)
def test_conv2d_bench(case) -> None:
    op = Conv2dFwdOp(**case.arguments)
    op.request_tune()
    _run_conv(op, case, rank=2)


@pytest.mark.parametrize("case", bench.cases(Conv3dFwdOp), ids=lambda case: case.id)
def test_conv3d_bench(case) -> None:
    op = Conv3dFwdOp(**case.arguments)
    op.request_tune()
    _run_conv(op, case, rank=3)
