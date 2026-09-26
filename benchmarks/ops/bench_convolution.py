"""Benchmarks for the convolution op family (1d/2d/3d).

Workload shapes, channel counts, kernel sizes, strides, paddings, and dtypes
are loaded from the ops manifest (``src/tileops/manifest/convolution.yaml``);
FLOP/byte counts come from each op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

One ``test_*_bench`` per op, over the op's manifest calls; a row passes bias when its
``some`` names it.

Every row is timed against flag_gems' Triton convolutions, ``F.convNd`` eager
(which is cuDNN, so cuDNN takes no tag of its own), and that reference through
inductor.
"""

from typing import Callable, Optional

import pytest
import torch

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import Conv1dFwdOp, Conv2dFwdOp, Conv3dFwdOp
from workloads.convolution import Conv1dWorkload, Conv2dWorkload, Conv3dWorkload

# Bench-local: autotuning is benchmark infrastructure, not a workload property.
_TUNE = True

# flag_gems' aten-level convolutions, by spatial rank.
_FLAGGEMS_CONV = {1: "conv1d", 2: "conv2d", 3: "conv3d"}

# Triton and cuDNN sum the same products in a different order, so agreement is
# relative to the output scale: across the manifest's workloads flag_gems lands
# within a fraction of cuDNN where the reference reaches the hundreds.
_BASELINE_RTOL = 2e-2
_BASELINE_ATOL = 2e-2


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


def _run_conv(
    op,
    bm: ManifestBenchmark,
    workload,
    *,
    rank: int,
    static_weight: bool = False,
) -> None:
    """Profile op against flag_gems and torch on the same inputs, recording all.

    flag_gems takes per-axis padding only, so a row padding by mode drops its tag.
    """
    inputs = workload.gen_inputs()
    baseline = workload.ref_program
    baselines = {"torch": baseline, TORCH_COMPILE_TAG: compiled_reference(baseline)}
    if not isinstance(workload.padding, str):
        flaggems = _flaggems_conv_baseline(rank, workload)
        assert_matches_reference(
            flaggems,
            baseline,
            *inputs,
            rtol=_BASELINE_RTOL,
            atol=_BASELINE_ATOL,
        )
        baselines = {FLAGGEMS_TAG: flaggems, **baselines}
    _profile_conv(op, bm, inputs, baselines, static_weight=static_weight)


def _profile_conv(
    op,
    bm: ManifestBenchmark,
    inputs: tuple[torch.Tensor, ...],
    baselines: dict[str, Callable],
    *,
    static_weight: bool = False,
) -> None:
    """Profile op and every baseline on the same inputs and record them all."""
    if static_weight:
        x, weight, *maybe_bias = inputs
        bias = maybe_bias[0] if maybe_bias else None

        def op_with_static_weight(x_i):
            if bias is None:
                return op(x_i, weight)
            return op(x_i, weight, bias)

        def bind_static_weight(fn: Callable) -> Callable:
            def run(x_i):
                return fn(x_i, weight, bias)

            return run

        bm.compare(
            {
                "tileops": op_with_static_weight,
                **{tag: bind_static_weight(fn) for tag, fn in baselines.items()},
            },
            x,
        )
        return

    bm.compare({"tileops": op, **baselines}, *inputs)


@pytest.mark.parametrize("call", manifest_calls(Conv1dFwdOp))
def test_conv1d_bench(call) -> None:
    workload = Conv1dWorkload.from_call(call)
    op = Conv1dFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)
    _run_conv(op, bm, workload, rank=1, static_weight=True)


@pytest.mark.parametrize("call", manifest_calls(Conv2dFwdOp))
def test_conv2d_bench(call) -> None:
    workload = Conv2dWorkload.from_call(call)
    op = Conv2dFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)
    _run_conv(op, bm, workload, rank=2)


@pytest.mark.parametrize("call", manifest_calls(Conv3dFwdOp))
def test_conv3d_bench(call) -> None:
    workload = Conv3dWorkload.from_call(call)
    op = Conv3dFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)
    _run_conv(op, bm, workload, rank=3)
