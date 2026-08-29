"""Benchmarks for the convolution op family (1d/2d/3d).

Workload shapes, channel counts, kernel sizes, strides, paddings, and dtypes
are loaded from the ops manifest (``src/tileops/manifest/convolution.yaml``);
FLOP/byte counts come from each op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

One ``test_*_bench`` per op, so every op this file is declared the benchmark
of records a row of its own. A row passes bias
when it declares ``bias_shape``.

Every row is timed against flag_gems' Triton convolutions, ``F.convNd`` eager
(which is cuDNN, so cuDNN takes no tag of its own), and that reference through
inductor.
"""

import functools
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import (
    FLAGGEMS_TAG,
    TORCH_COMPILE_TAG,
    assert_matches_reference,
    compiled_reference,
    flaggems_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops import Conv1dFwdOp, Conv2dFwdOp, Conv3dFwdOp

# Bench-local: autotuning is benchmark infrastructure, not a workload property.
_TUNE = True

# flag_gems' aten-level convolutions, by spatial rank.
_FLAGGEMS_CONV = {1: "conv1d", 2: "conv2d", 3: "conv3d"}

# Triton and cuDNN sum the same products in a different order, so agreement is
# relative to the output scale: over the 13 manifest workloads on an H200, flag_gems
# lands within 0.25 of cuDNN where |ref| reaches 564.
_BASELINE_RTOL = 2e-2
_BASELINE_ATOL = 2e-2


class ConvWorkload:
    """Minimal shape/dtype descriptor for the convolution family.

    Holds ``shape`` and ``dtype`` so :class:`ManifestBenchmark` can call
    ``op.eval_roofline()`` after ``forward()`` has bound the dynamic vars.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


@dataclass(frozen=True)
class ConvCase:
    """One manifest workload row, resolved to concrete convolution arguments."""

    input_shape: tuple[int, ...]
    c_out: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    dtype: torch.dtype
    with_bias: bool

    def as_record(self) -> dict:
        return asdict(self)


def _conv_args(w: dict, dtype: torch.dtype, kernel_keys: tuple[str, ...]) -> tuple:
    """One :class:`ConvCase` for this row and dtype.

    ``kernel_keys`` names the manifest spatial-extent keys in order, e.g.
    ``("kD", "kH", "kW")`` for 3d. Rows omitting ``stride`` / ``padding`` fall
    back to the manifest signature defaults; a scalar entry is broadcast across
    the spatial dims the way PyTorch broadcasts it.
    """
    n_spatial = len(kernel_keys)

    def spatial(value) -> tuple[int, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,) * n_spatial

    return (
        ConvCase(
            tuple(w["input_shape"]),
            w["C_out"],
            tuple(w[key] for key in kernel_keys),
            spatial(w.get("stride", 1)),
            spatial(w.get("padding", 0)),
            spatial(w.get("dilation", 1)),
            w.get("groups", 1),
            dtype,
            "bias_shape" in w,
        ),
    )


def _conv_inputs(
    input_shape: tuple[int, ...],
    c_out: int,
    kernel_size: tuple[int, ...],
    dtype: torch.dtype,
    *,
    groups: int,
    with_bias: bool,
) -> tuple[torch.Tensor, ...]:
    """Generate ``(input, weight[, bias])`` for a convolution workload."""
    c_in = input_shape[1]
    x = torch.randn(input_shape, device="cuda", dtype=dtype).contiguous()
    weight = torch.randn(
        c_out,
        c_in // groups,
        *kernel_size,
        device="cuda",
        dtype=dtype,
    ).contiguous()
    if not with_bias:
        return x, weight
    bias = torch.zeros(c_out, device="cuda", dtype=dtype).contiguous()
    return x, weight, bias


def _torch_conv_baseline(
    conv_fn: Callable,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
) -> Callable:
    """Return a ``torch.nn.functional`` conv baseline bound to these params."""

    def baseline_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return conv_fn(
            x,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    return baseline_fn


def _flaggems_conv_baseline(
    rank: int,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
) -> Callable:
    """Return flag_gems' convolution of this rank, bound to these params.

    Same parameter names as ``F.convNd``, but the spatial extents go in as lists.
    """
    conv_fn = flaggems_op(_FLAGGEMS_CONV[rank])

    def baseline_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return conv_fn(
            x,
            weight,
            bias,
            list(stride),
            list(padding),
            list(dilation),
            groups,
        )

    return baseline_fn


def _run_conv(
    op,
    bm: ManifestBenchmark,
    torch_fn: Callable,
    case: ConvCase,
    *,
    rank: int,
    with_bias: bool,
    static_weight: bool = False,
) -> None:
    """Profile op against flag_gems and torch on the same inputs, recording all.

    One caller per manifest op, so every op the manifest declares this file for
    records a row of its own.
    """
    inputs = _conv_inputs(
        case.input_shape,
        case.c_out,
        case.kernel_size,
        case.dtype,
        groups=case.groups,
        with_bias=with_bias,
    )
    baseline = _torch_conv_baseline(
        torch_fn,
        case.stride,
        case.padding,
        case.dilation,
        case.groups,
    )
    flaggems = _flaggems_conv_baseline(
        rank,
        case.stride,
        case.padding,
        case.dilation,
        case.groups,
    )
    assert_matches_reference(
        flaggems,
        baseline,
        *inputs,
        rtol=_BASELINE_RTOL,
        atol=_BASELINE_ATOL,
    )
    _profile_conv(
        op,
        bm,
        inputs,
        {
            FLAGGEMS_TAG: flaggems,
            "torch": baseline,
            TORCH_COMPILE_TAG: compiled_reference(baseline),
        },
        case.as_record(),
        static_weight=static_weight,
    )


def _profile_conv(
    op,
    bm: ManifestBenchmark,
    inputs: tuple[torch.Tensor, ...],
    baselines: dict[str, Callable],
    params: dict,
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
            record_as=op,
            params=params,
        )
        return

    bm.compare({"tileops": op, **baselines}, *inputs, record_as=op, params=params)


# Conv1d

_CONV1D_OP = "Conv1dFwdOp"
_CONV1D_KERNEL_KEYS = ("kW",)


@pytest.mark.parametrize(
    "case",
    workload_params(
        load_workloads(_CONV1D_OP),
        functools.partial(_conv_args, kernel_keys=_CONV1D_KERNEL_KEYS),
        smoke_first=True,
    ),
)
def test_conv1d_bench(case: ConvCase) -> None:
    op = Conv1dFwdOp(
        stride=case.stride,
        padding=case.padding,
        dilation=case.dilation,
        groups=case.groups,
        tune=_TUNE,
    )
    bm = ManifestBenchmark(op, ConvWorkload(case.input_shape, case.dtype))
    _run_conv(op, bm, F.conv1d, case, rank=1, with_bias=case.with_bias, static_weight=True)


# Conv2d

_CONV2D_OP = "Conv2dFwdOp"
_CONV2D_KERNEL_KEYS = ("kH", "kW")


@pytest.mark.parametrize(
    "case",
    workload_params(
        load_workloads(_CONV2D_OP),
        functools.partial(_conv_args, kernel_keys=_CONV2D_KERNEL_KEYS),
        smoke_first=True,
    ),
)
def test_conv2d_bench(case: ConvCase) -> None:
    op = Conv2dFwdOp(
        stride=case.stride,
        padding=case.padding,
        dilation=case.dilation,
        groups=case.groups,
        tune=_TUNE,
    )
    bm = ManifestBenchmark(op, ConvWorkload(case.input_shape, case.dtype))
    _run_conv(op, bm, F.conv2d, case, rank=2, with_bias=case.with_bias)


# Conv3d

_CONV3D_OP = "Conv3dFwdOp"
_CONV3D_KERNEL_KEYS = ("kD", "kH", "kW")


@pytest.mark.parametrize(
    "case",
    workload_params(
        load_workloads(_CONV3D_OP),
        functools.partial(_conv_args, kernel_keys=_CONV3D_KERNEL_KEYS),
        smoke_first=True,
    ),
)
def test_conv3d_bench(case: ConvCase) -> None:
    op = Conv3dFwdOp(
        stride=case.stride,
        padding=case.padding,
        dilation=case.dilation,
        groups=case.groups,
        tune=_TUNE,
    )
    bm = ManifestBenchmark(op, ConvWorkload(case.input_shape, case.dtype))
    _run_conv(op, bm, F.conv3d, case, rank=3, with_bias=case.with_bias)
