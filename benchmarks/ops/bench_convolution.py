"""Benchmarks for the convolution op family (1d/2d/3d, with and without bias).

Workload shapes, channel counts, kernel sizes, strides, paddings, and dtypes
are loaded from the ops manifest (``src/tileops/manifest/convolution.yaml``);
FLOP/byte counts come from each op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

One ``test_*_bench`` per op, so the validator's L4 AST check can tie each
``load_workloads("<OpName>")`` call to its manifest entry.
"""

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import (
    Conv1dBiasFwdOp,
    Conv1dFwdOp,
    Conv2dBiasFwdOp,
    Conv2dFwdOp,
    Conv3dBiasFwdOp,
    Conv3dFwdOp,
)

# Bench-local: autotuning is benchmark infrastructure, not a workload property.
_TUNE = True


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

    def as_record(self) -> dict:
        return asdict(self)


def _mark(idx: int):
    """First manifest workload of an op is the smoke case; the rest are full."""
    return pytest.mark.smoke if idx == 0 else pytest.mark.full


def _conv_params(workloads: list[dict], kernel_keys: tuple[str, ...]) -> list:
    """Build one :class:`ConvCase` parameter per manifest workload row and dtype.

    ``kernel_keys`` names the manifest spatial-extent keys in order, e.g.
    ``("kD", "kH", "kW")`` for 3d. Workload entries omitting ``stride`` /
    ``padding`` fall back to the manifest signature defaults; scalar entries
    are broadcast across the spatial dims the way PyTorch broadcasts them.
    """
    n_spatial = len(kernel_keys)

    def _spatial(value) -> tuple[int, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,) * n_spatial

    params = []
    for idx, w in enumerate(workloads):
        input_shape = tuple(w["input_shape"])
        kernel_size = tuple(w[key] for key in kernel_keys)
        stride = _spatial(w.get("stride", 1))
        padding = _spatial(w.get("padding", 0))
        dilation = _spatial(w.get("dilation", 1))
        groups = w.get("groups", 1)
        for dtype_name in w["dtypes"]:
            params.append(pytest.param(
                ConvCase(
                    input_shape, w["C_out"], kernel_size, stride, padding,
                    dilation, groups, getattr(torch, dtype_name),
                ),
                id=f"{w['label']}-{dtype_name}",
                marks=_mark(idx),
            ))
    return params


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
        c_out, c_in // groups, *kernel_size, device="cuda", dtype=dtype,
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
            x, weight, bias=bias,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups,
        )

    return baseline_fn


def _run_conv(
    op,
    bm: ManifestBenchmark,
    torch_fn: Callable,
    case: ConvCase,
    *,
    with_bias: bool,
    static_weight: bool = False,
) -> None:
    """Profile op and its torch baseline on the same inputs, recording both.

    One caller per manifest op, so each keeps a literal
    ``ManifestBenchmark(<op name>, ...)`` the manifest validator matches
    statically.
    """
    inputs = _conv_inputs(
        case.input_shape, case.c_out, case.kernel_size, case.dtype,
        groups=case.groups, with_bias=with_bias,
    )
    baseline = _torch_conv_baseline(
        torch_fn, case.stride, case.padding, case.dilation, case.groups,
    )
    _profile_conv(
        op,
        bm,
        inputs,
        baseline,
        case.as_record(),
        static_weight=static_weight,
    )


def _profile_conv(
    op,
    bm: ManifestBenchmark,
    inputs: tuple[torch.Tensor, ...],
    baseline_fn: Callable,
    params: dict,
    *,
    static_weight: bool = False,
) -> None:
    """Profile op and the torch baseline on the same inputs and record both."""
    if static_weight:
        x, weight, *maybe_bias = inputs
        bias = maybe_bias[0] if maybe_bias else None

        def op_with_static_weight(x_i):
            if bias is None:
                return op(x_i, weight)
            return op(x_i, weight, bias)

        def baseline_with_static_weight(x_i):
            return baseline_fn(x_i, weight, bias)

        result = bm.profile(op_with_static_weight, x)
        BenchmarkReport.record(op, params, result, tag="tileops")

        result_bl = bm.profile(baseline_with_static_weight, x)
        BenchmarkReport.record(op, params, result_bl, tag="torch")
        return

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, params, result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, params, result_bl, tag="torch")


# Conv1d

_CONV1D_OP = "Conv1dFwdOp"
_CONV1D_KERNEL_KEYS = ("kW",)


@pytest.mark.parametrize(
    "case",
    _conv_params(load_workloads(_CONV1D_OP), _CONV1D_KERNEL_KEYS),
)
def test_conv1d_bench(case: ConvCase) -> None:
    op = Conv1dFwdOp(
        stride=case.stride, padding=case.padding,
        dilation=case.dilation, groups=case.groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV1D_OP, op, ConvWorkload(case.input_shape, case.dtype))
    _run_conv(op, bm, F.conv1d, case, with_bias=False, static_weight=True)


_CONV1D_BIAS_OP = "Conv1dBiasFwdOp"


@pytest.mark.parametrize(
    "case",
    _conv_params(load_workloads(_CONV1D_BIAS_OP), _CONV1D_KERNEL_KEYS),
)
def test_conv1d_bias_bench(case: ConvCase) -> None:
    op = Conv1dBiasFwdOp(
        stride=case.stride, padding=case.padding,
        dilation=case.dilation, groups=case.groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV1D_BIAS_OP, op, ConvWorkload(case.input_shape, case.dtype))
    _run_conv(op, bm, F.conv1d, case, with_bias=True, static_weight=True)


# Conv2d

_CONV2D_OP = "Conv2dFwdOp"
_CONV2D_KERNEL_KEYS = ("kH", "kW")


@pytest.mark.parametrize(
    "case",
    _conv_params(load_workloads(_CONV2D_OP), _CONV2D_KERNEL_KEYS),
)
def test_conv2d_bench(case: ConvCase) -> None:
    op = Conv2dFwdOp(
        stride=case.stride, padding=case.padding,
        dilation=case.dilation, groups=case.groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV2D_OP, op, ConvWorkload(case.input_shape, case.dtype))
    _run_conv(op, bm, F.conv2d, case, with_bias=False)


_CONV2D_BIAS_OP = "Conv2dBiasFwdOp"


@pytest.mark.parametrize(
    "case",
    _conv_params(load_workloads(_CONV2D_BIAS_OP), _CONV2D_KERNEL_KEYS),
)
def test_conv2d_bias_bench(case: ConvCase) -> None:
    op = Conv2dBiasFwdOp(
        stride=case.stride, padding=case.padding,
        dilation=case.dilation, groups=case.groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV2D_BIAS_OP, op, ConvWorkload(case.input_shape, case.dtype))
    _run_conv(op, bm, F.conv2d, case, with_bias=True)


# Conv3d

_CONV3D_OP = "Conv3dFwdOp"
_CONV3D_KERNEL_KEYS = ("kD", "kH", "kW")


@pytest.mark.parametrize(
    "case",
    _conv_params(load_workloads(_CONV3D_OP), _CONV3D_KERNEL_KEYS),
)
def test_conv3d_bench(case: ConvCase) -> None:
    op = Conv3dFwdOp(
        stride=case.stride, padding=case.padding,
        dilation=case.dilation, groups=case.groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV3D_OP, op, ConvWorkload(case.input_shape, case.dtype))
    _run_conv(op, bm, F.conv3d, case, with_bias=False)


_CONV3D_BIAS_OP = "Conv3dBiasFwdOp"


@pytest.mark.parametrize(
    "case",
    _conv_params(load_workloads(_CONV3D_BIAS_OP), _CONV3D_KERNEL_KEYS),
)
def test_conv3d_bias_bench(case: ConvCase) -> None:
    op = Conv3dBiasFwdOp(
        stride=case.stride, padding=case.padding,
        dilation=case.dilation, groups=case.groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV3D_BIAS_OP, op, ConvWorkload(case.input_shape, case.dtype))
    _run_conv(op, bm, F.conv3d, case, with_bias=True)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
