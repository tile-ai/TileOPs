"""Benchmarks for the convolution op family (1d/2d/3d, with and without bias).

Workload shapes, channel counts, kernel sizes, strides, paddings, and dtypes
are loaded from the ops manifest (``tileops/manifest/convolution.yaml``);
FLOP/byte counts come from each op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

One ``test_*_bench`` per op, so the validator's L4 AST check can tie each
``load_workloads("<OpName>")`` call to its manifest entry.
"""

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


class _ConvWorkload:
    """Minimal :class:`ShapeDtypeWorkload` for the convolution family.

    Holds ``shape`` and ``dtype`` so :class:`ManifestBenchmark` can call
    ``op.eval_roofline()`` after ``forward()`` has bound the dynamic vars.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


def _mark(idx: int):
    """First manifest workload of an op is the smoke case; the rest are full."""
    return pytest.mark.smoke if idx == 0 else pytest.mark.full


def _conv_params(workloads: list[dict], kernel_keys: tuple[str, ...]) -> list:
    """Build ``(input_shape, c_out, kernel_size, stride, padding, dtype)`` params.

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
                input_shape, w["C_out"], kernel_size, stride, padding,
                dilation, groups, getattr(torch, dtype_name),
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


def _profile_conv(
    op,
    bm: ManifestBenchmark,
    inputs: tuple[torch.Tensor, ...],
    baseline_fn: Callable,
    params: dict,
) -> None:
    """Profile op and the torch baseline on the same inputs and record both."""
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, params, result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, params, result_bl, tag="torch")


# Conv1d

_CONV1D_OP = "Conv1dFwdOp"
_CONV1D_KERNEL_KEYS = ("kW",)


@pytest.mark.parametrize(
    "input_shape, c_out, kernel_size, stride, padding, dilation, groups, dtype",
    _conv_params(load_workloads(_CONV1D_OP), _CONV1D_KERNEL_KEYS),
)
def test_conv1d_bench(
    input_shape: tuple[int, ...],
    c_out: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    dtype: torch.dtype,
) -> None:
    inputs = _conv_inputs(input_shape, c_out, kernel_size, dtype,
                          groups=groups, with_bias=False)
    op = Conv1dFwdOp(
        stride=stride, padding=padding,
        dilation=dilation, groups=groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV1D_OP, op, _ConvWorkload(input_shape, dtype))
    _profile_conv(
        op, bm, inputs, _torch_conv_baseline(F.conv1d, stride, padding, dilation, groups),
        {"input_shape": input_shape, "c_out": c_out, "kernel_size": kernel_size,
         "stride": stride, "padding": padding, "dilation": dilation,
         "groups": groups, "dtype": dtype},
    )


_CONV1D_BIAS_OP = "Conv1dBiasFwdOp"


@pytest.mark.parametrize(
    "input_shape, c_out, kernel_size, stride, padding, dilation, groups, dtype",
    _conv_params(load_workloads(_CONV1D_BIAS_OP), _CONV1D_KERNEL_KEYS),
)
def test_conv1d_bias_bench(
    input_shape: tuple[int, ...],
    c_out: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    dtype: torch.dtype,
) -> None:
    inputs = _conv_inputs(input_shape, c_out, kernel_size, dtype,
                          groups=groups, with_bias=True)
    op = Conv1dBiasFwdOp(
        stride=stride, padding=padding,
        dilation=dilation, groups=groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV1D_BIAS_OP, op, _ConvWorkload(input_shape, dtype))
    _profile_conv(
        op, bm, inputs, _torch_conv_baseline(F.conv1d, stride, padding, dilation, groups),
        {"input_shape": input_shape, "c_out": c_out, "kernel_size": kernel_size,
         "stride": stride, "padding": padding, "dilation": dilation,
         "groups": groups, "dtype": dtype},
    )


# Conv2d

_CONV2D_OP = "Conv2dFwdOp"
_CONV2D_KERNEL_KEYS = ("kH", "kW")


@pytest.mark.parametrize(
    "input_shape, c_out, kernel_size, stride, padding, dilation, groups, dtype",
    _conv_params(load_workloads(_CONV2D_OP), _CONV2D_KERNEL_KEYS),
)
def test_conv2d_bench(
    input_shape: tuple[int, ...],
    c_out: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    dtype: torch.dtype,
) -> None:
    inputs = _conv_inputs(input_shape, c_out, kernel_size, dtype,
                          groups=groups, with_bias=False)
    op = Conv2dFwdOp(
        stride=stride, padding=padding,
        dilation=dilation, groups=groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV2D_OP, op, _ConvWorkload(input_shape, dtype))
    _profile_conv(
        op, bm, inputs, _torch_conv_baseline(F.conv2d, stride, padding, dilation, groups),
        {"input_shape": input_shape, "c_out": c_out, "kernel_size": kernel_size,
         "stride": stride, "padding": padding, "dilation": dilation,
         "groups": groups, "dtype": dtype},
    )


_CONV2D_BIAS_OP = "Conv2dBiasFwdOp"


@pytest.mark.parametrize(
    "input_shape, c_out, kernel_size, stride, padding, dilation, groups, dtype",
    _conv_params(load_workloads(_CONV2D_BIAS_OP), _CONV2D_KERNEL_KEYS),
)
def test_conv2d_bias_bench(
    input_shape: tuple[int, ...],
    c_out: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    dtype: torch.dtype,
) -> None:
    inputs = _conv_inputs(input_shape, c_out, kernel_size, dtype,
                          groups=groups, with_bias=True)
    op = Conv2dBiasFwdOp(
        stride=stride, padding=padding,
        dilation=dilation, groups=groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV2D_BIAS_OP, op, _ConvWorkload(input_shape, dtype))
    _profile_conv(
        op, bm, inputs, _torch_conv_baseline(F.conv2d, stride, padding, dilation, groups),
        {"input_shape": input_shape, "c_out": c_out, "kernel_size": kernel_size,
         "stride": stride, "padding": padding, "dilation": dilation,
         "groups": groups, "dtype": dtype},
    )


# Conv3d

_CONV3D_OP = "Conv3dFwdOp"
_CONV3D_KERNEL_KEYS = ("kD", "kH", "kW")


@pytest.mark.parametrize(
    "input_shape, c_out, kernel_size, stride, padding, dilation, groups, dtype",
    _conv_params(load_workloads(_CONV3D_OP), _CONV3D_KERNEL_KEYS),
)
def test_conv3d_bench(
    input_shape: tuple[int, ...],
    c_out: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    dtype: torch.dtype,
) -> None:
    inputs = _conv_inputs(input_shape, c_out, kernel_size, dtype,
                          groups=groups, with_bias=False)
    op = Conv3dFwdOp(
        stride=stride, padding=padding,
        dilation=dilation, groups=groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV3D_OP, op, _ConvWorkload(input_shape, dtype))
    _profile_conv(
        op, bm, inputs, _torch_conv_baseline(F.conv3d, stride, padding, dilation, groups),
        {"input_shape": input_shape, "c_out": c_out, "kernel_size": kernel_size,
         "stride": stride, "padding": padding, "dilation": dilation,
         "groups": groups, "dtype": dtype},
    )


_CONV3D_BIAS_OP = "Conv3dBiasFwdOp"


@pytest.mark.parametrize(
    "input_shape, c_out, kernel_size, stride, padding, dilation, groups, dtype",
    _conv_params(load_workloads(_CONV3D_BIAS_OP), _CONV3D_KERNEL_KEYS),
)
def test_conv3d_bias_bench(
    input_shape: tuple[int, ...],
    c_out: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    dtype: torch.dtype,
) -> None:
    inputs = _conv_inputs(input_shape, c_out, kernel_size, dtype,
                          groups=groups, with_bias=True)
    op = Conv3dBiasFwdOp(
        stride=stride, padding=padding,
        dilation=dilation, groups=groups, tune=_TUNE,
    )
    bm = ManifestBenchmark(_CONV3D_BIAS_OP, op, _ConvWorkload(input_shape, dtype))
    _profile_conv(
        op, bm, inputs, _torch_conv_baseline(F.conv3d, stride, padding, dilation, groups),
        {"input_shape": input_shape, "c_out": c_out, "kernel_size": kernel_size,
         "stride": stride, "padding": padding, "dilation": dilation,
         "groups": groups, "dtype": dtype},
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
