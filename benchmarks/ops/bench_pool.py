"""Pooling benchmarks.

Workloads are loaded from ``src/tileops/manifest/pool.yaml``. The 2D cases model
vision-backbone downsampling patterns such as ResNet/ConvNeXt feature stages.
The 3D cases model video CNN spatiotemporal pooling patterns such as
I3D/SlowFast-style feature stages.
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.kernels.pool.common import pool_output_dim
from tileops.manifest import load_workloads
from tileops.ops import (
    AdaptiveAvgPool2dFwdOp,
    AdaptiveMaxPool2dFwdOp,
    AdaptiveMaxPool2dIndicesFwdOp,
    AvgPool1dFwdOp,
    AvgPool2dFwdOp,
    AvgPool3dFwdOp,
    MaxPool1dFwdOp,
    MaxPool1dIndicesFwdOp,
    MaxPool2dFwdOp,
    MaxPool2dIndicesFwdOp,
    MaxPool3dFwdOp,
    MaxPool3dIndicesFwdOp,
)
from workloads.pool import (
    AdaptivePool2dWorkload,
    AvgPool1dBenchCase,
    AvgPool2dBenchCase,
    AvgPool3dBenchCase,
    MaxPool1dBenchCase,
    MaxPool2dBenchCase,
    MaxPool3dBenchCase,
)

_ADAPTIVE_AVG_POOL2D_OP_NAME = "AdaptiveAvgPool2dFwdOp"
_ADAPTIVE_MAX_POOL2D_OP_NAME = "AdaptiveMaxPool2dFwdOp"
_ADAPTIVE_MAX_POOL2D_INDICES_OP_NAME = "AdaptiveMaxPool2dIndicesFwdOp"
_AVG_POOL1D_OP_NAME = "AvgPool1dFwdOp"
_AVG_POOL2D_OP_NAME = "AvgPool2dFwdOp"
_AVG_POOL3D_OP_NAME = "AvgPool3dFwdOp"
_MAX_POOL1D_OP_NAME = "MaxPool1dFwdOp"
_MAX_POOL1D_INDICES_OP_NAME = "MaxPool1dIndicesFwdOp"
_MAX_POOL2D_OP_NAME = "MaxPool2dFwdOp"
_MAX_POOL2D_INDICES_OP_NAME = "MaxPool2dIndicesFwdOp"
_MAX_POOL3D_OP_NAME = "MaxPool3dFwdOp"
_MAX_POOL3D_INDICES_OP_NAME = "MaxPool3dIndicesFwdOp"


def _avg_pool1d_bench_params() -> list:
    params = []
    for workload in load_workloads(_AVG_POOL1D_OP_NAME):
        n, c_in, l_in = workload["input_shape"]
        kernel_size = workload["kernel_size"]
        stride = workload.get("stride")
        padding = workload.get("padding", 0)
        ceil_mode = workload.get("ceil_mode", False)
        count_include_pad = workload.get("count_include_pad", True)
        label = workload.get("label", f"{n}x{c_in}x{l_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    l_in,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


class AvgPool1dBenchmarkWorkload(AvgPool1dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
        )


class AvgPool2dBenchmarkWorkload(AvgPool2dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


class AvgPool3dBenchmarkWorkload(AvgPool3dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


class MaxPool2dBenchmarkWorkload(MaxPool2dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class MaxPool1dBenchmarkWorkload(MaxPool1dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return F.max_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class MaxPool3dBenchmarkWorkload(MaxPool3dBenchCase):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return F.max_pool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


@pytest.mark.parametrize(
    "n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype, tune",
    _avg_pool1d_bench_params(),
)
def test_avg_pool1d_bench(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: int,
    stride: Optional[int],
    padding: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool1dBenchmarkWorkload(
        n, c_in, l_in, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype
    )
    inputs = test.gen_inputs()

    op = AvgPool1dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        tune=tune,
    )
    bm = ManifestBenchmark(_AVG_POOL1D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


def _avg_pool2d_bench_params() -> list:
    params = []
    for workload in load_workloads(_AVG_POOL2D_OP_NAME):
        n, c_in, h_in, w_in = workload["input_shape"]
        kernel_size = tuple(workload["kernel_size"])
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride)
        padding = tuple(workload.get("padding", (0, 0)))
        ceil_mode = workload.get("ceil_mode", False)
        count_include_pad = workload.get("count_include_pad", True)
        divisor_override = workload.get("divisor_override")
        label = workload.get("label", f"{n}x{c_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    h_in,
                    w_in,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
    _avg_pool2d_bench_params(),
)
def test_avg_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool2dBenchmarkWorkload(
        n,
        c_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    inputs = test.gen_inputs()

    op = AvgPool2dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        tune=tune,
    )
    bm = ManifestBenchmark(_AVG_POOL2D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


def _avg_pool3d_bench_params() -> list:
    params = []
    for workload in load_workloads(_AVG_POOL3D_OP_NAME):
        n, c_in, d_in, h_in, w_in = workload["input_shape"]
        kernel_size = tuple(workload["kernel_size"])
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride)
        padding = tuple(workload.get("padding", (0, 0, 0)))
        ceil_mode = workload.get("ceil_mode", False)
        count_include_pad = workload.get("count_include_pad", True)
        divisor_override = workload.get("divisor_override")
        label = workload.get("label", f"{n}x{c_in}x{d_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    d_in,
                    h_in,
                    w_in,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, dtype, tune",
    _avg_pool3d_bench_params(),
)
def test_avg_pool3d_bench(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int, int],
    stride: Optional[tuple[int, int, int]],
    padding: tuple[int, int, int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AvgPool3dBenchmarkWorkload(
        n,
        c_in,
        d_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dtype,
    )
    inputs = test.gen_inputs()

    op = AvgPool3dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        tune=tune,
    )
    bm = ManifestBenchmark(_AVG_POOL3D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


def _max_pool2d_bench_params_from_workloads(workloads: list[dict]) -> list:
    params = []
    for workload in workloads:
        n, c_in, h_in, w_in = workload["input_shape"]
        kernel_size = tuple(workload["kernel_size"])
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride)
        padding = tuple(workload.get("padding", (0, 0)))
        dilation = tuple(workload.get("dilation", (1, 1)))
        ceil_mode = workload.get("ceil_mode", False)
        label = workload.get("label", f"{n}x{c_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    h_in,
                    w_in,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


def _max_pool2d_bench_params() -> list:
    return _max_pool2d_bench_params_from_workloads(load_workloads(_MAX_POOL2D_OP_NAME))


def _max_pool2d_indices_bench_params() -> list:
    return _max_pool2d_bench_params_from_workloads(load_workloads(_MAX_POOL2D_INDICES_OP_NAME))


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool2d_bench_params(),
)
def test_max_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool2dBenchmarkWorkload(
        n,
        c_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
    )
    inputs = test.gen_inputs()

    op = MaxPool2dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL2D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool2d_indices_bench_params(),
)
def test_max_pool2d_indices_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int],
    stride: Optional[tuple[int, int]],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool2dBenchmarkWorkload(
        n,
        c_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        return_indices=True,
    )
    inputs = test.gen_inputs()

    op = MaxPool2dIndicesFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL2D_INDICES_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


def _max_pool1d_bench_params_from_workloads(workloads: list[dict]) -> list:
    params = []
    for workload in workloads:
        n, c_in, l_in = workload["input_shape"]
        kernel_size = workload["kernel_size"]
        kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else (kernel_size,)
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride) if isinstance(stride, list) else (stride,)
        padding = workload.get("padding", 0)
        padding = tuple(padding) if isinstance(padding, list) else (padding,)
        dilation = workload.get("dilation", 1)
        dilation = tuple(dilation) if isinstance(dilation, list) else (dilation,)
        ceil_mode = workload.get("ceil_mode", False)
        label = workload.get("label", f"{n}x{c_in}x{l_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    l_in,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


def _max_pool1d_bench_params() -> list:
    return _max_pool1d_bench_params_from_workloads(load_workloads(_MAX_POOL1D_OP_NAME))


def _max_pool1d_indices_bench_params() -> list:
    return _max_pool1d_bench_params_from_workloads(load_workloads(_MAX_POOL1D_INDICES_OP_NAME))


@pytest.mark.parametrize(
    "n, c_in, l_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool1d_bench_params(),
)
def test_max_pool1d_bench(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: tuple[int],
    stride: Optional[tuple[int]],
    padding: tuple[int],
    dilation: tuple[int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool1dBenchmarkWorkload(
        n,
        c_in,
        l_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
    )
    inputs = test.gen_inputs()

    op = MaxPool1dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL1D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


@pytest.mark.parametrize(
    "n, c_in, l_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool1d_indices_bench_params(),
)
def test_max_pool1d_indices_bench(
    n: int,
    c_in: int,
    l_in: int,
    kernel_size: tuple[int],
    stride: Optional[tuple[int]],
    padding: tuple[int],
    dilation: tuple[int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool1dBenchmarkWorkload(
        n,
        c_in,
        l_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        return_indices=True,
    )
    inputs = test.gen_inputs()

    op = MaxPool1dIndicesFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL1D_INDICES_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


def _max_pool3d_bench_params_from_workloads(workloads: list[dict]) -> list:
    params = []
    for workload in workloads:
        n, c_in, d_in, h_in, w_in = workload["input_shape"]
        kernel_size = workload["kernel_size"]
        kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else (kernel_size,) * 3
        stride = workload.get("stride")
        if stride is not None:
            stride = tuple(stride) if isinstance(stride, list) else (stride,) * 3
        padding = workload.get("padding", 0)
        padding = tuple(padding) if isinstance(padding, list) else (padding,) * 3
        dilation = workload.get("dilation", 1)
        dilation = tuple(dilation) if isinstance(dilation, list) else (dilation,) * 3
        ceil_mode = workload.get("ceil_mode", False)
        label = workload.get("label", f"{n}x{c_in}x{d_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    d_in,
                    h_in,
                    w_in,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


def _max_pool3d_bench_params() -> list:
    return _max_pool3d_bench_params_from_workloads(load_workloads(_MAX_POOL3D_OP_NAME))


def _max_pool3d_indices_bench_params() -> list:
    return _max_pool3d_bench_params_from_workloads(load_workloads(_MAX_POOL3D_INDICES_OP_NAME))


@pytest.mark.parametrize(
    "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool3d_bench_params(),
)
def test_max_pool3d_bench(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int, int],
    stride: Optional[tuple[int, int, int]],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool3dBenchmarkWorkload(
        n,
        c_in,
        d_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
    )
    inputs = test.gen_inputs()

    op = MaxPool3dFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL3D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


@pytest.mark.parametrize(
    "n, c_in, d_in, h_in, w_in, kernel_size, stride, padding, dilation, ceil_mode, dtype, tune",
    _max_pool3d_indices_bench_params(),
)
def test_max_pool3d_indices_bench(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    kernel_size: tuple[int, int, int],
    stride: Optional[tuple[int, int, int]],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    ceil_mode: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MaxPool3dBenchmarkWorkload(
        n,
        c_in,
        d_in,
        h_in,
        w_in,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        dtype,
        return_indices=True,
    )
    inputs = test.gen_inputs()

    op = MaxPool3dIndicesFwdOp(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        tune=tune,
    )
    bm = ManifestBenchmark(_MAX_POOL3D_INDICES_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


class AdaptiveAvgPool2dBenchmarkWorkload(AdaptivePool2dWorkload):
    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # torch rejects a scalar None here; (None, None) means the same.
        size = (None, None) if self.output_size is None else self.output_size
        return F.adaptive_avg_pool2d(x, size)


class AdaptiveMaxPool2dBenchmarkWorkload(AdaptivePool2dWorkload):
    def __init__(self, *args, return_indices: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.return_indices = return_indices

    def ref_program(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # torch rejects a scalar None here; (None, None) means the same.
        size = (None, None) if self.output_size is None else self.output_size
        return F.adaptive_max_pool2d(
            x, size, return_indices=self.return_indices
        )


def _adaptive_pool2d_bench_params_from_workloads(workloads) -> list:
    params = []
    for workload in workloads:
        n, c_in, h_in, w_in = workload["input_shape"]
        output_size = tuple(workload["output_size"])
        label = workload.get("label", f"{n}x{c_in}x{h_in}x{w_in}")
        for dtype_str in workload["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    n,
                    c_in,
                    h_in,
                    w_in,
                    output_size,
                    dtype,
                    True,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, output_size, dtype, tune",
    _adaptive_pool2d_bench_params_from_workloads(load_workloads(_ADAPTIVE_AVG_POOL2D_OP_NAME)),
)
def test_adaptive_avg_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: tuple[int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AdaptiveAvgPool2dBenchmarkWorkload(n, c_in, h_in, w_in, output_size, dtype)
    inputs = test.gen_inputs()

    op = AdaptiveAvgPool2dFwdOp(output_size=output_size, tune=tune)
    bm = ManifestBenchmark(_ADAPTIVE_AVG_POOL2D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, output_size, dtype, tune",
    _adaptive_pool2d_bench_params_from_workloads(load_workloads(_ADAPTIVE_MAX_POOL2D_OP_NAME)),
)
def test_adaptive_max_pool2d_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: tuple[int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AdaptiveMaxPool2dBenchmarkWorkload(n, c_in, h_in, w_in, output_size, dtype)
    inputs = test.gen_inputs()

    op = AdaptiveMaxPool2dFwdOp(output_size=output_size, tune=tune)
    bm = ManifestBenchmark(_ADAPTIVE_MAX_POOL2D_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


@pytest.mark.parametrize(
    "n, c_in, h_in, w_in, output_size, dtype, tune",
    _adaptive_pool2d_bench_params_from_workloads(load_workloads(_ADAPTIVE_MAX_POOL2D_INDICES_OP_NAME)),
)
def test_adaptive_max_pool2d_indices_bench(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    output_size: tuple[int, int],
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = AdaptiveMaxPool2dBenchmarkWorkload(
        n, c_in, h_in, w_in, output_size, dtype, return_indices=True
    )
    inputs = test.gen_inputs()

    op = AdaptiveMaxPool2dIndicesFwdOp(output_size=output_size, tune=tune)
    bm = ManifestBenchmark(_ADAPTIVE_MAX_POOL2D_INDICES_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")
