"""Pooling benchmarks.

Workloads are loaded from ``tileops/manifest/pool.yaml``. The 2D cases model
vision-backbone downsampling patterns such as ResNet/ConvNeXt feature stages.
The 3D cases model video CNN spatiotemporal pooling patterns such as
I3D/SlowFast-style feature stages.
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.kernels.pool.common import pool_output_dim
from tileops.manifest import load_workloads
from tileops.ops import (
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

_AVG_POOL1D_OP_NAME = "AvgPool1dFwdOp"
_AVG_POOL2D_OP_NAME = "AvgPool2dFwdOp"
_AVG_POOL3D_OP_NAME = "AvgPool3dFwdOp"
_MAX_POOL1D_OP_NAME = "MaxPool1dFwdOp"
_MAX_POOL1D_INDICES_OP_NAME = "MaxPool1dIndicesFwdOp"
_MAX_POOL2D_OP_NAME = "MaxPool2dFwdOp"
_MAX_POOL2D_INDICES_OP_NAME = "MaxPool2dIndicesFwdOp"
_MAX_POOL3D_OP_NAME = "MaxPool3dFwdOp"
_MAX_POOL3D_INDICES_OP_NAME = "MaxPool3dIndicesFwdOp"


class AvgPool1dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_size: int,
        stride: Optional[int],
        padding: int,
        ceil_mode: bool,
        count_include_pad: bool,
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n, self.c_in, self.l_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
        )


class AvgPool1dBenchmark(BenchmarkBase[AvgPool1dBenchCase]):
    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test: AvgPool1dBenchCase, op: AvgPool1dFwdOp) -> None:
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


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
    test = AvgPool1dBenchCase(
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
    bm = AvgPool1dBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("avg_pool1d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("avg_pool1d", locals(), result_bl, tag="torch-ref")


class AvgPool2dBenchCase:
    def __init__(
        self,
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
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n, self.c_in, self.h_in, self.w_in, device="cuda", dtype=self.dtype
        ).contiguous()
        return (x,)

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


class AvgPool2dBenchmark(BenchmarkBase[AvgPool2dBenchCase]):
    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test: AvgPool2dBenchCase, op: AvgPool2dFwdOp) -> None:
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


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
    test = AvgPool2dBenchCase(
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
    bm = AvgPool2dBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("avg_pool2d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("avg_pool2d", locals(), result_bl, tag="torch-ref")


class AvgPool3dBenchCase:
    def __init__(
        self,
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
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.d_in,
            self.h_in,
            self.w_in,
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        return (x,)

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


class AvgPool3dBenchmark(BenchmarkBase[AvgPool3dBenchCase]):
    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(self, test: AvgPool3dBenchCase, op: AvgPool3dFwdOp) -> None:
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


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
    test = AvgPool3dBenchCase(
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
    bm = AvgPool3dBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("avg_pool3d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("avg_pool3d", locals(), result_bl, tag="torch-ref")


class MaxPool2dBenchCase:
    def __init__(
        self,
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
        return_indices: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.return_indices = return_indices

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n, self.c_in, self.h_in, self.w_in, device="cuda", dtype=self.dtype
        ).contiguous()
        return (x,)

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


class MaxPool2dBenchmark(BenchmarkBase[MaxPool2dBenchCase]):
    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(
        self,
        test: MaxPool2dBenchCase,
        op: MaxPool2dFwdOp | MaxPool2dIndicesFwdOp,
    ) -> None:
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


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
    test = MaxPool2dBenchCase(
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
    bm = MaxPool2dBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("max_pool2d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("max_pool2d", locals(), result_bl, tag="torch-ref")


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
    test = MaxPool2dBenchCase(
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
    bm = MaxPool2dBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("max_pool2d_indices", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("max_pool2d_indices", locals(), result_bl, tag="torch-ref")


class MaxPool1dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_size: tuple[int],
        stride: Optional[tuple[int]],
        padding: tuple[int],
        dilation: tuple[int],
        ceil_mode: bool,
        dtype: torch.dtype,
        return_indices: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.return_indices = return_indices

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n, self.c_in, self.l_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

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


class MaxPool1dBenchmark(BenchmarkBase[MaxPool1dBenchCase]):
    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(
        self,
        test: MaxPool1dBenchCase,
        op: MaxPool1dFwdOp | MaxPool1dIndicesFwdOp,
    ) -> None:
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


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
    test = MaxPool1dBenchCase(
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
    bm = MaxPool1dBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("max_pool1d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("max_pool1d", locals(), result_bl, tag="torch-ref")


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
    test = MaxPool1dBenchCase(
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
    bm = MaxPool1dBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("max_pool1d_indices", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("max_pool1d_indices", locals(), result_bl, tag="torch-ref")


class MaxPool3dBenchCase:
    def __init__(
        self,
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
        return_indices: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.return_indices = return_indices

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.d_in,
            self.h_in,
            self.w_in,
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        return (x,)

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


class MaxPool3dBenchmark(BenchmarkBase[MaxPool3dBenchCase]):
    _roofline_cache: Optional[tuple[float, float]] = None

    def __init__(
        self,
        test: MaxPool3dBenchCase,
        op: MaxPool3dFwdOp | MaxPool3dIndicesFwdOp,
    ) -> None:
        super().__init__(test)
        self._op = op

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            self._roofline_cache = self._op.eval_roofline()
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


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
    test = MaxPool3dBenchCase(
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
    bm = MaxPool3dBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("max_pool3d", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("max_pool3d", locals(), result_bl, tag="torch-ref")


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
    test = MaxPool3dBenchCase(
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
    bm = MaxPool3dBenchmark(test, op)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("max_pool3d_indices", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("max_pool3d_indices", locals(), result_bl, tag="torch-ref")
