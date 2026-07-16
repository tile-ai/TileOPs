from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool import (
    AvgPool1dKernel,
    AvgPool1dSpatialKernel,
    AvgPool2dKernel,
    AvgPool2dSpatialKernel,
    AvgPool3dKernel,
    AvgPool3dSpatialKernel,
    MaxPool2dKernel,
    MaxPool2dWithIndicesKernel,
)
from tileops.kernels.pool.common import (
    _normalize_pool_dims,
    pool_output_dim,
    validate_pool_params,
)

from .op_base import Op

__all__ = [
    "AvgPool1dFwdOp",
    "AvgPool2dFwdOp",
    "AvgPool3dFwdOp",
    "MaxPool2dFwdOp",
    "MaxPool2dIndicesFwdOp",
]


def _device_index(tensor: torch.Tensor) -> int | None:
    return tensor.device.index


class AvgPool1dFwdOp(Op):
    """Average pooling over PyTorch-compatible NCL inputs."""

    def __init__(
        self,
        kernel_size: int | Tuple[int],
        stride: Optional[int | Tuple[int]] = None,
        padding: int | Tuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = None
        self.c_in = None
        self.l_in = None
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, 1)[0]
        self.stride = (
            (self.kernel_size,) if stride is None else _normalize_pool_dims("stride", stride, 1)
        )[0]
        self.padding = _normalize_pool_dims("padding", padding, 1)[0]
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = None
        self.tune = tune
        validate_pool_params(
            ndim=1,
            kernel_size=(self.kernel_size,),
            stride=(self.stride,),
            padding=(self.padding,),
        )
        self.dispatch_kernel(kernel_map)
        if (
            "avg_pool1d_kernel" not in self.kernel_map
            and "avg_pool1d_spatial_kernel" not in self.kernel_map
        ):
            raise NotImplementedError(
                "AvgPool1dFwdOp requires 'avg_pool1d_kernel' or "
                "'avg_pool1d_spatial_kernel' in kernel_map"
            )
        self._has_explicit_generic_kernel = (
            kernel_map is not None and "avg_pool1d_kernel" in kernel_map
        )
        self._has_explicit_spatial_kernel = (
            kernel_map is not None and "avg_pool1d_spatial_kernel" in kernel_map
        )
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._last_roofline_spec: Optional[tuple] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "avg_pool1d_kernel": AvgPool1dKernel,
            "avg_pool1d_spatial_kernel": AvgPool1dSpatialKernel,
        }

    def _resolve_input_1d(
        self,
        input: torch.Tensor,
    ) -> tuple[int, int, int, int, torch.dtype]:
        if input.ndim != 3:
            raise ValueError("AvgPool1dFwdOp expects input to be a 3D NCL tensor")
        n, c_in, l_in = input.shape
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        self._validate_dtypes(input)
        out_l = pool_output_dim(l_in, self.kernel_size, self.stride, self.padding, self.ceil_mode)
        if out_l <= 0:
            raise ValueError(
                f"AvgPool1dFwdOp calculated output size must be greater than zero, got ({out_l},)"
            )
        return n, c_in, l_in, out_l, input.dtype

    def _get_kernel_1d(
        self,
        n: int,
        c_in: int,
        l_in: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        use_spatial_fast_path = (
            not self.ceil_mode
            and self.count_include_pad
            and "avg_pool1d_spatial_kernel" in self.kernel_map
            and (not self._has_explicit_generic_kernel or self._has_explicit_spatial_kernel)
        )
        kernel_name = "avg_pool1d_spatial_kernel" if use_spatial_fast_path else "avg_pool1d_kernel"
        key = (
            kernel_name,
            n,
            c_in,
            l_in,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            dtype,
            device_index,
            self.tune,
        )
        if key not in self._kernel_cache:
            kernel_kwargs = dict(
                n=n,
                c_in=c_in,
                l_in=l_in,
                kernel_l=self.kernel_size,
                stride_l=self.stride,
                pad_l=self.padding,
                dtype=dtype,
                tune=self.tune,
            )
            if use_spatial_fast_path:
                self._kernel_cache[key] = self.kernel_map[kernel_name](**kernel_kwargs)
            else:
                self._kernel_cache[key] = self.kernel_map[kernel_name](
                    **kernel_kwargs,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
        return self._kernel_cache[key]

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 3:
            raise ValueError("AvgPool1dFwdOp expects input_shape to be 3D NCL")
        n, c_in, l_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0)}
        out_l = pool_output_dim(l_in, kernel_size, stride, padding, ceil_mode)
        return {"output": (n, c_in, out_l)}

    def _validate_dtypes(self, input: torch.Tensor) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, c_in, l_in, out_l, dtype = self._resolve_input_1d(input)
        input = input.contiguous()
        kernel = self._get_kernel_1d(n, c_in, l_in, dtype, _device_index(input))
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.out_l = out_l
        self.dtype = dtype
        self._last_roofline_spec = (n, c_in, l_in, out_l, dtype)
        return kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError("AvgPool1dFwdOp.eval_roofline() requires a prior forward() call")
        n, c_in, l_in, out_l, dtype = self._last_roofline_spec
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        flops = n * c_in * out_l * self.kernel_size
        bytes_ = (n * c_in * l_in + n * c_in * out_l) * elem_bytes
        return flops, bytes_


class AvgPool2dFwdOp(Op):
    """Average pooling over PyTorch-compatible NCHW inputs."""

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = None
        self.c_in = None
        self.h_in = None
        self.w_in = None
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, 2)
        self.stride = (
            self.kernel_size if stride is None else _normalize_pool_dims("stride", stride, 2)
        )
        self.padding = _normalize_pool_dims("padding", padding, 2)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = None
        self.tune = tune
        validate_pool_params(
            ndim=2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            divisor_override=divisor_override,
        )
        self.dispatch_kernel(kernel_map)
        if (
            "avg_pool2d_kernel" not in self.kernel_map
            and "avg_pool2d_spatial_kernel" not in self.kernel_map
        ):
            raise NotImplementedError(
                "AvgPool2dFwdOp requires 'avg_pool2d_kernel' or "
                "'avg_pool2d_spatial_kernel' in kernel_map"
            )
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._last_roofline_spec: Optional[tuple] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "avg_pool2d_kernel": AvgPool2dKernel,
            "avg_pool2d_spatial_kernel": AvgPool2dSpatialKernel,
        }

    def _resolve_input_2d(
        self,
        input: torch.Tensor,
    ) -> tuple[int, int, int, int, int, int, torch.dtype]:
        if input.ndim != 4:
            raise ValueError("AvgPool2dFwdOp expects input to be a 4D NCHW tensor")
        n, c_in, h_in, w_in = input.shape
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        self._validate_dtypes(input)
        out_h = pool_output_dim(
            h_in, self.kernel_size[0], self.stride[0], self.padding[0], self.ceil_mode
        )
        out_w = pool_output_dim(
            w_in, self.kernel_size[1], self.stride[1], self.padding[1], self.ceil_mode
        )
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "AvgPool2dFwdOp calculated output size must be greater than zero, "
                f"got ({out_h}, {out_w})"
            )
        return n, c_in, h_in, w_in, out_h, out_w, input.dtype

    def _get_kernel_2d(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        use_spatial_fast_path = (
            not self.ceil_mode
            and self.count_include_pad
            and self.divisor_override is None
            and "avg_pool2d_spatial_kernel" in self.kernel_map
        )
        variant = "spatial" if use_spatial_fast_path else "general"
        key = (
            variant,
            n,
            c_in,
            h_in,
            w_in,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
            dtype,
            device_index,
            self.tune,
        )
        if key not in self._kernel_cache:
            kernel_kwargs = dict(
                n=n,
                c_in=c_in,
                h_in=h_in,
                w_in=w_in,
                kernel_h=self.kernel_size[0],
                kernel_w=self.kernel_size[1],
                stride_h=self.stride[0],
                stride_w=self.stride[1],
                pad_h=self.padding[0],
                pad_w=self.padding[1],
                dtype=dtype,
                tune=self.tune,
            )
            if use_spatial_fast_path:
                self._kernel_cache[key] = self.kernel_map["avg_pool2d_spatial_kernel"](
                    **kernel_kwargs
                )
            else:
                self._kernel_cache[key] = self.kernel_map["avg_pool2d_kernel"](
                    **kernel_kwargs,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                    divisor_override=self.divisor_override,
                )
        return self._kernel_cache[key]

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 4:
            raise ValueError("AvgPool2dFwdOp expects input_shape to be 4D NCHW")
        n, c_in, h_in, w_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0, 0)}
        out_h = pool_output_dim(h_in, kernel_size[0], stride[0], padding[0], ceil_mode)
        out_w = pool_output_dim(w_in, kernel_size[1], stride[1], padding[1], ceil_mode)
        return {"output": (n, c_in, out_h, out_w)}

    def _validate_dtypes(self, input: torch.Tensor) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, c_in, h_in, w_in, out_h, out_w, dtype = self._resolve_input_2d(input)
        input = input.contiguous()
        kernel = self._get_kernel_2d(n, c_in, h_in, w_in, dtype, _device_index(input))
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self._last_roofline_spec = (n, c_in, h_in, w_in, out_h, out_w, dtype)
        return kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "AvgPool2dFwdOp.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        n, c_in, h_in, w_in, out_h, out_w, dtype = self._last_roofline_spec
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        flops = n * c_in * out_h * out_w * self.kernel_size[0] * self.kernel_size[1]
        bytes_ = (n * c_in * h_in * w_in + n * c_in * out_h * out_w) * elem_bytes
        return flops, bytes_


class _MaxPool2dFwdOpBase(Op):
    """Shared implementation for MaxPool2dFwdOp and MaxPool2dIndicesFwdOp."""

    _kernel_slot: ClassVar[str] = ""

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        ceil_mode: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = None
        self.c_in = None
        self.h_in = None
        self.w_in = None
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, 2)
        self.stride = (
            self.kernel_size if stride is None else _normalize_pool_dims("stride", stride, 2)
        )
        self.padding = _normalize_pool_dims("padding", padding, 2)
        self.dilation = _normalize_pool_dims("dilation", dilation, 2)
        if not isinstance(ceil_mode, bool):
            raise TypeError("ceil_mode must be a bool")
        self.ceil_mode = ceil_mode
        self.dtype = None
        self.tune = tune
        validate_pool_params(
            ndim=2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.dispatch_kernel(kernel_map)
        if self._kernel_slot not in self.kernel_map:
            raise NotImplementedError(
                f"{self.__class__.__name__} requires {self._kernel_slot!r} in kernel_map"
            )
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._last_roofline_spec: Optional[tuple] = None

    def _resolve_input_2d(
        self,
        input: torch.Tensor,
    ) -> tuple[int, int, int, int, int, int, torch.dtype]:
        if input.ndim != 4:
            raise ValueError(f"{self.__class__.__name__} expects input to be a 4D NCHW tensor")
        n, c_in, h_in, w_in = input.shape
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        self._validate_dtypes(input)
        out_h = pool_output_dim(
            h_in,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
            self.ceil_mode,
            self.dilation[0],
        )
        out_w = pool_output_dim(
            w_in,
            self.kernel_size[1],
            self.stride[1],
            self.padding[1],
            self.ceil_mode,
            self.dilation[1],
        )
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"{self.__class__.__name__} calculated output size must be greater than zero, "
                f"got ({out_h}, {out_w})"
            )
        return n, c_in, h_in, w_in, out_h, out_w, input.dtype

    def _get_kernel_2d(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            n,
            c_in,
            h_in,
            w_in,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            dtype,
            device_index,
            self.tune,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map[self._kernel_slot](
                n=n,
                c_in=c_in,
                h_in=h_in,
                w_in=w_in,
                kernel_h=self.kernel_size[0],
                kernel_w=self.kernel_size[1],
                stride_h=self.stride[0],
                stride_w=self.stride[1],
                pad_h=self.padding[0],
                pad_w=self.padding[1],
                dilation_h=self.dilation[0],
                dilation_w=self.dilation[1],
                ceil_mode=self.ceil_mode,
                dtype=dtype,
                tune=self.tune,
            )
        return self._kernel_cache[key]


class MaxPool2dFwdOp(_MaxPool2dFwdOpBase):
    """Max pooling over PyTorch-compatible NCHW inputs (return_indices=False)."""

    _kernel_slot = "max_pool2d_kernel"

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool2d_kernel": MaxPool2dKernel,
        }

    def _validate_dtypes(self, input: torch.Tensor) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
            )

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 4:
            raise ValueError("MaxPool2dFwdOp expects input_shape to be 4D NCHW")
        n, c_in, h_in, w_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        dilation = getattr(self, "dilation", (1, 1))
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0, 0)}
        out_h = pool_output_dim(h_in, kernel_size[0], stride[0], padding[0], ceil_mode, dilation[0])
        out_w = pool_output_dim(w_in, kernel_size[1], stride[1], padding[1], ceil_mode, dilation[1])
        return {"output": (n, c_in, out_h, out_w)}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, c_in, h_in, w_in, out_h, out_w, dtype = self._resolve_input_2d(input)
        input = input.contiguous()
        kernel = self._get_kernel_2d(n, c_in, h_in, w_in, dtype, _device_index(input))
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self._last_roofline_spec = (n, c_in, h_in, w_in, out_h, out_w, dtype)
        return kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "MaxPool2dFwdOp.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        n, c_in, h_in, w_in, out_h, out_w, dtype = self._last_roofline_spec
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        flops = n * c_in * out_h * out_w * self.kernel_size[0] * self.kernel_size[1]
        bytes_ = (n * c_in * h_in * w_in + n * c_in * out_h * out_w) * elem_bytes
        return flops, bytes_


class MaxPool2dIndicesFwdOp(_MaxPool2dFwdOpBase):
    """Max pooling over PyTorch-compatible NCHW inputs (return_indices=True)."""

    _kernel_slot = "max_pool2d_with_indices_kernel"

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool2d_with_indices_kernel": MaxPool2dWithIndicesKernel,
        }

    def _validate_dtypes(self, input: torch.Tensor) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
            )

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 4:
            raise ValueError("MaxPool2dIndicesFwdOp expects input_shape to be 4D NCHW")
        n, c_in, h_in, w_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        dilation = getattr(self, "dilation", (1, 1))
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0, 0), "indices": (n, c_in, 0, 0)}
        out_h = pool_output_dim(h_in, kernel_size[0], stride[0], padding[0], ceil_mode, dilation[0])
        out_w = pool_output_dim(w_in, kernel_size[1], stride[1], padding[1], ceil_mode, dilation[1])
        return {"output": (n, c_in, out_h, out_w), "indices": (n, c_in, out_h, out_w)}

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, c_in, h_in, w_in, out_h, out_w, dtype = self._resolve_input_2d(input)
        input = input.contiguous()
        kernel = self._get_kernel_2d(n, c_in, h_in, w_in, dtype, _device_index(input))
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self._last_roofline_spec = (n, c_in, h_in, w_in, out_h, out_w, dtype)
        return kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "MaxPool2dIndicesFwdOp.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        n, c_in, h_in, w_in, out_h, out_w, dtype = self._last_roofline_spec
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        flops = n * c_in * out_h * out_w * self.kernel_size[0] * self.kernel_size[1]
        bytes_ = (
            n * c_in * h_in * w_in + n * c_in * out_h * out_w
        ) * elem_bytes + n * c_in * out_h * out_w * 8
        return flops, bytes_


class AvgPool3dFwdOp(Op):
    """Average pooling over PyTorch-compatible NCDHW inputs."""

    def __init__(
        self,
        kernel_size: int | Tuple[int, int, int],
        stride: Optional[int | Tuple[int, int, int]] = None,
        padding: int | Tuple[int, int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = None
        self.c_in = None
        self.d_in = None
        self.h_in = None
        self.w_in = None
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, 3)
        self.stride = (
            self.kernel_size if stride is None else _normalize_pool_dims("stride", stride, 3)
        )
        self.padding = _normalize_pool_dims("padding", padding, 3)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = None
        self.tune = tune
        validate_pool_params(
            ndim=3,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            divisor_override=divisor_override,
        )
        self.dispatch_kernel(kernel_map)
        if (
            "avg_pool3d_kernel" not in self.kernel_map
            and "avg_pool3d_spatial_kernel" not in self.kernel_map
        ):
            raise NotImplementedError(
                "AvgPool3dFwdOp requires 'avg_pool3d_kernel' or "
                "'avg_pool3d_spatial_kernel' in kernel_map"
            )
        self._has_explicit_generic_kernel = (
            kernel_map is not None and "avg_pool3d_kernel" in kernel_map
        )
        self._has_explicit_spatial_kernel = (
            kernel_map is not None and "avg_pool3d_spatial_kernel" in kernel_map
        )
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._last_roofline_spec: Optional[tuple] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "avg_pool3d_kernel": AvgPool3dKernel,
            "avg_pool3d_spatial_kernel": AvgPool3dSpatialKernel,
        }

    def _resolve_input_3d(
        self,
        input: torch.Tensor,
    ) -> tuple[int, int, int, int, int, int, int, int, torch.dtype]:
        if input.ndim != 5:
            raise ValueError("AvgPool3dFwdOp expects input to be a 5D NCDHW tensor")
        n, c_in, d_in, h_in, w_in = input.shape
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        self._validate_dtypes(input)
        out_d = pool_output_dim(
            d_in, self.kernel_size[0], self.stride[0], self.padding[0], self.ceil_mode
        )
        out_h = pool_output_dim(
            h_in, self.kernel_size[1], self.stride[1], self.padding[1], self.ceil_mode
        )
        out_w = pool_output_dim(
            w_in, self.kernel_size[2], self.stride[2], self.padding[2], self.ceil_mode
        )
        if out_d <= 0 or out_h <= 0 or out_w <= 0:
            raise ValueError(
                "AvgPool3dFwdOp calculated output size must be greater than zero, "
                f"got ({out_d}, {out_h}, {out_w})"
            )
        return n, c_in, d_in, h_in, w_in, out_d, out_h, out_w, input.dtype

    def _get_kernel_3d(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        use_spatial_fast_path = (
            not self.ceil_mode
            and self.count_include_pad
            and self.divisor_override is None
            and "avg_pool3d_spatial_kernel" in self.kernel_map
            and (not self._has_explicit_generic_kernel or self._has_explicit_spatial_kernel)
        )
        kernel_name = "avg_pool3d_spatial_kernel" if use_spatial_fast_path else "avg_pool3d_kernel"
        key = (
            kernel_name,
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
            dtype,
            device_index,
            self.tune,
        )
        if key not in self._kernel_cache:
            kernel_kwargs = dict(
                n=n,
                c_in=c_in,
                d_in=d_in,
                h_in=h_in,
                w_in=w_in,
                kernel_d=self.kernel_size[0],
                kernel_h=self.kernel_size[1],
                kernel_w=self.kernel_size[2],
                stride_d=self.stride[0],
                stride_h=self.stride[1],
                stride_w=self.stride[2],
                pad_d=self.padding[0],
                pad_h=self.padding[1],
                pad_w=self.padding[2],
                dtype=dtype,
                tune=self.tune,
            )
            if use_spatial_fast_path:
                self._kernel_cache[key] = self.kernel_map[kernel_name](**kernel_kwargs)
            else:
                self._kernel_cache[key] = self.kernel_map[kernel_name](
                    **kernel_kwargs,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                    divisor_override=self.divisor_override,
                )
        return self._kernel_cache[key]

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 5:
            raise ValueError("AvgPool3dFwdOp expects input_shape to be 5D NCDHW")
        n, c_in, d_in, h_in, w_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0, 0, 0)}
        out_d = pool_output_dim(d_in, kernel_size[0], stride[0], padding[0], ceil_mode)
        out_h = pool_output_dim(h_in, kernel_size[1], stride[1], padding[1], ceil_mode)
        out_w = pool_output_dim(w_in, kernel_size[2], stride[2], padding[2], ceil_mode)
        return {"output": (n, c_in, out_d, out_h, out_w)}

    def _validate_dtypes(self, input: torch.Tensor) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, c_in, d_in, h_in, w_in, out_d, out_h, out_w, dtype = self._resolve_input_3d(input)
        input = input.contiguous()
        kernel = self._get_kernel_3d(n, c_in, d_in, h_in, w_in, dtype, _device_index(input))
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_d = out_d
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self._last_roofline_spec = (
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            out_d,
            out_h,
            out_w,
            dtype,
        )
        return kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "AvgPool3dFwdOp.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        (
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            out_d,
            out_h,
            out_w,
            dtype,
        ) = self._last_roofline_spec
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        flops = (
            n
            * c_in
            * out_d
            * out_h
            * out_w
            * self.kernel_size[0]
            * self.kernel_size[1]
            * self.kernel_size[2]
        )
        bytes_ = (n * c_in * d_in * h_in * w_in + n * c_in * out_d * out_h * out_w) * elem_bytes
        return flops, bytes_
