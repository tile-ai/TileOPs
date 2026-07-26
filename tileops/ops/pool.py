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
    MaxPool1dKernel,
    MaxPool1dWithIndicesKernel,
    MaxPool2dKernel,
    MaxPool2dWithIndicesKernel,
    MaxPool3dKernel,
    MaxPool3dWithIndicesKernel,
)
from tileops.kernels.pool.common import (
    _normalize_pool_dims,
    pool_output_dim,
    validate_pool_params,
)

from .compile_boundary import get_instance
from .op_base import Op

__all__ = [
    "AvgPool1dFwdOp",
    "AvgPool2dFwdOp",
    "AvgPool3dFwdOp",
    "MaxPool1dFwdOp",
    "MaxPool1dIndicesFwdOp",
    "MaxPool2dFwdOp",
    "MaxPool2dIndicesFwdOp",
    "MaxPool3dFwdOp",
    "MaxPool3dIndicesFwdOp",
]


def _device_index(tensor: torch.Tensor) -> int | None:
    return tensor.device.index


# Layout token and per-axis name suffixes, indexed by spatial dimensionality.
_POOL_LAYOUTS: Dict[int, str] = {1: "NCL", 2: "NCHW", 3: "NCDHW"}
_POOL_DIM_NAMES: Dict[int, Tuple[str, ...]] = {1: ("l",), 2: ("h", "w"), 3: ("d", "h", "w")}
# Kernel-kwarg suffixes for kernel_size/stride/padding(/dilation).
# Why: the 1d max-pool kernels historically name their pooling axis `w`.
_AVG_POOL_PARAM_SUFFIXES: Dict[int, Tuple[str, ...]] = _POOL_DIM_NAMES
_MAX_POOL_PARAM_SUFFIXES: Dict[int, Tuple[str, ...]] = {
    1: ("w",),
    2: ("h", "w"),
    3: ("d", "h", "w"),
}


def _validate_pool_input_dtypes(self, input: torch.Tensor) -> None:
    """Shared pool-family dtype validator (bound per concrete class)."""
    if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(
            f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
        )


class _AvgPoolFwdOpBase(Op):
    """Generic average-pooling forward, parametrized by class-attribute ``ndim``.

    Concrete subclasses set ``ndim``, supply ``default_kernel_map``, and keep
    ``eval_roofline`` / ``_validate_dtypes`` in their own class body so
    manifest codegen resolves them per concrete class.
    """

    ndim: ClassVar[int]

    def __init__(
        self,
        kernel_size: int | Tuple[int, ...],
        stride: Optional[int | Tuple[int, ...]] = None,
        padding: int | Tuple[int, ...] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        nd = self.ndim
        self.n = None
        self.c_in = None
        for name in _POOL_DIM_NAMES[nd]:
            setattr(self, f"{name}_in", None)
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, nd)
        self.stride = (
            self.kernel_size if stride is None else _normalize_pool_dims("stride", stride, nd)
        )
        self.padding = _normalize_pool_dims("padding", padding, nd)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = None
        self.tune = tune
        validate_pool_params(
            ndim=nd,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            divisor_override=divisor_override,
        )
        self.dispatch_kernel(kernel_map)
        if (
            self._generic_slot not in self.kernel_map
            and self._spatial_slot not in self.kernel_map
        ):
            raise NotImplementedError(
                f"{type(self).__name__} requires {self._generic_slot!r} or "
                f"{self._spatial_slot!r} in kernel_map"
            )
        self._has_explicit_generic_kernel = (
            kernel_map is not None and self._generic_slot in kernel_map
        )
        self._has_explicit_spatial_kernel = (
            kernel_map is not None and self._spatial_slot in kernel_map
        )
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._last_roofline_spec: Optional[tuple] = None

    @property
    def _generic_slot(self) -> str:
        return f"avg_pool{self.ndim}d_kernel"

    @property
    def _spatial_slot(self) -> str:
        return f"avg_pool{self.ndim}d_spatial_kernel"

    def _param_tuples(self) -> tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Return (kernel_size, stride, padding) as ndim-tuples."""
        return self.kernel_size, self.stride, self.padding

    def _use_spatial_fast_path(self) -> bool:
        # Strict 1d/3d policy: an explicit generic-kernel override opts out of
        # the spatial fast path unless the spatial kernel is also explicit.
        # AvgPool2dFwdOp overrides this with its laxer historical policy.
        return (
            not self.ceil_mode
            and self.count_include_pad
            and self.divisor_override is None
            and self._spatial_slot in self.kernel_map
            and (not self._has_explicit_generic_kernel or self._has_explicit_spatial_kernel)
        )

    def _resolve_input(self, input: torch.Tensor) -> tuple:
        nd = self.ndim
        if input.ndim != nd + 2:
            raise ValueError(
                f"{type(self).__name__} expects input to be a "
                f"{nd + 2}D {_POOL_LAYOUTS[nd]} tensor"
            )
        n, c_in, *in_dims = input.shape
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        self._validate_dtypes(input)
        ks, st, pd = self._param_tuples()
        out_dims = tuple(
            pool_output_dim(size, ks[k], st[k], pd[k], self.ceil_mode)
            for k, size in enumerate(in_dims)
        )
        if any(v <= 0 for v in out_dims):
            raise ValueError(
                f"{type(self).__name__} calculated output size must be greater than zero, "
                f"got {out_dims}"
            )
        return (n, c_in, *in_dims, *out_dims, input.dtype)

    def _get_kernel(
        self,
        n: int,
        c_in: int,
        in_dims: Tuple[int, ...],
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        use_spatial_fast_path = self._use_spatial_fast_path()
        kernel_name = self._spatial_slot if use_spatial_fast_path else self._generic_slot
        key = (
            kernel_name,
            n,
            c_in,
            *in_dims,
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
            ks, st, pd = self._param_tuples()
            kernel_kwargs: Dict[str, object] = dict(n=n, c_in=c_in, dtype=dtype, tune=self.tune)
            for k, name in enumerate(_POOL_DIM_NAMES[self.ndim]):
                kernel_kwargs[f"{name}_in"] = in_dims[k]
            for k, name in enumerate(_AVG_POOL_PARAM_SUFFIXES[self.ndim]):
                kernel_kwargs[f"kernel_{name}"] = ks[k]
                kernel_kwargs[f"stride_{name}"] = st[k]
                kernel_kwargs[f"pad_{name}"] = pd[k]
            if use_spatial_fast_path:
                self._kernel_cache[key] = self.kernel_map[kernel_name](**kernel_kwargs)
            else:
                kernel_kwargs["ceil_mode"] = self.ceil_mode
                kernel_kwargs["count_include_pad"] = self.count_include_pad
                if self.ndim > 1:
                    # The 1d generic kernel has no divisor_override parameter.
                    kernel_kwargs["divisor_override"] = self.divisor_override
                self._kernel_cache[key] = self.kernel_map[kernel_name](**kernel_kwargs)
        return self._kernel_cache[key]

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        nd = self.ndim
        if len(input_shape) != nd + 2:
            raise ValueError(
                f"{type(self).__name__} expects input_shape to be "
                f"{nd + 2}D {_POOL_LAYOUTS[nd]}"
            )
        n, c_in, *in_dims = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in) + (0,) * nd}
        ks, st, pd = self._param_tuples()
        out_dims = tuple(
            pool_output_dim(size, ks[k], st[k], pd[k], ceil_mode)
            for k, size in enumerate(in_dims)
        )
        return {"output": (n, c_in, *out_dims)}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _pool_fwd(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        resolved = self._resolve_input(input)
        input = input.contiguous()
        nd = self.ndim
        n, c_in = resolved[0], resolved[1]
        in_dims = resolved[2:2 + nd]
        out_dims = resolved[2 + nd:2 + 2 * nd]
        dtype = resolved[-1]
        kernel = self._get_kernel(n, c_in, in_dims, dtype, _device_index(input))
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        for name, size in zip(_POOL_DIM_NAMES[nd], in_dims, strict=True):
            setattr(self, f"{name}_in", size)
        for name, size in zip(_POOL_DIM_NAMES[nd], out_dims, strict=True):
            setattr(self, f"out_{name}", size)
        self.dtype = dtype
        self._last_roofline_spec = resolved
        return kernel(input)


class AvgPool1dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCL inputs."""

    ndim = 1
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

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
        # No divisor_override: torch.nn.functional.avg_pool1d does not take one.
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            kernel_map=kernel_map,
            tune=tune,
        )
        # avg_pool1d exposes scalar pooling params; unwrap the normalized 1-tuples.
        self.kernel_size = self.kernel_size[0]
        self.stride = self.stride[0]
        self.padding = self.padding[0]

    def _param_tuples(self) -> tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        return (self.kernel_size,), (self.stride,), (self.padding,)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "avg_pool1d_kernel": AvgPool1dKernel,
            "avg_pool1d_spatial_kernel": AvgPool1dSpatialKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError("AvgPool1dFwdOp.eval_roofline() requires a prior forward() call")
        n, c_in, l_in, out_l, dtype = self._last_roofline_spec
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        flops = n * c_in * out_l * self.kernel_size
        bytes_ = (n * c_in * l_in + n * c_in * out_l) * elem_bytes
        return flops, bytes_


class AvgPool2dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCHW inputs."""

    ndim = 2
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "avg_pool2d_kernel": AvgPool2dKernel,
            "avg_pool2d_spatial_kernel": AvgPool2dSpatialKernel,
        }

    def _use_spatial_fast_path(self) -> bool:
        # Laxer historical 2d policy: an explicit generic-kernel override does
        # not opt out of the spatial fast path (asymmetric with 1d/3d).
        return (
            not self.ceil_mode
            and self.count_include_pad
            and self.divisor_override is None
            and self._spatial_slot in self.kernel_map
        )

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


class _MaxPool1dFwdOpBase(Op):
    """Shared implementation for MaxPool1dFwdOp and MaxPool1dIndicesFwdOp."""

    _kernel_slot: ClassVar[str] = ""

    def __init__(
        self,
        kernel_size: int | Tuple[int],
        stride: Optional[int | Tuple[int]] = None,
        padding: int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        ceil_mode: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.n = None
        self.c_in = None
        self.l_in = None
        self.kernel_size = _normalize_pool_dims("kernel_size", kernel_size, 1)
        self.stride = (
            self.kernel_size if stride is None else _normalize_pool_dims("stride", stride, 1)
        )
        self.padding = _normalize_pool_dims("padding", padding, 1)
        self.dilation = _normalize_pool_dims("dilation", dilation, 1)
        if not isinstance(ceil_mode, bool):
            raise TypeError("ceil_mode must be a bool")
        self.ceil_mode = ceil_mode
        self.dtype = None
        self.tune = tune
        validate_pool_params(
            ndim=1,
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

    def _validate_dtypes(self, input: torch.Tensor) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
            )

    def _resolve_input_1d(
        self,
        input: torch.Tensor,
    ) -> tuple[int, int, int, int, torch.dtype]:
        if input.ndim != 3:
            raise ValueError(f"{self.__class__.__name__} expects input to be a 3D NCL tensor")
        n, c_in, l_in = input.shape
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        self._validate_dtypes(input)
        out_l = pool_output_dim(
            l_in,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
            self.ceil_mode,
            self.dilation[0],
        )
        if out_l <= 0:
            raise ValueError(
                f"{self.__class__.__name__} calculated output size must be greater than zero, "
                f"got ({out_l},)"
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
        key = (
            n,
            c_in,
            l_in,
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
                l_in=l_in,
                kernel_w=self.kernel_size[0],
                stride_w=self.stride[0],
                pad_w=self.padding[0],
                dilation_w=self.dilation[0],
                ceil_mode=self.ceil_mode,
                dtype=dtype,
                tune=self.tune,
            )
        return self._kernel_cache[key]


class MaxPool1dFwdOp(_MaxPool1dFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=False)."""

    _kernel_slot = "max_pool1d_kernel"
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _MaxPool1dFwdOpBase._validate_dtypes

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool1d_kernel": MaxPool1dKernel,
        }

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 3:
            raise ValueError("MaxPool1dFwdOp expects input_shape to be 3D NCL")
        n, c_in, l_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        dilation = getattr(self, "dilation", (1,))
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0)}
        out_l = pool_output_dim(l_in, kernel_size[0], stride[0], padding[0], ceil_mode, dilation[0])
        return {"output": (n, c_in, out_l)}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _pool_fwd(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
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
            raise RuntimeError(
                "MaxPool1dFwdOp.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        n, c_in, l_in, out_l, dtype = self._last_roofline_spec
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        flops = n * c_in * out_l * self.kernel_size[0]
        bytes_ = (n * c_in * l_in + n * c_in * out_l) * elem_bytes
        return flops, bytes_


class MaxPool1dIndicesFwdOp(_MaxPool1dFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=True)."""

    _kernel_slot = "max_pool1d_with_indices_kernel"
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _MaxPool1dFwdOpBase._validate_dtypes

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool1d_with_indices_kernel": MaxPool1dWithIndicesKernel,
        }

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 3:
            raise ValueError("MaxPool1dIndicesFwdOp expects input_shape to be 3D NCL")
        n, c_in, l_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        dilation = getattr(self, "dilation", (1,))
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0), "indices": (n, c_in, 0)}
        out_l = pool_output_dim(l_in, kernel_size[0], stride[0], padding[0], ceil_mode, dilation[0])
        return {"output": (n, c_in, out_l), "indices": (n, c_in, out_l)}

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _pool_fwd_with_indices(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            raise RuntimeError(
                "MaxPool1dIndicesFwdOp.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        n, c_in, l_in, out_l, dtype = self._last_roofline_spec
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        flops = n * c_in * out_l * self.kernel_size[0]
        bytes_ = (n * c_in * l_in + n * c_in * out_l) * elem_bytes + n * c_in * out_l * 8
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

    def _validate_dtypes(self, input: torch.Tensor) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
            )

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
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _MaxPool2dFwdOpBase._validate_dtypes

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool2d_kernel": MaxPool2dKernel,
        }

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
        return _pool_fwd(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
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
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _MaxPool2dFwdOpBase._validate_dtypes

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool2d_with_indices_kernel": MaxPool2dWithIndicesKernel,
        }

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
        return _pool_fwd_with_indices(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class _MaxPool3dFwdOpBase(Op):
    """Shared implementation for MaxPool3dFwdOp and MaxPool3dIndicesFwdOp."""

    _kernel_slot: ClassVar[str] = ""

    def __init__(
        self,
        kernel_size: int | Tuple[int, int, int],
        stride: Optional[int | Tuple[int, int, int]] = None,
        padding: int | Tuple[int, int, int] = 0,
        dilation: int | Tuple[int, int, int] = 1,
        ceil_mode: bool = False,
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
        self.dilation = _normalize_pool_dims("dilation", dilation, 3)
        if not isinstance(ceil_mode, bool):
            raise TypeError("ceil_mode must be a bool")
        self.ceil_mode = ceil_mode
        self.dtype = None
        self.tune = tune
        validate_pool_params(
            ndim=3,
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

    def _validate_dtypes(self, input: torch.Tensor) -> None:
        if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}"
            )

    def _resolve_input_3d(
        self,
        input: torch.Tensor,
    ) -> tuple[int, int, int, int, int, int, int, int, torch.dtype]:
        if input.ndim != 5:
            raise ValueError(f"{self.__class__.__name__} expects input to be a 5D NCDHW tensor")
        n, c_in, d_in, h_in, w_in = input.shape
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        self._validate_dtypes(input)
        out_d = pool_output_dim(
            d_in,
            self.kernel_size[0],
            self.stride[0],
            self.padding[0],
            self.ceil_mode,
            self.dilation[0],
        )
        out_h = pool_output_dim(
            h_in,
            self.kernel_size[1],
            self.stride[1],
            self.padding[1],
            self.ceil_mode,
            self.dilation[1],
        )
        out_w = pool_output_dim(
            w_in,
            self.kernel_size[2],
            self.stride[2],
            self.padding[2],
            self.ceil_mode,
            self.dilation[2],
        )
        if out_d <= 0 or out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"{self.__class__.__name__} calculated output size must be greater than zero, "
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
        key = (
            n,
            c_in,
            d_in,
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
                dilation_d=self.dilation[0],
                dilation_h=self.dilation[1],
                dilation_w=self.dilation[2],
                ceil_mode=self.ceil_mode,
                dtype=dtype,
                tune=self.tune,
            )
        return self._kernel_cache[key]


class MaxPool3dFwdOp(_MaxPool3dFwdOpBase):
    """Max pooling over PyTorch-compatible NCDHW inputs (return_indices=False)."""

    _kernel_slot = "max_pool3d_kernel"
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _MaxPool3dFwdOpBase._validate_dtypes

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool3d_kernel": MaxPool3dKernel,
        }

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 5:
            raise ValueError("MaxPool3dFwdOp expects input_shape to be 5D NCDHW")
        n, c_in, d_in, h_in, w_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        dilation = getattr(self, "dilation", (1, 1, 1))
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0, 0, 0)}
        out_d = pool_output_dim(d_in, kernel_size[0], stride[0], padding[0], ceil_mode, dilation[0])
        out_h = pool_output_dim(h_in, kernel_size[1], stride[1], padding[1], ceil_mode, dilation[1])
        out_w = pool_output_dim(w_in, kernel_size[2], stride[2], padding[2], ceil_mode, dilation[2])
        return {"output": (n, c_in, out_d, out_h, out_w)}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _pool_fwd(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
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
        self._last_roofline_spec = (n, c_in, d_in, h_in, w_in, out_d, out_h, out_w, dtype)
        return kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "MaxPool3dFwdOp.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        n, c_in, d_in, h_in, w_in, out_d, out_h, out_w, dtype = self._last_roofline_spec
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


class MaxPool3dIndicesFwdOp(_MaxPool3dFwdOpBase):
    """Max pooling over PyTorch-compatible NCDHW inputs (return_indices=True)."""

    _kernel_slot = "max_pool3d_with_indices_kernel"
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _MaxPool3dFwdOpBase._validate_dtypes

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool3d_with_indices_kernel": MaxPool3dWithIndicesKernel,
        }

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) != 5:
            raise ValueError("MaxPool3dIndicesFwdOp expects input_shape to be 5D NCDHW")
        n, c_in, d_in, h_in, w_in = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        dilation = getattr(self, "dilation", (1, 1, 1))
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            return {"output": (n, c_in, 0, 0, 0), "indices": (n, c_in, 0, 0, 0)}
        out_d = pool_output_dim(d_in, kernel_size[0], stride[0], padding[0], ceil_mode, dilation[0])
        out_h = pool_output_dim(h_in, kernel_size[1], stride[1], padding[1], ceil_mode, dilation[1])
        out_w = pool_output_dim(w_in, kernel_size[2], stride[2], padding[2], ceil_mode, dilation[2])
        return {
            "output": (n, c_in, out_d, out_h, out_w),
            "indices": (n, c_in, out_d, out_h, out_w),
        }

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _pool_fwd_with_indices(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self._last_roofline_spec = (n, c_in, d_in, h_in, w_in, out_d, out_h, out_w, dtype)
        return kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "MaxPool3dIndicesFwdOp.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        n, c_in, d_in, h_in, w_in, out_d, out_h, out_w, dtype = self._last_roofline_spec
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
        bytes_ = (
            n * c_in * d_in * h_in * w_in + n * c_in * out_d * out_h * out_w
        ) * elem_bytes + n * c_in * out_d * out_h * out_w * 8
        return flops, bytes_


class AvgPool3dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCDHW inputs."""

    ndim = 3
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "avg_pool3d_kernel": AvgPool3dKernel,
            "avg_pool3d_spatial_kernel": AvgPool3dSpatialKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "AvgPool3dFwdOp.eval_roofline() requires a prior forward() "
                "call to bind input shape and dtype"
            )
        n, c_in, d_in, h_in, w_in, out_d, out_h, out_w, dtype = self._last_roofline_spec
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


# torch.compile dispatch boundary (see tileops/ops/compile_boundary.py)


@torch.library.custom_op("top::pool_fwd", mutates_args=())
def _pool_fwd(input: torch.Tensor, instance_key: str) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(input)


@_pool_fwd.register_fake
def _pool_fwd_fake(input: torch.Tensor, instance_key: str) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(input.shape))
    return input.new_empty(shapes["output"])


@torch.library.custom_op("top::pool_fwd_with_indices", mutates_args=())
def _pool_fwd_with_indices(
    input: torch.Tensor, instance_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return get_instance(instance_key)._eager_forward(input)


@_pool_fwd_with_indices.register_fake
def _pool_fwd_with_indices_fake(
    input: torch.Tensor, instance_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(input.shape))
    return (
        input.new_empty(shapes["output"]),
        input.new_empty(shapes["indices"], dtype=torch.int64),
    )
