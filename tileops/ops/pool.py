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

    def _kernel_cache_key(
        self,
        kernel_name: str,
        use_spatial_fast_path: bool,
        n: int,
        c_in: int,
        in_dims: Tuple[int, ...],
        dtype: torch.dtype,
        device_index: int | None,
    ) -> tuple:
        return (
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
        key = self._kernel_cache_key(
            kernel_name, use_spatial_fast_path, n, c_in, in_dims, dtype, device_index,
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

    def _kernel_cache_key(
        self,
        kernel_name: str,
        use_spatial_fast_path: bool,
        n: int,
        c_in: int,
        in_dims: Tuple[int, ...],
        dtype: torch.dtype,
        device_index: int | None,
    ) -> tuple:
        # avg_pool1d has no divisor_override; its key never carried one.
        return (
            kernel_name,
            n,
            c_in,
            *in_dims,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            dtype,
            device_index,
            self.tune,
        )

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
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            kernel_map=kernel_map,
            tune=tune,
        )


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

    def _kernel_cache_key(
        self,
        kernel_name: str,
        use_spatial_fast_path: bool,
        n: int,
        c_in: int,
        in_dims: Tuple[int, ...],
        dtype: torch.dtype,
        device_index: int | None,
    ) -> tuple:
        # avg_pool2d keys historically discriminate on "spatial"/"general".
        variant = "spatial" if use_spatial_fast_path else "general"
        return (
            variant,
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



def _max_pool_roofline(op: "_MaxPoolFwdOpBase", *, indices: bool) -> tuple[int, int]:
    """Shared max-pool roofline: flops = out_elems * prod(kernel); bytes in+out."""
    if op._last_roofline_spec is None:
        raise RuntimeError(
            f"{type(op).__name__}.eval_roofline() requires a prior forward() "
            "call to bind input shape and dtype"
        )
    spec = op._last_roofline_spec
    nd = op.ndim
    n, c_in = spec[0], spec[1]
    in_dims = spec[2:2 + nd]
    out_dims = spec[2 + nd:2 + 2 * nd]
    dtype = spec[-1]
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    in_elems = n * c_in
    out_elems = n * c_in
    for size in in_dims:
        in_elems *= size
    for size in out_dims:
        out_elems *= size
    flops = out_elems
    for k in op.kernel_size:
        flops *= k
    bytes_ = (in_elems + out_elems) * elem_bytes
    if indices:
        bytes_ += out_elems * 8
    return flops, bytes_

def _make_max_pool_forward(returns_indices: bool):
    """Build the compile-boundary forward for one max-pool output variant."""
    if returns_indices:
        def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return _pool_fwd_with_indices(input, self._instance_key)
    else:
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return _pool_fwd(input, self._instance_key)
    return forward


class _MaxPoolFwdOpBase(Op):
    """Generic max-pooling forward, parametrized by class attributes.

    Concrete subclasses set ``ndim`` / ``_kernel_slot`` / ``_returns_indices``,
    supply ``default_kernel_map``, and keep ``eval_roofline`` /
    ``_validate_dtypes`` in their own class body so manifest codegen resolves
    them per concrete class.
    """

    ndim: ClassVar[int]
    _kernel_slot: ClassVar[str] = ""
    _returns_indices: ClassVar[bool] = False

    def __init__(
        self,
        kernel_size: int | Tuple[int, ...],
        stride: Optional[int | Tuple[int, ...]] = None,
        padding: int | Tuple[int, ...] = 0,
        dilation: int | Tuple[int, ...] = 1,
        ceil_mode: bool = False,
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
        self.dilation = _normalize_pool_dims("dilation", dilation, nd)
        if not isinstance(ceil_mode, bool):
            raise TypeError("ceil_mode must be a bool")
        self.ceil_mode = ceil_mode
        self.dtype = None
        self.tune = tune
        validate_pool_params(
            ndim=nd,
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

    def _resolve_input(self, input: torch.Tensor) -> tuple:
        nd = self.ndim
        if input.ndim != nd + 2:
            raise ValueError(
                f"{self.__class__.__name__} expects input to be a "
                f"{nd + 2}D {_POOL_LAYOUTS[nd]} tensor"
            )
        n, c_in, *in_dims = input.shape
        if not input.is_cuda:
            raise ValueError("input must be a CUDA tensor")
        self._validate_dtypes(input)
        out_dims = tuple(
            pool_output_dim(
                size,
                self.kernel_size[k],
                self.stride[k],
                self.padding[k],
                self.ceil_mode,
                self.dilation[k],
            )
            for k, size in enumerate(in_dims)
        )
        if any(v <= 0 for v in out_dims):
            raise ValueError(
                f"{self.__class__.__name__} calculated output size must be greater than zero, "
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
        key = (
            n,
            c_in,
            *in_dims,
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
            kernel_kwargs: Dict[str, object] = dict(
                n=n, c_in=c_in, ceil_mode=self.ceil_mode, dtype=dtype, tune=self.tune,
            )
            for k, name in enumerate(_POOL_DIM_NAMES[self.ndim]):
                kernel_kwargs[f"{name}_in"] = in_dims[k]
            for k, name in enumerate(_MAX_POOL_PARAM_SUFFIXES[self.ndim]):
                kernel_kwargs[f"kernel_{name}"] = self.kernel_size[k]
                kernel_kwargs[f"stride_{name}"] = self.stride[k]
                kernel_kwargs[f"pad_{name}"] = self.padding[k]
                kernel_kwargs[f"dilation_{name}"] = self.dilation[k]
            self._kernel_cache[key] = self.kernel_map[self._kernel_slot](**kernel_kwargs)
        return self._kernel_cache[key]

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        nd = self.ndim
        if len(input_shape) != nd + 2:
            raise ValueError(
                f"{self.__class__.__name__} expects input_shape to be "
                f"{nd + 2}D {_POOL_LAYOUTS[nd]}"
            )
        n, c_in, *in_dims = input_shape
        kernel_size = getattr(self, "kernel_size", None)
        stride = getattr(self, "stride", None)
        padding = getattr(self, "padding", None)
        dilation = getattr(self, "dilation", (1,) * nd)
        ceil_mode = getattr(self, "ceil_mode", False)
        if kernel_size is None or stride is None or padding is None:
            zero = (n, c_in) + (0,) * nd
            if self._returns_indices:
                return {"output": zero, "indices": zero}
            return {"output": zero}
        out_dims = tuple(
            pool_output_dim(size, kernel_size[k], stride[k], padding[k], ceil_mode, dilation[k])
            for k, size in enumerate(in_dims)
        )
        full = (n, c_in, *out_dims)
        if self._returns_indices:
            return {"output": full, "indices": full}
        return {"output": full}

    def __init_subclass__(cls, **kwargs) -> None:
        # _returns_indices selects the forward variant at class-definition
        # time so every concrete class carries the exact return annotation
        # its manifest outputs declare (Tensor vs Tuple[Tensor, Tensor]).
        super().__init_subclass__(**kwargs)
        if "forward" not in cls.__dict__:
            cls.forward = _make_max_pool_forward(cls._returns_indices)

    def _eager_forward(self, input: torch.Tensor):
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


class MaxPool1dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=False)."""

    ndim = 1
    _kernel_slot = "max_pool1d_kernel"
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

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
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            kernel_map=kernel_map,
            tune=tune,
        )


    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool1d_kernel": MaxPool1dKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=False)


class MaxPool1dIndicesFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=True)."""

    ndim = 1
    _kernel_slot = "max_pool1d_with_indices_kernel"
    _returns_indices = True
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

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
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            kernel_map=kernel_map,
            tune=tune,
        )


    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool1d_with_indices_kernel": MaxPool1dWithIndicesKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=True)


class MaxPool2dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCHW inputs (return_indices=False)."""

    ndim = 2
    _kernel_slot = "max_pool2d_kernel"
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

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
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            kernel_map=kernel_map,
            tune=tune,
        )


    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool2d_kernel": MaxPool2dKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=False)


class MaxPool2dIndicesFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCHW inputs (return_indices=True)."""

    ndim = 2
    _kernel_slot = "max_pool2d_with_indices_kernel"
    _returns_indices = True
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

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
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            kernel_map=kernel_map,
            tune=tune,
        )


    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool2d_with_indices_kernel": MaxPool2dWithIndicesKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=True)


class MaxPool3dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCDHW inputs (return_indices=False)."""

    ndim = 3
    _kernel_slot = "max_pool3d_kernel"
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

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
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            kernel_map=kernel_map,
            tune=tune,
        )


    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool3d_kernel": MaxPool3dKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=False)


class MaxPool3dIndicesFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCDHW inputs (return_indices=True)."""

    ndim = 3
    _kernel_slot = "max_pool3d_with_indices_kernel"
    _returns_indices = True
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

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
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            kernel_map=kernel_map,
            tune=tune,
        )


    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool3d_with_indices_kernel": MaxPool3dWithIndicesKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=True)


class AvgPool3dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCDHW inputs."""

    ndim = 3
    # Keep a concrete binding so manifest dtype codegen honors the shared validator.
    _validate_dtypes = _validate_pool_input_dtypes

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
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            kernel_map=kernel_map,
            tune=tune,
        )


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
