from collections.abc import Sequence
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool import (
    AdaptiveAvgPool2dKernel,
    AdaptiveMaxPool2dKernel,
    AdaptiveMaxPool2dWithIndicesKernel,
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
    MeanPoolingFwdKernel,
)
from tileops.kernels.pool.common import pool_output_dim

from .compile_boundary import get_instance
from .op_base import Op

__all__ = [
    "AdaptiveAvgPool2dFwdOp",
    "AdaptiveMaxPool2dFwdOp",
    "AdaptiveMaxPool2dIndicesFwdOp",
    "AvgPool1dFwdOp",
    "AvgPool2dFwdOp",
    "AvgPool3dFwdOp",
    "MaxPool1dFwdOp",
    "MaxPool1dIndicesFwdOp",
    "MaxPool2dFwdOp",
    "MaxPool2dIndicesFwdOp",
    "MaxPool3dFwdOp",
    "MaxPool3dIndicesFwdOp",
    "MeanPoolingForwardOp",
]

# Normalizing and checking the pooling parameters is the op layer's, and runs for every
# target. ``pool_output_dim`` above stays with the kernels: they compute their own output
# extents with it, and this layer needs the same arithmetic for ``_infer_output_shapes``.


def _normalize_pool_dims(name: str, value: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int or a tuple of {ndim} ints")

    if isinstance(value, int):
        return (value,) * ndim

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an int or a tuple of {ndim} ints")

    if len(value) != ndim:
        raise ValueError(f"{name} must be an int or a tuple of {ndim} ints")

    if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise TypeError(f"{name} must contain only ints")

    return tuple(value)


def validate_pool_params(
    *,
    ndim: int,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...] | None = None,
    divisor_override: int | None = None,
) -> None:
    if len(kernel_size) != ndim or len(stride) != ndim or len(padding) != ndim:
        raise ValueError("kernel_size, stride, and padding must match pooling dimensionality")

    if dilation is None:
        dilation = (1,) * ndim
    if len(dilation) != ndim:
        raise ValueError("dilation must match pooling dimensionality")

    for name, values in (
        ("kernel_size", kernel_size),
        ("stride", stride),
        ("padding", padding),
        ("dilation", dilation),
    ):
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise TypeError(f"{name} must contain only ints")

    if any(v <= 0 for v in kernel_size):
        raise ValueError("kernel_size must be greater than zero")

    if any(v <= 0 for v in stride):
        raise ValueError("stride must be greater than zero")

    if any(v <= 0 for v in dilation):
        raise ValueError("dilation must be greater than zero")

    if any(v < 0 for v in padding):
        raise ValueError("padding must be non-negative")

    for pad, kernel in zip(padding, kernel_size, strict=True):
        if pad > kernel // 2:
            raise ValueError("padding must be at most half of the effective kernel size")

    if divisor_override is not None and (
        not isinstance(divisor_override, int) or isinstance(divisor_override, bool)
    ):
        raise TypeError("divisor_override must be an int or None")

    if divisor_override == 0:
        raise ValueError("divisor_override must not be zero")


class MeanPoolingForwardOp(Op):
    """Chunked mean over the sequence axis of a ``[batch, seq, heads, dim]`` tensor.

    This is not a PyTorch pooling op and has no torch counterpart. The sequence axis
    is cut into chunks of ``chunk_size`` and each chunk is averaged, giving one output
    row per chunk. With ``use_offsets=1`` the chunks follow the ragged sequence
    boundaries ``offsets`` describes instead of a uniform split, and ``indices`` names
    the ``(sequence, chunk-within-sequence)`` pair each output row belongs to.

    A sequence's last chunk may be shorter than ``chunk_size``. It is divided by the
    count it actually holds, so no padding is averaged in. On a uniform split that
    makes the op equal to ``torch.nn.functional.avg_pool1d(kernel_size=chunk_size,
    stride=chunk_size, ceil_mode=True)`` over a view with the sequence axis last.
    Chunk sums accumulate in
    ``accum_dtype`` and are cast back to the input dtype at the boundary, so a
    ``float16`` input with ``accum_dtype=torch.float32`` does not lose the sum to
    rounding.

    Every shape is fixed at construction rather than read off the call, and
    ``chunks_per_batch`` and ``seq_num`` have to agree with what ``offsets`` implies —
    the kernel does not check.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        heads: int,
        dim: int,
        chunk_size: int,
        chunks_per_batch: int,
        seq_num: int,
        use_offsets: int,
        accum_dtype: torch.dtype,
        tune: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        """Build the op for one set of shapes.

        Args:
            batch_size: Manifest ``params.batch_size``, ``int`` — ``x.shape[0]``.
            seq_len: Manifest ``params.seq_len``, ``int`` — ``x.shape[1]``.
            heads: Manifest ``params.heads``, ``int`` — ``x.shape[2]``.
            dim: Manifest ``params.dim``, ``int`` — ``x.shape[3]``. The kernel tiles it by
                a tuned width of at most 128, and a tile that is neither the whole ``dim``
                nor an exact divisor of it has no valid layout, so ``dim`` is either at
                most 128 or a multiple of it.
            chunk_size: Manifest ``params.chunk_size``, ``int``. A warp reduction spans
                the chunk, so it must be a multiple of 32.
            chunks_per_batch: Manifest ``params.chunks_per_batch``, ``int`` — the output's
                chunk axis. For a uniform split that is ``ceil(seq_len / chunk_size)``;
                for a ragged one, the chunk count summed over the sequences.
            seq_num: Manifest ``params.seq_num``, ``int`` — how many sequences ``offsets``
                describes, so ``offsets`` holds ``seq_num + 1`` entries.
            use_offsets: Manifest ``params.use_offsets``, ``int`` — 1 to take the chunk
                bounds from ``offsets`` and ``indices``, 0 for a uniform split.
            accum_dtype: Manifest ``params.accum_dtype``, ``torch.dtype`` — what the chunk
                sum accumulates in.
            tune: Whether to autotune, applied when a kernel is first built.
            target: Backend target to serve this op, or ``None`` to decide from the input
                device.
            kernel_map: Optional kernel override dict.
        """
        params = {k: v for k, v in locals().items() if k not in ("self", "kernel_map", "target")}
        for key, value in params.items():
            setattr(self, key, value)

        self._kernel_params = params
        self.target = target
        self.dispatch_kernel(kernel_map)

    def _infer_output_shapes(
        self,
        x_shape: tuple[int, ...],
        offsets_shape: tuple[int, ...],
        indices_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        # Read off the constructor, not off the inputs: the chunk count is what the
        # caller declared, and for a ragged split the inputs do not determine it.
        return {"output": (self.batch_size, self.chunks_per_batch, self.heads, self.dim)}

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "mean_pooling_fwd_kernel",
            inputs,
            key=dtype,
            build=lambda: self.kernel_map["mean_pooling_fwd_kernel"](
                **self._kernel_params,
                dtype=dtype,
            ),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mean_pooling_fwd_kernel": MeanPoolingFwdKernel}

    def forward(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Average each chunk of ``x``'s sequence axis.

        Args:
            x: Input tensor, ``[batch_size, seq_len, heads, dim]``, dtype ``float16``,
                ``bfloat16`` or ``float32``.
            offsets: Sequence boundaries, ``[seq_num + 1]``, dtype ``int32``. Read only
                when ``use_offsets`` is 1, but required either way.
            indices: One ``(sequence, chunk-within-sequence)`` pair per output chunk,
                ``[chunks_per_batch, 2]``, dtype ``int32``. Read only when
                ``use_offsets`` is 1, but required either way.

        Returns:
            ``output``, ``[batch_size, chunks_per_batch, heads, dim]``, dtype as ``x``.

        Raises:
            ValueError: An input's dtype is outside what the manifest declares.
        """
        self._validate_dtypes(x, offsets, indices)
        kernel = self._get_kernel((x, offsets, indices), x.dtype)
        out = kernel(x, offsets, indices=indices)
        self.dtype = x.dtype
        return out


def _device_index(tensor: torch.Tensor) -> int | None:
    return tensor.device.index


# Layout token and per-axis name suffixes, indexed by spatial dimensionality.
_POOL_LAYOUTS: Dict[int, str] = {1: "NCL", 2: "NCHW", 3: "NCDHW"}
_POOL_DIM_NAMES: Dict[int, Tuple[str, ...]] = {1: ("l",), 2: ("h", "w"), 3: ("d", "h", "w")}
# Kernel-kwarg suffixes for kernel_size/stride/padding(/dilation).
# Why: the 1d max-pool kernels name their pooling axis `w`, not `l`.
_AVG_POOL_PARAM_SUFFIXES: Dict[int, Tuple[str, ...]] = _POOL_DIM_NAMES
_MAX_POOL_PARAM_SUFFIXES: Dict[int, Tuple[str, ...]] = {
    1: ("w",),
    2: ("h", "w"),
    3: ("d", "h", "w"),
}


def _validate_pool_input_dtypes(self, input: torch.Tensor) -> None:
    """Shared pool-family dtype validator (bound per concrete class)."""
    if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"input.dtype must be float16, bfloat16, or float32, got {input.dtype}")


class _AvgPoolFwdOpBase(Op):
    """Generic average-pooling forward, parametrized by class-attribute ``ndim``.

    Concrete subclasses set ``ndim``, supply ``default_kernel_map``, and keep
    ``eval_roofline`` / ``_validate_dtypes`` in their own class body so
    manifest codegen resolves them per concrete class.
    """

    ndim: ClassVar[int]
    # Average pooling has one output; the registration below reads this.
    _returns_indices: ClassVar[bool] = False

    # This op's operator, and its name; both set by the registrations at the bottom of
    # this module, one per concrete op class.
    _wrapped: ClassVar[Any]
    compile_op_names: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        kernel_size: int | Tuple[int, ...],
        stride: Optional[int | Tuple[int, ...]] = None,
        padding: int | Tuple[int, ...] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
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
        self.target = target
        self.tune = tune
        validate_pool_params(
            ndim=nd,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            divisor_override=divisor_override,
        )
        self.dispatch_kernel(kernel_map)
        if self._generic_slot not in self.kernel_map and self._spatial_slot not in self.kernel_map:
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
                f"{type(self).__name__} expects input to be a {nd + 2}D {_POOL_LAYOUTS[nd]} tensor"
            )
        n, c_in, *in_dims = input.shape
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
        input: torch.Tensor,
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

        def build() -> Kernel:
            ks, st, pd = self._param_tuples()
            kernel_kwargs: Dict[str, object] = dict(n=n, c_in=c_in, dtype=dtype, tune=self.tune)
            for k, name in enumerate(_POOL_DIM_NAMES[self.ndim]):
                kernel_kwargs[f"{name}_in"] = in_dims[k]
            for k, name in enumerate(_AVG_POOL_PARAM_SUFFIXES[self.ndim]):
                kernel_kwargs[f"kernel_{name}"] = ks[k]
                kernel_kwargs[f"stride_{name}"] = st[k]
                kernel_kwargs[f"pad_{name}"] = pd[k]
            if not use_spatial_fast_path:
                kernel_kwargs["ceil_mode"] = self.ceil_mode
                kernel_kwargs["count_include_pad"] = self.count_include_pad
                if self.ndim > 1:
                    # The 1d generic kernel has no divisor_override parameter.
                    kernel_kwargs["divisor_override"] = self.divisor_override
            return self.kernel_map[kernel_name](**kernel_kwargs)

        return self.get_or_build_kernel(kernel_name, (input,), key=key, build=build)

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        nd = self.ndim
        if len(input_shape) != nd + 2:
            raise ValueError(
                f"{type(self).__name__} expects input_shape to be {nd + 2}D {_POOL_LAYOUTS[nd]}"
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
            pool_output_dim(size, ks[k], st[k], pd[k], ceil_mode) for k, size in enumerate(in_dims)
        )
        return {"output": (n, c_in, *out_dims)}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return type(self)._wrapped(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        resolved = self._resolve_input(input)
        input = input.contiguous()
        nd = self.ndim
        n, c_in = resolved[0], resolved[1]
        in_dims = resolved[2 : 2 + nd]
        out_dims = resolved[2 + nd : 2 + 2 * nd]
        dtype = resolved[-1]
        kernel = self._get_kernel(input, n, c_in, in_dims, dtype, _device_index(input))
        out = kernel(input)
        # Recorded after the launch: eval_roofline and profiling read these, and a call that
        # raised described nothing.
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        for name, size in zip(_POOL_DIM_NAMES[nd], in_dims, strict=True):
            setattr(self, f"{name}_in", size)
        for name, size in zip(_POOL_DIM_NAMES[nd], out_dims, strict=True):
            setattr(self, f"out_{name}", size)
        self.dtype = dtype
        self._last_roofline_spec = resolved
        return out


class AvgPool1dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCL inputs."""

    ndim = 1
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int],
        stride: Optional[int | Tuple[int]] = None,
        padding: int | Tuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        # No divisor_override: torch.nn.functional.avg_pool1d does not take one.
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int]``, default ``0``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            count_include_pad: Manifest ``params.count_include_pad``, ``bool``, default ``True``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            target=target,
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
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int, int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int, int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int, int]``, default ``0``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            count_include_pad: Manifest ``params.count_include_pad``, ``bool``, default ``True``.
            divisor_override: Manifest ``params.divisor_override``, ``int | None``, default ``None``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            target=target,
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
    in_dims = spec[2 : 2 + nd]
    out_dims = spec[2 + nd : 2 + 2 * nd]
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

    # This op's operator, and its name; both set by the registrations at the bottom of
    # this module, one per concrete op class.
    _wrapped: ClassVar[Any]
    compile_op_names: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        kernel_size: int | Tuple[int, ...],
        stride: Optional[int | Tuple[int, ...]] = None,
        padding: int | Tuple[int, ...] = 0,
        dilation: int | Tuple[int, ...] = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
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
        self.target = target
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
        self._last_roofline_spec: Optional[tuple] = None

    def _resolve_input(self, input: torch.Tensor) -> tuple:
        nd = self.ndim
        if input.ndim != nd + 2:
            raise ValueError(
                f"{self.__class__.__name__} expects input to be a "
                f"{nd + 2}D {_POOL_LAYOUTS[nd]} tensor"
            )
        n, c_in, *in_dims = input.shape
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
        input: torch.Tensor,
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

        def build() -> Kernel:
            kernel_kwargs: Dict[str, object] = dict(
                n=n,
                c_in=c_in,
                ceil_mode=self.ceil_mode,
                dtype=dtype,
                tune=self.tune,
            )
            for k, name in enumerate(_POOL_DIM_NAMES[self.ndim]):
                kernel_kwargs[f"{name}_in"] = in_dims[k]
            for k, name in enumerate(_MAX_POOL_PARAM_SUFFIXES[self.ndim]):
                kernel_kwargs[f"kernel_{name}"] = self.kernel_size[k]
                kernel_kwargs[f"stride_{name}"] = self.stride[k]
                kernel_kwargs[f"pad_{name}"] = self.padding[k]
                kernel_kwargs[f"dilation_{name}"] = self.dilation[k]
            return self.kernel_map[self._kernel_slot](**kernel_kwargs)

        return self.get_or_build_kernel(self._kernel_slot, (input,), key=key, build=build)

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        nd = self.ndim
        if len(input_shape) != nd + 2:
            raise ValueError(
                f"{self.__class__.__name__} expects input_shape to be {nd + 2}D {_POOL_LAYOUTS[nd]}"
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return type(self)._wrapped(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor):
        resolved = self._resolve_input(input)
        input = input.contiguous()
        nd = self.ndim
        n, c_in = resolved[0], resolved[1]
        in_dims = resolved[2 : 2 + nd]
        out_dims = resolved[2 + nd : 2 + 2 * nd]
        dtype = resolved[-1]
        kernel = self._get_kernel(input, n, c_in, in_dims, dtype, _device_index(input))
        out = kernel(input)
        # Recorded after the launch: eval_roofline and profiling read these, and a call that
        # raised described nothing.
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        for name, size in zip(_POOL_DIM_NAMES[nd], in_dims, strict=True):
            setattr(self, f"{name}_in", size)
        for name, size in zip(_POOL_DIM_NAMES[nd], out_dims, strict=True):
            setattr(self, f"out_{name}", size)
        self.dtype = dtype
        self._last_roofline_spec = resolved
        return out


class MaxPool1dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=False)."""

    ndim = 1
    _kernel_slot = "max_pool1d_kernel"
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int],
        stride: Optional[int | Tuple[int]] = None,
        padding: int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int]``, default ``0``.
            dilation: Manifest ``params.dilation``, ``int | tuple[int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
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
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int],
        stride: Optional[int | Tuple[int]] = None,
        padding: int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int]``, default ``0``.
            dilation: Manifest ``params.dilation``, ``int | tuple[int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool1d_with_indices_kernel": MaxPool1dWithIndicesKernel,
        }

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return type(self)._wrapped(input, self._instance_key)

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=True)


class MaxPool2dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCHW inputs (return_indices=False)."""

    ndim = 2
    _kernel_slot = "max_pool2d_kernel"
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int, int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int, int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int, int]``, default ``0``.
            dilation: Manifest ``params.dilation``, ``int | tuple[int, int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
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
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int, int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int, int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int, int]``, default ``0``.
            dilation: Manifest ``params.dilation``, ``int | tuple[int, int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool2d_with_indices_kernel": MaxPool2dWithIndicesKernel,
        }

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return type(self)._wrapped(input, self._instance_key)

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=True)


class MaxPool3dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCDHW inputs (return_indices=False)."""

    ndim = 3
    _kernel_slot = "max_pool3d_kernel"
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int, int, int],
        stride: Optional[int | Tuple[int, int, int]] = None,
        padding: int | Tuple[int, int, int] = 0,
        dilation: int | Tuple[int, int, int] = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int, int, int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int, int, int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int, int, int]``, default ``0``.
            dilation: Manifest ``params.dilation``, ``int | tuple[int, int, int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
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
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int, int, int],
        stride: Optional[int | Tuple[int, int, int]] = None,
        padding: int | Tuple[int, int, int] = 0,
        dilation: int | Tuple[int, int, int] = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int, int, int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int, int, int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int, int, int]``, default ``0``.
            dilation: Manifest ``params.dilation``, ``int | tuple[int, int, int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "max_pool3d_with_indices_kernel": MaxPool3dWithIndicesKernel,
        }

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return type(self)._wrapped(input, self._instance_key)

    def eval_roofline(self) -> tuple[int, int]:
        return _max_pool_roofline(self, indices=True)


class AvgPool3dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCDHW inputs."""

    ndim = 3
    _validate_dtypes = _validate_pool_input_dtypes

    def __init__(
        self,
        kernel_size: int | Tuple[int, int, int],
        stride: Optional[int | Tuple[int, int, int]] = None,
        padding: int | Tuple[int, int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int, int, int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int, int, int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int, int, int]``, default ``0``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            count_include_pad: Manifest ``params.count_include_pad``, ``bool``, default ``True``.
            divisor_override: Manifest ``params.divisor_override``, ``int | None``, default ``None``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            target=target,
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


def _normalize_output_size(
    output_size: int | None | Tuple[Optional[int], Optional[int]],
) -> Tuple[Optional[int], Optional[int]]:
    """Normalize adaptive-pool ``output_size`` to a 2-tuple of ``int | None``.

    Accepts ``None``, an int, or a 2-element tuple/list of ``int | None``
    (PyTorch-style leniency for list inputs); ``None`` entries — including a
    scalar ``None`` — resolve to the input size.
    """
    if output_size is None:
        return (None, None)
    if isinstance(output_size, bool):
        raise TypeError("output_size must be None, an int, or a tuple of (int | None, int | None)")
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if (
        not isinstance(output_size, (tuple, list))
        or len(output_size) != 2
        or any(
            isinstance(v, bool) or (v is not None and not isinstance(v, int)) for v in output_size
        )
    ):
        raise TypeError("output_size must be None, an int, or a tuple of (int | None, int | None)")
    if any(v is not None and v <= 0 for v in output_size):
        raise ValueError("output_size entries must be positive or None")
    return tuple(output_size)


def _validate_adaptive_pool_input_dtypes(self, input: torch.Tensor) -> None:
    """Adaptive-pool dtype validator: FP16/BF16 only (bound per concrete class)."""
    if input.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"input.dtype must be float16 or bfloat16, got {input.dtype}")


def _adaptive_pool2d_roofline(op: "_AdaptivePool2dFwdOpBase", *, indices: bool) -> tuple[int, int]:
    """Shared adaptive-pool roofline mirroring the manifest formulas."""
    if op._last_roofline_spec is None:
        raise RuntimeError(
            f"{type(op).__name__}.eval_roofline() requires a prior forward() "
            "call to bind input shape and dtype"
        )
    n, c_in, h_in, w_in, out_h, out_w, dtype = op._last_roofline_spec
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    # Exact adaptive-bin scan: bins are [floor(o*in/out), ceil((o+1)*in/out))
    # and adjacent bins overlap by one row/col unless out | j*in.
    scan_h = h_in + sum(1 for j in range(1, out_h) if (j * h_in) % out_h != 0)
    scan_w = w_in + sum(1 for o in range(1, out_w) if (o * w_in) % out_w != 0)
    flops = n * c_in * scan_h * scan_w
    bytes_ = (n * c_in * h_in * w_in + n * c_in * out_h * out_w) * elem_bytes
    if indices:
        bytes_ += n * c_in * out_h * out_w * 8
    return flops, bytes_


class _AdaptivePool2dFwdOpBase(Op):
    """Generic adaptive 2D pooling forward over CHW/NCHW inputs.

    Concrete subclasses set ``_kernel_slot`` / ``_returns_indices``, supply
    ``default_kernel_map``, and keep ``eval_roofline`` / ``_validate_dtypes``
    in their own class body so manifest codegen resolves them per concrete
    class.
    """

    _kernel_slot: ClassVar[str] = ""
    _returns_indices: ClassVar[bool] = False

    # This op's operator, and its name; both set by the registrations at the bottom of
    # this module, one per concrete op class.
    _wrapped: ClassVar[Any]
    compile_op_names: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]],
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.n = None
        self.c_in = None
        self.h_in = None
        self.w_in = None
        self.output_size = _normalize_output_size(output_size)
        self.dtype = None
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        if self._kernel_slot not in self.kernel_map:
            raise NotImplementedError(
                f"{type(self).__name__} requires {self._kernel_slot!r} in kernel_map"
            )
        self._last_roofline_spec: Optional[tuple] = None

    def _resolve_out_dims(self, h_in: int, w_in: int) -> tuple[int, int]:
        # Manifest parity probes call _infer_output_shapes on a mock self
        # without __init__; a missing output_size acts as (None, None),
        # i.e. pool to the input spatial size.
        output_size = getattr(self, "output_size", (None, None))
        out_h = h_in if output_size[0] is None else output_size[0]
        out_w = w_in if output_size[1] is None else output_size[1]
        return out_h, out_w

    def _infer_output_shapes(self, input_shape: tuple[int, ...]) -> Dict[str, tuple[int, ...]]:
        if len(input_shape) == 3:
            c_in, h_in, w_in = input_shape
            out_h, out_w = self._resolve_out_dims(h_in, w_in)
            full = (c_in, out_h, out_w)
        elif len(input_shape) == 4:
            n, c_in, h_in, w_in = input_shape
            out_h, out_w = self._resolve_out_dims(h_in, w_in)
            full = (n, c_in, out_h, out_w)
        else:
            raise ValueError(f"{type(self).__name__} expects input_shape to be 3D CHW or 4D NCHW")
        if self._returns_indices:
            return {"output": full, "indices": full}
        return {"output": full}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return type(self)._wrapped(input, self._instance_key)

    def _eager_forward(self, input: torch.Tensor):
        # A 3D CHW call is the torch-parity convenience; the manifest declares NCHW only. The
        # rank is normalized here, like contiguity, so what crosses to a kernel is the shape
        # the manifest describes, and the batch axis is dropped again on the way out. A CHW
        # call therefore shares its kernel with the batch-1 NCHW call, which computes it.
        if input.ndim == 3:
            squeezed = True
            x = input.unsqueeze(0)
        elif input.ndim == 4:
            squeezed = False
            x = input
        else:
            raise ValueError(
                f"{type(self).__name__} expects input to be a 3D CHW or 4D NCHW tensor"
            )
        self._validate_dtypes(x)
        n, c_in, h_in, w_in = x.shape
        out_h, out_w = self._resolve_out_dims(h_in, w_in)
        x = x.contiguous()
        dtype = x.dtype
        key = (n, c_in, h_in, w_in, out_h, out_w, dtype, _device_index(x), self.tune)
        kernel = self.get_or_build_kernel(
            self._kernel_slot,
            (x,),
            key=key,
            build=lambda: self.kernel_map[self._kernel_slot](
                n=n,
                c_in=c_in,
                h_in=h_in,
                w_in=w_in,
                out_h=out_h,
                out_w=out_w,
                dtype=dtype,
                tune=self.tune,
            ),
        )
        result = kernel(x)
        # Recorded after the launch: eval_roofline and profiling read these, and a call that
        # raised described nothing.
        self.kernel = kernel
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self._last_roofline_spec = (n, c_in, h_in, w_in, out_h, out_w, dtype)
        if self._returns_indices:
            out, indices = result
            if squeezed:
                return out.squeeze(0), indices.squeeze(0)
            return out, indices
        if squeezed:
            return result.squeeze(0)
        return result


class AdaptiveAvgPool2dFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive average pooling over PyTorch-compatible CHW/NCHW inputs."""

    _kernel_slot = "adaptive_avg_pool2d_kernel"
    _validate_dtypes = _validate_adaptive_pool_input_dtypes

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]],
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            output_size: Manifest ``params.output_size``, ``int | None | tuple[int | None, int | None] | list[int | None]``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            output_size=output_size,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "adaptive_avg_pool2d_kernel": AdaptiveAvgPool2dKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _adaptive_pool2d_roofline(self, indices=False)


class AdaptiveMaxPool2dFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive max pooling over CHW/NCHW inputs (return_indices=False)."""

    _kernel_slot = "adaptive_max_pool2d_kernel"
    _validate_dtypes = _validate_adaptive_pool_input_dtypes

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]],
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            output_size: Manifest ``params.output_size``, ``int | None | tuple[int | None, int | None] | list[int | None]``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            output_size=output_size,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "adaptive_max_pool2d_kernel": AdaptiveMaxPool2dKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        return _adaptive_pool2d_roofline(self, indices=False)


class AdaptiveMaxPool2dIndicesFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive max pooling over CHW/NCHW inputs (return_indices=True)."""

    _kernel_slot = "adaptive_max_pool2d_with_indices_kernel"
    _returns_indices = True
    _validate_dtypes = _validate_adaptive_pool_input_dtypes

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]],
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            output_size: Manifest ``params.output_size``, ``int | None | tuple[int | None, int | None] | list[int | None]``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            output_size=output_size,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "adaptive_max_pool2d_with_indices_kernel": AdaptiveMaxPool2dWithIndicesKernel,
        }

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return type(self)._wrapped(input, self._instance_key)

    def eval_roofline(self) -> tuple[int, int]:
        return _adaptive_pool2d_roofline(self, indices=True)


# The compile boundary: one operator per concrete op class, registered at import time.
# The op's key crosses it, and the body trades the key back for the instance — see
# src/tileops/ops/compile_boundary.py. Per class rather than per family, so the name a
# traced graph carries identifies the op that produced it, and a target that replaces one
# pool op leaves what the others' graphs hold alone.


def _register_pool_operator(op_cls: type, name: str) -> None:
    """Register *name* as *op_cls*'s operator and record it on the class."""
    if op_cls._returns_indices:

        @torch.library.custom_op(name, mutates_args=())
        def _fwd(input: torch.Tensor, instance_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
            return get_instance(instance_key)._eager_forward(input)

        @_fwd.register_fake
        def _fwd_fake(
            input: torch.Tensor,
            instance_key: str,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            shapes = get_instance(instance_key)._infer_output_shapes(tuple(input.shape))
            # ``new_empty``, not ``empty_like``: a non-contiguous input's strides must not reach the fake.
            return (
                input.new_empty(shapes["output"]),
                input.new_empty(shapes["indices"], dtype=torch.int64),
            )

    else:

        @torch.library.custom_op(name, mutates_args=())
        def _fwd(input: torch.Tensor, instance_key: str) -> torch.Tensor:  # noqa: F811
            return get_instance(instance_key)._eager_forward(input)

        @_fwd.register_fake
        def _fwd_fake(input: torch.Tensor, instance_key: str) -> torch.Tensor:  # noqa: F811
            shapes = get_instance(instance_key)._infer_output_shapes(tuple(input.shape))
            return input.new_empty(shapes["output"])

    op_cls._wrapped = _fwd
    op_cls.compile_op_names = (name,)


for _op_cls, _op_name in (
    (AvgPool1dFwdOp, "tileops::pool_avg_pool1d_fwd"),
    (AvgPool2dFwdOp, "tileops::pool_avg_pool2d_fwd"),
    (AvgPool3dFwdOp, "tileops::pool_avg_pool3d_fwd"),
    (MaxPool1dFwdOp, "tileops::pool_max_pool1d_fwd"),
    (MaxPool2dFwdOp, "tileops::pool_max_pool2d_fwd"),
    (MaxPool3dFwdOp, "tileops::pool_max_pool3d_fwd"),
    (MaxPool1dIndicesFwdOp, "tileops::pool_max_pool1d_indices_fwd"),
    (MaxPool2dIndicesFwdOp, "tileops::pool_max_pool2d_indices_fwd"),
    (MaxPool3dIndicesFwdOp, "tileops::pool_max_pool3d_indices_fwd"),
    (AdaptiveAvgPool2dFwdOp, "tileops::pool_adaptive_avg_pool2d_fwd"),
    (AdaptiveMaxPool2dFwdOp, "tileops::pool_adaptive_max_pool2d_fwd"),
    (AdaptiveMaxPool2dIndicesFwdOp, "tileops::pool_adaptive_max_pool2d_indices_fwd"),
):
    _register_pool_operator(_op_cls, _op_name)
