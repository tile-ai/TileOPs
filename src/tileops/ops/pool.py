import weakref
from collections.abc import Sequence
from math import prod
from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
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
    "MeanPoolingFwdOp",
]


def _per_axis(value: "int | Sequence[int]", ndim: int) -> tuple[int, ...]:
    """A pooling parameter as one value per spatial axis, as ``per_axis`` reads it."""
    return (value,) * ndim if isinstance(value, int) else tuple(value)


class _CheckedChunkMaps:
    """The chunk maps whose values have already been through `MeanPoolingFwdOp`'s checks.

    An entry stands only while both weak references still resolve to the tensors that were
    checked and neither `_version` has moved, so a map written in place, freed, or replaced
    is checked again.
    """

    # Distinct maps remembered at once. A dead tensor leaves its key behind, so the table is
    # dropped rather than grown once a caller has cycled through this many.
    _ENTRIES = 8

    def __init__(self) -> None:
        """Start empty. An entry maps a key to (weak offsets, weak indices, both versions)."""
        self._seen: Dict[tuple, tuple] = {}

    @staticmethod
    def _key(offsets: torch.Tensor, indices: torch.Tensor, seq_len: int, chunks: int) -> tuple:
        return (id(offsets), id(indices), seq_len, chunks)

    def checked(
        self, offsets: torch.Tensor, indices: torch.Tensor, seq_len: int, chunks: int
    ) -> bool:
        """Whether this map, unchanged since it was checked, may skip the checks."""
        entry = self._seen.get(self._key(offsets, indices, seq_len, chunks))
        if entry is None:
            return False
        offsets_ref, indices_ref, versions = entry
        return (
            offsets_ref() is offsets
            and indices_ref() is indices
            and versions == (offsets._version, indices._version)
        )

    def remember(
        self, offsets: torch.Tensor, indices: torch.Tensor, seq_len: int, chunks: int
    ) -> None:
        """Record that this map passed the checks."""
        if len(self._seen) >= self._ENTRIES:
            self._seen.clear()
        self._seen[self._key(offsets, indices, seq_len, chunks)] = (
            weakref.ref(offsets),
            weakref.ref(indices),
            (offsets._version, indices._version),
        )


class MeanPoolingFwdOp(Op):
    """Chunked mean over the sequence axis of a ``[batch, seq, heads, dim]`` tensor.

    Not a PyTorch pooling op, and torch has no counterpart. The sequence axis is cut into
    chunks of ``chunk_size`` and each chunk is averaged, giving one output row per chunk.
    Pass ``offsets`` and ``indices`` and the chunks follow the ragged sequence boundaries
    ``offsets`` describes instead of a uniform split; ``indices`` then names the
    ``(sequence, chunk-within-sequence)`` pair each output row belongs to, one row per
    chunk.

    A sequence's last chunk may be shorter than ``chunk_size``. It is divided by the count
    it actually holds, so no padding is averaged in. On a uniform split that makes the op
    equal to ``torch.nn.functional.avg_pool1d(kernel_size=chunk_size, stride=chunk_size,
    ceil_mode=True)`` over a view with the sequence axis last. Chunk sums accumulate in
    ``accum_dtype`` and are cast back to the input dtype at the boundary, so a ``float16``
    input with ``accum_dtype=torch.float32`` does not lose the sum to rounding.

    Example:
        ```python linenums="1"
        op = MeanPoolingFwdOp(chunk_size=64, accum_dtype=torch.float32)
        chunk_means = op(x)                        # uniform split
        chunk_means = op(x, offsets, indices)      # ragged, one row of indices per chunk
        ```
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mean_pooling_fwd_kernel": MeanPoolingFwdKernel
    }

    def __init__(
        self,
        chunk_size: int,
        accum_dtype: torch.dtype,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            chunk_size: Manifest ``params.chunk_size``, ``int``, a positive multiple of 32.
            accum_dtype: Manifest ``params.accum_dtype``, ``torch.dtype`` — what a chunk sum
                accumulates in.
            target: Backend target to serve this op, or ``None`` to decide from the input
                device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.chunk_size = chunk_size
        self.accum_dtype = accum_dtype
        self.target = target
        self.tune = tune
        # Keyed by (device, shape): the uniform path hands the kernel tensors it never
        # reads, and a placeholder on the wrong device would route the launch there.
        self._placeholders: Dict[tuple, torch.Tensor] = {}
        self._checked_maps = _CheckedChunkMaps()
        self.dispatch_kernel(kernel_map)

    def _placeholder(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        key = (device, shape)
        if key not in self._placeholders:
            self._placeholders[key] = torch.zeros(shape, dtype=torch.int32, device=device)
        return self._placeholders[key]

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, chunking and offsets presence."""
        (
            batch_size,
            seq_len,
            heads,
            dim,
            chunks_per_batch,
            seq_num,
            use_offsets,
            dtype,
        ) = call
        return call, lambda: self.kernel_map["mean_pooling_fwd_kernel"](
            batch_size=batch_size,
            seq_len=seq_len,
            heads=heads,
            dim=dim,
            chunk_size=self.chunk_size,
            chunks_per_batch=chunks_per_batch,
            seq_num=seq_num,
            use_offsets=use_offsets,
            dtype=dtype,
            accum_dtype=self.accum_dtype,
            tune=self.tune,
        )

    def forward(
        self,
        x: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Average each chunk of ``x``'s sequence axis.

        Args:
            x: Input tensor, ``[batch, seq, heads, dim]``, dtype ``float16``, ``bfloat16``
                or ``float32``. ``dim`` is at most 128 or a multiple of it, which is what the
                manifest declares rather than what the kernel needs.
            offsets: Sequence boundaries, ``[seq_num + 1]``, dtype ``int32``. Passing it
                selects the ragged split, and it comes with ``indices``.
            indices: One ``(sequence, chunk-within-sequence)`` pair per output chunk,
                ``[chunks, 2]``, dtype ``int32``.

        Returns:
            ``output``, ``[batch, chunks, heads, dim]``, dtype as ``x``. ``chunks`` is
            ``ceil(seq / chunk_size)`` for a uniform split and ``indices.shape[0]`` for a
            ragged one.

        Raises:
            ValueError: ``indices`` does not hold one row per chunk ``offsets`` implies, or
                ``offsets`` does not partition ``x``'s sequence axis.
        """
        # Heads and dim are read as one width.
        x = x.contiguous()
        batch_size, seq_len, heads, dim = x.shape
        ragged = offsets is not None
        if ragged:
            chunks = indices.shape[0]
            self._validate_ragged(offsets, indices, seq_len, chunks)
            seq_num = offsets.shape[0] - 1
            offsets_arg, indices_arg = offsets.contiguous(), indices.contiguous()
        else:
            chunks = -(-seq_len // self.chunk_size)
            # The kernel takes both tensors whether or not it reads them; `inputs` keeps
            # the caller's `None`s, which is where presence is read from. `seq_num = 0`
            # divides by zero in the autotune supply, so one whole-axis sequence it is.
            seq_num = 1
            offsets_arg = self._placeholder((2,), x.device)
            indices_arg = self._placeholder((chunks, 2), x.device)

        kernel = self.kernel_for(
            "mean_pooling_fwd_kernel",
            (x, offsets, indices),
            (batch_size, seq_len, heads, dim, chunks, seq_num, int(ragged), x.dtype),
        )
        return kernel(x, offsets_arg, indices=indices_arg)

    def _validate_ragged(
        self, offsets: torch.Tensor, indices: torch.Tensor, seq_len: int, chunks: int
    ) -> None:
        """Check a chunk map once, then skip it while the same tensors come back unchanged.

        `_check_ragged` reads `offsets` and `indices` element by element, costing a dozen
        device launches and as many syncs that a ragged call would otherwise pay before
        every pooling. A chunk map is built once and passed to many calls.
        """
        if self._checked_maps.checked(offsets, indices, seq_len, chunks):
            return
        self._check_ragged(offsets, indices, seq_len, chunks)
        self._checked_maps.remember(offsets, indices, seq_len, chunks)

    def _check_ragged(
        self, offsets: torch.Tensor, indices: torch.Tensor, seq_len: int, chunks: int
    ) -> None:
        """Check `indices` against `offsets` rather than believing its row count.

        The output's chunk axis has to come from a shape, because that is all the compile
        fake is handed, so it comes from `indices`. The values are here, so this is where
        an `indices` that disagrees with `offsets` is caught.
        """
        lengths = offsets[1:] - offsets[:-1]
        if int(lengths.min()) < 0:
            raise ValueError("offsets must be non-decreasing")
        # A partition, not a window: every token belongs to exactly one sequence.
        if int(offsets[0]) != 0 or int(offsets[-1]) != seq_len:
            raise ValueError(
                f"offsets must run 0 to x's sequence axis of {seq_len}; got "
                f"{int(offsets[0])} to {int(offsets[-1])}"
            )
        per_seq = -(-lengths // self.chunk_size)
        implied = int(per_seq.sum())
        if chunks != implied:
            raise ValueError(
                f"indices holds {chunks} chunks but offsets imply {implied} for "
                f"chunk_size={self.chunk_size}"
            )
        # The kernel reads the chunk each row names, in range or not.
        seq_ids, chunk_ids = indices[:, 0].long(), indices[:, 1]
        if int(seq_ids.min()) < 0 or int(seq_ids.max()) >= per_seq.shape[0]:
            raise ValueError(
                f"indices names a sequence outside offsets' {per_seq.shape[0]} sequences"
            )
        if not bool(((chunk_ids >= 0) & (chunk_ids < per_seq[seq_ids])).all()):
            raise ValueError("indices names a chunk its sequence does not have")
        # A repeat would emit one chunk twice and drop another.
        chunk_base = per_seq.cumsum(0) - per_seq
        if int(torch.unique(chunk_base[seq_ids] + chunk_ids).numel()) != chunks:
            raise ValueError("indices must name each chunk offsets implies exactly once")


# Per-axis name suffixes, indexed by spatial dimensionality.
_POOL_DIM_NAMES: Dict[int, Tuple[str, ...]] = {1: ("l",), 2: ("h", "w"), 3: ("d", "h", "w")}
# Kernel-kwarg suffixes for kernel_size/stride/padding(/dilation).
# Why: the 1d max-pool kernels name their pooling axis `w`, not `l`.
_MAX_POOL_PARAM_SUFFIXES: Dict[int, Tuple[str, ...]] = {
    1: ("w",),
    2: ("h", "w"),
    3: ("d", "h", "w"),
}


class _AvgPoolFwdOpBase(Op):
    """Generic average-pooling forward, parametrized by class-attribute ``ndim``.

    Concrete subclasses set ``ndim`` and ``kernel_types`` and state the manifest's
    ``__init__``; the signature's checks, shape inference and roofline are generated per
    concrete class.
    """

    ndim: ClassVar[int]
    compile_boundary = True

    def _setup(self, kernel_map: Optional[Dict[str, Kernel]]) -> None:
        """Resolve the kernel map, then the per-axis parameters the kernels take."""
        self.dispatch_kernel(kernel_map)
        nd = self.ndim
        self._kernel_size = _per_axis(self.kernel_size, nd)
        self._stride = self._kernel_size if self.stride is None else _per_axis(self.stride, nd)
        self._padding = _per_axis(self.padding, nd)
        self._divisor_override = getattr(self, "divisor_override", None)
        self._has_explicit_generic_kernel = (
            kernel_map is not None and self._generic_slot in kernel_map
        )
        self._has_explicit_spatial_kernel = (
            kernel_map is not None and self._spatial_slot in kernel_map
        )

    @property
    def _generic_slot(self) -> str:
        return f"avg_pool{self.ndim}d_kernel"

    @property
    def _spatial_slot(self) -> str:
        return f"avg_pool{self.ndim}d_spatial_kernel"

    def _use_spatial_fast_path(self) -> bool:
        # Strict 1d/3d policy: an explicit generic-kernel override opts out of
        # the spatial fast path unless the spatial kernel is also explicit.
        # AvgPool2dFwdOp overrides this with a laxer 2d policy.
        return (
            not self.ceil_mode
            and self.count_include_pad
            and self._divisor_override is None
            and (not self._has_explicit_generic_kernel or self._has_explicit_spatial_kernel)
        )

    def entry_for(self, role: str, call: tuple) -> Entry:
        """The spatial fast path picks the implementation, so its name is in the identity."""
        n, c_in, in_dims, dtype, device_index = call
        use_spatial_fast_path = self._use_spatial_fast_path()
        kernel_name = self._spatial_slot if use_spatial_fast_path else self._generic_slot
        key = (
            kernel_name,
            n,
            c_in,
            *in_dims,
            self._kernel_size,
            self._stride,
            self._padding,
            self.ceil_mode,
            self.count_include_pad,
            self._divisor_override,
            dtype,
            device_index,
        )

        def build() -> Kernel:
            kernel_kwargs: Dict[str, object] = dict(n=n, c_in=c_in, dtype=dtype, tune=self.tune)
            for k, name in enumerate(_POOL_DIM_NAMES[self.ndim]):
                kernel_kwargs[f"{name}_in"] = in_dims[k]
                kernel_kwargs[f"kernel_{name}"] = self._kernel_size[k]
                kernel_kwargs[f"stride_{name}"] = self._stride[k]
                kernel_kwargs[f"pad_{name}"] = self._padding[k]
            if not use_spatial_fast_path:
                kernel_kwargs["ceil_mode"] = self.ceil_mode
                kernel_kwargs["count_include_pad"] = self.count_include_pad
                if self.ndim > 1:
                    # The 1d generic kernel has no divisor_override parameter.
                    kernel_kwargs["divisor_override"] = self._divisor_override
            return self.kernel_map[kernel_name](**kernel_kwargs)

        return key, build

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return self._call_boundary(input)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.contiguous()
        n, c_in, *in_dims = input.shape
        self.kernel = self.kernel_for(
            "avg_pool", (input,), (n, c_in, tuple(in_dims), input.dtype, input.device.index)
        )
        return self.kernel(input)


class AvgPool1dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCL inputs."""

    ndim = 1
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "avg_pool1d_kernel": AvgPool1dKernel,
        "avg_pool1d_spatial_kernel": AvgPool1dSpatialKernel,
    }

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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.target = target
        self.tune = tune
        self._setup(kernel_map)


class AvgPool2dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCHW inputs."""

    ndim = 2
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "avg_pool2d_kernel": AvgPool2dKernel,
        "avg_pool2d_spatial_kernel": AvgPool2dSpatialKernel,
    }

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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.target = target
        self.tune = tune
        self._setup(kernel_map)

    def _use_spatial_fast_path(self) -> bool:
        # Laxer 2d policy: an explicit generic-kernel override does not opt out
        # of the spatial fast path (asymmetric with 1d/3d).
        return not self.ceil_mode and self.count_include_pad and self._divisor_override is None


class AvgPool3dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCDHW inputs."""

    ndim = 3
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "avg_pool3d_kernel": AvgPool3dKernel,
        "avg_pool3d_spatial_kernel": AvgPool3dSpatialKernel,
    }

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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.target = target
        self.tune = tune
        self._setup(kernel_map)


class _MaxPoolFwdOpBase(Op):
    """Generic max-pooling forward, parametrized by class attributes.

    Concrete subclasses set ``ndim``, ``_kernel_slot`` and ``kernel_types`` and state the
    manifest's ``__init__``.
    """

    ndim: ClassVar[int]
    _kernel_slot: ClassVar[str] = ""
    compile_boundary = True

    def __init__(
        self,
        kernel_size: "int | Tuple[int, ...]",
        stride: "Optional[int | Tuple[int, ...]]" = None,
        padding: "int | Tuple[int, ...]" = 0,
        dilation: "int | Tuple[int, ...]" = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, an int or one per spatial axis.
            stride: Manifest ``params.stride``, an int, one per axis, or ``None``.
            padding: Manifest ``params.padding``, an int or one per spatial axis.
            dilation: Manifest ``params.dilation``, an int or one per spatial axis.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        nd = self.ndim
        self._kernel_size = _per_axis(kernel_size, nd)
        self._stride = self._kernel_size if stride is None else _per_axis(stride, nd)
        self._padding = _per_axis(padding, nd)
        self._dilation = _per_axis(dilation, nd)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, window, stride, padding and dilation."""
        n, c_in, in_dims, dtype, device_index = call
        key = (
            n,
            c_in,
            *in_dims,
            self._kernel_size,
            self._stride,
            self._padding,
            self._dilation,
            self.ceil_mode,
            dtype,
            device_index,
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
                kernel_kwargs[f"kernel_{name}"] = self._kernel_size[k]
                kernel_kwargs[f"stride_{name}"] = self._stride[k]
                kernel_kwargs[f"pad_{name}"] = self._padding[k]
                kernel_kwargs[f"dilation_{name}"] = self._dilation[k]
            return self.kernel_map[self._kernel_slot](**kernel_kwargs)

        return key, build

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return self._call_boundary(input)

    def _eager_forward(self, input: torch.Tensor):
        input = input.contiguous()
        n, c_in, *in_dims = input.shape
        self.kernel = self.kernel_for(
            "max_pool", (input,), (n, c_in, tuple(in_dims), input.dtype, input.device.index)
        )
        return self.kernel(input)


class MaxPool1dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=False)."""

    ndim = 1
    _kernel_slot = "max_pool1d_kernel"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"max_pool1d_kernel": MaxPool1dKernel}

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


class MaxPool1dIndicesFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=True)."""

    ndim = 1
    _kernel_slot = "max_pool1d_with_indices_kernel"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "max_pool1d_with_indices_kernel": MaxPool1dWithIndicesKernel
    }

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

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return self._call_boundary(input)


class MaxPool2dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCHW inputs (return_indices=False)."""

    ndim = 2
    _kernel_slot = "max_pool2d_kernel"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"max_pool2d_kernel": MaxPool2dKernel}

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


class MaxPool2dIndicesFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCHW inputs (return_indices=True)."""

    ndim = 2
    _kernel_slot = "max_pool2d_with_indices_kernel"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "max_pool2d_with_indices_kernel": MaxPool2dWithIndicesKernel
    }

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

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return self._call_boundary(input)


class MaxPool3dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCDHW inputs (return_indices=False)."""

    ndim = 3
    _kernel_slot = "max_pool3d_kernel"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"max_pool3d_kernel": MaxPool3dKernel}

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


class MaxPool3dIndicesFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCDHW inputs (return_indices=True)."""

    ndim = 3
    _kernel_slot = "max_pool3d_with_indices_kernel"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "max_pool3d_with_indices_kernel": MaxPool3dWithIndicesKernel
    }

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

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return self._call_boundary(input)


class _AdaptivePool2dFwdOpBase(Op):
    """Generic adaptive 2D pooling forward over CHW/NCHW inputs.

    Concrete subclasses set ``_kernel_slot`` and ``kernel_types`` and state the manifest's
    ``__init__``. A CHW input is handed to the kernel as it is; the kernel adds and drops
    the batch axis.
    """

    _kernel_slot: ClassVar[str] = ""
    compile_boundary = True

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]] | list[Optional[int]],
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            output_size: Manifest ``params.output_size``, ``int | None | tuple[int | None, int | None] | list[int | None]``;
                a ``None`` extent keeps the input's.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.output_size = output_size
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._output_size = (
            (output_size, output_size)
            if output_size is None or isinstance(output_size, int)
            else tuple(output_size)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return self._call_boundary(input)

    def _eager_forward(self, input: torch.Tensor):
        input = input.contiguous()
        c_in, h_in, w_in = input.shape[-3:]
        out_h = h_in if self._output_size[0] is None else self._output_size[0]
        out_w = w_in if self._output_size[1] is None else self._output_size[1]
        key = (
            prod(input.shape[:-3]),
            c_in,
            h_in,
            w_in,
            out_h,
            out_w,
            input.dtype,
            input.device.index,
        )
        self.kernel = self.kernel_for("adaptive_pool", (input,), key)
        return self.kernel(input)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per input and output extents, dtype and device."""
        n, c_in, h_in, w_in, out_h, out_w, dtype, _device = call
        return call, lambda: self.kernel_map[self._kernel_slot](
            n=n,
            c_in=c_in,
            h_in=h_in,
            w_in=w_in,
            out_h=out_h,
            out_w=out_w,
            dtype=dtype,
            tune=self.tune,
        )


class AdaptiveAvgPool2dFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive average pooling over PyTorch-compatible CHW/NCHW inputs."""

    _kernel_slot = "adaptive_avg_pool2d_kernel"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "adaptive_avg_pool2d_kernel": AdaptiveAvgPool2dKernel
    }

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]] | list[Optional[int]],
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
        super().__init__(output_size=output_size, target=target, kernel_map=kernel_map, tune=tune)


class AdaptiveMaxPool2dFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive max pooling over CHW/NCHW inputs (return_indices=False)."""

    _kernel_slot = "adaptive_max_pool2d_kernel"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "adaptive_max_pool2d_kernel": AdaptiveMaxPool2dKernel
    }

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]] | list[Optional[int]],
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
        super().__init__(output_size=output_size, target=target, kernel_map=kernel_map, tune=tune)


class AdaptiveMaxPool2dIndicesFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive max pooling over CHW/NCHW inputs (return_indices=True)."""

    _kernel_slot = "adaptive_max_pool2d_with_indices_kernel"
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "adaptive_max_pool2d_with_indices_kernel": AdaptiveMaxPool2dWithIndicesKernel
    }

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]] | list[Optional[int]],
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
        super().__init__(output_size=output_size, target=target, kernel_map=kernel_map, tune=tune)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return self._call_boundary(input)
