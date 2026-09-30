"""The facts of one pooling call that its in-tree kernels select and build on, and the kernel
interfaces their implementations inherit."""

import dataclasses
from abc import abstractmethod
from typing import Optional, Tuple

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "AdaptiveAvgPool2dFwdInterface",
    "AdaptiveMaxPool2dFwdInterface",
    "AdaptiveMaxPool2dIndicesFwdInterface",
    "AdaptivePool2dCall",
    "AvgPool1dFwdInterface",
    "AvgPool2dFwdInterface",
    "AvgPool3dFwdInterface",
    "AvgPoolCall",
    "MaxPool1dFwdInterface",
    "MaxPool1dIndicesFwdInterface",
    "MaxPool2dFwdInterface",
    "MaxPool2dIndicesFwdInterface",
    "MaxPool3dFwdInterface",
    "MaxPool3dIndicesFwdInterface",
    "MaxPoolCall",
    "MeanPoolingCall",
    "MeanPoolingFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class AvgPoolCall(CallSpec):
    """One average-pooling call over an ``(n, c_in, *size)`` input.

    ``size``, ``window``, ``stride`` and ``pad`` hold one entry per spatial axis, in the
    input's axis order, so the same record serves the 1d, 2d and 3d interfaces.
    """

    n: int = 0
    c_in: int = 0
    size: Tuple[int, ...] = ()
    window: Tuple[int, ...] = ()
    stride: Tuple[int, ...] = ()
    pad: Tuple[int, ...] = ()
    ceil_mode: bool = False
    count_include_pad: bool = True
    divisor_override: Optional[int] = None
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class MaxPoolCall(CallSpec):
    """One max-pooling call over an ``(n, c_in, *size)`` input.

    ``size``, ``window``, ``stride``, ``pad`` and ``dilation`` hold one entry per spatial
    axis, in the input's axis order.
    """

    n: int = 0
    c_in: int = 0
    size: Tuple[int, ...] = ()
    window: Tuple[int, ...] = ()
    stride: Tuple[int, ...] = ()
    pad: Tuple[int, ...] = ()
    dilation: Tuple[int, ...] = ()
    ceil_mode: bool = False
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class AdaptivePool2dCall(CallSpec):
    """One adaptive 2D pooling call, with the output extents the op resolved."""

    n: int = 0
    c_in: int = 0
    h_in: int = 0
    w_in: int = 0
    out_h: int = 0
    out_w: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class MeanPoolingCall(CallSpec):
    """One chunked sequence-mean call, with the chunking the op resolved from its inputs."""

    batch_size: int = 0
    seq_len: int = 0
    heads: int = 0
    dim: int = 0
    chunk_size: int = 0
    chunks_per_batch: int = 0
    seq_num: int = 0
    use_offsets: bool = False
    dtype: Optional[torch.dtype] = None
    accum_dtype: Optional[torch.dtype] = None


class AvgPool1dFwdInterface(KernelInterface):
    """Average pooling over the length axis of an NCL input."""

    request = AvgPoolCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average each window of ``x``'s length axis; nothing is written in place.

        Padding is not read: a window overlapping it sums the input elements it covers.
        Nothing is written in place, and the output aliases no input.

        Args:
            x: The input, ``(call.n, call.c_in, call.size[0])`` in ``call.dtype``, on
                ``call.device``, contiguous in that axis order. The op makes it contiguous
                before the call, so an implementation may read it as one flat run.

        Returns:
            A new contiguous ``(call.n, call.c_in, out_l)`` tensor in ``call.dtype`` on
            ``call.device``, where ``out_l`` follows ``call.window``, ``call.stride``,
            ``call.pad`` and ``call.ceil_mode`` as ``torch.nn.functional.avg_pool1d``
            defines it. A window's divisor is the count of positions it covers, the padded
            ones included exactly when ``call.count_include_pad``. Sums accumulate in
            ``float32`` and are cast back at the store.
        """


class AvgPool2dFwdInterface(KernelInterface):
    """Average pooling over the two spatial axes of an NCHW input."""

    request = AvgPoolCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average each window of ``x``'s spatial axes; nothing is written in place.

        Padding is not read: a window overlapping it sums the input elements it covers.
        Nothing is written in place, and the output aliases no input.

        Args:
            x: The input, ``(call.n, call.c_in, *call.size)`` in ``call.dtype``, on
                ``call.device``, contiguous in that axis order. The op makes it contiguous
                before the call, so an implementation may read each plane as one flat run.

        Returns:
            A new contiguous ``(call.n, call.c_in, out_h, out_w)`` tensor in ``call.dtype``
            on ``call.device``, with the output extents
            ``torch.nn.functional.avg_pool2d`` gives for ``call.window``, ``call.stride``,
            ``call.pad`` and ``call.ceil_mode``. A window's divisor is
            ``call.divisor_override`` where it is set, else the count of positions the
            window covers with the padded ones included exactly when
            ``call.count_include_pad``. Sums accumulate in ``float32``.
        """


class AvgPool3dFwdInterface(KernelInterface):
    """Average pooling over the three spatial axes of an NCDHW input."""

    request = AvgPoolCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average each window of ``x``'s spatial axes; nothing is written in place.

        Padding is not read: a window overlapping it sums the input elements it covers.
        Nothing is written in place, and the output aliases no input.

        Args:
            x: The input, ``(call.n, call.c_in, *call.size)`` in ``call.dtype``, on
                ``call.device``, contiguous in that axis order. The op makes it contiguous
                before the call, so an implementation may read each volume as one flat run.

        Returns:
            A new contiguous ``(call.n, call.c_in, out_d, out_h, out_w)`` tensor in
            ``call.dtype`` on ``call.device``, with the output extents
            ``torch.nn.functional.avg_pool3d`` gives for ``call.window``, ``call.stride``,
            ``call.pad`` and ``call.ceil_mode``. A window's divisor is
            ``call.divisor_override`` where it is set, else the count of positions the
            window covers with the padded ones included exactly when
            ``call.count_include_pad``. Sums accumulate in ``float32``.
        """


class MaxPool1dFwdInterface(KernelInterface):
    """Max pooling over the length axis of an NCL input, values only."""

    request = MaxPoolCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take each window's maximum; nothing is written in place.

        A window's positions step by ``call.dilation``. The padded positions read as
        negative infinity, so a window covering only padding yields it. Nothing is written
        in place, and the output aliases no input.

        Args:
            x: The input, ``(call.n, call.c_in, call.size[0])`` in ``call.dtype``, on
                ``call.device``, contiguous in that axis order. The op makes it contiguous
                before the call.

        Returns:
            A new contiguous ``(call.n, call.c_in, out_l)`` tensor in ``call.dtype`` on
            ``call.device``, with the output extents ``torch.nn.functional.max_pool1d``
            gives for ``call.window``, ``call.stride``, ``call.pad``, ``call.dilation``
            and ``call.ceil_mode``. The comparison runs in ``call.dtype``.
        """


class MaxPool2dFwdInterface(KernelInterface):
    """Max pooling over the two spatial axes of an NCHW input, values only."""

    request = MaxPoolCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take each window's maximum; nothing is written in place.

        A window's positions step by ``call.dilation``. The padded positions read as
        negative infinity, so a window covering only padding yields it. Nothing is written
        in place, and the output aliases no input.

        Args:
            x: The input, ``(call.n, call.c_in, *call.size)`` in ``call.dtype``, on
                ``call.device``, contiguous in that axis order. The op makes it contiguous
                before the call.

        Returns:
            A new contiguous ``(call.n, call.c_in, out_h, out_w)`` tensor in ``call.dtype`` on
            ``call.device``, with the output extents ``torch.nn.functional.max_pool2d``
            gives for ``call.window``, ``call.stride``, ``call.pad``, ``call.dilation``
            and ``call.ceil_mode``. The comparison runs in ``call.dtype``.
        """


class MaxPool3dFwdInterface(KernelInterface):
    """Max pooling over the three spatial axes of an NCDHW input, values only."""

    request = MaxPoolCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take each window's maximum; nothing is written in place.

        A window's positions step by ``call.dilation``. The padded positions read as
        negative infinity, so a window covering only padding yields it. Nothing is written
        in place, and the output aliases no input.

        Args:
            x: The input, ``(call.n, call.c_in, *call.size)`` in ``call.dtype``, on
                ``call.device``, contiguous in that axis order. The op makes it contiguous
                before the call.

        Returns:
            A new contiguous ``(call.n, call.c_in, out_d, out_h, out_w)`` tensor in ``call.dtype`` on
            ``call.device``, with the output extents ``torch.nn.functional.max_pool3d``
            gives for ``call.window``, ``call.stride``, ``call.pad``, ``call.dilation``
            and ``call.ceil_mode``. The comparison runs in ``call.dtype``.
        """


class MaxPool1dIndicesFwdInterface(KernelInterface):
    """Max pooling over the length axis of an NCL input, values and argmax positions."""

    request = MaxPoolCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take each window's maximum and where it sat; nothing is written in place.

        A window's positions step by ``call.dilation``. The padded positions read as
        negative infinity. Nothing is written in place, and neither output aliases an input.

        Args:
            x: The input, ``(call.n, call.c_in, call.size[0])`` in ``call.dtype``, on
                ``call.device``, contiguous in that axis order. The op makes it contiguous
                before the call.

        Returns:
            New contiguous ``(values, indices)`` on ``call.device``, both
            ``(call.n, call.c_in, out_l)`` with the extents
            ``torch.nn.functional.max_pool1d`` gives: ``values`` in ``call.dtype``, and
            ``indices`` in ``int64``, each the flat position of its maximum within the
            input's length axis. A tie takes the lowest position.
        """


class MaxPool2dIndicesFwdInterface(KernelInterface):
    """Max pooling over the two spatial axes of an NCHW input, values and argmax positions."""

    request = MaxPoolCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take each window's maximum and where it sat; nothing is written in place.

        A window's positions step by ``call.dilation``. The padded positions read as
        negative infinity. Nothing is written in place, and neither output aliases an input.

        Args:
            x: The input, ``(call.n, call.c_in, *call.size)`` in ``call.dtype``, on
                ``call.device``, contiguous in that axis order. The op makes it contiguous
                before the call.

        Returns:
            New contiguous ``(values, indices)`` on ``call.device``, both
            ``(call.n, call.c_in, out_h, out_w)`` with the extents
            ``torch.nn.functional.max_pool2d`` gives: ``values`` in ``call.dtype``, and
            ``indices`` in ``int64``, each the flat position of its maximum within the
            input's ``(h, w)`` plane. A tie takes the lowest position.
        """


class MaxPool3dIndicesFwdInterface(KernelInterface):
    """Max pooling over the three spatial axes of an NCDHW input, values and argmax positions."""

    request = MaxPoolCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take each window's maximum and where it sat; nothing is written in place.

        A window's positions step by ``call.dilation``. The padded positions read as
        negative infinity. Nothing is written in place, and neither output aliases an input.

        Args:
            x: The input, ``(call.n, call.c_in, *call.size)`` in ``call.dtype``, on
                ``call.device``, contiguous in that axis order. The op makes it contiguous
                before the call.

        Returns:
            New contiguous ``(values, indices)`` on ``call.device``, both
            ``(call.n, call.c_in, out_d, out_h, out_w)`` with the extents
            ``torch.nn.functional.max_pool3d`` gives: ``values`` in ``call.dtype``, and
            ``indices`` in ``int64``, each the flat position of its maximum within the
            input's ``(d, h, w)`` volume. A tie takes the lowest position.
        """


class AdaptiveAvgPool2dFwdInterface(KernelInterface):
    """Adaptive average pooling to a fixed output grid over a CHW or NCHW input."""

    request = AdaptivePool2dCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average each output cell's input window; nothing is written in place.

        Output cell ``(i, j)`` averages the input rows ``floor(i * h_in / out_h)`` up to
        ``ceil((i + 1) * h_in / out_h)`` and the columns given the same way, as
        ``torch.nn.functional.adaptive_avg_pool2d`` defines it. Windows may overlap, the
        divisor is the count of positions the window covers, and sums accumulate in
        ``float32``. Nothing is written in place, and the output aliases no input.

        Args:
            x: The input, ``(call.n, call.c_in, call.h_in, call.w_in)``, or
                ``(call.c_in, call.h_in, call.w_in)`` when ``call.n`` is 1 and the caller
                passed an unbatched input, in ``call.dtype`` on ``call.device``, contiguous
                in that axis order.

        Returns:
            A new contiguous tensor in ``call.dtype`` on ``call.device``, of ``x``'s rank,
            with the trailing two axes replaced by ``(call.out_h, call.out_w)``.
        """


class AdaptiveMaxPool2dFwdInterface(KernelInterface):
    """Adaptive max pooling to a fixed output grid over a CHW or NCHW input, values only."""

    request = AdaptivePool2dCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take each output cell's window maximum; nothing is written in place.

        Output cell ``(i, j)`` takes the maximum over the input rows
        ``floor(i * h_in / out_h)`` up to ``ceil((i + 1) * h_in / out_h)`` and the columns
        given the same way, which is what ``torch.nn.functional.adaptive_max_pool2d`` uses.
        Every window holds at least one position, so no fill value is read. Nothing is
        written in place, and the output aliases no input.

        Args:
            x: The input, ``(call.n, call.c_in, call.h_in, call.w_in)``, or
                ``(call.c_in, call.h_in, call.w_in)`` when ``call.n`` is 1 and the caller
                passed an unbatched input, in ``call.dtype`` on ``call.device``, contiguous
                in that axis order.

        Returns:
            A new contiguous tensor in ``call.dtype`` on ``call.device``, of ``x``'s rank,
            with the trailing two axes replaced by ``(call.out_h, call.out_w)``.
        """


class AdaptiveMaxPool2dIndicesFwdInterface(KernelInterface):
    """Adaptive max pooling over a CHW or NCHW input, values and argmax positions."""

    request = AdaptivePool2dCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take each output cell's window maximum and where it sat; nothing is written in place.

        The windows are the ones ``torch.nn.functional.adaptive_max_pool2d`` uses: cell
        ``(i, j)`` covers the rows ``floor(i * h_in / out_h)`` up to
        ``ceil((i + 1) * h_in / out_h)`` and the columns given the same way. Every window
        holds at least one position. Nothing is written in place, and neither output
        aliases an input.

        Args:
            x: The input, ``(call.n, call.c_in, call.h_in, call.w_in)``, or
                ``(call.c_in, call.h_in, call.w_in)`` when ``call.n`` is 1 and the caller
                passed an unbatched input, in ``call.dtype`` on ``call.device``, contiguous
                in that axis order.

        Returns:
            New contiguous ``(values, indices)`` on ``call.device``, of ``x``'s rank, with
            the trailing two axes replaced by ``(call.out_h, call.out_w)``: ``values`` in
            ``call.dtype``, and ``indices`` in ``int64``, each the flat position of its
            maximum within the input's ``(h_in, w_in)`` plane. A tie takes the lowest
            position.
        """


class MeanPoolingFwdInterface(KernelInterface):
    """The mean of each chunk of a ``(batch, seq, heads, dim)`` tensor's sequence axis."""

    request = MeanPoolingCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, offsets: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Average each chunk over the sequence axis; nothing is written in place.

        Every tensor is contiguous on ``call.device``. A chunk is divided by the tokens it
        actually holds, so a short last chunk averages no padding. Sums accumulate in
        ``call.accum_dtype`` and are cast back at the boundary.

        Args:
            x: ``(call.batch_size, call.seq_len, call.heads, call.dim)`` in ``call.dtype``.
            offsets: ``int32`` ``(call.seq_num + 1,)`` sequence boundaries. Read only when
                ``call.use_offsets``; otherwise it fills the slot and the built program cuts
                the axis into uniform ``call.chunk_size`` chunks.
            indices: ``int32`` ``(call.chunks_per_batch, 2)``, one
                ``(sequence, chunk-within-sequence)`` pair per output chunk. Read under the
                same condition as *offsets*.

        Returns:
            A new ``(call.batch_size, call.chunks_per_batch, call.heads, call.dim)`` tensor
            in ``call.dtype``.
        """
