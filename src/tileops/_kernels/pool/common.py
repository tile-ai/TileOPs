import itertools
import math
from collections.abc import Callable
from typing import Any, ClassVar, NamedTuple

import tilelang.language as T
import torch

from tileops._kernels.constants import STATIC_SHARED_BYTES, VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.pool.call_spec import AdaptivePool2dCall, AvgPoolCall, MaxPoolCall

# Every pool kernel reduces its windows in fp32 and casts back at the store: a narrow
# accumulator loses the low bits of a wide window's sum.
ACCUM_DTYPE = "float"

# The largest input whose max-pool loads carry an L2 eviction hint; a larger one loads with
# the default policy. The 1d and the 2d register kernels cut at the same size.
EVICT_HINT_MAX_BYTES = 128 << 20


def fits_static_shared(elements: int, dtype: str) -> bool:
    """Whether a tile of *elements* fits the shared memory a block gets by default.

    Args:
        elements: Elements the tile holds.
        dtype: TileLang name of the element type.

    Returns:
        True when the tile fits :data:`STATIC_SHARED_BYTES`.
    """
    return elements * getattr(torch, dtype).itemsize <= STATIC_SHARED_BYTES


class WindowSpan(NamedTuple):
    """The stretch of a row one block stages, in the row's own coordinates."""

    # Elements one full-width access covers.
    vector_elems: int
    # Elements staged in front of the block's leftmost window.
    head: int
    # Elements the block stages.
    span: int

    @property
    def vectors(self) -> int:
        return self.span // self.vector_elems


def window_span(
    tile_outputs: int,
    step: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dilation_l: int,
    dtype: str,
) -> WindowSpan:
    """The stretch of a row a block stages to cover ``tile_outputs`` outputs.

    A group of ``vector_elems`` is loaded as one, so the width is narrowed until the
    head, the row end and *step* all divide it and no group straddles the row's edge.

    Args:
        tile_outputs: Outputs one block covers.
        step: Elements between the origins of two consecutive blocks of one row, or 0
            when one block covers the whole row and every origin is the same.
        l_in: Elements one row holds.
        kernel_l: Elements one window reaches over, before dilation.
        stride_l: Elements between two consecutive windows.
        pad_l: Elements the leftmost window reaches in front of the row.
        dilation_l: Elements between two taps of one window.
        dtype: TileLang name of the element type.

    Returns:
        The access width, the elements staged in front of the first window, and the
        elements staged in all.
    """

    def round_up(value: int, step: int) -> int:
        return ((value + step - 1) // step) * step

    vector_elems = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    while vector_elems > 1 and (l_in % vector_elems or step % vector_elems):
        vector_elems //= 2
    head = round_up(pad_l, vector_elems)
    reach = head + (tile_outputs - 1) * stride_l + dilation_l * (kernel_l - 1) + 1
    return WindowSpan(vector_elems, head, round_up(reach, vector_elems))


def axis_inside(
    size_in: int, size_out: int, kernel: int, stride: int, pad: int, dilation: int
) -> bool:
    """Whether every window on this axis lies inside the input.

    True makes the axis's bounds test a compile-time truth, and it drops out.
    """
    return pad == 0 and (size_out - 1) * stride + (kernel - 1) * dilation < size_in


def pool_output_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    dilation: int = 1,
) -> int:
    """The output extent of one axis, as ``torch.nn.functional`` pooling computes it."""
    effective_kernel = dilation * (kernel_size - 1) + 1
    if ceil_mode:
        out = (input_size + 2 * padding - effective_kernel + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - effective_kernel) // stride + 1

    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1

    return max(out, 0)


def register_plane_refusal(call: AvgPoolCall | MaxPoolCall) -> str | None:
    """Why a kernel reading a thread's windows into registers cannot take *call*'s plane.

    A thread takes the outputs one 16-byte load of an input row covers, so the stride must
    divide that load and the outputs must divide into the loads of a row. ``None`` where the
    plane fits.
    """
    if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"requires float16, bfloat16 or float32, got {call.dtype}"
    if len(call.size) != 2:
        return "serves a 2-d window"
    (h_in, w_in), (kernel_h, kernel_w) = call.size, call.window
    (stride_h, stride_w), (pad_h, pad_w) = call.stride, call.pad
    run = VECTOR_ACCESS_BYTES // call.dtype.itemsize
    if run % stride_w:
        return f"requires a stride dividing the {run}-element load, got {stride_w}"
    out_h = (h_in + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (w_in + 2 * pad_w - kernel_w) // stride_w + 1
    # The window-row loop is unrolled, so a taller window stays on the one-output-a-thread
    # kernel.
    max_rows = 16
    # Each element a window reaches past a load is a load of its own.
    max_reach = 1
    if kernel_h > max_rows:
        return f"unrolls at most {max_rows} window rows, got {kernel_h}"
    if pad_w > max_reach or kernel_w - stride_w - pad_w > max_reach:
        return f"reads at most {max_reach} element past a load"
    if not (
        out_h > 0
        and out_w > 0
        and out_w % (run // stride_w) == 0
        and out_w * stride_w <= w_in
        and w_in % run == 0
    ):
        return f"requires whole {run}-element loads along the row"
    return None


class AvgPoolWindow(NamedTuple):
    """One average-pooling problem, and the extents and facts that follow from it.

    The per-axis fields carry one entry per spatial axis, so the same type serves 1d, 2d
    and 3d. A kernel builder is cached on this, so everything it derives is derived here
    once and cannot differ between the builders of one family.
    """

    rows: int
    size: tuple[int, ...]
    kernel: tuple[int, ...]
    stride: tuple[int, ...]
    pad: tuple[int, ...]
    ceil_mode: bool
    count_include_pad: bool
    divisor_override: int | None

    @classmethod
    def of(cls, call: AvgPoolCall) -> "AvgPoolWindow":
        """The window of *call*."""
        return cls(
            rows=call.n * call.c_in,
            size=call.size,
            kernel=call.window,
            stride=call.stride,
            pad=call.pad,
            ceil_mode=call.ceil_mode,
            count_include_pad=call.count_include_pad,
            divisor_override=call.divisor_override,
        )

    @property
    def out(self) -> tuple[int, ...]:
        return tuple(
            pool_output_dim(size, k, s, p, self.ceil_mode)
            for size, k, s, p in zip(self.size, self.kernel, self.stride, self.pad, strict=True)
        )

    @property
    def outputs(self) -> int:
        return self.rows * math.prod(self.out)

    @property
    def window_inside(self) -> bool:
        """Whether every window lies inside the input, so no tap carries a test.

        True also settles the output extent: a window that fits is one the ceil-mode and
        the floor-mode formula both count.
        """
        return all(
            p == 0 and (o - 1) * s + k <= size
            for size, o, k, s, p in zip(
                self.size, self.out, self.kernel, self.stride, self.pad, strict=True
            )
        )

    @property
    def whole_window_divides(self) -> bool:
        """Whether one divisor covers every output.

        Without ceil mode a window reaches ``size + pad`` at the furthest, so counting
        the padding gives every output the whole kernel. Ceil mode can overhang that, and
        uncounted padding shortens the windows that do.
        """
        return self.window_inside or (self.count_include_pad and not self.ceil_mode)

    @property
    def divisor(self) -> int:
        """The divisor every window takes where one covers them all.

        An explicit divisor is used as given, negative included.
        """
        if self.divisor_override is not None:
            return self.divisor_override
        return math.prod(self.kernel)

    @property
    def overlap(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Per axis, the half-open range a window's own divisor counts.

        Uncounted padding cuts the range back to the input itself.
        """
        if self.count_include_pad:
            return (
                tuple(-p for p in self.pad),
                tuple(size + p for size, p in zip(self.size, self.pad, strict=True)),
            )
        return ((0,) * len(self.pad), self.size)


def adaptive_bin(o, size_in: int, size_out: int):
    """Half-open input range `[start, end)` feeding output index ``o``.

    PyTorch partitions each spatial axis as
    ``[floor(o*in/out), ceil((o+1)*in/out))``. Bins are never empty, including
    when ``size_out > size_in``. Evaluated while tracing, so the emitted
    expression is the same as writing the arithmetic inline.

    The products before the division reach ``size_out * (size_in + 1)``; a traced ``o``
    is widened to int64 where that passes int32.
    """
    if not isinstance(o, int) and size_out * (size_in + 1) > 2**31 - 1:
        o = T.cast(o, "int64")
    return (o * size_in) // size_out, ((o + 1) * size_in + size_out - 1) // size_out


def max_adaptive_bin_extent(size_in: int, size_out: int) -> int:
    """Widest bin on this axis — a compile-time bound for a `T.serial` loop.

    `ceil(in/out)` is not a valid bound: in=55/out=7 has a bin of 9 against
    ceil = 8, and expanding in=8/out=12 gives bins of 2 against 1.
    """
    extents = (adaptive_bin(o, size_in, size_out) for o in range(size_out))
    return max(end - start for start, end in extents)


class AdaptivePool2dKernelBase(Kernel):
    """Shared scaffold for the adaptive 2D pool kernels.

    The variants differ only in their reduction and in whether they emit
    indices. Everything around that -- the dtype gate, the config space, the
    argument marshalling -- is the same, so a subclass binds two callables and
    declares nothing else:

    - ``_build`` the cached builder that traces the prim_func.
    - ``_launch`` the run of the compiled program the forward call goes through.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    _build: ClassVar[Callable[..., Any]]

    @classmethod
    def entry_for(cls, call: AdaptivePool2dCall) -> Entry:
        return call, lambda: cls(
            call.n, call.c_in, call.h_in, call.w_in, call.out_h, call.out_w, call.dtype
        )

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        out_h: int,
        out_w: int,
        dtype: torch.dtype,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(f"{type(self).__name__} supports float16 and bfloat16, got {dtype}")
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self.kernel = type(self)._build(n, c_in, h_in, w_in, out_h, out_w, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"block_m": 256, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([128, 256, 512], [128, 256, 512])
        ]

    def forward(self, x: torch.Tensor):
        """Pool ``x``, ``[N, C, H, W]`` or unbatched ``[C, H, W]``; outputs follow its rank."""
        self._require_devices(x=x)
        unbatched = x.dim() == 3
        result = self._launch(self.program(self._compile), x.unsqueeze(0) if unbatched else x)
        if not unbatched:
            return result
        if isinstance(result, (tuple, list)):
            return tuple(t.squeeze(0) for t in result)
        return result.squeeze(0)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)

    def _launch(self, program, x: torch.Tensor) -> Any:
        """Run *program* on the batched ``x`` and shape its outputs as ``[N, C, out_h, out_w]``."""
        raise NotImplementedError


class MaxPoolKernelBase(Kernel):
    """What the 2d and 3d max-pool kernels share: how a call constructs one, the config
    space and the launch.

    A subclass builds ``self.kernel`` in its constructor and supplies ``_launch``.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: MaxPoolCall) -> Entry:
        return call, lambda: cls(
            call.n,
            call.c_in,
            *call.size,
            *call.window,
            *call.stride,
            *call.pad,
            *call.dilation,
            call.ceil_mode,
            call.dtype,
        )

    @property
    def default_config(self) -> dict:
        return {"block_m": 512, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([256, 512, 1024, 2048], [128, 256, 512])
            if threads <= block_m
        ]

    def forward(self, x: torch.Tensor) -> Any:
        self._require_devices(x=x)
        return self._launch(self.program(self._compile), x)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(**cfg)

    def _launch(self, program, x: torch.Tensor) -> Any:
        """Run *program* on ``x`` and shape its outputs as the interface publishes them."""
        raise NotImplementedError
