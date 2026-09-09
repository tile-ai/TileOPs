import itertools
from typing import Any, Callable, ClassVar, NamedTuple, Optional

import torch

from tileops.kernels.constants import STATIC_SHARED_BYTES, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel


def dtype_itemsize(dtype: str) -> int:
    """Bytes one element of *dtype* takes, over the dtypes these kernels accept."""
    return 4 if dtype in ("float", "float32") else 2


def fits_static_shared(elements: int, dtype: str) -> bool:
    """Whether a tile of *elements* fits the shared memory a block gets by default.

    Args:
        elements: Elements the tile holds.
        dtype: TileLang name of the element type.

    Returns:
        True when the tile fits :data:`STATIC_SHARED_BYTES`.
    """
    return elements * dtype_itemsize(dtype) <= STATIC_SHARED_BYTES


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


def round_up(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step


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
    vector_elems = VECTOR_ACCESS_BYTES // dtype_itemsize(dtype)
    while vector_elems > 1 and (l_in % vector_elems or step % vector_elems):
        vector_elems //= 2
    head = round_up(pad_l, vector_elems)
    reach = head + (tile_outputs - 1) * stride_l + dilation_l * (kernel_l - 1) + 1
    return WindowSpan(vector_elems, head, round_up(reach, vector_elems))


def pool_output_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
    dilation: int = 1,
) -> int:
    effective_kernel = dilation * (kernel_size - 1) + 1
    if ceil_mode:
        out = (input_size + 2 * padding - effective_kernel + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - effective_kernel) // stride + 1

    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1

    return max(out, 0)


def adaptive_bin(o, size_in: int, size_out: int):
    """Half-open input range `[start, end)` feeding output index ``o``.

    PyTorch partitions each spatial axis as
    ``[floor(o*in/out), ceil((o+1)*in/out))``. Bins are never empty, including
    when ``size_out > size_in``. Evaluated while tracing, so the emitted
    expression is the same as writing the arithmetic inline.
    """
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
    - ``_dispatch`` the launch the forward call goes through.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    _build: ClassVar[Callable[..., Any]]
    _dispatch: ClassVar[Callable[..., Any]]

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        out_h: int,
        out_w: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
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
        self.init_config(config, tune)

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
        self._require_cuda(x=x)
        return type(self)._dispatch(
            self.n,
            self.c_in,
            self.h_in,
            self.w_in,
            self.out_h,
            self.out_w,
            self.dtype_str,
            self.config,
            x,
        )
