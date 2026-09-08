import functools
import itertools
from typing import Any, Callable, ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import fits_static_shared, pool_output_dim

__all__ = ["MaxPool1dKernel", "MaxPool1dWithIndicesKernel"]


def _window_geometry(
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
) -> Tuple[int, bool]:
    """Outputs along the pooled axis, and whether every window stays in its row.

    The second decides whether a tap carries a bounds test, and both 1d max-pool
    kernels read it, so the test for it has one home.
    """
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    last_tap = (out_l - 1) * stride_w + dilation_w * (kernel_w - 1)
    return out_l, pad_w == 0 and last_tap < l_in


def _stage_rows(
    block_m: int, out_l: int, c_in: int, l_in: int, dtype: str
) -> Optional[Tuple[int, int]]:
    """Rows of one image a block stages, and the width of one staged row, or None.

    None means the block reads global instead, which every shape does unless its
    outputs form whole rows of one image and those rows fit in shared memory. The
    width exceeds `l_in` by the pad that keeps two staged rows off one bank.
    """
    # A warp's width. At or below it a warp spans more than one (batch, channel)
    # row, so neighbouring threads read global memory `l_in` elements apart.
    max_out_l = 32
    # Off a whole number of banks. Measured on an H200; re-measure elsewhere.
    row_pad = 8

    if out_l > max_out_l or block_m % out_l:
        return None
    rows = block_m // out_l
    # Dividing c_in leaves no ragged last block, so the staged body needs no test.
    if rows > c_in or c_in % rows:
        return None
    row_width = l_in + row_pad
    if not fits_static_shared(rows * row_width, dtype):
        return None
    return rows, row_width


@functools.lru_cache(maxsize=32)
def _max_pool1d_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l, always_in_bounds = _window_geometry(
        l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode
    )
    total_output = n * c_in * out_l

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool1d_func(block_m: int, threads: int):
        staged = _stage_rows(block_m, out_l, c_in, l_in, dtype)

        @T.macro
        def _reduce_window_global(src, src_c, src_row, ow, out, out_c, out_row):
            """Store the max over one window, for taps that come from global.

            ``src`` is indexed with a leading axis so the staged tile and ``x`` are
            read the same way.

            NaN enters ``max_val`` and never leaves: a later value fails
            ``val > NaN``, which is how PyTorch propagates it. Spending a select per
            tap to save the accumulator's register is the trade a scan that waits on
            memory wants.
            """
            max_val = T.alloc_var(T.float32)
            max_val = T.cast(float("-inf"), accum_dtype)
            for kw in T.serial(kernel_w):
                iw = ow * stride_w - pad_w + kw * dilation_w
                if always_in_bounds or (iw >= 0 and iw < l_in):
                    val = T.cast(src[src_c, src_row, iw], accum_dtype)
                    max_val = T.if_then_else(T.isnan(val) or (val > max_val), val, max_val)

            out[out_c, out_row, ow] = T.cast(max_val, dtype)

        @T.macro
        def _reduce_window_shared(src, src_c, src_row, ow, out, out_c, out_row):
            """Store the max over one window, for taps that come from shared.

            ``src`` is indexed with a leading axis so the staged tile and ``x`` are
            read the same way.

            Resident taps make the scan arithmetic-bound, which wants the opposite
            trade: ``T.max`` drops NaN, so NaN rides in an accumulator instead of
            costing a select per tap.
            """
            max_val = T.alloc_var(T.float32)
            has_nan = T.alloc_var(T.bool)
            max_val = T.cast(float("-inf"), accum_dtype)
            has_nan = False
            for kw in T.serial(kernel_w):
                iw = ow * stride_w - pad_w + kw * dilation_w
                if always_in_bounds or (iw >= 0 and iw < l_in):
                    val = T.cast(src[src_c, src_row, iw], accum_dtype)
                    has_nan = has_nan | T.isnan(val)
                    max_val = T.max(max_val, val)

            out[out_c, out_row, ow] = T.cast(
                T.if_then_else(has_nan, T.cast(float("nan"), accum_dtype), max_val), dtype
            )

        @T.prim_func
        def _max_pool1d_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_l), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                if staged is None:
                    for i in T.Parallel(block_m):
                        out_idx = bx * block_m + i
                        if out_idx < total_output:
                            ow = out_idx % out_l
                            channel_batch_idx = out_idx // out_l
                            c_idx = channel_batch_idx % c_in
                            batch = channel_batch_idx // c_in
                            _reduce_window_global(x, batch, c_idx, ow, out, batch, c_idx)
                else:
                    stage_rows, row_width = staged
                    tile = T.alloc_shared((1, stage_rows, row_width), dtype)
                    batch = bx * stage_rows // c_in
                    c_base = bx * stage_rows % c_in
                    T.copy(x[batch, c_base : c_base + stage_rows, 0:l_in], tile[0, :, 0:l_in])
                    for i in T.Parallel(block_m):
                        ow = i % out_l
                        row = i // out_l
                        _reduce_window_shared(tile, 0, row, ow, out, batch, c_base + row)

        return _max_pool1d_main

    return _max_pool1d_func


def _launch_max_pool1d(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _max_pool1d_kernel(
        n,
        c_in,
        l_in,
        kernel_w,
        stride_w,
        pad_w,
        dilation_w,
        ceil_mode,
        dtype,
    )(block_m, threads)(x)


class _MaxPool1dKernelBase(Kernel):
    """Shared construction and dispatch for the 1d max-pool kernels.

    Concrete kernels supply ``_build`` and ``_dispatch``; everything else —
    parameter capture, output extents, config and launch — is identical
    between the value-only and with-indices variants.
    """

    _build: ClassVar[Callable[..., Any]]
    _dispatch: ClassVar[Callable[..., Any]]

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_w: int,
        stride_w: int,
        pad_w: int,
        dilation_w: int,
        ceil_mode: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(
                f"{type(self).__name__} supports float16, bfloat16, and float32, got {dtype}"
            )
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_w = kernel_w
        self.stride_w = stride_w
        self.pad_w = pad_w
        self.dilation_w = dilation_w
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
        self.kernel = type(self)._build(
            n,
            c_in,
            l_in,
            kernel_w,
            stride_w,
            pad_w,
            dilation_w,
            ceil_mode,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 256,
            "threads": 256,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # `block_m` counts outputs, and a row of a real workload runs into the
        # thousands of them, so the widest block here is four per thread at 256.
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([128, 256, 512, 1024], [128, 256, 512])
        ]

    def forward(self, x: torch.Tensor) -> Any:
        self._require_cuda(x=x)
        return type(self)._dispatch(
            self.n,
            self.c_in,
            self.l_in,
            self.kernel_w,
            self.stride_w,
            self.pad_w,
            self.dilation_w,
            self.ceil_mode,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )


class MaxPool1dKernel(_MaxPool1dKernelBase):
    """Max pooling forward kernel (return_indices=False)."""

    _build = staticmethod(_max_pool1d_kernel)
    _dispatch = staticmethod(_launch_max_pool1d)


@functools.lru_cache(maxsize=32)
def _max_pool1d_with_indices_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l, always_in_bounds = _window_geometry(
        l_in, kernel_w, stride_w, pad_w, dilation_w, ceil_mode
    )
    total_output = n * c_in * out_l

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool1d_with_indices_func(block_m: int, threads: int):
        staged = _stage_rows(block_m, out_l, c_in, l_in, dtype)

        @T.macro
        def _reduce_window(src, src_c, src_row, ow, out, indices, out_c, out_row):
            """Store the max and its index over one window of ``src[src_c, src_row]``."""
            max_val = T.alloc_var(T.float32)
            max_idx = T.alloc_var(T.int32)
            # -1 until a NaN is seen, so it is also the flag saying one was.
            nan_idx = T.alloc_var(T.int32)
            max_val = T.cast(float("-inf"), accum_dtype)
            nan_idx = -1
            if always_in_bounds:
                # An expression, not a variable: a variable is opaque to the range
                # analysis, and each tap would load under a bounds check this
                # window's extent rules out.
                iw0 = ow * stride_w - pad_w
                max_idx = iw0
            else:
                # A variable, because each tap is bounds-tested anyway and this
                # then stays out of all of them.
                iw0 = T.alloc_var(T.int32)
                iw0 = ow * stride_w - pad_w
                # The first tap the row holds, on the dilation grid. Only padding
                # starts a window before position 0, and that tap is the position
                # PyTorch reports when every tap the row holds is -inf.
                max_idx = iw0 + dilation_w * T.ceildiv(T.max(-iw0, 0), dilation_w)
            for kw in T.serial(kernel_w):
                iw = iw0 + kw * dilation_w
                if always_in_bounds or (iw >= 0 and iw < l_in):
                    val = T.cast(src[src_c, src_row, iw], accum_dtype)
                    # `max_val` is never NaN and NaN fails `>`, so the compare
                    # rejects NaN without a separate test. Strict `>` also keeps
                    # the seed against a tap equal to it, and the first maximum
                    # against a later equal one, which is what PyTorch reports.
                    take = val > max_val
                    max_val = T.if_then_else(take, val, max_val)
                    max_idx = T.if_then_else(take, iw, max_idx)
                    nan_idx = T.if_then_else(T.isnan(val), iw, nan_idx)

            # PyTorch reports the last NaN a window visited.
            out[out_c, out_row, ow] = T.cast(
                T.if_then_else(nan_idx >= 0, T.cast(float("nan"), accum_dtype), max_val), dtype
            )
            indices[out_c, out_row, ow] = T.cast(
                T.if_then_else(nan_idx >= 0, nan_idx, max_idx), "int64"
            )

        @T.prim_func
        def _max_pool1d_with_indices_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_l), dtype),  # type: ignore
            indices: T.Tensor((n, c_in, out_l), "int64"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                if staged is None:
                    for i in T.Parallel(block_m):
                        out_idx = bx * block_m + i
                        if out_idx < total_output:
                            ow = out_idx % out_l
                            channel_batch_idx = out_idx // out_l
                            c_idx = channel_batch_idx % c_in
                            batch = channel_batch_idx // c_in
                            _reduce_window(x, batch, c_idx, ow, out, indices, batch, c_idx)
                else:
                    stage_rows, row_width = staged
                    tile = T.alloc_shared((1, stage_rows, row_width), dtype)
                    batch = bx * stage_rows // c_in
                    c_base = bx * stage_rows % c_in
                    T.copy(x[batch, c_base : c_base + stage_rows, 0:l_in], tile[0, :, 0:l_in])
                    for i in T.Parallel(block_m):
                        ow = i % out_l
                        row = i // out_l
                        _reduce_window(tile, 0, row, ow, out, indices, batch, c_base + row)

        return _max_pool1d_with_indices_main

    return _max_pool1d_with_indices_func


def _launch_max_pool1d_with_indices(
    n: int,
    c_in: int,
    l_in: int,
    kernel_w: int,
    stride_w: int,
    pad_w: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _max_pool1d_with_indices_kernel(
        n,
        c_in,
        l_in,
        kernel_w,
        stride_w,
        pad_w,
        dilation_w,
        ceil_mode,
        dtype,
    )(block_m, threads)(x)


class MaxPool1dWithIndicesKernel(_MaxPool1dKernelBase):
    """Max pooling forward-with-indices kernel."""

    _build = staticmethod(_max_pool1d_with_indices_kernel)
    _dispatch = staticmethod(_launch_max_pool1d_with_indices)
