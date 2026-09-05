import functools
import itertools
from typing import Any, Callable, ClassVar, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import fits_static_shared, pool_output_dim

__all__ = ["MaxPool1dKernel", "MaxPool1dWithIndicesKernel"]


# One warp's width. At or below it a warp spans more than one (batch, channel)
# row, so neighbouring threads read global memory `l_in` elements apart.
_STAGE_MAX_OUT_L = 32
# Pad the shared row off a whole number of banks, or the staged rows collide.
# The width is measured, on an H200; re-measure it on other hardware.
_STAGE_ROW_PAD = 8


def _stage_rows(block_m: int, out_l: int, c_in: int, l_in: int, dtype: str) -> Optional[int]:
    """Rows of one image a block stages in shared memory, or None to read global."""
    if out_l > _STAGE_MAX_OUT_L or block_m % out_l:
        return None
    rows = block_m // out_l
    # Dividing c_in leaves no ragged last block, so the staged body needs no test.
    if rows > c_in or c_in % rows:
        return None
    if not fits_static_shared(rows * (l_in + _STAGE_ROW_PAD), dtype):
        return None
    return rows


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
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    total_output = n * c_in * out_l

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool1d_func(block_m: int, threads: int):
        stage_rows = _stage_rows(block_m, out_l, c_in, l_in, dtype)

        @T.macro
        def _reduce_window(src, src_c, src_row, ow, out, out_c, out_row):
            """Store the max over one window of ``src[src_c, src_row]``."""
            max_val = T.alloc_var(T.float32)
            has_nan = T.alloc_var(T.bool)
            max_val = T.cast(float("-inf"), accum_dtype)
            has_nan = False
            for kw in T.serial(kernel_w):
                iw = ow * stride_w - pad_w + kw * dilation_w
                if iw >= 0 and iw < l_in:
                    val = T.cast(src[src_c, src_row, iw], accum_dtype)
                    if T.isnan(val):
                        has_nan = True
                    max_val = T.max(max_val, val)

            result = T.if_then_else(
                has_nan,
                T.cast(float("nan"), accum_dtype),
                max_val,
            )
            out[out_c, out_row, ow] = T.cast(result, dtype)

        @T.prim_func
        def _max_pool1d_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_l), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                for i in T.Parallel(block_m):
                    out_idx = bx * block_m + i
                    if out_idx < total_output:
                        ow = out_idx % out_l
                        channel_batch_idx = out_idx // out_l
                        c_idx = channel_batch_idx % c_in
                        batch = channel_batch_idx // c_in
                        _reduce_window(x, batch, c_idx, ow, out, batch, c_idx)

        if stage_rows is not None:
            # Leading axis of 1 so the macro indexes the tile and x alike.
            @T.prim_func
            def _max_pool1d_staged_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                out: T.Tensor((n, c_in, out_l), dtype),  # type: ignore
            ):
                with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                    tile = T.alloc_shared((1, stage_rows, l_in + _STAGE_ROW_PAD), dtype)
                    batch = bx * stage_rows // c_in
                    c_base = bx * stage_rows % c_in
                    T.copy(x[batch, c_base : c_base + stage_rows, 0:l_in], tile[0, :, 0:l_in])
                    for i in T.Parallel(block_m):
                        ow = i % out_l
                        row = i // out_l
                        _reduce_window(tile, 0, row, ow, out, batch, c_base + row)

            return _max_pool1d_staged_main

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
        return [
            {"block_m": block_m, "threads": threads}
            for block_m, threads in itertools.product([128, 256, 512], [128, 256, 512])
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
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    total_output = n * c_in * out_l
    # Static specialization: with zero padding and no ceil overshoot every window
    # lies fully inside the input, so the per-element bounds check can be dropped.
    always_in_bounds = pad_w == 0 and (out_l - 1) * stride_w + (kernel_w - 1) * dilation_w < l_in

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool1d_with_indices_func(block_m: int, threads: int):
        stage_rows = _stage_rows(block_m, out_l, c_in, l_in, dtype)

        @T.macro
        def _reduce_window(src, src_c, src_row, ow, out, indices, out_c, out_row):
            """Store the max and its index over one window of ``src[src_c, src_row]``."""
            max_val = T.alloc_var(T.float32)
            has_nan = T.alloc_var(T.bool)
            max_idx = T.alloc_var(T.int32)
            nan_idx = T.alloc_var(T.int32)
            first_valid = T.alloc_var(T.bool)
            # Loop-invariant window corner, materialized so it is not
            # re-inlined into every window element.
            iw0 = T.alloc_var(T.int32)
            max_val = T.cast(float("-inf"), accum_dtype)
            has_nan = False
            first_valid = True
            iw0 = ow * stride_w - pad_w
            if always_in_bounds:
                # Window element 0 is in bounds here, so its flat index is
                # the correct seed: an all--inf window reports the first
                # position, matching PyTorch, and first_valid is unneeded.
                max_idx = iw0
                nan_idx = iw0
            else:
                max_idx = 0
                nan_idx = 0
            for kw in T.serial(kernel_w):
                iw = iw0 + kw * dilation_w
                if always_in_bounds:
                    val = T.cast(src[src_c, src_row, iw], accum_dtype)
                    is_nan = T.isnan(val)
                    # Branch-free update. Strict > keeps the first
                    # maximum; NaN never touches max_val/max_idx and
                    # records the last NaN visited, matching PyTorch.
                    take = (not is_nan) and (val > max_val)
                    max_val = T.if_then_else(take, val, max_val)
                    max_idx = T.if_then_else(take, iw, max_idx)
                    nan_idx = T.if_then_else(is_nan, iw, nan_idx)
                    has_nan = has_nan or is_nan
                elif iw >= 0 and iw < l_in:
                    val = T.cast(src[src_c, src_row, iw], accum_dtype)
                    is_nan = T.isnan(val)
                    take = (not is_nan) and (first_valid or (val > max_val))
                    max_val = T.if_then_else(take, val, max_val)
                    max_idx = T.if_then_else(take, iw, max_idx)
                    first_valid = first_valid and is_nan
                    nan_idx = T.if_then_else(is_nan, iw, nan_idx)
                    has_nan = has_nan or is_nan

            result = T.if_then_else(
                has_nan,
                T.cast(float("nan"), accum_dtype),
                max_val,
            )
            out[out_c, out_row, ow] = T.cast(result, dtype)
            indices[out_c, out_row, ow] = T.cast(T.if_then_else(has_nan, nan_idx, max_idx), "int64")

        @T.prim_func
        def _max_pool1d_with_indices_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_l), dtype),  # type: ignore
            indices: T.Tensor((n, c_in, out_l), "int64"),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                for i in T.Parallel(block_m):
                    out_idx = bx * block_m + i
                    if out_idx < total_output:
                        ow = out_idx % out_l
                        channel_batch_idx = out_idx // out_l
                        c_idx = channel_batch_idx % c_in
                        batch = channel_batch_idx // c_in
                        _reduce_window(x, batch, c_idx, ow, out, indices, batch, c_idx)

        if stage_rows is not None:
            # Leading axis of 1 so the macro indexes the tile and x alike.
            @T.prim_func
            def _max_pool1d_with_indices_staged_main(
                x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
                out: T.Tensor((n, c_in, out_l), dtype),  # type: ignore
                indices: T.Tensor((n, c_in, out_l), "int64"),  # type: ignore
            ):
                with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                    tile = T.alloc_shared((1, stage_rows, l_in + _STAGE_ROW_PAD), dtype)
                    batch = bx * stage_rows // c_in
                    c_base = bx * stage_rows % c_in
                    T.copy(x[batch, c_base : c_base + stage_rows, 0:l_in], tile[0, :, 0:l_in])
                    for i in T.Parallel(block_m):
                        ow = i % out_l
                        row = i // out_l
                        _reduce_window(tile, 0, row, ow, out, indices, batch, c_base + row)

            return _max_pool1d_with_indices_staged_main

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
