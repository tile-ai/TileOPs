import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import fits_static_shared, pool_output_dim

__all__ = ["AvgPool1dKernel", "AvgPool1dSpatialKernel"]


@functools.lru_cache(maxsize=64)
def _avg_pool1d_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, ceil_mode)
    rows = n * c_in
    total = rows * out_l
    window_inside = pad_l == 0 and (out_l - 1) * stride_l + kernel_l <= l_in
    # Otherwise a window can overhang, and the divisor comes from its own extent.
    whole_window_divides = window_inside or (count_include_pad and not ceil_mode)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool1d_func(block_m: int, threads: int):
        tile_full = total % block_m == 0
        # Neighbouring outputs read windows `stride_l` apart, so a warp's taps span
        # several lines; a block holding a whole row reads it once instead.
        stage_row = fits_static_shared(l_in, dtype)

        @T.macro
        def _mean_window(src, src_row, ol, out, out_row):
            """Store the mean of one window of ``src[src_row]``.

            ``src`` is indexed with a leading axis so the staged tile and ``x`` are
            read the same way.
            """
            start = ol * stride_l - pad_l
            total_val = T.alloc_var(T.float32)
            total_val = T.cast(0.0, accum_dtype)
            for k in T.serial(kernel_l):
                il = start + k
                if window_inside:
                    total_val += T.cast(src[src_row, il], accum_dtype)
                else:
                    # A select, not a branch, so every thread walks the same window.
                    # The clamp only keeps the discarded read in range.
                    total_val += T.if_then_else(
                        (il >= 0) and (il < l_in),
                        T.cast(src[src_row, T.max(0, T.min(il, l_in - 1))], accum_dtype),
                        T.cast(0.0, accum_dtype),
                    )
            if whole_window_divides:
                divisor = T.cast(kernel_l, accum_dtype)
            elif count_include_pad:
                divisor = T.cast(
                    T.max(T.min(start + kernel_l, l_in + pad_l) - T.max(start, -pad_l), 1),
                    accum_dtype,
                )
            else:
                divisor = T.cast(
                    T.max(T.min(start + kernel_l, l_in) - T.max(start, 0), 1),
                    accum_dtype,
                )
            out[out_row, ol] = T.cast(total_val / divisor, dtype)

        @T.prim_func
        def _avg_pool1d_main(
            x: T.Tensor((rows, l_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_l), dtype),  # type: ignore
        ):
            grid = rows if stage_row else T.ceildiv(total, block_m)
            with T.Kernel(grid, threads=threads) as bx:
                if stage_row:
                    tile = T.alloc_shared((1, l_in), dtype)
                    T.copy(x[bx, 0:l_in], tile[0, 0:l_in])
                    for ol in T.Parallel(out_l):
                        _mean_window(tile, 0, ol, out, bx)
                else:
                    for i in T.Parallel(block_m):
                        idx = bx * block_m + i
                        if tile_full or idx < total:
                            row = idx // out_l
                            _mean_window(x, row, idx % out_l, out, row)

        return _avg_pool1d_main

    return _avg_pool1d_func


def _avg_pool1d_spatial_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dtype: str = "float16",
):
    """Zero-padded, floor-mode 1d average pooling.

    Every window then spans the full kernel once the padding is counted, which is
    what ``_avg_pool1d_kernel`` emits for these two flags.
    """
    return _avg_pool1d_kernel(n, c_in, l_in, kernel_l, stride_l, pad_l, False, True, dtype)


def _launch_avg_pool1d(
    n: int,
    c_in: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    ceil_mode: bool,
    count_include_pad: bool,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, ceil_mode)
    kernel = _avg_pool1d_kernel(
        n,
        c_in,
        l_in,
        kernel_l,
        stride_l,
        pad_l,
        ceil_mode,
        count_include_pad,
        dtype,
    )(block_m, threads)

    return kernel(x.contiguous().view(n * c_in, l_in)).view(n, c_in, out_l)


def _launch_avg_pool1d_spatial(
    n: int,
    c_in: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _launch_avg_pool1d(
        n, c_in, l_in, kernel_l, stride_l, pad_l, False, True, dtype, block_m, threads, x
    )


class AvgPool1dSpatialKernel(Kernel):
    """Fast path for common NCL avg_pool1d workloads."""

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_l: int,
        stride_l: int,
        pad_l: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.dtype = dtype
        self.out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, False)

        self.kernel = _avg_pool1d_spatial_kernel(
            n,
            c_in,
            l_in,
            kernel_l,
            stride_l,
            pad_l,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        return _launch_avg_pool1d_spatial(
            self.n,
            self.c_in,
            self.l_in,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )


class AvgPool1dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_l: int,
        stride_l: int,
        pad_l: int,
        ceil_mode: bool,
        count_include_pad: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_l = kernel_l
        self.stride_l = stride_l
        self.pad_l = pad_l
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype
        self.out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, ceil_mode)

        self.kernel = _avg_pool1d_kernel(
            n,
            c_in,
            l_in,
            kernel_l,
            stride_l,
            pad_l,
            ceil_mode,
            count_include_pad,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        return _launch_avg_pool1d(
            self.n,
            self.c_in,
            self.l_in,
            self.kernel_l,
            self.stride_l,
            self.pad_l,
            self.ceil_mode,
            self.count_include_pad,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )
