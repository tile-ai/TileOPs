import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

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
    total_output = n * c_in * out_l

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool1d_func(block_m: int, threads: int):
        @T.prim_func
        def _avg_pool1d_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_l), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                T.use_swizzle(10)
                tile_out_start = bx * block_m
                tile_out_end = tile_out_start + block_m - 1
                tile_ol_start = tile_out_start % out_l
                tile_ol_end = tile_out_end % out_l
                tile_same_row = tile_out_start // out_l == tile_out_end // out_l
                tile_input_start = tile_ol_start * stride_l - pad_l
                tile_input_end = tile_ol_end * stride_l + kernel_l - 1 - pad_l
                tile_spatial_full = (
                    tile_same_row
                    & (tile_input_start >= 0)
                    & (tile_input_end < l_in)
                )
                for i in T.Parallel(block_m):
                    out_idx = bx * block_m + i
                    if out_idx < total_output:
                        ol = out_idx % out_l
                        nc_idx = out_idx // out_l
                        c_idx = nc_idx % c_in
                        batch = nc_idx // c_in

                        sum_val = T.alloc_var(T.float32)
                        sum_val = T.cast(0.0, accum_dtype)

                        if tile_spatial_full:
                            for kw in T.serial(kernel_l):
                                il = ol * stride_l + kw - pad_l
                                sum_val += T.cast(x[batch, c_idx, il], accum_dtype)
                            out[batch, c_idx, ol] = T.cast(
                                sum_val / T.cast(kernel_l, accum_dtype),
                                dtype,
                            )
                        else:
                            for kw in T.serial(kernel_l):
                                il = ol * stride_l + kw - pad_l
                                if il >= 0 and il < l_in:
                                    sum_val += T.cast(x[batch, c_idx, il], accum_dtype)

                            window_start = ol * stride_l - pad_l
                            window_end = window_start + kernel_l
                            valid_start = T.max(window_start, 0)
                            valid_end = T.min(window_end, l_in)
                            valid_count = T.max(valid_end - valid_start, 0)
                            padded_start = T.max(window_start, -pad_l)
                            padded_end = T.min(window_end, l_in + pad_l)
                            padded_count = T.max(padded_end - padded_start, 0)
                            divisor = T.max(
                                T.if_then_else(count_include_pad, padded_count, valid_count),
                                1,
                            )
                            out[batch, c_idx, ol] = T.cast(
                                sum_val / T.cast(divisor, accum_dtype),
                                dtype,
                            )

        return _avg_pool1d_main

    return _avg_pool1d_func


@functools.lru_cache(maxsize=64)
def _avg_pool1d_spatial_kernel(
    n: int,
    c_in: int,
    l_in: int,
    kernel_l: int,
    stride_l: int,
    pad_l: int,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, False)
    total_output = n * c_in * out_l

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool1d_spatial_func(block_m: int, threads: int):
        @T.prim_func
        def _avg_pool1d_spatial_main(
            x: T.Tensor((n, c_in, l_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_l), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                T.use_swizzle(10)
                tile_out_start = bx * block_m
                tile_out_end = tile_out_start + block_m - 1
                tile_ol_start = tile_out_start % out_l
                tile_ol_end = tile_out_end % out_l
                tile_same_row = tile_out_start // out_l == tile_out_end // out_l
                tile_input_start = tile_ol_start * stride_l - pad_l
                tile_input_end = tile_ol_end * stride_l + kernel_l - 1 - pad_l
                tile_spatial_full = (
                    tile_same_row
                    & (tile_input_start >= 0)
                    & (tile_input_end < l_in)
                )
                for i in T.Parallel(block_m):
                    out_idx = bx * block_m + i
                    if out_idx < total_output:
                        ol = out_idx % out_l
                        nc_idx = out_idx // out_l
                        c_idx = nc_idx % c_in
                        batch = nc_idx // c_in

                        sum_val = T.alloc_var(T.float32)
                        sum_val = T.cast(0.0, accum_dtype)

                        if tile_spatial_full:
                            for kw in T.serial(kernel_l):
                                il = ol * stride_l + kw - pad_l
                                sum_val += T.cast(x[batch, c_idx, il], accum_dtype)
                        else:
                            for kw in T.serial(kernel_l):
                                il = ol * stride_l + kw - pad_l
                                if il >= 0 and il < l_in:
                                    sum_val += T.cast(x[batch, c_idx, il], accum_dtype)

                        out[batch, c_idx, ol] = T.cast(
                            sum_val / T.cast(kernel_l, accum_dtype),
                            dtype,
                        )

        return _avg_pool1d_spatial_main

    return _avg_pool1d_spatial_func


@torch.library.custom_op("top::avg_pool1d_spatial_wrapped_kernel", mutates_args=())
def _avg_pool1d_spatial_wrapped_kernel(
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
    return _avg_pool1d_spatial_kernel(
        n,
        c_in,
        l_in,
        kernel_l,
        stride_l,
        pad_l,
        dtype,
    )(block_m, threads)(x)


@_avg_pool1d_spatial_wrapped_kernel.register_fake
def _(
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
    _ = (dtype, block_m, threads)
    out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, False)
    return torch.empty((n, c_in, out_l), dtype=x.dtype, device=x.device)


@torch.library.custom_op("top::avg_pool1d_wrapped_kernel", mutates_args=())
def _avg_pool1d_wrapped_kernel(
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
    return _avg_pool1d_kernel(
        n,
        c_in,
        l_in,
        kernel_l,
        stride_l,
        pad_l,
        ceil_mode,
        count_include_pad,
        dtype,
    )(block_m, threads)(x)


@_avg_pool1d_wrapped_kernel.register_fake
def _(
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
    _ = (count_include_pad, dtype, block_m, threads)
    out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, ceil_mode)
    return torch.empty((n, c_in, out_l), dtype=x.dtype, device=x.device)


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
        return _avg_pool1d_spatial_wrapped_kernel(
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
        return _avg_pool1d_wrapped_kernel(
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
