import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["AvgPool2dKernel"]


@functools.lru_cache(maxsize=64)
def _avg_pool2d_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    ceil_mode: bool,
    count_include_pad: bool,
    use_divisor_override: bool,
    divisor_override: int,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool2d_func(block_m: int, block_c: int, threads: int):
        @T.prim_func
        def _avg_pool2d_main(
            x: T.Tensor((n, c_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(out_h * out_w, block_m),
                T.ceildiv(c_in, block_c),
                n,
                threads=threads,
            ) as (bx, by, bz):
                T.use_swizzle(10)
                tile_spatial_start = bx * block_m
                tile_spatial_end = tile_spatial_start + block_m - 1
                tile_oh_start = tile_spatial_start // out_w
                tile_oh_end = tile_spatial_end // out_w
                tile_spatial_same_row = tile_oh_start == tile_oh_end
                tile_ow_start = tile_spatial_start % out_w
                tile_ow_end = tile_spatial_end % out_w
                tile_input_h_start = tile_oh_start * stride_h - pad_h
                tile_input_h_end = tile_oh_end * stride_h + kernel_h - 1 - pad_h
                tile_input_w_start = tile_ow_start * stride_w - pad_w
                tile_input_w_end = tile_ow_end * stride_w + kernel_w - 1 - pad_w
                tile_spatial_full = (
                    tile_spatial_same_row
                    & (tile_input_h_start >= 0)
                    & (tile_input_h_end < h_in)
                    & (tile_input_w_start >= 0)
                    & (tile_input_w_end < w_in)
                )
                use_fixed_kernel_divisor = count_include_pad and not use_divisor_override

                for i, j in T.Parallel(block_m, block_c):
                    out_spatial = bx * block_m + i
                    oh = out_spatial // out_w
                    ow = out_spatial % out_w
                    c_idx = by * block_c + j
                    batch = bz
                    if out_spatial < out_h * out_w and c_idx < c_in:
                        sum_val = T.alloc_var(T.float32)
                        sum_val = T.cast(0.0, accum_dtype)

                        if tile_spatial_full and use_fixed_kernel_divisor:
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    ih = oh * stride_h + kh - pad_h
                                    iw = ow * stride_w + kw - pad_w
                                    sum_val += T.cast(
                                        x[batch, c_idx, ih, iw], accum_dtype
                                    )
                            out[batch, c_idx, oh, ow] = T.cast(
                                sum_val / T.cast(kernel_h * kernel_w, accum_dtype),
                                dtype,
                            )
                        else:
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    ih = oh * stride_h + kh - pad_h
                                    iw = ow * stride_w + kw - pad_w
                                    if ih >= 0 and ih < h_in and iw >= 0 and iw < w_in:
                                        sum_val += T.cast(
                                            x[batch, c_idx, ih, iw], accum_dtype
                                        )

                            start_h = oh * stride_h - pad_h
                            start_w = ow * stride_w - pad_w
                            end_h = start_h + kernel_h
                            end_w = start_w + kernel_w
                            valid_h = T.max(T.min(end_h, h_in) - T.max(start_h, 0), 0)
                            valid_w = T.max(T.min(end_w, w_in) - T.max(start_w, 0), 0)
                            valid_count = valid_h * valid_w
                            padded_h = T.max(
                                T.min(end_h, h_in + pad_h) - T.max(start_h, -pad_h), 0
                            )
                            padded_w = T.max(
                                T.min(end_w, w_in + pad_w) - T.max(start_w, -pad_w), 0
                            )
                            padded_count = padded_h * padded_w
                            auto_divisor = T.max(
                                T.if_then_else(
                                    count_include_pad, padded_count, valid_count
                                ),
                                1,
                            )
                            divisor = T.if_then_else(
                                use_divisor_override,
                                divisor_override,
                                auto_divisor,
                            )
                            out[batch, c_idx, oh, ow] = T.cast(
                                sum_val / T.cast(divisor, accum_dtype),
                                dtype,
                            )

        return _avg_pool2d_main

    return _avg_pool2d_func


@torch.library.custom_op("top::avg_pool2d_wrapped_kernel", mutates_args=())
def _avg_pool2d_wrapped_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    ceil_mode: bool,
    count_include_pad: bool,
    use_divisor_override: bool,
    divisor_override: int,
    dtype: str,
    block_m: int,
    block_c: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _avg_pool2d_kernel(
        n,
        c_in,
        h_in,
        w_in,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        ceil_mode,
        count_include_pad,
        use_divisor_override,
        divisor_override,
        dtype,
    )(block_m, block_c, threads)(x)


@_avg_pool2d_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    ceil_mode: bool,
    count_include_pad: bool,
    use_divisor_override: bool,
    divisor_override: int,
    dtype: str,
    block_m: int,
    block_c: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (
        count_include_pad,
        use_divisor_override,
        divisor_override,
        dtype,
        block_m,
        block_c,
        threads,
    )
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode)
    return torch.empty((n, c_in, out_h, out_w), dtype=x.dtype, device=x.device)


class AvgPool2dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.use_divisor_override = divisor_override is not None
        self.divisor_override = divisor_override or 0
        self.dtype = dtype
        self.out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode)
        self.out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode)

        self.kernel = _avg_pool2d_kernel(
            n,
            c_in,
            h_in,
            w_in,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            ceil_mode,
            count_include_pad,
            self.use_divisor_override,
            self.divisor_override,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_c": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = itertools.product([64, 128, 256], [32, 64, 128], [128, 256])
        return [
            {
                "block_m": block_m,
                "block_c": block_c,
                "threads": threads,
            }
            for block_m, block_c, threads in configs
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _avg_pool2d_wrapped_kernel(
            self.n,
            self.c_in,
            self.h_in,
            self.w_in,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.ceil_mode,
            self.count_include_pad,
            self.use_divisor_override,
            self.divisor_override,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_c"],
            self.config["threads"],
            x,
        )
