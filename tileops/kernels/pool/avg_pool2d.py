import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
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
            x: T.Tensor((n, h_in, w_in, c_in), dtype),  # type: ignore
            out: T.Tensor((n, out_h, out_w, c_in), dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(c_in, block_c),
                T.ceildiv(n * out_h * out_w, block_m),
                threads=threads,
            ) as (bx, by):
                out_flat = T.Tensor((n * out_h * out_w, c_in), dtype, out.data)

                for i, j in T.Parallel(block_m, block_c):
                    m_idx = by * block_m + i
                    c_idx = bx * block_c + j
                    if m_idx < n * out_h * out_w and c_idx < c_in:
                        batch = m_idx // (out_h * out_w)
                        out_idx = m_idx % (out_h * out_w)
                        oh = out_idx // out_w
                        ow = out_idx % out_w
                        sum_val = T.alloc_var(T.float32)
                        sum_val = T.cast(0.0, accum_dtype)

                        for kh in T.serial(kernel_h):
                            for kw in T.serial(kernel_w):
                                ih = oh * stride_h + kh - pad_h
                                iw = ow * stride_w + kw - pad_w
                                if ih >= 0 and ih < h_in and iw >= 0 and iw < w_in:
                                    sum_val += T.cast(x[batch, ih, iw, c_idx], accum_dtype)

                        start_h = oh * stride_h - pad_h
                        start_w = ow * stride_w - pad_w
                        end_h = start_h + kernel_h
                        end_w = start_w + kernel_w
                        valid_h = T.max(T.min(end_h, h_in) - T.max(start_h, 0), 0)
                        valid_w = T.max(T.min(end_w, w_in) - T.max(start_w, 0), 0)
                        valid_count = valid_h * valid_w
                        padded_h = T.max(T.min(end_h, h_in + pad_h) - T.max(start_h, -pad_h), 0)
                        padded_w = T.max(T.min(end_w, w_in + pad_w) - T.max(start_w, -pad_w), 0)
                        padded_count = padded_h * padded_w
                        divisor = T.max(
                            T.if_then_else(
                                use_divisor_override,
                                divisor_override,
                                T.if_then_else(count_include_pad, padded_count, valid_count),
                            ),
                            1,
                        )
                        out_flat[m_idx, c_idx] = T.cast(
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
    _ = (count_include_pad, use_divisor_override, divisor_override, dtype, block_m, block_c, threads)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode)
    return torch.empty((n, out_h, out_w, c_in), dtype=x.dtype, device=x.device)


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
