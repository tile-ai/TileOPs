import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["AvgPool1dKernel"]


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

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool1d_func(block_m: int, block_c: int, threads: int):
        @T.prim_func
        def _avg_pool1d_main(
            x: T.Tensor((n, l_in, c_in), dtype),  # type: ignore
            out: T.Tensor((n, out_l, c_in), dtype),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(c_in, block_c),
                T.ceildiv(n * out_l, block_m),
                threads=threads,
            ) as (bx, by):
                out_local = T.alloc_fragment((block_m, block_c), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_c), dtype)
                out_flat = T.Tensor((n * out_l, c_in), dtype, out.data)

                T.clear(out_local)

                for kw in T.serial(kernel_l):
                    for i, j in T.Parallel(block_m, block_c):
                        m_idx = by * block_m + i
                        c_idx = bx * block_c + j
                        batch = m_idx // out_l
                        ol = m_idx % out_l
                        il = ol * stride_l + kw - pad_l
                        in_bound = (
                            (m_idx < n * out_l)
                            & (c_idx < c_in)
                            & (il >= 0)
                            & (il < l_in)
                        )
                        out_local[i, j] += T.if_then_else(
                            in_bound,
                            T.cast(x[batch, il, c_idx], accum_dtype),
                            T.cast(0.0, accum_dtype),
                        )

                for i, j in T.Parallel(block_m, block_c):
                    m_idx = by * block_m + i
                    c_idx = bx * block_c + j
                    ol = m_idx % out_l
                    window_start = ol * stride_l - pad_l
                    window_end = window_start + kernel_l
                    valid_start = T.max(window_start, 0)
                    valid_end = T.min(window_end, l_in)
                    valid_count = T.max(valid_end - valid_start, 0)
                    padded_start = T.max(window_start, -pad_l)
                    padded_end = T.min(window_end, l_in + pad_l)
                    padded_count = T.max(padded_end - padded_start, 0)
                    divisor = T.if_then_else(count_include_pad, padded_count, valid_count)
                    divisor = T.max(divisor, 1)
                    out_shared[i, j] = T.if_then_else(
                        (m_idx < n * out_l) & (c_idx < c_in),
                        T.cast(out_local[i, j] / T.cast(divisor, accum_dtype), dtype),
                        T.cast(0.0, dtype),
                    )

                for i, j in T.Parallel(block_m, block_c):
                    m_idx = by * block_m + i
                    c_idx = bx * block_c + j
                    if m_idx < n * out_l and c_idx < c_in:
                        out_flat[m_idx, c_idx] = out_shared[i, j]

        return _avg_pool1d_main

    return _avg_pool1d_func


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
    block_c: int,
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
    )(block_m, block_c, threads)(x)


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
    block_c: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (count_include_pad, dtype, block_m, block_c, threads)
    out_l = pool_output_dim(l_in, kernel_l, stride_l, pad_l, ceil_mode)
    return torch.empty((n, out_l, c_in), dtype=x.dtype, device=x.device)


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
            self.config["block_c"],
            self.config["threads"],
            x,
        )
