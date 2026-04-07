import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["MaxPool2dKernel"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@functools.lru_cache(maxsize=64)
def _max_pool2d_values_kernel(
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
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    accum_dtype = "float32"
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool2d_func(block_m: int, block_c: int, threads: int):
        @T.prim_func
        def _max_pool2d_main(
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
                        max_val = T.alloc_var(T.float32)
                        max_val = -T.infinity(accum_dtype)

                        for kh in T.serial(kernel_h):
                            for kw in T.serial(kernel_w):
                                ih = oh * stride_h + kh * dilation_h - pad_h
                                iw = ow * stride_w + kw * dilation_w - pad_w
                                if ih >= 0 and ih < h_in and iw >= 0 and iw < w_in:
                                    candidate = T.cast(x[batch, ih, iw, c_idx], accum_dtype)
                                    max_val = T.max(max_val, candidate)

                        out_flat[m_idx, c_idx] = T.cast(max_val, dtype)

        return _max_pool2d_main

    return _max_pool2d_func


@functools.lru_cache(maxsize=64)
def _max_pool2d_values_indices_kernel(
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
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str = "float16",
):
    accum_dtype = "float32"
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool2d_func(block_m: int, block_c: int, threads: int):
        @T.prim_func
        def _max_pool2d_main(
            x: T.Tensor((n, h_in, w_in, c_in), dtype),  # type: ignore
            out: T.Tensor((n, out_h, out_w, c_in), dtype),  # type: ignore
            out_indices: T.Tensor((n, out_h, out_w, c_in), "int64"),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(c_in, block_c),
                T.ceildiv(n * out_h * out_w, block_m),
                threads=threads,
            ) as (bx, by):
                out_flat = T.Tensor((n * out_h * out_w, c_in), dtype, out.data)
                indices_flat = T.Tensor((n * out_h * out_w, c_in), "int64", out_indices.data)

                for i, j in T.Parallel(block_m, block_c):
                    m_idx = by * block_m + i
                    c_idx = bx * block_c + j
                    if m_idx < n * out_h * out_w and c_idx < c_in:
                        batch = m_idx // (out_h * out_w)
                        out_idx = m_idx % (out_h * out_w)
                        oh = out_idx // out_w
                        ow = out_idx % out_w
                        max_val = T.alloc_var(T.float32)
                        max_index = T.alloc_var(T.int64)
                        max_val = -T.infinity(accum_dtype)
                        max_index = T.int64(0)

                        for kh in T.serial(kernel_h):
                            for kw in T.serial(kernel_w):
                                ih = oh * stride_h + kh * dilation_h - pad_h
                                iw = ow * stride_w + kw * dilation_w - pad_w
                                if ih >= 0 and ih < h_in and iw >= 0 and iw < w_in:
                                    candidate = T.cast(x[batch, ih, iw, c_idx], accum_dtype)
                                    candidate_index = T.cast(ih * w_in + iw, "int64")
                                    should_update = candidate > max_val
                                    max_val = T.if_then_else(should_update, candidate, max_val)
                                    max_index = T.if_then_else(should_update, candidate_index, max_index)

                        out_flat[m_idx, c_idx] = T.cast(max_val, dtype)
                        indices_flat[m_idx, c_idx] = max_index

        return _max_pool2d_main

    return _max_pool2d_func


@torch.library.custom_op("top::max_pool2d_values_wrapped_kernel", mutates_args=())
def _max_pool2d_values_wrapped_kernel(
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
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    block_c: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _max_pool2d_values_kernel(
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
        dilation_h,
        dilation_w,
        ceil_mode,
        dtype,
    )(block_m, block_c, threads)(x)


@_max_pool2d_values_wrapped_kernel.register_fake
def _max_pool2d_values_wrapped_kernel_fake(
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
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    block_c: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (dtype, block_m, block_c, threads)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    return torch.empty((n, out_h, out_w, c_in), dtype=x.dtype, device=x.device)


@torch.library.custom_op("top::max_pool2d_values_indices_wrapped_kernel", mutates_args=())
def _max_pool2d_values_indices_wrapped_kernel(
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
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    block_c: int,
    threads: int,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _max_pool2d_values_indices_kernel(
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
        dilation_h,
        dilation_w,
        ceil_mode,
        dtype,
    )(block_m, block_c, threads)(x)


@_max_pool2d_values_indices_wrapped_kernel.register_fake
def _max_pool2d_values_indices_wrapped_kernel_fake(
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
    dilation_h: int,
    dilation_w: int,
    ceil_mode: bool,
    dtype: str,
    block_m: int,
    block_c: int,
    threads: int,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ = (dtype, block_m, block_c, threads)
    out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
    out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    return (
        torch.empty((n, out_h, out_w, c_in), dtype=x.dtype, device=x.device),
        torch.empty((n, out_h, out_w, c_in), dtype=torch.int64, device=x.device),
    )


class MaxPool2dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _SUPPORTED_DTYPES

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
        dilation_h: int,
        dilation_w: int,
        ceil_mode: bool,
        return_indices: bool,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
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
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        self.dtype = dtype
        self.out_h = pool_output_dim(h_in, kernel_h, stride_h, pad_h, ceil_mode, dilation_h)
        self.out_w = pool_output_dim(w_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)

        if return_indices:
            self.kernel = _max_pool2d_values_indices_kernel(
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
                dilation_h,
                dilation_w,
                ceil_mode,
                self.dtype_str,
            )
        else:
            self.kernel = _max_pool2d_values_kernel(
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
                dilation_h,
                dilation_w,
                ceil_mode,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.return_indices:
            return _max_pool2d_values_indices_wrapped_kernel(
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
                self.dilation_h,
                self.dilation_w,
                self.ceil_mode,
                self.dtype_str,
                self.config["block_m"],
                self.config["block_c"],
                self.config["threads"],
                x,
            )
        return _max_pool2d_values_wrapped_kernel(
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
            self.dilation_h,
            self.dilation_w,
            self.ceil_mode,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_c"],
            self.config["threads"],
            x,
        )
