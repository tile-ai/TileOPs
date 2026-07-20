import functools
import itertools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import pool_output_dim

__all__ = ["MaxPool1dKernel", "MaxPool1dWithIndicesKernel"]


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

                        max_val = T.alloc_var(T.float32)
                        has_nan = T.alloc_var(T.bool)
                        max_val = T.cast(float("-inf"), accum_dtype)
                        has_nan = False
                        for kw in T.serial(kernel_w):
                            iw = ow * stride_w - pad_w + kw * dilation_w
                            if iw >= 0 and iw < l_in:
                                val = T.cast(x[batch, c_idx, iw], accum_dtype)
                                has_nan = has_nan | T.isnan(val)
                                max_val = T.max(max_val, val)

                        result = T.if_then_else(
                            has_nan,
                            T.cast(float("nan"), accum_dtype),
                            max_val,
                        )
                        out[batch, c_idx, ow] = T.cast(result, dtype)

        return _max_pool1d_main

    return _max_pool1d_func


@torch.library.custom_op("top::max_pool1d_wrapped_kernel", mutates_args=())
def _max_pool1d_wrapped_kernel(
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


@_max_pool1d_wrapped_kernel.register_fake
def _(
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
    _ = (dtype, block_m, threads)
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    return torch.empty((n, c_in, out_l), dtype=x.dtype, device=x.device)


class MaxPool1dKernel(Kernel):
    """Max pooling forward kernel for NCL inputs (return_indices=False)."""

    supported_archs: list[int] = [80, 86, 89, 90]

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
                f"MaxPool1dKernel supports float16, bfloat16, and float32, got {dtype}"
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
        self.kernel = _max_pool1d_kernel(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _max_pool1d_wrapped_kernel(
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

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _max_pool1d_with_indices_func(block_m: int, threads: int):
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

                        max_val = T.alloc_var(T.float32)
                        has_nan = T.alloc_var(T.bool)
                        max_idx = T.alloc_var(T.int64)
                        nan_idx = T.alloc_var(T.int64)
                        first_valid = T.alloc_var(T.bool)
                        max_val = T.cast(float("-inf"), accum_dtype)
                        has_nan = False
                        max_idx = T.cast(0, "int64")
                        nan_idx = T.cast(0, "int64")
                        first_valid = True
                        for kw in T.serial(kernel_w):
                            iw = ow * stride_w - pad_w + kw * dilation_w
                            if iw >= 0 and iw < l_in:
                                val = T.cast(x[batch, c_idx, iw], accum_dtype)
                                flat_idx = T.cast(iw, "int64")
                                is_nan = T.isnan(val)
                                if is_nan:
                                    # PyTorch records the last NaN visited in
                                    # a pooling window.
                                    nan_idx = flat_idx
                                    has_nan = True
                                elif first_valid:
                                    max_val = val
                                    max_idx = flat_idx
                                    first_valid = False
                                elif val > max_val:
                                    max_val = val
                                    max_idx = flat_idx

                        result = T.if_then_else(
                            has_nan,
                            T.cast(float("nan"), accum_dtype),
                            max_val,
                        )
                        out[batch, c_idx, ow] = T.cast(result, dtype)
                        indices[batch, c_idx, ow] = T.if_then_else(
                            has_nan,
                            nan_idx,
                            max_idx,
                        )

        return _max_pool1d_with_indices_main

    return _max_pool1d_with_indices_func


@torch.library.custom_op("top::max_pool1d_with_indices_wrapped_kernel", mutates_args=())
def _max_pool1d_with_indices_wrapped_kernel(
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


@_max_pool1d_with_indices_wrapped_kernel.register_fake
def _(
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
    _ = (dtype, block_m, threads)
    out_l = pool_output_dim(l_in, kernel_w, stride_w, pad_w, ceil_mode, dilation_w)
    return (
        torch.empty((n, c_in, out_l), dtype=x.dtype, device=x.device),
        torch.empty((n, c_in, out_l), dtype=torch.int64, device=x.device),
    )


class MaxPool1dWithIndicesKernel(Kernel):
    """Max pooling forward-with-indices kernel for NCL inputs."""

    supported_archs: list[int] = [80, 86, 89, 90]

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
                f"MaxPool1dWithIndicesKernel supports float16, bfloat16, and float32, got {dtype}"
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
        self.kernel = _max_pool1d_with_indices_kernel(
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _max_pool1d_with_indices_wrapped_kernel(
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
