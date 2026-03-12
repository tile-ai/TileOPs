from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["PointwiseConvKernel"]


def _pointwise_conv_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    out_h: int,
    out_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _pointwise_conv_func(
        block_m: int,
        block_n: int,
        block_k: int,
        threads: int,
        num_stages: int,
        enable_rasteration: bool,
    ) -> Callable:
        m = n * out_h * out_w

        @T.prim_func
        def _pointwise_conv(
            x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
            weight: T.Tensor((c_in, c_out), dtype),  # type: ignore
            y: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(m, block_m),
                threads=threads,
            ) as (bx, by):
                a_shared = T.alloc_shared((block_m, block_k), dtype)
                b_shared = T.alloc_shared((block_k, block_n), dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                c_shared = T.alloc_shared((block_m, block_n), dtype)

                T.annotate_layout({
                    c_shared: tilelang.layout.make_swizzled_layout(c_shared),
                })
                T.use_swizzle(10, enable=enable_rasteration)

                row_start = by * block_m
                col_start = bx * block_n
                actual_rows = T.min(block_m, m - row_start)
                actual_cols = T.min(block_n, c_out - col_start)

                T.clear(c_local)

                for ko in T.Pipelined(T.ceildiv(c_in, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_m, block_k):
                        row = row_start + i
                        cin_idx = ko * block_k + j
                        if i < actual_rows and cin_idx < c_in:
                            batch_idx = row // (out_h * out_w)
                            spatial_idx = row % (out_h * out_w)
                            oh = spatial_idx // out_w
                            ow = spatial_idx % out_w
                            if pad_h == 0 and pad_w == 0:
                                a_shared[i, j] = x[
                                    batch_idx,
                                    cin_idx,
                                    oh * stride_h,
                                    ow * stride_w,
                                ]
                            else:
                                ih = oh * stride_h - pad_h
                                iw = ow * stride_w - pad_w
                                a_shared[i, j] = T.if_then_else(
                                    (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w),
                                    x[batch_idx, cin_idx, ih, iw],
                                    T.cast(0, dtype),
                                )
                        else:
                            a_shared[i, j] = T.cast(0, dtype)

                    T.copy(weight[ko * block_k, col_start], b_shared)
                    T.gemm(a_shared, b_shared, c_local)

                T.copy(c_local, c_shared)
                for i, j in T.Parallel(block_m, block_n):
                    if i < actual_rows and j < actual_cols:
                        row = row_start + i
                        batch_idx = row // (out_h * out_w)
                        spatial_idx = row % (out_h * out_w)
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w
                        y[batch_idx, col_start + j, oh, ow] = c_shared[i, j]

        return _pointwise_conv

    return _pointwise_conv_func


@torch.library.custom_op("top::pointwise_conv_wrapped_kernel", mutates_args=())
def _pointwise_conv_wrapped_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    out_h: int,
    out_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    threads: int,
    num_stages: int,
    enable_rasteration: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return _pointwise_conv_kernel(
        n,
        c_in,
        h,
        w,
        c_out,
        out_h,
        out_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dtype,
    )(block_m, block_n, block_k, threads, num_stages, enable_rasteration)(x, weight)


@_pointwise_conv_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    out_h: int,
    out_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    threads: int,
    num_stages: int,
    enable_rasteration: bool,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((n, c_out, out_h, out_w), dtype=x.dtype, device=x.device)


class PointwiseConvKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        out_h: int,
        out_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.out_h = out_h
        self.out_w = out_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dtype = dtype
        self.kernel = _pointwise_conv_kernel(
            n,
            c_in,
            h,
            w,
            c_out,
            out_h,
            out_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 32,
            "threads": 128,
            "num_stages": 2,
            "enable_rasteration": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {
                "block_m": 64,
                "block_n": 128,
                "block_k": 32,
                "threads": 128,
                "num_stages": 1,
                "enable_rasteration": True,
            },
            {
                "block_m": 128,
                "block_n": 128,
                "block_k": 32,
                "threads": 128,
                "num_stages": 2,
                "enable_rasteration": True,
            },
            {
                "block_m": 128,
                "block_n": 256,
                "block_k": 32,
                "threads": 128,
                "num_stages": 2,
                "enable_rasteration": True,
            },
            {
                "block_m": 128,
                "block_n": 256,
                "block_k": 64,
                "threads": 256,
                "num_stages": 2,
                "enable_rasteration": True,
            },
            {
                "block_m": 256,
                "block_n": 128,
                "block_k": 32,
                "threads": 256,
                "num_stages": 3,
                "enable_rasteration": True,
            },
            {
                "block_m": 128,
                "block_n": 256,
                "block_k": 64,
                "threads": 256,
                "num_stages": 3,
                "enable_rasteration": False,
            },
        ]

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _pointwise_conv_wrapped_kernel(
            self.n,
            self.c_in,
            self.h,
            self.w,
            self.c_out,
            self.out_h,
            self.out_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["threads"],
            self.config["num_stages"],
            self.config["enable_rasteration"],
            x,
            weight,
        )
