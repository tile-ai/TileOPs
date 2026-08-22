"""3-D convolution kernels: dense and grouped."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

from ._common import _launch, conv_autotune_configs

__all__ = [
    "Conv3dKernel",
    "GroupConv3dKernel",
]


@functools.lru_cache(maxsize=64)
def _conv3d_kernel(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_d = (d_in + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
    out_h = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    k_total = kernel_d * kernel_h * kernel_w * c_in

    # Re-enable automatic async copy once TileLang lowers scalar cp.async
    # widening for vectorized manual data loads. Keep weight T.copy eligible for TMA.
    @tilelang.jit(
        out_idx=[2],
        compile_flags=["-O3", "-DENABLE_BF16"],
        pass_configs={"tl.enable_async_copy": False},
    )
    def _conv3d_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv3d_body(x, weight, out, bias):
            out_dhw = out_d * out_h * out_w
            with T.Kernel(
                T.ceildiv(out_dhw, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                weight_flat = T.Tensor((c_out, k_total), dtype, weight.data)
                out_flat = T.Tensor((n, c_out, out_dhw), dtype, out.data)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + i
                        spatial_idx = bx * block_n + j
                        ci = k_idx // (kernel_d * kernel_h * kernel_w)
                        kernel_idx = k_idx % (kernel_d * kernel_h * kernel_w)
                        kd = kernel_idx // (kernel_h * kernel_w)
                        kh = (kernel_idx // kernel_w) % kernel_h
                        kw = kernel_idx % kernel_w
                        od = spatial_idx // (out_h * out_w)
                        oh = (spatial_idx // out_w) % out_h
                        ow = spatial_idx % out_w
                        id_ = od * stride_d + kd * dilation_d - pad_d
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        in_bound = (
                            (spatial_idx < out_dhw)
                            & (k_idx < k_total)
                            & (id_ >= 0)
                            & (ih >= 0)
                            & (iw >= 0)
                            & (id_ < d_in)
                            & (ih < h_in)
                            & (iw < w_in)
                        )
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            x[bz, ci, id_, ih, iw],
                            T.cast(0.0, dtype),
                        )

                    T.copy(weight_flat[by * block_m, k_iter * block_k], weight_shared)
                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    spatial_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (spatial_idx < out_dhw),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (spatial_idx < out_dhw),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out_flat[bz, by * block_m, bx * block_n])

        if has_bias:

            @T.prim_func
            def _conv3d_bias_main(
                x: T.Tensor((n, c_in, d_in, h_in, w_in), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in, kernel_d, kernel_h, kernel_w), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_d, out_h, out_w), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv3d_body(x, weight, out, bias)

            return _conv3d_bias_main

        @T.prim_func
        def _conv3d_main(
            x: T.Tensor((n, c_in, d_in, h_in, w_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in, kernel_d, kernel_h, kernel_w), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_d, out_h, out_w), dtype),  # type: ignore
        ):
            _conv3d_body(x, weight, out, None)

        return _conv3d_main

    return _conv3d_func


@functools.lru_cache(maxsize=64)
def _conv3d_group_kernel(
    n: int,
    c_in: int,
    d_in: int,
    h_in: int,
    w_in: int,
    c_out: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    has_bias: bool,
    dtype: str = "float16",
    groups: int = 1,
    c_in_g: int = 0,
    c_out_g: int = 0,
):
    accum_dtype = "float"
    out_d = (d_in + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
    out_h = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    out_dhw = out_d * out_h * out_w
    c_in_g = c_in_g if c_in_g > 0 else c_in // groups
    c_out_g = c_out_g if c_out_g > 0 else c_out // groups
    k_total = kernel_d * kernel_h * kernel_w * c_in_g

    @tilelang.jit(
        out_idx=[2],
        compile_flags=["-O3", "-DENABLE_BF16"],
        pass_configs={"tl.enable_async_copy": False},
    )
    def _conv3d_group_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv3d_group_body(x, weight, out, bias):
            with T.Kernel(
                T.ceildiv(out_dhw, block_n),
                T.ceildiv(c_out_g, block_m),
                n * groups,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                # k runs over (c_in_g, kernel_d, kernel_h, kernel_w) in that order, which is
                # how the weight is already laid out, so staging it is a tile copy rather
                # than a gather. The rows past this group's c_out_g read the next group's
                # weights; the epilogue masks the accumulator rows they feed.
                weight_flat = T.Tensor((c_out, k_total), dtype, weight.data)
                out_flat = T.Tensor((n, c_out, out_dhw), dtype, out.data)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                batch_id = bz // groups
                group_id = bz % groups
                oc_base = group_id * c_out_g + by * block_m

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    T.copy(weight_flat[oc_base, k_iter * block_k], weight_shared)

                    for k, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + k
                        spatial_idx = bx * block_n + j
                        ci_g = k_idx // (kernel_d * kernel_h * kernel_w)
                        ci = group_id * c_in_g + ci_g
                        kernel_idx = k_idx % (kernel_d * kernel_h * kernel_w)
                        kd = kernel_idx // (kernel_h * kernel_w)
                        kh = (kernel_idx // kernel_w) % kernel_h
                        kw = kernel_idx % kernel_w
                        od = spatial_idx // (out_h * out_w)
                        oh = (spatial_idx // out_w) % out_h
                        ow = spatial_idx % out_w
                        id_ = od * stride_d + kd * dilation_d - pad_d
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        data_shared[k, j] = T.if_then_else(
                            (spatial_idx < out_dhw)
                            & (k_idx < k_total)
                            & (id_ >= 0)
                            & (ih >= 0)
                            & (iw >= 0)
                            & (id_ < d_in)
                            & (ih < h_in)
                            & (iw < w_in),
                            x[batch_id, ci, id_, ih, iw],
                            T.cast(0.0, dtype),
                        )

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc_g = by * block_m + i
                    oc = group_id * c_out_g + oc_g
                    spatial_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc_g < c_out_g) & (spatial_idx < out_dhw),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc_g < c_out_g) & (spatial_idx < out_dhw),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                if c_out_g % block_m == 0:
                    # The tile ends on this group's last channel, so the copy cannot spill
                    # into the next group's rows.
                    T.copy(out_shared, out_flat[batch_id, oc_base, bx * block_n])
                else:
                    for i, j in T.Parallel(block_m, block_n):
                        oc_g = by * block_m + i
                        oc = group_id * c_out_g + oc_g
                        spatial_idx = bx * block_n + j
                        od = spatial_idx // (out_h * out_w)
                        oh = (spatial_idx // out_w) % out_h
                        ow = spatial_idx % out_w
                        if oc_g < c_out_g and spatial_idx < out_dhw:
                            out[batch_id, oc, od, oh, ow] = out_shared[i, j]

        if has_bias:

            @T.prim_func
            def _conv3d_group_bias_main(
                x: T.Tensor((n, c_in, d_in, h_in, w_in), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in_g, kernel_d, kernel_h, kernel_w), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_d, out_h, out_w), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv3d_group_body(x, weight, out, bias)

            return _conv3d_group_bias_main

        @T.prim_func
        def _conv3d_group_main(
            x: T.Tensor((n, c_in, d_in, h_in, w_in), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in_g, kernel_d, kernel_h, kernel_w), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_d, out_h, out_w), dtype),  # type: ignore
        ):
            _conv3d_group_body(x, weight, out, None)

        return _conv3d_group_main

    return _conv3d_group_func


class Conv3dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        c_out: int,
        kernel_d: int,
        kernel_h: int,
        kernel_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.kernel_d = kernel_d
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_d = stride_d
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_d = pad_d
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_d = dilation_d
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_d = (d_in + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
        self.out_h = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        self.out_w = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        self.m = n * self.out_d * self.out_h * self.out_w
        self.k_total = c_in * kernel_d * kernel_h * kernel_w

        self.kernel = _conv3d_kernel(
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            c_out,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            dilation_d,
            dilation_h,
            dilation_w,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "num_stages": 3,
                "threads": 128,
                "enable_rasterization": True,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 2,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return conv_autotune_configs(
            self.dtype,
            block_n=[32, 64, 128],
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _launch(self, x, weight, bias=bias)


class GroupConv3dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        c_out: int,
        kernel_d: int,
        kernel_h: int,
        kernel_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        dilation_d: int,
        dilation_h: int,
        dilation_w: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        groups: int = 1,
        c_in_g: Optional[int] = None,
        c_out_g: Optional[int] = None,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.c_out = c_out
        self.kernel_d = kernel_d
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_d = stride_d
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_d = pad_d
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_d = dilation_d
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.c_in_g = c_in_g if c_in_g is not None else c_in // groups
        self.c_out_g = c_out_g if c_out_g is not None else c_out // groups
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_d = (d_in + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
        self.out_h = (h_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        self.out_w = (w_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        self.m = n * self.groups * self.out_d * self.out_h * self.out_w
        self.k_total = self.c_in_g * kernel_d * kernel_h * kernel_w
        self._validate_group_shape()

        self.kernel = _conv3d_group_kernel(
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            c_out,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            dilation_d,
            dilation_h,
            dilation_w,
            has_bias,
            self.dtype_str,
            groups,
            self.c_in_g,
            self.c_out_g,
        )
        self.init_config(config, tune)

    def _validate_group_shape(self) -> None:
        if self.groups <= 1:
            raise ValueError("GroupConv3dKernel requires groups > 1")
        if self.c_in % self.groups != 0 or self.c_out % self.groups != 0:
            raise ValueError(
                f"GroupConv3dKernel requires c_in and c_out divisible by groups; "
                f"got c_in={self.c_in}, c_out={self.c_out}, groups={self.groups}"
            )

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "num_stages": 3,
                "threads": 128,
                "enable_rasterization": True,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 2,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return conv_autotune_configs(
            self.dtype,
            block_n=[32, 64, 128],
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _launch(self, x, weight, bias=bias)
