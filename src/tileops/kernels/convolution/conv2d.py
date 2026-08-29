"""2-D convolution kernels: dense, grouped, depthwise, symmetric, and 1x1."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

from ._common import _launch, conv_autotune_configs
from .call_spec import (
    Conv2dCall,
    conv2d_dense_region,
    conv2d_group_region,
    conv2d_pointwise_region,
    conv2d_symmetric_region,
)

__all__ = [
    "Conv2d1x1Kernel",
    "Conv2dKernel",
    "Conv2dSymmetricKernel",
    "GroupConv2dKernel",
]


@functools.lru_cache(maxsize=32)
def _conv2d_1x1_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    if stride_h != 1 or stride_w != 1 or pad_h != 0 or pad_w != 0:
        raise ValueError("Conv2d1x1Kernel requires stride=1 and padding=0")
    hw = h * w

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv2d_1x1_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv2d_1x1_body(x, weight, out, bias):
            x_flat = T.Tensor((n, c_in, hw), dtype, x.data)
            out_flat = T.Tensor((n, c_out, hw), dtype, out.data)
            with T.Kernel(
                T.ceildiv(hw, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(c_in, block_k), num_stages=num_stages):
                    T.copy(weight[by * block_m, k_iter * block_k], weight_shared)
                    T.copy(x_flat[bz, k_iter * block_k, bx * block_n], data_shared)
                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    hw_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (hw_idx < hw),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (hw_idx < hw),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out_flat[bz, by * block_m, bx * block_n])

        if has_bias:

            @T.prim_func
            def _conv2d_1x1_bias_main(
                x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in), dtype),  # type: ignore
                out: T.Tensor((n, c_out, h, w), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv2d_1x1_body(x, weight, out, bias)

            return _conv2d_1x1_bias_main

        @T.prim_func
        def _conv2d_1x1_main(
            x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in), dtype),  # type: ignore
            out: T.Tensor((n, c_out, h, w), dtype),  # type: ignore
        ):
            _conv2d_1x1_body(x, weight, out, None)

        return _conv2d_1x1_main

    return _conv2d_1x1_func


@functools.lru_cache(maxsize=32)
def _conv2d_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    k_total = kernel_h * kernel_w * c_in

    # Re-enable automatic async copy once TileLang lowers scalar cp.async
    # widening for vectorized manual data loads. Keep weight T.copy eligible for TMA.
    @tilelang.jit(
        out_idx=[2],
        compile_flags=["-O3", "-DENABLE_BF16"],
        pass_configs={"tl.enable_async_copy": False},
    )
    def _conv2d_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv2d_body(x, weight, out, bias):
            out_hw = out_h * out_w
            with T.Kernel(
                T.ceildiv(out_hw, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                weight_flat = T.Tensor((c_out, k_total), dtype, weight.data)
                out_flat = T.Tensor((n, c_out, out_hw), dtype, out.data)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    for i, j in T.Parallel(block_k, block_n):
                        k_idx = k_iter * block_k + i
                        spatial_idx = bx * block_n + j
                        ci = k_idx // (kernel_h * kernel_w)
                        kernel_idx = k_idx % (kernel_h * kernel_w)
                        kh = kernel_idx // kernel_w
                        kw = kernel_idx % kernel_w
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        in_bound = (
                            (spatial_idx < out_hw)
                            & (k_idx < k_total)
                            & (ih >= 0)
                            & (iw >= 0)
                            & (ih < h)
                            & (iw < w)
                        )
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            x[bz, ci, ih, iw],
                            T.cast(0.0, dtype),
                        )

                    T.copy(weight_flat[by * block_m, k_iter * block_k], weight_shared)

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    spatial_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (spatial_idx < out_hw),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc < c_out) & (spatial_idx < out_hw),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out_flat[bz, by * block_m, bx * block_n])

        if has_bias:

            @T.prim_func
            def _conv2d_bias_main(
                x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in, kernel_h, kernel_w), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv2d_body(x, weight, out, bias)

            return _conv2d_bias_main

        @T.prim_func
        def _conv2d_main(
            x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in, kernel_h, kernel_w), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
        ):
            _conv2d_body(x, weight, out, None)

        return _conv2d_main

    return _conv2d_func


@functools.lru_cache(maxsize=64)
def _conv2d_group_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    has_bias: bool,
    dtype: str = "float16",
    groups: int = 1,
    c_in_g: int = 0,
    c_out_g: int = 0,
):
    accum_dtype = "float"
    out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    out_hw = out_h * out_w
    c_in_g = c_in_g if c_in_g > 0 else c_in // groups
    c_out_g = c_out_g if c_out_g > 0 else c_out // groups
    k_total = kernel_h * kernel_w * c_in_g

    @tilelang.jit(
        out_idx=[2],
        compile_flags=["-O3", "-DENABLE_BF16"],
        pass_configs={"tl.enable_async_copy": False},
    )
    def _conv2d_group_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv2d_group_body(x, weight, out, bias):
            with T.Kernel(
                T.ceildiv(out_hw, block_n),
                T.ceildiv(c_out_g, block_m),
                n * groups,
                threads=threads,
            ) as (bx, by, bz):
                weight_shared = T.alloc_shared((block_m, block_k), dtype)
                data_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                # k runs over (c_in_g, kernel_h, kernel_w) in that order, which is how the
                # weight is already laid out, so staging it is a tile copy rather than a
                # gather. The rows past this group's c_out_g read the next group's weights;
                # the epilogue masks the accumulator rows they feed.
                weight_flat = T.Tensor((c_out, k_total), dtype, weight.data)
                out_flat = T.Tensor((n, c_out, out_hw), dtype, out.data)

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
                        ci_g = k_idx // (kernel_h * kernel_w)
                        ci = group_id * c_in_g + ci_g
                        kernel_idx = k_idx % (kernel_h * kernel_w)
                        kh = kernel_idx // kernel_w
                        kw = kernel_idx % kernel_w
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        data_shared[k, j] = T.if_then_else(
                            (spatial_idx < out_hw)
                            & (k_idx < k_total)
                            & (ih >= 0)
                            & (iw >= 0)
                            & (ih < h)
                            & (iw < w),
                            x[batch_id, ci, ih, iw],
                            T.cast(0.0, dtype),
                        )

                    T.gemm(weight_shared, data_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    oc_g = by * block_m + i
                    oc = group_id * c_out_g + oc_g
                    spatial_idx = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (oc_g < c_out_g) & (spatial_idx < out_hw),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (oc_g < c_out_g) & (spatial_idx < out_hw),
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
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w
                        if oc_g < c_out_g and spatial_idx < out_hw:
                            out[batch_id, oc, oh, ow] = out_shared[i, j]

        if has_bias:

            @T.prim_func
            def _conv2d_group_bias_main(
                x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
                weight: T.Tensor((c_out, c_in_g, kernel_h, kernel_w), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv2d_group_body(x, weight, out, bias)

            return _conv2d_group_bias_main

        @T.prim_func
        def _conv2d_group_main(
            x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
            weight: T.Tensor((c_out, c_in_g, kernel_h, kernel_w), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
        ):
            _conv2d_group_body(x, weight, out, None)

        return _conv2d_group_main

    return _conv2d_group_func


@functools.lru_cache(maxsize=32)
def _conv2d_depthwise_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    has_bias: bool,
    dtype: str = "float16",
):
    """Build the depthwise Conv2d program: one input channel feeds one output channel.

    The grouped kernel reaches this shape with a GEMM whose M is the group's one output
    channel, so a tile of ``block_m`` rows carries one useful row. This one multiplies and
    accumulates directly instead, the way the Conv1d depthwise path does.
    """
    accum_dtype = "float"
    out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    out_hw = out_h * out_w

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv2d_depthwise_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def _conv2d_depthwise_body(x, weight, out, bias):
            out_flat = T.Tensor((n, c_out, out_hw), dtype, out.data)
            with T.Kernel(
                T.ceildiv(out_hw, block_n),
                T.ceildiv(c_out, block_m),
                n,
                threads=threads,
            ) as (bx, by, bz):
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for kh, kw in T.grid(kernel_h, kernel_w):
                    for i, j in T.Parallel(block_m, block_n):
                        oc = by * block_m + i
                        spatial_idx = bx * block_n + j
                        oh = spatial_idx // out_w
                        ow = spatial_idx % out_w
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        iw = ow * stride_w + kw * dilation_w - pad_w
                        valid = (
                            (oc < c_out)
                            & (spatial_idx < out_hw)
                            & (ih >= 0)
                            & (iw >= 0)
                            & (ih < h)
                            & (iw < w)
                        )
                        out_local[i, j] += T.if_then_else(
                            valid,
                            T.cast(x[bz, oc, ih, iw], accum_dtype)
                            * T.cast(weight[oc, 0, kh, kw], accum_dtype),
                            T.cast(0.0, accum_dtype),
                        )

                for i, j in T.Parallel(block_m, block_n):
                    oc = by * block_m + i
                    spatial_idx = bx * block_n + j
                    if oc < c_out and spatial_idx < out_hw:
                        if has_bias:
                            out_flat[bz, oc, spatial_idx] = T.cast(
                                out_local[i, j] + T.cast(bias[oc], accum_dtype),
                                dtype,
                            )
                        else:
                            out_flat[bz, oc, spatial_idx] = T.cast(out_local[i, j], dtype)

        if has_bias:

            @T.prim_func
            def _conv2d_depthwise_bias_main(
                x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
                weight: T.Tensor((c_out, 1, kernel_h, kernel_w), dtype),  # type: ignore
                out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
                bias: T.Tensor((c_out,), dtype),  # type: ignore
            ):
                _conv2d_depthwise_body(x, weight, out, bias)

            return _conv2d_depthwise_bias_main

        @T.prim_func
        def _conv2d_depthwise_main(
            x: T.Tensor((n, c_in, h, w), dtype),  # type: ignore
            weight: T.Tensor((c_out, 1, kernel_h, kernel_w), dtype),  # type: ignore
            out: T.Tensor((n, c_out, out_h, out_w), dtype),  # type: ignore
        ):
            _conv2d_depthwise_body(x, weight, out, None)

        return _conv2d_depthwise_main

    return _conv2d_depthwise_func


@functools.lru_cache(maxsize=8)
def _conv2d_symmetric_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    kernel_size: int,
    stride: int,
    pad: int,
    dilation: int,
    has_bias: bool,
    dtype: str = "float16",
):
    accum_dtype = "float"
    out_h = (h + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1
    out_hw = out_h * out_w
    k_total = c_in * kernel_size * kernel_size

    @tilelang.jit(
        out_idx=[5],
        compile_flags=["-O3", "-DENABLE_BF16"],
        pass_configs={"tl.enable_async_copy": False},
    )
    def _conv2d_symmetric_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
    ):
        @T.macro
        def transpose_spatial_channel(
            src: T.Tensor,
            dst: T.Tensor,
            batch_size: int,
            spatial_size: int,
            channel_size: int,
            width: int,
            spatial_block: int,
            channel_block: int,
            channel_lanes: int,
            channel_fastest: bool,
            is_nchw_to_nhwc: bool,
        ):
            assert spatial_block * channel_lanes == 256, (
                "spatial_block * channel_lanes must equal 256"
            )
            assert channel_block % channel_lanes == 0, "channel_lanes must divide channel_block"
            if not channel_fastest:
                assert channel_block * spatial_block == 256, (
                    "channel_block * spatial_block must equal 256 when channel_fastest=False"
                )

            with T.Kernel(
                T.ceildiv(spatial_size, spatial_block),
                T.ceildiv(channel_size, channel_block),
                batch_size,
                threads=256,
            ) as (bx, by, bz):
                if channel_fastest:
                    values_per_thread = channel_block // channel_lanes
                    for spatial_inner, channel_lane in T.Parallel(spatial_block, channel_lanes):
                        spatial = bx * spatial_block + spatial_inner
                        h_idx = spatial // width
                        w_idx = spatial - h_idx * width
                        channel_base = by * channel_block
                        for channel_offset in T.serial(values_per_thread):
                            c = channel_base + channel_offset * channel_lanes + channel_lane
                            if (spatial < spatial_size) & (c < channel_size):
                                if is_nchw_to_nhwc:
                                    dst[bz, h_idx, w_idx, c] = src[bz, c, h_idx, w_idx]
                                else:
                                    dst[bz, c, h_idx, w_idx] = src[bz, h_idx, w_idx, c]
                else:
                    for channel_inner, spatial_inner in T.Parallel(channel_block, spatial_block):
                        spatial = bx * spatial_block + spatial_inner
                        h_idx = spatial // width
                        w_idx = spatial - h_idx * width
                        c = by * channel_block + channel_inner
                        if (spatial < spatial_size) & (c < channel_size):
                            if is_nchw_to_nhwc:
                                dst[bz, h_idx, w_idx, c] = src[bz, c, h_idx, w_idx]
                            else:
                                dst[bz, c, h_idx, w_idx] = src[bz, h_idx, w_idx, c]

        @T.macro
        def conv_nhwc_implicit_gemm(x_nhwc, weight_krsc, out_nhwc, bias):
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(n * out_h * out_w, block_m),
                threads=threads,
            ) as (bx, by):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_n, block_k), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                weight_flat = T.Tensor((c_out, k_total), dtype, weight_krsc.data)
                out_flat = T.Tensor((n * out_h * out_w, c_out), dtype, out_nhwc.data)

                T.use_swizzle(10, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    T.im2col(x_nhwc, data_shared, by, k_iter, kernel_size, stride, dilation, pad)
                    T.copy(weight_flat[bx * block_n, k_iter * block_k], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local, transpose_B=True)

                for i, j in T.Parallel(block_m, block_n):
                    spatial_idx = by * block_m + i
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (spatial_idx < n * out_hw) & (oc < c_out),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (spatial_idx < n * out_hw) & (oc < c_out),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out_flat[by * block_m, bx * block_n])

        @T.macro
        def _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out_nhwc, out, bias):
            transpose_spatial_channel(
                x,
                x_nhwc,
                batch_size=n,
                spatial_size=h * w,
                channel_size=c_in,
                width=w,
                spatial_block=32,
                channel_block=32,
                channel_lanes=8,
                channel_fastest=True,
                is_nchw_to_nhwc=True,
            )
            transpose_spatial_channel(
                weight,
                weight_krsc,
                batch_size=c_out,
                spatial_size=kernel_size * kernel_size,
                channel_size=c_in,
                width=kernel_size,
                spatial_block=16,
                channel_block=32,
                channel_lanes=16,
                channel_fastest=True,
                is_nchw_to_nhwc=True,
            )
            conv_nhwc_implicit_gemm(x_nhwc, weight_krsc, out_nhwc, bias)
            transpose_spatial_channel(
                out_nhwc,
                out,
                batch_size=n,
                spatial_size=out_h * out_w,
                channel_size=c_out,
                width=out_w,
                spatial_block=128,
                channel_block=2,
                channel_lanes=2,
                channel_fastest=False,
                is_nchw_to_nhwc=False,
            )

        if has_bias:

            @T.prim_func
            def _conv2d_symmetric_bias_main(
                x: T.Tensor((n, c_in, h, w), dtype),
                weight: T.Tensor((c_out, c_in, kernel_size, kernel_size), dtype),
                x_nhwc: T.Tensor((n, h, w, c_in), dtype),
                weight_krsc: T.Tensor((c_out, kernel_size, kernel_size, c_in), dtype),
                out_nhwc: T.Tensor((n, out_h, out_w, c_out), dtype),
                out: T.Tensor((n, c_out, out_h, out_w), dtype),
                bias: T.Tensor((c_out,), dtype),
            ):
                _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out_nhwc, out, bias)

            return _conv2d_symmetric_bias_main

        @T.prim_func
        def _conv2d_symmetric_main(
            x: T.Tensor((n, c_in, h, w), dtype),
            weight: T.Tensor((c_out, c_in, kernel_size, kernel_size), dtype),
            x_nhwc: T.Tensor((n, h, w, c_in), dtype),
            weight_krsc: T.Tensor((c_out, kernel_size, kernel_size, c_in), dtype),
            out_nhwc: T.Tensor((n, out_h, out_w, c_out), dtype),
            out: T.Tensor((n, c_out, out_h, out_w), dtype),
        ):
            _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out_nhwc, out, None)

        return _conv2d_symmetric_main

    return _conv2d_symmetric_func


class Conv2dSymmetricKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: Conv2dCall) -> bool:
        return conv2d_symmetric_region(call)

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        pad: int,
        dilation: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_h = (h + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1
        self.out_w = (w + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1
        self.m = n * self.out_h * self.out_w
        self.k_total = c_in * kernel_size * kernel_size

        self.kernel = _conv2d_symmetric_kernel(
            n,
            c_in,
            h,
            w,
            c_out,
            kernel_size,
            stride,
            pad,
            dilation,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 256,
            "block_k": 32,
            "num_stages": 3,
            "threads": 256,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        configs = conv_autotune_configs(
            self.dtype,
            block_m=[64, 128],
            block_k=[16, 32, 64],
        )
        return [c for c in configs if self.c_in % c["block_k"] == 0]

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_nhwc = torch.empty(
            (self.n, self.h, self.w, self.c_in),
            device=x.device,
            dtype=x.dtype,
        )
        weight_krsc = torch.empty(
            (self.c_out, self.kernel_size, self.kernel_size, self.c_in),
            device=weight.device,
            dtype=weight.dtype,
        )
        out_nhwc = torch.empty(
            (self.n, self.out_h, self.out_w, self.c_out),
            device=x.device,
            dtype=x.dtype,
        )
        return _launch(self, x, weight, x_nhwc, weight_krsc, out_nhwc, bias=bias)


class Conv2dKernel(Kernel):
    general = True
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: Conv2dCall) -> bool:
        return conv2d_dense_region(call)

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
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
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        self.out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        self.m = n * self.out_h * self.out_w
        self.k_total = c_in * kernel_h * kernel_w

        self.kernel = _conv2d_kernel(
            n,
            c_in,
            h,
            w,
            c_out,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
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
                "enable_rasterization": False,
            }
        if sm_version in {80}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "threads": 128,
                "num_stages": 2,
                "enable_rasterization": True,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "threads": 128,
            "num_stages": 2,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return conv_autotune_configs(
            self.dtype,
            block_m=[64, 128],
            block_k=[64, 128],
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _launch(self, x, weight, bias=bias)


class GroupConv2dKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: Conv2dCall) -> bool:
        return conv2d_group_region(call)

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
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
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.c_in_g = c_in_g if c_in_g is not None else c_in // groups
        self.c_out_g = c_out_g if c_out_g is not None else c_out // groups
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        self.out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        self.m = n * self.groups * self.out_h * self.out_w
        self.k_total = self.c_in_g * kernel_h * kernel_w
        self.use_direct = self.c_in_g == 1 and self.c_out_g == 1
        self._validate_group_shape()

        if self.use_direct:
            self.kernel = _conv2d_depthwise_kernel(
                n,
                c_in,
                h,
                w,
                c_out,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                has_bias,
                self.dtype_str,
            )
        else:
            self.kernel = _conv2d_group_kernel(
                n,
                c_in,
                h,
                w,
                c_out,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
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
            raise ValueError("GroupConv2dKernel requires groups > 1")
        if self.c_in % self.groups != 0 or self.c_out % self.groups != 0:
            raise ValueError(
                f"GroupConv2dKernel requires c_in and c_out divisible by groups; "
                f"got c_in={self.c_in}, c_out={self.c_out}, groups={self.groups}"
            )

    @property
    def default_config(self) -> dict:
        if self.use_direct:
            # No GEMM here, so block_k and num_stages carry nothing: one channel per block
            # row, a strip of outputs per block column.
            return {
                "block_m": 1,
                "block_n": 128,
                "block_k": 1,
                "num_stages": 1,
                "threads": 128,
                "enable_rasterization": True,
            }
        sm_version = get_sm_version()
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "num_stages": 3,
                "threads": 128,
                "enable_rasterization": False,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "threads": 128,
            "num_stages": 2,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        if self.use_direct:
            return [self.default_config]
        return conv_autotune_configs(self.dtype)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _launch(self, x, weight, bias=bias)


class Conv2d1x1Kernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: Conv2dCall) -> bool:
        return conv2d_pointwise_region(call)

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dtype = dtype
        self.has_bias = has_bias

        self.kernel = _conv2d_1x1_kernel(
            n,
            c_in,
            h,
            w,
            c_out,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            has_bias,
            self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()
        if sm_version in {80}:
            return {
                "block_m": 64,
                "block_n": 64,
                "block_k": 64,
                "num_stages": 1,
                "threads": 128,
                "enable_rasterization": True,
            }
        if sm_version in {90}:
            return {
                "block_m": 64,
                "block_n": 128,
                "block_k": 128,
                "num_stages": 2,
                "threads": 128,
                "enable_rasterization": True,
            }
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 1,
            "threads": 128,
            "enable_rasterization": True,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return conv_autotune_configs(
            self.dtype,
            block_m=[64, 128, 256],
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # OIHW -> OC,IC since the 1x1 kernel consumes a dense [C_out, C_in] weight matrix.
        weight_oc_ci = weight.view(self.c_out, self.c_in).contiguous()
        return _launch(self, x, weight_oc_ci, bias=bias)
