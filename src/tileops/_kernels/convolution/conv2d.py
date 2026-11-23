"""2-D convolution kernels: dense, grouped, depthwise, symmetric, and 1x1."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.convolution._common import (
    SWIZZLE_PANEL,
    conv_autotune_configs,
    grid_refusal,
    launch,
    operand_refusal,
    stage_output_tile,
)
from tileops._kernels.convolution.call_spec import Conv2dCall, Conv2dFwdInterface
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._perf import find_profile, resolve_roof, tensor_core_roof
from tileops._utils import WARP_LANES

__all__ = [
    "Conv2d1x1Kernel",
    "Conv2dKernel",
    "Conv2dSymmetricKernel",
    "DepthwiseConv2dKernel",
    "DepthwiseConv2dWarpKernel",
    "GroupConv2dKernel",
]


@target_cache(maxsize=32)
def _conv2d_1x1_kernel(
    n: int,
    c_in: int,
    h: int,
    w: int,
    c_out: int,
    has_bias: bool,
    dtype: str,
):
    accum_dtype = "float"
    hw = h * w

    @tilelang.jit(out_idx=[2], compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"])
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

                T.use_swizzle(SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(c_in, block_k), num_stages=num_stages):
                    T.copy(weight[by * block_m, k_iter * block_k], weight_shared)
                    T.copy(x_flat[bz, k_iter * block_k, bx * block_n], data_shared)
                    T.gemm(weight_shared, data_shared, out_local)

                stage_output_tile(
                    out_local,
                    out_shared,
                    bias,
                    block_m,
                    block_n,
                    by * block_m,
                    c_out,
                    bx * block_n,
                    hw,
                    0,
                    dtype,
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


@target_cache(maxsize=32)
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
    dtype: str,
    *,
    pad_h_end: int,
    pad_w_end: int,
):
    # The end of each axis may pad more than the start.
    accum_dtype = "float"
    out_h = (h + pad_h + pad_h_end - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w + pad_w + pad_w_end - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    k_total = kernel_h * kernel_w * c_in

    # Automatic async copy is off: TileLang does not widen the scalar cp.async it emits for
    # the gathered data loads. The weight T.copy stays eligible for TMA.
    @tilelang.jit(
        out_idx=[2],
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
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

                T.use_swizzle(SWIZZLE_PANEL, enable=enable_rasterization)
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

                stage_output_tile(
                    out_local,
                    out_shared,
                    bias,
                    block_m,
                    block_n,
                    by * block_m,
                    c_out,
                    bx * block_n,
                    out_hw,
                    0,
                    dtype,
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


@target_cache(maxsize=64)
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
    dtype: str,
    groups: int,
    c_in_g: int,
    c_out_g: int,
    *,
    pad_h_end: int,
    pad_w_end: int,
):
    # The end of each axis may pad more than the start.
    accum_dtype = "float"
    out_h = (h + pad_h + pad_h_end - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w + pad_w + pad_w_end - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    out_hw = out_h * out_w
    k_total = kernel_h * kernel_w * c_in_g

    @tilelang.jit(
        out_idx=[2],
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
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

                T.use_swizzle(SWIZZLE_PANEL, enable=enable_rasterization)
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

                stage_output_tile(
                    out_local,
                    out_shared,
                    bias,
                    block_m,
                    block_n,
                    by * block_m,
                    c_out_g,
                    bx * block_n,
                    out_hw,
                    group_id * c_out_g,
                    dtype,
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


@target_cache(maxsize=32)
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
    dtype: str,
    *,
    pad_h_end: int,
    pad_w_end: int,
):
    """Build the depthwise Conv2d program: one input channel feeds one output channel.

    The grouped kernel reaches this shape with a GEMM whose M is the group's one output
    channel, so a tile of ``block_m`` rows carries one useful row. This one multiplies and
    accumulates directly instead, the way the Conv1d depthwise path does.
    """
    # The end of each axis may pad more than the start.
    accum_dtype = "float"
    out_h = (h + pad_h + pad_h_end - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (w + pad_w + pad_w_end - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    out_hw = out_h * out_w

    @tilelang.jit(out_idx=[2], compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"])
    def _conv2d_depthwise_func(
        block_m: int,
        block_n: int,
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
                T.use_swizzle(SWIZZLE_PANEL, enable=enable_rasterization)
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


@target_cache(maxsize=8)
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
    dtype: str,
):
    accum_dtype = "float"
    out_h = (h + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (w + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1
    out_hw = out_h * out_w
    k_total = c_in * kernel_size * kernel_size

    @tilelang.jit(
        out_idx=[4],
        compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"],
        pass_configs={"tl.enable_async_copy": False},
    )
    def _conv2d_symmetric_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasterization: bool,
        split_k: int,
    ):
        # One kernel stages both operands: each is the same transpose over a plane of
        # (channel, spatial), and the block index selects which one a block serves.
        # Both walk c_in in the same channel block, so one block count over the
        # channel axis serves both.
        staging_threads = 256
        channel_block = 32
        x_lanes, x_spatial_block = 8, 32
        w_lanes, w_spatial_block = 16, 16
        assert (
            x_spatial_block * x_lanes == staging_threads
            and w_spatial_block * w_lanes == staging_threads
        ), "each operand's tile must occupy the threads the staging kernel launches"
        assert channel_block % x_lanes == 0 and channel_block % w_lanes == 0, (
            "a thread's channel stride must divide the channel block it walks"
        )
        channel_blocks = -(-c_in // channel_block)
        x_spatial_blocks = -(-(h * w) // x_spatial_block)
        w_spatial_blocks = -(-(kernel_size * kernel_size) // w_spatial_block)
        x_blocks = x_spatial_blocks * channel_blocks * n
        w_blocks = w_spatial_blocks * channel_blocks * c_out

        @T.macro
        def plane_to_channel_last(
            src: T.Tensor,
            dst: T.Tensor,
            block: T.int32,
            spatial_size: int,
            spatial_block: int,
            spatial_blocks: int,
            channel_lanes: int,
        ):
            """One block's share of ``src[p, c, s] -> dst[p, s, c]``."""
            spatial_base = (block % spatial_blocks) * spatial_block
            channel_base = ((block // spatial_blocks) % channel_blocks) * channel_block
            plane = block // (spatial_blocks * channel_blocks)
            for spatial_inner, channel_lane in T.Parallel(spatial_block, channel_lanes):
                spatial = spatial_base + spatial_inner
                for channel_offset in T.serial(channel_block // channel_lanes):
                    c = channel_base + channel_offset * channel_lanes + channel_lane
                    if (spatial < spatial_size) & (c < c_in):
                        dst[plane, spatial, c] = src[plane, c, spatial]

        @T.macro
        def stage_operands(x, weight, x_nhwc, weight_krsc):
            with T.Kernel(x_blocks + w_blocks, threads=staging_threads) as block:
                if block < x_blocks:
                    plane_to_channel_last(
                        T.Tensor((n, c_in, h * w), dtype, x.data),
                        T.Tensor((n, h * w, c_in), dtype, x_nhwc.data),
                        block,
                        spatial_size=h * w,
                        spatial_block=x_spatial_block,
                        spatial_blocks=x_spatial_blocks,
                        channel_lanes=x_lanes,
                    )
                else:
                    plane_to_channel_last(
                        T.Tensor((c_out, c_in, kernel_size * kernel_size), dtype, weight.data),
                        T.Tensor((c_out, kernel_size * kernel_size, c_in), dtype, weight_krsc.data),
                        block - x_blocks,
                        spatial_size=kernel_size * kernel_size,
                        spatial_block=w_spatial_block,
                        spatial_blocks=w_spatial_blocks,
                        channel_lanes=w_lanes,
                    )

        @T.macro
        def conv_nhwc_implicit_gemm(x_nhwc, weight_krsc, out, bias):
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(n * out_h * out_w, block_m),
                threads=threads,
            ) as (bx, by):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_n, block_k), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                # Channel-major, so the store walks the spatial axis -- the contiguous
                # one in NCHW.
                out_shared = T.alloc_shared((block_n, block_m), dtype)

                weight_flat = T.Tensor((c_out, k_total), dtype, weight_krsc.data)
                out_nchw = T.Tensor((n, c_out, out_hw), dtype, out.data)

                T.use_swizzle(SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    T.im2col(x_nhwc, data_shared, by, k_iter, kernel_size, stride, dilation, pad)
                    T.copy(weight_flat[bx * block_n, k_iter * block_k], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local, transpose_B=True)

                for i, j in T.Parallel(block_m, block_n):
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[j, i] = T.cast(
                            out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype
                        )
                    else:
                        out_shared[j, i] = T.cast(out_local[i, j], dtype)

                # The region predicate and the config filter keep an m tile inside one
                # image, so the whole tile is one 2-D region of ``out[image]``.
                m_base = by * block_m
                T.copy(out_shared, out_nchw[m_base // out_hw, bx * block_n, m_base % out_hw])

        # A split of the contraction: grid z runs split_k CTAs over each output tile, each
        # over its own contiguous k_iters / split_k blocks, writing an f32 partial tile.
        k_blocks = -(-k_total // block_k)
        if k_blocks % split_k:
            raise ValueError(f"split_k={split_k} must divide the {k_blocks} k blocks")
        k_split = k_blocks // split_k
        # The partial sum runs one channel of one image per block row, four outputs a thread.
        sum_threads = 128
        sum_block = 4 * sum_threads

        @T.macro
        def conv_nhwc_implicit_gemm_split(x_nhwc, weight_krsc, partial):
            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(n * out_h * out_w, block_m),
                split_k,
                threads=threads,
            ) as (bx, by, bz):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_n, block_k), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_n, block_m), accum_dtype)

                weight_flat = T.Tensor((c_out, k_total), dtype, weight_krsc.data)

                T.use_swizzle(SWIZZLE_PANEL, enable=enable_rasterization)
                T.clear(out_local)

                for k_iter in T.Pipelined(k_split, num_stages=num_stages):
                    k_block = bz * k_split + k_iter
                    T.im2col(x_nhwc, data_shared, by, k_block, kernel_size, stride, dilation, pad)
                    T.copy(weight_flat[bx * block_n, k_block * block_k], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local, transpose_B=True)

                for i, j in T.Parallel(block_m, block_n):
                    out_shared[j, i] = out_local[i, j]

                m_base = by * block_m
                T.copy(out_shared, partial[bz, m_base // out_hw, bx * block_n, m_base % out_hw])

        def partial_sum(partial, bias, image, c, spatial):
            """The split_k partials of one output element, summed, plus its bias."""
            total = partial[0, image, c, spatial]
            for s in range(1, split_k):
                total = total + partial[s, image, c, spatial]
            if bias is not None:
                total = total + T.cast(bias[c], accum_dtype)
            return total

        @T.macro
        def sum_partials(partial, out, bias):
            out_nchw = T.Tensor((n, c_out, out_hw), dtype, out.data)
            with T.Kernel(T.ceildiv(out_hw, sum_block), c_out, n, threads=sum_threads) as (
                bx,
                c,
                image,
            ):
                for t in T.Parallel(sum_block):
                    spatial = bx * sum_block + t
                    if spatial < out_hw:
                        out_nchw[image, c, spatial] = T.cast(
                            partial_sum(partial, bias, image, c, spatial), dtype
                        )

        @T.macro
        def _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out, partial, bias):
            stage_operands(x, weight, x_nhwc, weight_krsc)
            if split_k == 1:
                conv_nhwc_implicit_gemm(x_nhwc, weight_krsc, out, bias)
            else:
                conv_nhwc_implicit_gemm_split(x_nhwc, weight_krsc, partial)
                sum_partials(partial, out, bias)

        if split_k > 1:
            partial_shape = (split_k, n, c_out, out_hw)
            if has_bias:

                @T.prim_func
                def _conv2d_symmetric_split_bias_main(
                    x: T.Tensor((n, c_in, h, w), dtype),
                    weight: T.Tensor((c_out, c_in, kernel_size, kernel_size), dtype),
                    x_nhwc: T.Tensor((n, h, w, c_in), dtype),
                    weight_krsc: T.Tensor((c_out, kernel_size, kernel_size, c_in), dtype),
                    out: T.Tensor((n, c_out, out_h, out_w), dtype),
                    partial: T.Tensor(partial_shape, accum_dtype),
                    bias: T.Tensor((c_out,), dtype),
                ):
                    _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out, partial, bias)

                return _conv2d_symmetric_split_bias_main

            @T.prim_func
            def _conv2d_symmetric_split_main(
                x: T.Tensor((n, c_in, h, w), dtype),
                weight: T.Tensor((c_out, c_in, kernel_size, kernel_size), dtype),
                x_nhwc: T.Tensor((n, h, w, c_in), dtype),
                weight_krsc: T.Tensor((c_out, kernel_size, kernel_size, c_in), dtype),
                out: T.Tensor((n, c_out, out_h, out_w), dtype),
                partial: T.Tensor(partial_shape, accum_dtype),
            ):
                _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out, partial, None)

            return _conv2d_symmetric_split_main

        if has_bias:

            @T.prim_func
            def _conv2d_symmetric_bias_main(
                x: T.Tensor((n, c_in, h, w), dtype),
                weight: T.Tensor((c_out, c_in, kernel_size, kernel_size), dtype),
                x_nhwc: T.Tensor((n, h, w, c_in), dtype),
                weight_krsc: T.Tensor((c_out, kernel_size, kernel_size, c_in), dtype),
                out: T.Tensor((n, c_out, out_h, out_w), dtype),
                bias: T.Tensor((c_out,), dtype),
            ):
                _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out, None, bias)

            return _conv2d_symmetric_bias_main

        @T.prim_func
        def _conv2d_symmetric_main(
            x: T.Tensor((n, c_in, h, w), dtype),
            weight: T.Tensor((c_out, c_in, kernel_size, kernel_size), dtype),
            x_nhwc: T.Tensor((n, h, w, c_in), dtype),
            weight_krsc: T.Tensor((c_out, kernel_size, kernel_size, c_in), dtype),
            out: T.Tensor((n, c_out, out_h, out_w), dtype),
        ):
            _conv2d_symmetric_body(x, weight, x_nhwc, weight_krsc, out, None, None)

        return _conv2d_symmetric_main

    return _conv2d_symmetric_func


def _gathered_gemm_default_config(arch: int) -> dict:
    """The tile ``Conv2dKernel`` and ``GroupConv2dKernel`` launch with before tuning."""
    if arch == 90:
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


class Conv2dSymmetricKernel(Kernel, Conv2dFwdInterface):
    """Dense Conv2d staged through an NHWC implicit GEMM.

    Serves the calls whose window, stride, padding and dilation are equal on both axes
    and whose channel count is whole 32-element steps, which is what makes the
    activation gather channel-contiguous.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    # The m tiles this kernel builds. One tuple serves the region, the default
    # configuration and the autotune filter, so widening it cannot leave the three
    # disagreeing about what an m tile can be.
    block_m_candidates: tuple[int, ...] = (64, 128)

    @classmethod
    def refusal(cls, call: Conv2dCall) -> str | None:
        if call.groups != 1:
            return "serves ungrouped convolution"
        if not (
            call.kernel_h == call.kernel_w
            and call.stride[0] == call.stride[1]
            and call.padding[0] == call.padding[1]
            and call.padding_end in (None, call.padding)
            and call.dilation[0] == call.dilation[1]
        ):
            return "serves a square kernel with equal stride, padding and dilation on both axes"
        if call.c_in % 32 != 0:
            return f"requires input channels a multiple of 32, got {call.c_in}"
        if not cls.tile_stays_in_one_image(call.n, call.out_hw, min(cls.block_m_candidates)):
            return "requires every output tile to stay in one image"
        m_tiles = -(-(call.n * call.out_hw) // min(cls.block_m_candidates))
        return super().refusal(call) or grid_refusal(y=m_tiles)

    @classmethod
    def entry_for(cls, call: Conv2dCall) -> Entry:
        """The cache identity and the thunk that builds this class for *call*.

        Its region is the symmetric one, so one extent of each pair speaks for both.
        """
        index = call.device_index
        args = (
            call.n,
            call.c_in,
            call.h,
            call.w,
            call.c_out,
            call.kernel_h,
            call.stride[0],
            call.padding[0],
            call.dilation[0],
            call.dtype,
        )
        return (*args, call.has_bias, index), lambda: cls(
            *args, has_bias=call.has_bias, device_index=index
        )

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
        config: dict | None = None,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.out_h = (h + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1
        self.out_w = (w + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1

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
        self.init_config(config)

    @staticmethod
    def tile_stays_in_one_image(n: int, out_hw: int, block_m: int) -> bool:
        """Whether an m tile of *block_m* rows never spans two of *n* images.

        The m axis runs over ``n * out_h * out_w``, and both ``T.im2col`` and the
        epilogue read the image index off the tile's first row. A tile that starts in
        one image and ends in the next would take the wrong image for its tail.
        """
        return n == 1 or out_hw % block_m == 0

    @property
    def default_config(self) -> dict:
        block_m, block_n, block_k = min(self.block_m_candidates), 256, 32
        # The output tile reuses the pipeline's shared memory, so the stages are all of
        # it: two where three do not fit (float32 on SM89's 99 KB).
        stage_bytes = (block_m + block_n) * block_k * self.dtype.itemsize
        fits = 3 * stage_bytes <= self.build_target.smem_budget
        tile = {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "num_stages": 3 if fits else 2,
            "threads": 256,
            "enable_rasterization": True,
        }
        if self.build_target.arch == 90 and self.dtype in (torch.float16, torch.bfloat16):
            # The wide tile for large channel counts.
            wide_m, wide_n, wide_k, wide_stages = 128, 256, 64, 3
            if self.c_out <= 64:
                tile = {
                    "block_m": block_m,
                    "block_n": 64,
                    "block_k": 32,
                    "num_stages": 3,
                    "threads": 128,
                    "enable_rasterization": True,
                }
            elif (
                self.c_out >= 256
                and self.c_in >= 256
                and self.out_h * self.out_w >= 8192
                and wide_m in self.block_m_candidates
                and self.tile_stays_in_one_image(self.n, self.out_h * self.out_w, wide_m)
                and self.c_in % wide_k == 0
                and wide_stages * (wide_m + wide_n) * wide_k * self.dtype.itemsize
                <= self.build_target.smem_budget
            ):
                tile = {
                    "block_m": wide_m,
                    "block_n": wide_n,
                    "block_k": wide_k,
                    "num_stages": wide_stages,
                    "threads": 256,
                    "enable_rasterization": False,
                }
        return {**tile, "split_k": self.split_k(tile)}

    @functools.cached_property
    def device_rates(self) -> tuple[float, float] | None:
        """The calibrated tensor-core rate for this dtype, in flop/s, and DRAM rate, in
        bytes/s, of the build device's GPU profile; ``None`` for a device without one."""
        device = self._build_device
        if device is None or device.type != "cuda":
            return None
        profile = find_profile(torch.cuda.get_device_name(device))
        roof = None if profile is None else resolve_roof(profile, tensor_core_roof(self.dtype))
        if roof is None or "effective" not in profile.get("hbm", {}):
            return None
        return roof["effective"], profile["hbm"]["effective"]

    def split_k(self, tile: dict) -> int:
        """How many CTAs share the contraction of each output tile of *tile*'s grid.

        A grid of ``tiles`` output tiles leaves ``sm_count - tiles`` SMs idle through the
        whole K loop. Split ``s`` ways, each CTA runs ``k_blocks / s`` of the K blocks and
        ``tiles * s`` CTAs stay in one wave. A split saves the ``k_blocks - k_blocks / s``
        blocks a CTA no longer runs, each priced at one SM's share of the tensor-core rate,
        and costs the f32 partials written and read back at the DRAM rate. ``s`` is the
        divisor of ``k_blocks`` that saves the most, 1 when none saves anything or the
        device has no GPU profile to price them. The partial tile is staged in shared
        memory beside the pipeline, so a split that does not fit there is not taken.
        """
        rates = self.device_rates
        tiles = -(-self.c_out // tile["block_n"]) * -(
            -(self.n * self.out_h * self.out_w) // tile["block_m"]
        )
        k_blocks = self.c_in * self.kernel_size**2 // tile["block_k"]
        pipeline_bytes = (
            tile["num_stages"]
            * (tile["block_m"] + tile["block_n"])
            * tile["block_k"]
            * self.dtype.itemsize
        )
        partial_bytes = tile["block_m"] * tile["block_n"] * 4
        if rates is None or pipeline_bytes + partial_bytes > self.build_target.smem_budget:
            return 1
        flops, dram = rates
        block_s = (
            2
            * tile["block_m"]
            * tile["block_n"]
            * tile["block_k"]
            / (flops / self.build_target.sm_count)
        )
        outputs = self.n * self.c_out * self.out_h * self.out_w

        def saved_s(split: int) -> float:
            return (k_blocks - k_blocks // split) * block_s - 2 * split * outputs * 4 / dram

        most = min(self.build_target.sm_count // tiles, k_blocks // tile["num_stages"])
        splits = [s for s in range(2, most + 1) if k_blocks % s == 0 and saved_s(s) > 0]
        return max(splits, key=saved_s, default=1)

    @property
    def autotune_configs(self) -> list[dict]:
        # No block_k of 16: the region already requires c_in to be a multiple of 32,
        # so a 32-wide k tile always divides it. Each tile is offered whole and, where
        # its grid leaves SMs idle, split.
        configs = conv_autotune_configs(
            self.dtype,
            self.build_target.smem_budget,
            block_m=list(self.block_m_candidates),
            block_k=[32, 64],
        )
        out_hw = self.out_h * self.out_w
        tiles = [
            c
            for c in configs
            if self.c_in % c["block_k"] == 0
            and self.tile_stays_in_one_image(self.n, out_hw, c["block_m"])
        ]
        return [
            {**tile, "split_k": split}
            for tile in tiles
            for split in sorted({1, self.split_k(tile)})
        ]

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
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
        split_k = self.config["split_k"]
        if split_k == 1:
            return launch(self, x, weight, x_nhwc, weight_krsc, bias=bias)
        partial = torch.empty(
            (split_k, self.n, self.c_out, self.out_h * self.out_w),
            device=x.device,
            dtype=torch.float32,
        )
        return launch(self, x, weight, x_nhwc, weight_krsc, partial, bias=bias)


class Conv2dKernel(Kernel, Conv2dFwdInterface):
    """Dense Conv2d over the full kernel window; serves every ungrouped call."""

    general = True
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: Conv2dCall) -> str | None:
        if call.groups != 1:
            return "serves ungrouped convolution"
        return super().refusal(call) or grid_refusal(z=call.n)

    @classmethod
    def entry_for(cls, call: Conv2dCall) -> Entry:
        index = call.device_index
        args = (
            call.n,
            call.c_in,
            call.h,
            call.w,
            call.c_out,
            call.kernel_h,
            call.kernel_w,
            call.stride[0],
            call.stride[1],
            call.padding[0],
            call.padding[1],
            call.dilation[0],
            call.dilation[1],
            call.dtype,
        )
        return (*args, call.has_bias, call.padding_end, index), lambda: cls(
            *args,
            pad_end=call.padding_end,
            has_bias=call.has_bias,
            device_index=index,
        )

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
        pad_end: tuple[int, ...] | None = None,
        config: dict | None = None,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dtype = dtype
        pad_h_end, pad_w_end = pad_end if pad_end is not None else (pad_h, pad_w)

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
            pad_h_end=pad_h_end,
            pad_w_end=pad_w_end,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return _gathered_gemm_default_config(self.build_target.arch)

    @property
    def autotune_configs(self) -> list[dict]:
        return conv_autotune_configs(
            self.dtype,
            self.build_target.smem_budget,
            block_m=[64, 128],
            block_k=[64, 128],
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return launch(self, x, weight, bias=bias)


class GroupConv2dKernel(Kernel, Conv2dFwdInterface):
    """Grouped Conv2d: one implicit GEMM per group over that group's channels."""

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def _grid_z(cls, call: Conv2dCall) -> int:
        """Blocks the program launches along grid z: one per image and group."""
        return call.n * call.groups

    @classmethod
    def refusal(cls, call: Conv2dCall) -> str | None:
        if call.groups <= 1:
            return "serves grouped convolution"
        return super().refusal(call) or grid_refusal(z=cls._grid_z(call))

    @classmethod
    def entry_for(cls, call: Conv2dCall) -> Entry:
        index = call.device_index
        args = (
            call.n,
            call.c_in,
            call.h,
            call.w,
            call.c_out,
            call.kernel_h,
            call.kernel_w,
            call.stride[0],
            call.stride[1],
            call.padding[0],
            call.padding[1],
            call.dilation[0],
            call.dilation[1],
            call.dtype,
        )
        group = (call.groups, call.c_in_g, call.c_out_g)
        return (*args, call.has_bias, *group, call.padding_end, index), lambda: cls(
            *args,
            pad_end=call.padding_end,
            has_bias=call.has_bias,
            groups=group[0],
            c_in_g=group[1],
            c_out_g=group[2],
            device_index=index,
        )

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
        pad_end: tuple[int, ...] | None = None,
        config: dict | None = None,
        *,
        groups: int,
        c_in_g: int,
        c_out_g: int,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
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
        self.pad_h_end, self.pad_w_end = pad_end if pad_end is not None else (pad_h, pad_w)
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.c_in_g = c_in_g
        self.c_out_g = c_out_g
        self.dtype = dtype
        self.has_bias = has_bias
        self.out_h = (h + pad_h + self.pad_h_end - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        self.out_w = (w + pad_w + self.pad_w_end - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        if self.groups <= 1:
            raise ValueError(f"{type(self).__name__} requires groups > 1")
        if self.c_in % self.groups or self.c_out % self.groups:
            raise ValueError(
                f"{type(self).__name__} requires c_in and c_out divisible by groups; "
                f"got c_in={self.c_in}, c_out={self.c_out}, groups={self.groups}"
            )
        self._build_program()
        self.init_config(config)

    def _build_program(self) -> None:
        """Compile the grouped implicit GEMM over this call's group shape."""
        self.kernel = _conv2d_group_kernel(
            self.n,
            self.c_in,
            self.h,
            self.w,
            self.c_out,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.dilation_h,
            self.dilation_w,
            self.has_bias,
            self.dtype_str,
            self.groups,
            self.c_in_g,
            self.c_out_g,
            pad_h_end=self.pad_h_end,
            pad_w_end=self.pad_w_end,
        )

    @property
    def default_config(self) -> dict:
        return _gathered_gemm_default_config(self.build_target.arch)

    @property
    def autotune_configs(self) -> list[dict]:
        return conv_autotune_configs(self.dtype, self.build_target.smem_budget, threads=[128])

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return launch(self, x, weight, bias=bias)


class DepthwiseConv2dKernel(GroupConv2dKernel):
    """Depthwise Conv2d: one input channel per group, one output channel per group.

    A different program from :class:`GroupConv2dKernel`: with a single channel per group
    there is no contraction to tile, so each thread accumulates a strip of output
    positions over the kernel window instead of running an implicit GEMM.
    """

    preferred_over = frozenset({"group_conv2d"})

    @classmethod
    def refusal(cls, call: Conv2dCall) -> str | None:
        if call.c_in_g != 1 or call.c_out_g != 1:
            return "serves one input and one output channel per group"
        return super().refusal(call)

    @classmethod
    def _grid_z(cls, call: Conv2dCall) -> int:
        """One block row per image, not per group."""
        return call.n

    def _build_program(self) -> None:
        self.kernel = _conv2d_depthwise_kernel(
            self.n,
            self.c_in,
            self.h,
            self.w,
            self.c_out,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.dilation_h,
            self.dilation_w,
            self.has_bias,
            self.dtype_str,
            pad_h_end=self.pad_h_end,
            pad_w_end=self.pad_w_end,
        )

    @property
    def default_config(self) -> dict:
        # One channel per block row, a strip of outputs per block column.
        return {
            "block_m": 1,
            "block_n": 128,
            "threads": 128,
            "enable_rasterization": not (
                self.build_target.arch == 90 and self.out_h * self.out_w >= 4096
            ),
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # One tile shape, since there is no GEMM to tile. The swizzle is still a
        # choice, so both ways are searched.
        return [{**self.default_config, "enable_rasterization": value} for value in (False, True)]


@target_cache(maxsize=32)
def _conv2d_depthwise_warp_kernel(
    n: int, channels: int, h: int, w: int, dtype: str, has_bias: bool
):
    @tilelang.jit(out_idx=[2])
    def build():
        @T.macro
        def convolve(x, weight, y, bias):
            # Four warps of two output rows cover an eight-row band; a lane takes two columns.
            with T.Kernel(h // 8, channels, n, threads=128) as (band, channel, image):
                thread = T.get_thread_binding()
                column = thread % WARP_LANES * 2
                row = band * 8 + thread // WARP_LANES * 2
                even = T.alloc_local((4,), "float32")
                odd = T.alloc_local((4,), "float32")
                left = T.alloc_local((4,), "float32")
                right = T.alloc_local((4,), "float32")
                values = T.alloc_local((2, 2), "float32")
                out = T.alloc_local((2, 2), dtype)
                T.clear(values)

                # Two output rows reuse four input rows. Adjacent lanes exchange the
                # outer columns of their two-pixel strips.
                for tap in T.unroll(4):
                    ih = row + tap - 1
                    if column < w and 0 <= ih < h:
                        even[tap] = T.cast(x[image, channel, ih, column], "float32")
                        odd[tap] = T.cast(x[image, channel, ih, column + 1], "float32")
                    else:
                        even[tap] = 0.0
                        odd[tap] = 0.0
                    left[tap] = T.shfl_up(odd[tap], 1)
                    right[tap] = T.shfl_down(even[tap], 1)

                for tap in T.unroll(3):
                    wl = T.cast(weight[channel, 0, tap, 0], "float32")
                    wc = T.cast(weight[channel, 0, tap, 1], "float32")
                    wr = T.cast(weight[channel, 0, tap, 2], "float32")
                    for j in T.unroll(2):
                        i = j + tap
                        if 0 <= row + i - 1 < h:
                            values[j, 0] += T.if_then_else(column > 0, left[i] * wl, 0.0)
                            values[j, 0] += even[i] * wc
                            values[j, 0] += odd[i] * wr
                            values[j, 1] += even[i] * wl
                            values[j, 1] += odd[i] * wc
                            values[j, 1] += T.if_then_else(column + 2 < w, right[i] * wr, 0.0)

                if column < w:
                    for j in T.unroll(2):
                        for k in T.unroll(2):
                            if has_bias:
                                out[j, k] = T.cast(
                                    values[j, k] + T.cast(bias[channel], "float32"), dtype
                                )
                            else:
                                out[j, k] = T.cast(values[j, k], dtype)
                        for k in T.vectorized(2):
                            y[image, channel, row + j, column + k] = out[j, k]

        if has_bias:

            @T.prim_func
            def main(
                x: T.Tensor[(n, channels, h, w), dtype],
                weight: T.Tensor[(channels, 1, 3, 3), dtype],
                y: T.Tensor[(n, channels, h, w), dtype],
                bias: T.Tensor[(channels,), dtype],
            ):
                convolve(x, weight, y, bias)

        else:

            @T.prim_func
            def main(
                x: T.Tensor[(n, channels, h, w), dtype],
                weight: T.Tensor[(channels, 1, 3, 3), dtype],
                y: T.Tensor[(n, channels, h, w), dtype],
            ):
                convolve(x, weight, y, None)

        return main

    return build


class DepthwiseConv2dWarpKernel(Kernel, Conv2dFwdInterface):
    """Reuse overlapping 3-by-3 depthwise windows across registers and warp lanes."""

    supported_archs = [90]
    preferred_over = frozenset({"depthwise_conv2d"})

    @classmethod
    def refusal(cls, call: Conv2dCall) -> str | None:
        if not (
            call.c_in == call.c_out == call.groups > 1
            and call.kernel_h == call.kernel_w == 3
            and call.stride == call.dilation == (1, 1)
            and call.padding == (1, 1)
            and call.padding_end in (None, (1, 1))
        ):
            return "requires a depthwise 3-by-3 window with one pixel of symmetric padding"
        if not (call.h % 8 == 0 and 16 <= call.w <= 64 and call.w % 2 == 0):
            return "requires full eight-row bands and an even width from 16 to 64"
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires fp16 or bf16"
        return (
            super().refusal(call)
            or grid_refusal(y=call.c_out, z=call.n)
            or operand_refusal(call.n * call.c_in * call.h * call.w)
        )

    @classmethod
    def entry_for(cls, call: Conv2dCall) -> Entry:
        index = call.device_index
        args = (call.n, call.c_in, call.h, call.w, call.dtype, call.has_bias)
        return (*args, index), lambda: cls(*args, device_index=index)

    def __init__(self, n, channels, h, w, dtype, has_bias, *, device_index=None):
        super().__init__(device_index=device_index)
        self.dtype = dtype
        self.kernel = _conv2d_depthwise_warp_kernel(n, channels, h, w, self.dtype_str, has_bias)
        self.init_config()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return launch(self, x, weight, bias=bias)


class Conv2d1x1Kernel(Kernel, Conv2dFwdInterface):
    """Dense 1x1 Conv2d, which lowers to a pointwise GEMM over the channel axis.

    A 1x1 window is symmetric, so this region nests inside
    :class:`Conv2dSymmetricKernel`'s wherever the channel count admits it; the
    pointwise program has no window to gather and wins there.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"conv2d_symmetric"})

    @classmethod
    def refusal(cls, call: Conv2dCall) -> str | None:
        if not (
            call.groups == 1
            and call.kernel_h == 1
            and call.kernel_w == 1
            and call.stride == (1, 1)
            and call.padding == (0, 0)
            and call.padding_end in (None, (0, 0))
            and call.dilation == (1, 1)
        ):
            return "serves an ungrouped 1x1 convolution: stride 1, no padding or dilation"
        return super().refusal(call) or grid_refusal(z=call.n)

    @classmethod
    def entry_for(cls, call: Conv2dCall) -> Entry:
        index = call.device_index
        args = (
            call.n,
            call.c_in,
            call.h,
            call.w,
            call.c_out,
            call.dtype,
        )
        return (*args, call.has_bias, index), lambda: cls(
            *args, has_bias=call.has_bias, device_index=index
        )

    def __init__(
        self,
        n: int,
        c_in: int,
        h: int,
        w: int,
        c_out: int,
        dtype: torch.dtype,
        has_bias: bool = False,
        config: dict | None = None,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.c_in = c_in
        self.c_out = c_out
        self.dtype = dtype
        self.kernel = _conv2d_1x1_kernel(n, c_in, h, w, c_out, has_bias, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        if self.build_target.arch == 90:
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
        # A 32-wide n tile, which the rest of the family does not reach: the block
        # count is C_out * H * W / (block_m * block_n), and a pointwise row over a
        # small map has no other way to fill the device.
        return conv_autotune_configs(
            self.dtype,
            self.build_target.smem_budget,
            block_m=[64, 128],
            block_n=[32, 64, 128, 256],
            threads=[128],
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # OIHW -> OC,IC since the 1x1 kernel consumes a dense [C_out, C_in] weight matrix.
        weight_oc_ci = weight.view(self.c_out, self.c_in).contiguous()
        return launch(self, x, weight_oc_ci, bias=bias)
