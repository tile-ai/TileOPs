import functools

import tilelang
import tilelang.language as T
import torch

from .common import AdaptivePool2dKernelBase, adaptive_bin, max_adaptive_bin_extent

__all__ = ["AdaptiveAvgPool2dKernel"]


@functools.lru_cache(maxsize=32)
def _adaptive_avg_pool2d_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str = "float16",
):
    accum_dtype = "float"
    total_output = n * c_in * out_h * out_w
    max_kh = max_adaptive_bin_extent(h_in, out_h)
    max_kw = max_adaptive_bin_extent(w_in, out_w)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _adaptive_avg_pool2d_func(block_m: int, threads: int):
        @T.prim_func
        def _adaptive_avg_pool2d_main(
            x: T.Tensor((n, c_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, block_m), threads=threads) as bx:
                for i in T.Parallel(block_m):
                    out_idx = bx * block_m + i
                    if out_idx < total_output:
                        ow = out_idx % out_w
                        spatial_idx = out_idx // out_w
                        oh = spatial_idx % out_h
                        channel_batch_idx = spatial_idx // out_h
                        c_idx = channel_batch_idx % c_in
                        batch = channel_batch_idx // c_in

                        ih_start, ih_end = adaptive_bin(oh, h_in, out_h)
                        iw_start, iw_end = adaptive_bin(ow, w_in, out_w)

                        sum_val = T.alloc_var(T.float32)
                        sum_val = T.cast(0.0, accum_dtype)
                        # Static-bound loops (TileLang rejects dynamic T.serial
                        # bounds); guard skips lanes outside this output's bin.
                        for kh in T.serial(max_kh):
                            for kw in T.serial(max_kw):
                                if ih_start + kh < ih_end and iw_start + kw < iw_end:
                                    sum_val += T.cast(
                                        x[batch, c_idx, ih_start + kh, iw_start + kw],
                                        accum_dtype,
                                    )

                        bin_size = (ih_end - ih_start) * (iw_end - iw_start)
                        out[batch, c_idx, oh, ow] = T.cast(
                            sum_val / T.cast(bin_size, accum_dtype),
                            dtype,
                        )

        return _adaptive_avg_pool2d_main

    return _adaptive_avg_pool2d_func


@torch.library.custom_op("top::adaptive_avg_pool2d_wrapped_kernel", mutates_args=())
def _adaptive_avg_pool2d_wrapped_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _adaptive_avg_pool2d_kernel(
        n, c_in, h_in, w_in, out_h, out_w, dtype
    )(block_m, threads)(x)


@_adaptive_avg_pool2d_wrapped_kernel.register_fake
def _(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    _ = (dtype, block_m, threads)
    return torch.empty((n, c_in, out_h, out_w), dtype=x.dtype, device=x.device)


class AdaptiveAvgPool2dKernel(AdaptivePool2dKernelBase):
    """Adaptive average pooling forward kernel for NCHW inputs."""

    _build = staticmethod(_adaptive_avg_pool2d_kernel)
    _dispatch = staticmethod(_adaptive_avg_pool2d_wrapped_kernel)
