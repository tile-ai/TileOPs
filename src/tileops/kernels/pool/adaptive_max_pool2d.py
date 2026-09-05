import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from .common import (
    AdaptivePool2dKernelBase,
    adaptive_bin,
    fits_static_shared,
    max_adaptive_bin_extent,
)

__all__ = ["AdaptiveMaxPool2dKernel", "AdaptiveMaxPool2dWithIndicesKernel"]


def _stage_planes(
    block_m: int, out_h: int, out_w: int, c_in: int, h_in: int, w_in: int, dtype: str
) -> Optional[int]:
    """(batch, channel) planes a block stages in shared, or None to read global.

    Neighbouring outputs read bins that start ``w_in // out_w`` apart. A block
    holding whole planes reads each plane once instead.
    """
    planes = max(1, block_m // (out_h * out_w))
    # Dividing c_in keeps every block inside one image, so the copy is one slice.
    if planes > c_in or c_in % planes:
        return None
    if not fits_static_shared(planes * h_in * w_in, dtype):
        return None
    return planes


@functools.lru_cache(maxsize=32)
def _adaptive_max_pool2d_kernel(
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
    def _adaptive_max_pool2d_func(block_m: int, threads: int):
        @T.prim_func
        def _adaptive_max_pool2d_main(
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

                        max_val = T.alloc_var(T.float32)
                        has_nan = T.alloc_var(T.bool)
                        max_val = T.cast(float("-inf"), accum_dtype)
                        has_nan = False
                        # Static-bound loops (TileLang rejects dynamic T.serial
                        # bounds); guard skips lanes outside this output's bin.
                        for kh in T.serial(max_kh):
                            for kw in T.serial(max_kw):
                                if ih_start + kh < ih_end and iw_start + kw < iw_end:
                                    val = T.cast(
                                        x[batch, c_idx, ih_start + kh, iw_start + kw],
                                        accum_dtype,
                                    )
                                    has_nan = has_nan | T.isnan(val)
                                    max_val = T.max(max_val, val)

                        result = T.if_then_else(
                            has_nan,
                            T.cast(float("nan"), accum_dtype),
                            max_val,
                        )
                        out[batch, c_idx, oh, ow] = T.cast(result, dtype)

        return _adaptive_max_pool2d_main

    return _adaptive_max_pool2d_func


def _launch_adaptive_max_pool2d(
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
    return _adaptive_max_pool2d_kernel(n, c_in, h_in, w_in, out_h, out_w, dtype)(block_m, threads)(
        x
    )


class AdaptiveMaxPool2dKernel(AdaptivePool2dKernelBase):
    """Adaptive max pooling forward kernel for NCHW inputs."""

    _build = staticmethod(_adaptive_max_pool2d_kernel)
    _dispatch = staticmethod(_launch_adaptive_max_pool2d)


@functools.lru_cache(maxsize=32)
def _adaptive_max_pool2d_with_indices_kernel(
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

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _adaptive_max_pool2d_with_indices_func(block_m: int, threads: int):
        stage_planes = _stage_planes(block_m, out_h, out_w, c_in, h_in, w_in, dtype)
        # The flat index spans one h_in * w_in plane. Carrying it as int32
        # keeps the div/mod the compiler sinks into the update path off the
        # 64-bit path; the stored index stays int64 to match PyTorch.
        idx_dtype = "int32" if h_in * w_in < 2**31 else "int64"

        @T.macro
        def _reduce_bin(src, src_c, src_p, oh, ow, out, indices, out_c, out_p):
            """Store the max and its index over one adaptive bin of ``src[src_c, src_p]``."""
            ih_start, ih_end = adaptive_bin(oh, h_in, out_h)
            iw_start, iw_end = adaptive_bin(ow, w_in, out_w)

            max_val = T.alloc_var(T.float32)
            has_nan = T.alloc_var(T.bool)
            max_idx = T.alloc_var(idx_dtype)
            nan_idx = T.alloc_var(idx_dtype)
            first_valid = T.alloc_var(T.bool)
            max_val = T.cast(float("-inf"), accum_dtype)
            has_nan = False
            max_idx = T.cast(0, idx_dtype)
            nan_idx = T.cast(0, idx_dtype)
            first_valid = True
            # Static-bound loops (TileLang rejects dynamic T.serial
            # bounds); guard skips lanes outside this output's bin.
            for kh in T.serial(max_kh):
                for kw in T.serial(max_kw):
                    if ih_start + kh < ih_end and iw_start + kw < iw_end:
                        ih = ih_start + kh
                        iw = iw_start + kw
                        val = T.cast(src[src_c, src_p, ih, iw], accum_dtype)
                        flat_idx = T.cast(ih * w_in + iw, idx_dtype)
                        is_nan = T.isnan(val)
                        has_nan = has_nan | is_nan
                        # PyTorch records the last NaN visited in a window.
                        nan_idx = T.if_then_else(is_nan, flat_idx, nan_idx)
                        # first_valid: an all--inf window still takes its first element.
                        take = (not is_nan) and (first_valid or val > max_val)
                        max_val = T.if_then_else(take, val, max_val)
                        max_idx = T.if_then_else(take, flat_idx, max_idx)
                        first_valid = first_valid and is_nan

            result = T.if_then_else(
                has_nan,
                T.cast(float("nan"), accum_dtype),
                max_val,
            )
            out[out_c, out_p, oh, ow] = T.cast(result, dtype)
            indices[out_c, out_p, oh, ow] = T.cast(
                T.if_then_else(has_nan, nan_idx, max_idx), "int64"
            )

        @T.prim_func
        def _adaptive_max_pool2d_with_indices_main(
            x: T.Tensor((n, c_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((n, c_in, out_h, out_w), dtype),  # type: ignore
            indices: T.Tensor((n, c_in, out_h, out_w), "int64"),  # type: ignore
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
                        _reduce_bin(x, batch, c_idx, oh, ow, out, indices, batch, c_idx)

        if stage_planes is not None:
            # Leading axis of 1 so the macro indexes the tile and x alike.
            @T.prim_func
            def _adaptive_max_pool2d_with_indices_staged_main(
                x: T.Tensor((n, c_in, h_in, w_in), dtype),  # type: ignore
                out: T.Tensor((n, c_in, out_h, out_w), dtype),  # type: ignore
                indices: T.Tensor((n, c_in, out_h, out_w), "int64"),  # type: ignore
            ):
                with T.Kernel(T.ceildiv(n * c_in, stage_planes), threads=threads) as bx:
                    tile = T.alloc_shared((1, stage_planes, h_in, w_in), dtype)
                    batch = bx * stage_planes // c_in
                    c_base = bx * stage_planes % c_in
                    T.copy(
                        x[batch, c_base : c_base + stage_planes, 0:h_in, 0:w_in],
                        tile[0, :, 0:h_in, 0:w_in],
                    )
                    for i in T.Parallel(stage_planes * out_h * out_w):
                        ow = i % out_w
                        spatial_idx = i // out_w
                        oh = spatial_idx % out_h
                        plane = spatial_idx // out_h
                        _reduce_bin(tile, 0, plane, oh, ow, out, indices, batch, c_base + plane)

            return _adaptive_max_pool2d_with_indices_staged_main

        return _adaptive_max_pool2d_with_indices_main

    return _adaptive_max_pool2d_with_indices_func


def _launch_adaptive_max_pool2d_with_indices(
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _adaptive_max_pool2d_with_indices_kernel(n, c_in, h_in, w_in, out_h, out_w, dtype)(
        block_m, threads
    )(x)


class AdaptiveMaxPool2dWithIndicesKernel(AdaptivePool2dKernelBase):
    """Adaptive max pooling forward kernel returning values and int64 indices."""

    _build = staticmethod(_adaptive_max_pool2d_with_indices_kernel)
    _dispatch = staticmethod(_launch_adaptive_max_pool2d_with_indices)
