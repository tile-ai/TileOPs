import functools
from typing import Tuple

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

# Staged bytes a block aims for. Small keeps the grid longer than the device has
# multiprocessors, which is what these shapes are short of: the whole input is a
# megabyte or two, so a block that takes more planes only empties the grid.
_TILE_BYTES = 4096

# Elements one thread carries in the staging copy. The copy is the kernel's whole
# memory cost, so the block width follows from it rather than from the output count.
_COPY_RUN = 8

# Widest staged tile the tuned space offers. Past this a block holds more shared memory
# and the grid holds fewer blocks, which is the wrong direction for a shape whose whole
# input is a megabyte or two, so those plane counts are not worth a tuning run.
_TUNE_TILE_BYTES = 4 * _TILE_BYTES


def _divisors(value: int) -> Tuple[int, ...]:
    return tuple(d for d in range(1, value + 1) if value % d == 0)


def _spread(values: Tuple[int, ...], limit: int) -> Tuple[int, ...]:
    """At most ``limit`` of ``values``, both ends kept and the rest evenly spaced."""
    if len(values) <= limit:
        return values
    step = (len(values) - 1) / (limit - 1)
    return tuple(sorted({values[round(i * step)] for i in range(limit)}))


def _plane_counts(rows: int, plane: int, dtype: str) -> Tuple[int, ...]:
    """Planes a block may take at once: divisors of ``rows`` whose tile fits shared.

    A run of planes is contiguous in the flat ``(rows, h_in, w_in)`` view whatever the
    batch and channel extents are, so the only constraints are the shared budget and an
    even split of the grid. Every divisor is offered when one plane alone will not fit,
    where the reduction reads global memory instead.
    """
    if not fits_static_shared(plane, dtype):
        return _divisors(rows)
    return tuple(d for d in _divisors(rows) if fits_static_shared(d * plane, dtype))


def _check_planes(planes: int, rows: int) -> None:
    """Refuse a plane count the grid cannot cover.

    The grid is ``rows // planes`` blocks of ``planes`` planes each, so a count that does
    not divide ``rows`` would leave the last planes unwritten. Both config sources draw
    from :func:`_plane_counts`, which offers divisors only; this catches a hand-written
    config before it returns uninitialized output.
    """
    if rows % planes:
        raise ValueError(f"planes={planes} must divide rows={rows}")


def _block_threads(planes: int, plane: int) -> int:
    width = 1 << max(0, (planes * plane // _COPY_RUN - 1).bit_length())
    return min(512, max(128, width))


def _default_config(rows: int, h_in: int, w_in: int, dtype: str) -> dict:
    plane = h_in * w_in
    itemsize = 4 if dtype in ("float", "float32") else 2
    counts = _plane_counts(rows, plane, dtype)
    fitting = [p for p in counts if p * plane * itemsize <= _TILE_BYTES]
    planes = max(fitting) if fitting else min(counts)
    return {"planes": planes, "threads": _block_threads(planes, plane)}


def _autotune_configs(rows: int, h_in: int, w_in: int, dtype: str) -> list[dict]:
    plane = h_in * w_in
    itemsize = 4 if dtype in ("float", "float32") else 2
    counts = _plane_counts(rows, plane, dtype)
    worth = tuple(p for p in counts if p * plane * itemsize <= _TUNE_TILE_BYTES)
    return [
        {"planes": planes, "threads": threads}
        for planes in _spread(worth or counts[:1], 4)
        for threads in (128, 256, 512)
    ]


def _bin_extent(size_in: int, size_out: int) -> Tuple[int, bool]:
    """The widest bin on this axis, and whether every bin on it is that wide."""
    extent = max_adaptive_bin_extent(size_in, size_out)
    return extent, all(
        e - s == extent for s, e in (adaptive_bin(o, size_in, size_out) for o in range(size_out))
    )


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
    rows = n * c_in
    plane = h_in * w_in
    out_plane = out_h * out_w
    max_kh, uniform_h = _bin_extent(h_in, out_h)
    max_kw, uniform_w = _bin_extent(w_in, out_w)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _adaptive_max_pool2d_func(planes: int, threads: int):
        _check_planes(planes, rows)
        staged = fits_static_shared(planes * plane, dtype)

        @T.macro
        def _max_bin(src, src_plane, dst, dst_plane, oh, ow):
            """Store the max over one adaptive bin of ``src[src_plane]``."""
            ih_start, ih_end = adaptive_bin(oh, h_in, out_h)
            iw_start, iw_end = adaptive_bin(ow, w_in, out_w)
            run = T.alloc_var(T.float32)
            run = -T.infinity(accum_dtype)
            # Static-bound loops: TileLang rejects a dynamic T.serial bound, so the
            # widest bin sets the trip count and a short bin skips its missing taps.
            for kh in T.serial(max_kh):
                ih = ih_start + kh
                if uniform_h or ih < ih_end:
                    for kw in T.serial(max_kw):
                        iw = iw_start + kw
                        if uniform_w or iw < iw_end:
                            v = T.cast(src[src_plane, ih, iw], accum_dtype)
                            # NaN enters `run` and never leaves, since a later value
                            # fails `v > NaN`.
                            run = T.if_then_else(T.isnan(v) or (v > run), v, run)
            dst[dst_plane, oh, ow] = T.cast(run, dtype)

        @T.prim_func
        def _adaptive_max_pool2d_main(
            x: T.Tensor((rows, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(rows // planes, threads=threads) as bx:
                base = bx * planes
                if staged:
                    tile = T.alloc_shared((planes, h_in, w_in), dtype)
                    T.copy(x[base : base + planes, :, :], tile)
                for i in T.Parallel(planes * out_plane):
                    p = i // out_plane
                    o = i - p * out_plane
                    oh = o // out_w
                    ow = o - oh * out_w
                    if staged:
                        _max_bin(tile, p, out, base + p, oh, ow)
                    else:
                        _max_bin(x, base + p, out, base + p, oh, ow)

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
    config: dict,
    x: torch.Tensor,
) -> torch.Tensor:
    kernel = _adaptive_max_pool2d_kernel(n, c_in, h_in, w_in, out_h, out_w, dtype)(**config)
    return kernel(x.reshape(n * c_in, h_in, w_in)).view(n, c_in, out_h, out_w)


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
    rows = n * c_in
    plane = h_in * w_in
    out_plane = out_h * out_w
    max_kh, uniform_h = _bin_extent(h_in, out_h)
    max_kw, uniform_w = _bin_extent(w_in, out_w)
    # The flat index spans one h_in * w_in plane. Carrying it as int32 keeps the
    # arithmetic on the update path off the 64-bit path; the stored index is int64 to
    # match PyTorch.
    idx_dtype = "int32" if plane < 2**31 else "int64"

    @tilelang.jit(out_idx=[1, 2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _adaptive_max_pool2d_with_indices_func(planes: int, threads: int):
        _check_planes(planes, rows)
        staged = fits_static_shared(planes * plane, dtype)

        @T.macro
        def _argmax_bin(src, src_plane, dst, indices, dst_plane, oh, ow):
            """Store the max and its flat position over one adaptive bin of ``src``."""
            ih_start, ih_end = adaptive_bin(oh, h_in, out_h)
            iw_start, iw_end = adaptive_bin(ow, w_in, out_w)
            run = T.alloc_var(T.float32)
            best = T.alloc_var(idx_dtype)
            run = -T.infinity(accum_dtype)
            # Bins are never empty, so the first tap is in range and seeds the
            # position: a bin holding nothing but -inf reports that tap.
            best = T.cast(ih_start * w_in + iw_start, idx_dtype)
            for kh in T.serial(max_kh):
                ih = ih_start + kh
                if uniform_h or ih < ih_end:
                    for kw in T.serial(max_kw):
                        iw = iw_start + kw
                        if uniform_w or iw < iw_end:
                            v = T.cast(src[src_plane, ih, iw], accum_dtype)
                            # Strict > keeps the first maximum; a NaN takes the position
                            # and holds it, so the last NaN in the bin wins.
                            take = T.isnan(v) or (v > run)
                            run = T.if_then_else(take, v, run)
                            best = T.if_then_else(take, T.cast(ih * w_in + iw, idx_dtype), best)
            dst[dst_plane, oh, ow] = T.cast(run, dtype)
            indices[dst_plane, oh, ow] = T.cast(best, "int64")

        @T.prim_func
        def _adaptive_max_pool2d_with_indices_main(
            x: T.Tensor((rows, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_h, out_w), dtype),  # type: ignore
            indices: T.Tensor((rows, out_h, out_w), "int64"),  # type: ignore
        ):
            with T.Kernel(rows // planes, threads=threads) as bx:
                base = bx * planes
                if staged:
                    tile = T.alloc_shared((planes, h_in, w_in), dtype)
                    T.copy(x[base : base + planes, :, :], tile)
                for i in T.Parallel(planes * out_plane):
                    p = i // out_plane
                    o = i - p * out_plane
                    oh = o // out_w
                    ow = o - oh * out_w
                    if staged:
                        _argmax_bin(tile, p, out, indices, base + p, oh, ow)
                    else:
                        _argmax_bin(x, base + p, out, indices, base + p, oh, ow)

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
    config: dict,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel = _adaptive_max_pool2d_with_indices_kernel(n, c_in, h_in, w_in, out_h, out_w, dtype)(
        **config
    )
    values, indices = kernel(x.reshape(n * c_in, h_in, w_in))
    return values.view(n, c_in, out_h, out_w), indices.view(n, c_in, out_h, out_w)


class _AdaptiveMaxPool2dKernelBase(AdaptivePool2dKernelBase):
    """Plane-staged config policy shared by the two adaptive max-pool kernels."""

    @property
    def default_config(self) -> dict:
        return _default_config(self.n * self.c_in, self.h_in, self.w_in, self.dtype_str)

    @property
    def autotune_configs(self) -> list[dict]:
        return _autotune_configs(self.n * self.c_in, self.h_in, self.w_in, self.dtype_str)


class AdaptiveMaxPool2dKernel(_AdaptiveMaxPool2dKernelBase):
    """Adaptive max pooling forward kernel for NCHW inputs."""

    _build = staticmethod(_adaptive_max_pool2d_kernel)
    _dispatch = staticmethod(_launch_adaptive_max_pool2d)


class AdaptiveMaxPool2dWithIndicesKernel(_AdaptiveMaxPool2dKernelBase):
    """Adaptive max pooling forward kernel returning values and int64 indices."""

    _build = staticmethod(_adaptive_max_pool2d_with_indices_kernel)
    _dispatch = staticmethod(_launch_adaptive_max_pool2d_with_indices)
