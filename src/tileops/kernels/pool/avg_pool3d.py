import functools
from typing import ClassVar, NamedTuple, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.pool.common import dtype_itemsize, pool_output_dim

__all__ = ["AvgPool3dKernel", "AvgPool3dSpatialKernel"]

# Window sums promote to fp32 and cast back at the store: a fp16 accumulator loses the
# low bits of a window this wide.
_ACCUM_DTYPE = "float"


class _Window(NamedTuple):
    """One avg_pool3d problem, and the extents and facts that follow from it.

    The builders below are cached on this, so everything they derive is derived here
    once and cannot differ between them.
    """

    rows: int
    size: Tuple[int, int, int]
    kernel: Tuple[int, int, int]
    stride: Tuple[int, int, int]
    pad: Tuple[int, int, int]
    ceil_mode: bool
    count_include_pad: bool
    divisor_override: Optional[int]

    @property
    def out(self) -> Tuple[int, int, int]:
        return tuple(
            pool_output_dim(size, k, s, p, self.ceil_mode)
            for size, k, s, p in zip(self.size, self.kernel, self.stride, self.pad, strict=True)
        )

    @property
    def outputs(self) -> int:
        out_d, out_h, out_w = self.out
        return self.rows * out_d * out_h * out_w

    @property
    def window_inside(self) -> bool:
        """Whether every window lies inside the volume, so no tap carries a test.

        True also settles the output extent: a window that fits is one the ceil-mode and
        the floor-mode formula both count.
        """
        return all(
            p == 0 and (o - 1) * s + k <= size
            for size, o, k, s, p in zip(
                self.size, self.out, self.kernel, self.stride, self.pad, strict=True
            )
        )

    @property
    def whole_window_divides(self) -> bool:
        """Whether one divisor covers every output.

        Without ceil mode a window reaches ``size + pad`` at the furthest, so counting
        the padding gives every output the whole kernel. Ceil mode can overhang that, and
        uncounted padding shortens the windows that do.
        """
        return self.window_inside or (self.count_include_pad and not self.ceil_mode)

    @property
    def divisor(self) -> int:
        """The divisor every window takes where one covers them all.

        An explicit divisor is used as given, negative included.
        """
        if self.divisor_override is not None:
            return self.divisor_override
        kernel_d, kernel_h, kernel_w = self.kernel
        return kernel_d * kernel_h * kernel_w

    @property
    def overlap(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Per axis, the half-open range a window's own divisor counts.

        Uncounted padding cuts the range back to the volume itself.
        """
        if self.count_include_pad:
            return (
                tuple(-p for p in self.pad),
                tuple(size + p for size, p in zip(self.size, self.pad, strict=True)),
            )
        return ((0, 0, 0), self.size)


class _WideRun(NamedTuple):
    """Outputs one thread takes along w, and the access width it reads them with."""

    run: int
    vector_elems: int


def _wide_run(window: _Window, dtype: str) -> Optional[_WideRun]:
    """The narrowest run of outputs whose taps reach the widest access the w axis admits.

    ``run`` consecutive outputs cover ``run * stride_w`` consecutive inputs, and the
    windows tile that stretch, so a thread that takes them reads each of its input rows
    as whole groups rather than one element per tap, and reads nothing twice or in vain.
    A narrower run holding the same width costs fewer registers and leaves more threads,
    so the search takes the first run that reaches the widest access.

    Returns None where the w axis admits no such run: the windows have to lie inside the
    volume and tile the row, and the group width has to divide both the run's span and
    the input row, so that every group starts at a multiple of its own width.
    """
    kernel_w, stride_w, out_w = window.kernel[2], window.stride[2], window.out[2]
    w_in = window.size[2]
    if not window.window_inside or kernel_w != stride_w:
        return None
    # Past one full-width access a longer run only adds groups, so it bounds the search.
    widest = VECTOR_ACCESS_BYTES // dtype_itemsize(dtype)
    best = None
    run = 1
    while run <= widest:
        span = run * stride_w
        vector_elems = min(widest, span)
        while vector_elems > 1 and (span % vector_elems or w_in % vector_elems):
            vector_elems //= 2
        if out_w % run == 0 and vector_elems > (best.vector_elems if best else 1):
            best = _WideRun(run, vector_elems)
        run *= 2
    return best


@functools.lru_cache(maxsize=32)
def _avg_pool3d_kernel(window: _Window, dtype: str):
    """One output per thread, every tap read where it lies."""
    d_in, h_in, w_in = window.size
    kernel_d, kernel_h, kernel_w = window.kernel
    stride_d, stride_h, stride_w = window.stride
    pad_d, pad_h, pad_w = window.pad
    out_d, out_h, out_w = window.out
    (low_d, low_h, low_w), (limit_d, limit_h, limit_w) = window.overlap
    rows, total = window.rows, window.outputs
    window_inside = window.window_inside
    one_divisor = window.whole_window_divides or window.divisor_override is not None
    divisor = window.divisor

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool3d_func(threads: int):
        # A compile-time truth when the launch divides, and the tail test drops out.
        block_full = total % threads == 0

        @T.macro
        def _store(total_val, od, oh, ow, out, row):
            """Store one output, taking its divisor from wherever this shape has it."""
            if one_divisor:
                out[row, od, oh, ow] = T.cast(total_val / T.cast(divisor, _ACCUM_DTYPE), dtype)
            else:
                front = od * stride_d - pad_d
                top = oh * stride_h - pad_h
                left = ow * stride_w - pad_w
                # An empty window would divide by zero, so the overlap takes a floor.
                extent = T.max(
                    T.max(T.min(front + kernel_d, limit_d) - T.max(front, low_d), 0)
                    * T.max(T.min(top + kernel_h, limit_h) - T.max(top, low_h), 0)
                    * T.max(T.min(left + kernel_w, limit_w) - T.max(left, low_w), 0),
                    1,
                )
                out[row, od, oh, ow] = T.cast(total_val / T.cast(extent, _ACCUM_DTYPE), dtype)

        @T.prim_func
        def _avg_pool3d_main(
            x: T.Tensor((rows, d_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_d, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total, threads), threads=threads) as tile:
                for i in T.Parallel(threads):
                    idx = tile * threads + i
                    if block_full or idx < total:
                        plane_row = idx // out_w
                        ow = idx - plane_row * out_w
                        depth_row = plane_row // out_h
                        oh = plane_row - depth_row * out_h
                        row = depth_row // out_d
                        od = depth_row - row * out_d
                        front = od * stride_d - pad_d
                        top = oh * stride_h - pad_h
                        left = ow * stride_w - pad_w
                        total_val = T.alloc_var(T.float32)
                        total_val = T.cast(0.0, _ACCUM_DTYPE)
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                for kw in T.serial(kernel_w):
                                    id_ = front + kd
                                    ih = top + kh
                                    iw = left + kw
                                    # Tested where it is read, so the test predicates the
                                    # load; hoisting it per axis makes it a branch.
                                    if window_inside or (
                                        (id_ >= 0)
                                        and (id_ < d_in)
                                        and (ih >= 0)
                                        and (ih < h_in)
                                        and (iw >= 0)
                                        and (iw < w_in)
                                    ):
                                        total_val += T.cast(x[row, id_, ih, iw], _ACCUM_DTYPE)
                        _store(total_val, od, oh, ow, out, row)

        return _avg_pool3d_main

    return _avg_pool3d_func


@functools.lru_cache(maxsize=32)
def _avg_pool3d_wide_kernel(window: _Window, dtype: str):
    """A run of outputs per thread, their shared input stretch read as whole groups.

    Reached only where :func:`_wide_run` returns a plan, so every window lies inside the
    volume and spans the whole kernel: the taps carry no test and one divisor covers them
    all.
    """
    d_in, h_in, w_in = window.size
    kernel_d, kernel_h, kernel_w = window.kernel
    stride_d, stride_h, stride_w = window.stride
    out_d, out_h, out_w = window.out
    rows, divisor = window.rows, window.divisor
    # The jit builder below may close over scalars only.
    run, vector_elems = _wide_run(window, dtype)

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _avg_pool3d_wide_func(threads: int):
        span = run * stride_w
        groups = span // vector_elems
        runs_per_row = out_w // run
        items = rows * out_d * out_h * runs_per_row
        # A compile-time truth when the launch divides, and the tail test drops out.
        block_full = items % threads == 0

        @T.prim_func
        def _avg_pool3d_main(
            x: T.Tensor((rows, d_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_d, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(items, threads), threads=threads) as tile:
                held = T.alloc_local((span,), dtype)
                sums = T.alloc_local((run,), _ACCUM_DTYPE)
                for i in T.Parallel(threads):
                    idx = tile * threads + i
                    if block_full or idx < items:
                        plane_row = idx // runs_per_row
                        jr = idx - plane_row * runs_per_row
                        depth_row = plane_row // out_h
                        oh = plane_row - depth_row * out_h
                        row = depth_row // out_d
                        od = depth_row - row * out_d
                        front = od * stride_d
                        top = oh * stride_h
                        left = jr * span
                        for j in T.serial(run):
                            sums[j] = T.cast(0.0, _ACCUM_DTYPE)
                        for kd in T.serial(kernel_d):
                            for kh in T.serial(kernel_h):
                                # The run covers one contiguous stretch of the row, so
                                # each (kd, kh) row it reads is one group of accesses.
                                for g in T.serial(groups):
                                    for v in T.vectorized(vector_elems):
                                        held[g * vector_elems + v] = x[
                                            row,
                                            front + kd,
                                            top + kh,
                                            left + g * vector_elems + v,
                                        ]
                                for j in T.serial(run):
                                    for kw in T.serial(kernel_w):
                                        sums[j] += T.cast(held[j * stride_w + kw], _ACCUM_DTYPE)
                        for j in T.serial(run):
                            out[row, od, oh, jr * run + j] = T.cast(
                                sums[j] / T.cast(divisor, _ACCUM_DTYPE), dtype
                            )

        return _avg_pool3d_main

    return _avg_pool3d_wide_func


class _AvgPool3dKernelBase(Kernel):
    """Shape, launch planning and dispatch shared by the two avg_pool3d kernels."""

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    # One output per thread, so the block size is the whole launch space: a parallel loop
    # wider than the block serializes it, and one narrower idles lanes.
    # Why a policy and not a tuned knob: these launches are too short for the autotuner
    # to rank, so a candidate list would pick from noise. Move this value on a
    # measurement, and keep `autotune_configs` naming whatever it holds.
    _BLOCK_THREADS: ClassVar[int] = 256

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        kernel_d: int,
        kernel_h: int,
        kernel_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.dtype = dtype
        self.window = _Window(
            rows=n * c_in,
            size=(d_in, h_in, w_in),
            kernel=(kernel_d, kernel_h, kernel_w),
            stride=(stride_d, stride_h, stride_w),
            pad=(pad_d, pad_h, pad_w),
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        build = (
            _avg_pool3d_wide_kernel
            if _wide_run(self.window, self.dtype_str) is not None
            else _avg_pool3d_kernel
        )
        self.kernel = build(self.window, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": self._BLOCK_THREADS}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_cuda(x=x)
        kernel = self.kernel(self.config["threads"])
        volumes = kernel(x.contiguous().view(self.window.rows, *self.window.size))
        return volumes.view(self.n, self.c_in, *self.window.out)


class AvgPool3dSpatialKernel(_AvgPool3dKernelBase):
    """Fast path for common NCDHW avg_pool3d workloads: zero-padded, floor-mode."""

    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        kernel_d: int,
        kernel_h: int,
        kernel_w: int,
        stride_d: int,
        stride_h: int,
        stride_w: int,
        pad_d: int,
        pad_h: int,
        pad_w: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__(
            n,
            c_in,
            d_in,
            h_in,
            w_in,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            pad_d,
            pad_h,
            pad_w,
            False,
            True,
            None,
            dtype,
            config,
            tune,
        )


class AvgPool3dKernel(_AvgPool3dKernelBase):
    """Average pooling over an NCDHW volume, with every PyTorch flag combination."""
