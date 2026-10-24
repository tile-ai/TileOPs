import functools
from typing import ClassVar, NamedTuple, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.pool.call_spec import AvgPool3dFwdInterface, AvgPoolCall
from tileops.kernels.pool.common import ACCUM_DTYPE, AvgPoolWindow, dtype_itemsize

__all__ = ["AvgPool3dKernel"]

# Every kernel here reads through the same csrc header, so one flag list keys their caches.
_JIT_FLAGS = ["-O3", "-DENABLE_BF16", *csrc_include("streaming_load.h")]


def _by_divisor(total, divisor: int):
    """``total / divisor`` as the float32 IEEE divide rounds it, without its slow-path branch.

    A power of two is an exact multiply, which the compiler makes of the divide. Up to 97 in
    magnitude, the float64 product with ``1 / divisor`` rounds to the IEEE quotient: checked
    bit for bit for every float32 numerator on H200 at divisors 1 to 97. 98 is the first
    divisor that differs: 39,896 positive numerators round differently, about half of them
    normal and half subnormal. The divide's slow-path branch cost the ``divisor`` row
    (2x24x10x20x22 bfloat16, divisor 7) one 32 ns tick on H200: 1.664 us against 1.632.
    """
    if abs(divisor) & (abs(divisor) - 1) == 0 or abs(divisor) > 97:
        return total / T.cast(divisor, ACCUM_DTYPE)
    return T.cast(T.cast(total, "float64") * T.cast(1.0 / divisor, "float64"), ACCUM_DTYPE)


class _WideRun(NamedTuple):
    """Outputs one thread takes along w, and the access width it reads them with."""

    run: int
    vector_elems: int


def _wide_run(window: AvgPoolWindow, dtype: str) -> Optional[_WideRun]:
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
def _avg_pool3d_kernel(window: AvgPoolWindow, dtype: str):
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

    @tilelang.jit(out_idx=[1], compile_flags=_JIT_FLAGS)
    def _avg_pool3d_func(threads: int):
        # A compile-time truth when the launch divides, and the tail test drops out.
        block_full = total % threads == 0
        # The clamped taps unroll, so a window past 32 of them keeps the serial loop; the
        # window measured below has 12.
        clamped_taps = not window_inside and one_divisor and kernel_d * kernel_h * kernel_w <= 32

        @T.macro
        def _store(total_val, od, oh, ow, out, row):
            """Store one output, taking its divisor from wherever this shape has it."""
            if one_divisor:
                out[row, od, oh, ow] = T.cast(_by_divisor(total_val, divisor), dtype)
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
                out[row, od, oh, ow] = T.cast(total_val / T.cast(extent, ACCUM_DTYPE), dtype)

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
                        total_val = T.cast(0.0, ACCUM_DTYPE)
                        if clamped_taps:
                            # Every tap reads a clamped address, evict-last as Inductor's
                            # loads are, and a select drops the outside ones: no load waits
                            # on a predicate. With the branch-free divide, divisor
                            # (2x24x10x20x22 bfloat16) took 1.632 us against 1.792 with
                            # predicated loads (H200). ceil-video, whose divisor each output
                            # computes, took 3.104 against 2.912 this way.
                            for kd in T.unroll(kernel_d):
                                for kh in T.unroll(kernel_h):
                                    for kw in T.unroll(kernel_w):
                                        id_ = front + kd
                                        ih = top + kh
                                        iw = left + kw
                                        tap = T.call_extern(
                                            dtype,
                                            "tileops::load_evict_last",
                                            T.address_of(
                                                x[
                                                    row,
                                                    T.min(T.max(id_, 0), d_in - 1),
                                                    T.min(T.max(ih, 0), h_in - 1),
                                                    T.min(T.max(iw, 0), w_in - 1),
                                                ]
                                            ),
                                        )
                                        total_val += T.if_then_else(
                                            (id_ >= 0)
                                            and (id_ < d_in)
                                            and (ih >= 0)
                                            and (ih < h_in)
                                            and (iw >= 0)
                                            and (iw < w_in),
                                            T.cast(tap, ACCUM_DTYPE),
                                            T.cast(0.0, ACCUM_DTYPE),
                                        )
                        else:
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
                                            total_val += T.cast(x[row, id_, ih, iw], ACCUM_DTYPE)
                        _store(total_val, od, oh, ow, out, row)

        return _avg_pool3d_main

    return _avg_pool3d_func


@functools.lru_cache(maxsize=32)
def _avg_pool3d_wide_kernel(window: AvgPoolWindow, dtype: str):
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

    @tilelang.jit(out_idx=[1], compile_flags=_JIT_FLAGS)
    def _avg_pool3d_wide_func(threads: int):
        span = run * stride_w
        groups = span // vector_elems
        runs_per_row = out_w // run
        items = rows * out_d * out_h * runs_per_row
        # A compile-time truth when the launch divides, and the tail test drops out.
        block_full = items % threads == 0
        whole_vector = vector_elems * dtype_itemsize(dtype) == VECTOR_ACCESS_BYTES
        # Every (kd, kh) row of the run is read before the first add, so the reads are in
        # flight together: video-2x2x2 float32 (1x32x16x56x56) took 3.775 us against 3.968
        # reading a row at a time (H200). The rows a thread holds bound the registers, so a
        # window past 64 elements of them keeps reading a row at a time.
        hoisted = kernel_d * kernel_h * span <= 64

        @T.macro
        def _load_row(held, base, x, row, d, h, left):
            """One (d, h) row of the run into ``held`` from ``base``, evict-last in L2 where
            a group is a whole 16-byte vector."""
            for g in T.serial(groups):
                if whole_vector:
                    T.call_extern(
                        "handle",
                        "tileops::load16_evict_last",
                        T.address_of(held[base + g * vector_elems]),
                        T.address_of(x[row, d, h, left + g * vector_elems]),
                    )
                else:
                    for v in T.vectorized(vector_elems):
                        held[base + g * vector_elems + v] = x[
                            row, d, h, left + g * vector_elems + v
                        ]

        @T.prim_func
        def _avg_pool3d_main(
            x: T.Tensor((rows, d_in, h_in, w_in), dtype),  # type: ignore
            out: T.Tensor((rows, out_d, out_h, out_w), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(items, threads), threads=threads) as tile:
                held = T.alloc_local(((kernel_d * kernel_h if hoisted else 1) * span,), dtype)
                sums = T.alloc_local((run,), ACCUM_DTYPE)
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
                            sums[j] = T.cast(0.0, ACCUM_DTYPE)
                        if hoisted:
                            for kd in T.unroll(kernel_d):
                                for kh in T.unroll(kernel_h):
                                    _load_row(
                                        held,
                                        (kd * kernel_h + kh) * span,
                                        x,
                                        row,
                                        front + kd,
                                        top + kh,
                                        left,
                                    )
                            for kd in T.unroll(kernel_d):
                                for kh in T.unroll(kernel_h):
                                    for j in T.unroll(run):
                                        for kw in T.unroll(kernel_w):
                                            sums[j] += T.cast(
                                                held[
                                                    (kd * kernel_h + kh) * span + j * stride_w + kw
                                                ],
                                                ACCUM_DTYPE,
                                            )
                        else:
                            for kd in T.serial(kernel_d):
                                for kh in T.serial(kernel_h):
                                    _load_row(held, 0, x, row, front + kd, top + kh, left)
                                    for j in T.serial(run):
                                        for kw in T.serial(kernel_w):
                                            sums[j] += T.cast(held[j * stride_w + kw], ACCUM_DTYPE)
                        for j in T.serial(run):
                            out[row, od, oh, jr * run + j] = T.cast(
                                _by_divisor(sums[j], divisor), dtype
                            )

        return _avg_pool3d_main

    return _avg_pool3d_wide_func


class AvgPool3dKernel(Kernel, AvgPool3dFwdInterface):
    """Average pooling over an NCDHW volume, with every PyTorch flag combination."""

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    # One output per thread, so the block size is the whole launch space: a parallel loop
    # wider than the block serializes it, and one narrower idles lanes.
    # Why a policy and not a tuned knob: these launches are too short for the autotuner
    # to rank, so a candidate list would pick from noise. Move this value on a
    # measurement, and keep `autotune_configs` naming whatever it holds.
    _BLOCK_THREADS: ClassVar[int] = 256

    @classmethod
    def entry_for(cls, call: AvgPoolCall) -> Entry:
        return call, lambda: cls(
            call.n,
            call.c_in,
            *call.size,
            *call.window,
            *call.stride,
            *call.pad,
            call.ceil_mode,
            call.count_include_pad,
            call.divisor_override,
            call.dtype,
        )

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
    ) -> None:
        super().__init__()
        self.n = n
        self.c_in = c_in
        self.dtype = dtype
        self.window = AvgPoolWindow(
            rows=n * c_in,
            size=(d_in, h_in, w_in),
            kernel=(kernel_d, kernel_h, kernel_w),
            stride=(stride_d, stride_h, stride_w),
            pad=(pad_d, pad_h, pad_w),
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        # Only the wide-run program reads x in 16-byte vectors.
        self._wide = _wide_run(self.window, self.dtype_str) is not None
        build = _avg_pool3d_wide_kernel if self._wide else _avg_pool3d_kernel
        self.kernel = build(self.window, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"threads": self._BLOCK_THREADS}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_devices(x=x)
        x = x.contiguous()
        if self._wide:
            x = vector_aligned(x)
        kernel = self.kernel(self.config["threads"])
        volumes = kernel(x.view(self.window.rows, *self.window.size))
        return volumes.view(self.n, self.c_in, *self.window.out)
