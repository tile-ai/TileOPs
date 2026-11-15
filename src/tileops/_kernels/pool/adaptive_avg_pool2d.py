import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry
from tileops._kernels.pool.call_spec import AdaptiveAvgPool2dFwdInterface, AdaptivePool2dCall
from tileops._kernels.pool.common import (
    ACCUM_DTYPE,
    AdaptivePool2dKernelBase,
    adaptive_bin,
    dtype_itemsize,
    max_adaptive_bin_extent,
)
from tileops._utils import WARP_LANES

__all__ = ["AdaptiveAvgPool2dKernel", "AdaptiveAvgPool2dLanesKernel"]


@target_cache(maxsize=32)
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


class AdaptiveAvgPool2dKernel(AdaptivePool2dKernelBase, AdaptiveAvgPool2dFwdInterface):
    """Adaptive average pooling forward kernel for NCHW inputs."""

    def _launch(self, program, x: torch.Tensor) -> torch.Tensor:
        return program(x)

    _build = staticmethod(_adaptive_avg_pool2d_kernel)


def _bin_lanes(call: AdaptivePool2dCall) -> int:
    """Threads one bin is split over, a power of two up to a warp.

    Doubled while a lane would still read more than one 16-byte access of its bin and the
    grid's threads do not yet fill the device once: past that, another lane repeats the
    bin's index arithmetic, fold and division without adding threads that run at once.
    """
    window_bytes = (
        max_adaptive_bin_extent(call.h_in, call.out_h)
        * max_adaptive_bin_extent(call.w_in, call.out_w)
        * call.dtype.itemsize
    )
    outputs = call.n * call.c_in * call.out_h * call.out_w
    resident = call.sm_count * call.threads_per_sm
    lanes = 1
    while (
        lanes < WARP_LANES
        and lanes * VECTOR_ACCESS_BYTES < window_bytes
        and outputs * lanes < resident
    ):
        lanes *= 2
    return lanes


@target_cache(maxsize=32)
def _adaptive_avg_pool2d_lanes_kernel(
    n: int,
    c_in: int,
    h_in: int,
    w_in: int,
    out_h: int,
    out_w: int,
    dtype: str = "float16",
):
    rows = n * c_in
    total_output = rows * out_h * out_w
    max_kh = max_adaptive_bin_extent(h_in, out_h)
    max_kw = max_adaptive_bin_extent(w_in, out_w)
    # Elements one load of a bin row takes: the widest access every bin edge lands on.
    vector = VECTOR_ACCESS_BYTES // dtype_itemsize(dtype)
    edges = {e for o in range(out_w) for e in adaptive_bin(o, w_in, out_w)}
    while vector > 1 and any(e % vector for e in edges):
        vector //= 2
    # A bin row's loads; a bin as wide as the input row is one contiguous run of its rows.
    row_slots = (w_in if out_w == 1 else max_kw) // vector

    @tilelang.jit(out_idx=[1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _adaptive_avg_pool2d_lanes_func(lanes: int, threads: int):
        bins_per_block = threads // lanes
        # Lanes split a bin row first and its rows with the rest.
        lanes_w = min(lanes, 1 << (row_slots.bit_length() - 1))
        lanes_h = lanes // lanes_w
        run_steps = -(-max_kh * row_slots // lanes)
        row_steps, col_steps = -(-max_kh // lanes_h), -(-row_slots // lanes_w)
        # A lane's loop unrolls at most this many loads. Unrolled whole, a wide bin's loop
        # grows the program with the bin; 64 loads still unroll a lane's run whole up to
        # 32 lanes x 64 x 16 bytes.
        unrolled = 64

        @T.macro
        def _add(acc, held, x, at, inside):
            # Every lane loads from an address inside its bin, clamped where its slot runs
            # past it, and only the sum is masked: the loads issue with no branch around them.
            for v in T.vectorized(vector):
                held[v] = x[at + v]
            for v in T.unroll(vector):
                acc[0] += T.if_then_else(
                    inside, T.cast(held[v], ACCUM_DTYPE), T.cast(0, ACCUM_DTYPE)
                )

        @T.prim_func
        def _adaptive_avg_pool2d_lanes_main(
            x: T.Tensor((rows * h_in * w_in,), dtype),  # type: ignore
            out: T.Tensor((total_output,), dtype),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(total_output, bins_per_block), threads=threads) as bx:
                tx = T.get_thread_binding()
                lane = tx % lanes
                # A thread past the last bin reads the last bin and stores nothing.
                o_raw = bx * bins_per_block + tx // lanes
                o = T.min(o_raw, total_output - 1)
                ow = o % out_w
                oh = (o // out_w) % out_h
                plane = o // (out_w * out_h)
                ih_start, ih_end = adaptive_bin(oh, h_in, out_h)
                iw_start, iw_end = adaptive_bin(ow, w_in, out_w)
                acc = T.alloc_local([1], ACCUM_DTYPE)
                held = T.alloc_local([vector], dtype)
                acc[0] = T.cast(0, ACCUM_DTYPE)
                if out_w == 1:
                    span = (ih_end - ih_start) * row_slots
                    for step in T.unroll(run_steps, unroll_factor=unrolled):
                        slot = step * lanes + lane
                        at = (plane * h_in + ih_start) * w_in + T.min(slot, span - 1) * vector
                        _add(acc, held, x, at, slot < span)
                else:
                    for i in T.unroll(row_steps, unroll_factor=max(1, unrolled // col_steps)):
                        ih = ih_start + lane // lanes_w + i * lanes_h
                        for j in T.unroll(col_steps, unroll_factor=unrolled):
                            iw = iw_start + (lane % lanes_w + j * lanes_w) * vector
                            at = (plane * h_in + T.min(ih, ih_end - 1)) * w_in + T.min(
                                iw, iw_end - vector
                            )
                            _add(acc, held, x, at, ih < ih_end and iw < iw_end)
                for level in T.unroll(lanes.bit_length() - 1):
                    acc[0] += T.shfl_xor(acc[0], T.shift_left(1, level))
                if o_raw < total_output and lane == 0:
                    count = (ih_end - ih_start) * (iw_end - iw_start)
                    # A division, not a multiply by the reciprocal: torch divides, and the
                    # two round apart.
                    out[o] = T.cast(acc[0] / T.cast(count, ACCUM_DTYPE), dtype)

        return _adaptive_avg_pool2d_lanes_main

    return _adaptive_avg_pool2d_lanes_func


class AdaptiveAvgPool2dLanesKernel(AdaptivePool2dKernelBase, AdaptiveAvgPool2dFwdInterface):
    """Adaptive average pooling, each bin's taps split over lanes that fold by shuffle.

    Serves calls that :func:`_bin_lanes` gives two or more lanes a bin.
    """

    preferred_over = frozenset({"adaptive_avg_pool2d_kernel"})

    @classmethod
    def refusal(cls, call: AdaptivePool2dCall) -> "str | None":
        if call.dtype not in (torch.float16, torch.bfloat16):
            return f"requires float16 or bfloat16, got {call.dtype}"
        if _bin_lanes(call) == 1:
            return (
                f"a bin fits one {VECTOR_ACCESS_BYTES}-byte access, or one thread a bin "
                "already fills the device"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: AdaptivePool2dCall) -> Entry:
        return call, lambda: cls(
            call.n,
            call.c_in,
            call.h_in,
            call.w_in,
            call.out_h,
            call.out_w,
            call.dtype,
            lanes=_bin_lanes(call),
        )

    def __init__(self, *args, lanes: int, **kwargs) -> None:
        self.lanes = lanes
        super().__init__(*args, **kwargs)

    @property
    def default_config(self) -> dict:
        # Four warps a block: a grid of few bins still spreads over the SMs, and a grid
        # that fills the device stays under an SM's block limit.
        return {"lanes": self.lanes, "threads": 128}

    @property
    def autotune_configs(self) -> None:
        return None

    def _launch(self, program, x: torch.Tensor) -> torch.Tensor:
        out = program(x.reshape(-1))
        return out.view(self.n, self.c_in, self.out_h, self.out_w)

    _build = staticmethod(_adaptive_avg_pool2d_lanes_kernel)
