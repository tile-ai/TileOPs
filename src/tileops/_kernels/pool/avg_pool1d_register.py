"""Average pooling over an NCL row, each thread's windows read into registers.

A thread takes the outputs one 16-byte load of the row covers, or two such loads, and
reads the few elements their windows reach past each load on either side; no block
stages the row in shared memory.
"""

from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import (
    MAX_BLOCK_THREADS,
    MAX_GRID_YZ_BLOCKS,
    SM_RESIDENT_BLOCKS,
    VECTOR_ACCESS_BYTES,
)
from tileops._kernels.pool.call_spec import AvgPool1dFwdInterface, AvgPoolCall
from tileops._kernels.pool.common import ACCUM_DTYPE, AvgPoolKernelBase, AvgPoolWindow
from tileops._utils import WARP_LANES

__all__ = ["AvgPool1dRegisterKernel"]


def _run(itemsize: int, stride_l: int) -> tuple[int, int]:
    """The elements of one 16-byte load, and the outputs whose windows start in it."""
    run = VECTOR_ACCESS_BYTES // itemsize
    return run, run // stride_l


def _one_block_an_sm_covers(warps: int, sm_count: int) -> bool:
    """Whether one block of at most MAX_BLOCK_THREADS an SM holds ``warps`` warps."""
    return warps * WARP_LANES <= sm_count * MAX_BLOCK_THREADS


def _runs_a_thread(itemsize: int, stride_l: int, units: int, sm_count: int) -> int:
    """The runs one thread pools for a launch of ``units`` runs.

    Two, half the launch apart, where one run's outputs take at most 4 bytes and one block
    an SM cannot cover the launch even at two runs a thread; otherwise one, as also past
    the one-axis grid's int32 thread index.
    """
    # A run storing one 32-bit register's worth leaves its thread mostly index math, which
    # a second run shares.
    paired_store_bytes = 4
    if units > AvgPool1dRegisterKernel._FLAT_UNITS_MAX:
        return 1
    _, outputs = _run(itemsize, stride_l)
    covered = _one_block_an_sm_covers(-(-units // (2 * WARP_LANES)), sm_count)
    return 2 if outputs * itemsize <= paired_store_bytes and not covered else 1


@target_cache(maxsize=32)
def _avg_pool1d_register_kernel(spec: AvgPoolWindow, dtype: str):
    """Build the program for ``spec.rows`` rows of ``l_in`` elements.

    A run is ``VECTOR_ACCESS_BYTES`` of the row, ``outputs`` outputs; a thread pools
    ``runs`` runs, ``span`` groups apart on the flat grid. The windows of a run's outputs
    start ``pad_l`` elements before it and end ``after`` past it; ``window`` holds them in
    float32, zero outside the row. Where one divisor does not cover every window, a window
    divides by the positions of ``[lo, hi)`` it covers. ``flat`` lays every row's groups
    along one grid axis; otherwise the grid is ``(groups, rows)`` and a thread pools one run.
    """
    rows = spec.rows
    (l_in,), (kernel_l,), (stride_l,), (pad_l,), (out_l,) = (
        spec.size,
        spec.kernel,
        spec.stride,
        spec.pad,
        spec.out,
    )
    whole = spec.whole_window_divides
    ((lo,), (hi,)) = ((0,), (0,)) if whole else spec.overlap
    run, outputs = _run(getattr(torch, dtype).itemsize, stride_l)
    after = max(kernel_l - stride_l - pad_l, 0)
    groups = out_l // outputs
    units = rows * groups
    # An input of 2**31 elements or more takes int64 extents: int32 cannot hold an extent of
    # exactly 2**31, and TileLang widens the offsets of a longer input by itself.
    wide = rows * l_in >= 2**31
    # A position of a row that long, a load's run or the window's width past an element,
    # can pass 2**31 - 1.
    position_dtype = "int64" if l_in + max(kernel_l, VECTOR_ACCESS_BYTES) >= 2**31 else "int32"

    @tilelang.jit(out_idx=[1])
    def build(threads: int, flat: bool, runs: int):
        span = -(-units // runs)
        x_shape = (T.int64(rows), T.int64(l_in)) if wide else (rows, l_in)
        y_shape = (T.int64(rows), T.int64(out_l)) if wide else (rows, out_l)

        @T.macro
        def side(x, row, at, width, window, first):
            """``x[row, at:at + width]`` in float32 into ``window[first:]`` as one load, zero
            where the span lies outside the row. A run's side lies wholly inside the row or
            wholly outside it, since the row is whole runs and a side is at most one run."""
            inside = T.max(T.min(at, l_in - width), 0)
            for i in T.vectorized(width):
                window[first + i] = T.if_then_else(
                    inside == at, T.cast(x[row, inside + i], ACCUM_DTYPE), T.cast(0, ACCUM_DTYPE)
                )

        @T.macro
        def load(x, row, group, values, v):
            """``group``'s run of ``row`` into ``values[v]``, as one load."""
            # Widened at the group, so every position of the row derived from it is.
            start = T.cast(group, position_dtype) * run
            for i in T.vectorized(run):
                values[v, i] = x[row, start + i]

        @T.macro
        def pool(x, row, group, values, means, v):
            """The means of ``group``'s outputs of ``row`` from its run in ``values[v]`` and
            the elements its windows reach past it, into ``means[v]``."""
            window = T.alloc_local([pad_l + run + after], ACCUM_DTYPE)
            totals = T.alloc_local([outputs], ACCUM_DTYPE)
            start = T.cast(group, position_dtype) * run
            for i in T.unroll(run):
                window[pad_l + i] = T.cast(values[v, i], ACCUM_DTYPE)
            if pad_l:
                side(x, row, start - pad_l, pad_l, window, 0)
            if after:
                side(x, row, start + run, after, window, pad_l + run)
            for e in T.unroll(outputs):
                totals[e] = T.cast(0, ACCUM_DTYPE)
                for t in T.unroll(kernel_l):
                    totals[e] += window[e * stride_l + t]
                # A division, not a multiply by the reciprocal: torch divides, and the
                # two round apart.
                if whole:
                    means[v, e] = T.cast(totals[e] / T.cast(kernel_l, ACCUM_DTYPE), dtype)
                else:
                    begin = (T.cast(group, position_dtype) * outputs + e) * stride_l - pad_l
                    divisor = T.min(begin + kernel_l, hi) - T.max(begin, lo)
                    # A whole window divides by the constant; only a window cut by the
                    # row's edge takes the division by a computed divisor.
                    means[v, e] = T.cast(
                        T.if_then_else(
                            divisor == kernel_l,
                            totals[e] / T.cast(kernel_l, ACCUM_DTYPE),
                            totals[e] / T.cast(divisor, ACCUM_DTYPE),
                        ),
                        dtype,
                    )

        # In either grid a thread past the last group reads the last group's run and stores
        # nothing: the loads then issue without a branch around them.
        if flat:

            @T.prim_func
            def main(
                x: T.Tensor[x_shape, dtype],
                y: T.Tensor[y_shape, dtype],
            ):
                with T.Kernel(-(-span // threads), threads=threads) as bx:
                    tx = T.get_thread_binding()
                    values = T.alloc_local([runs, run], dtype)
                    means = T.alloc_local([runs, outputs], dtype)
                    # Every run's 16-byte load issues before any run's arithmetic.
                    for v in T.unroll(runs):
                        last = T.min(bx * threads + tx + v * span, units - 1)
                        load(x, last // groups, last % groups, values, v)
                    for v in T.unroll(runs):
                        unit = bx * threads + tx + v * span
                        last = T.min(unit, units - 1)
                        row, group = last // groups, last % groups
                        pool(x, row, group, values, means, v)
                        # Run v of the launch's threads covers units [v * span, (v + 1) * span):
                        # a thread past the last of them stores nothing, so each output has
                        # one writer.
                        if unit < T.min(units, (v + 1) * span):
                            for e in T.vectorized(outputs):
                                y[row, T.cast(group, position_dtype) * outputs + e] = means[v, e]

        else:

            @T.prim_func
            def main(
                x: T.Tensor[x_shape, dtype],
                y: T.Tensor[y_shape, dtype],
            ):
                with T.Kernel(-(-groups // threads), rows, threads=threads) as (bx, row):
                    tx = T.get_thread_binding()
                    group = bx * threads + tx
                    last = T.min(group, groups - 1)
                    values = T.alloc_local([1, run], dtype)
                    means = T.alloc_local([1, outputs], dtype)
                    load(x, row, last, values, 0)
                    pool(x, row, last, values, means, 0)
                    if group < groups:
                        for e in T.vectorized(outputs):
                            y[row, T.cast(group, position_dtype) * outputs + e] = means[0, e]

        return main

    return build


def _launch(call: AvgPoolCall) -> dict:
    """The block width, the grid and the runs a thread for *call* on its device."""
    window = AvgPoolWindow.of(call)
    # A group of outputs a run, ``groups`` a row; ``runs`` runs a thread, ``lanes``
    # threads in all.
    _, outputs = _run(call.dtype.itemsize, call.stride[0])
    groups = window.out[0] // outputs
    units = window.rows * groups
    runs = _runs_a_thread(call.dtype.itemsize, call.stride[0], units, call.sm_count)
    lanes = -(-units // runs)
    warps = -(-lanes // WARP_LANES)
    # The row grid: 128-thread blocks along each row. Past one block an SM, the threads
    # resident on an SM, not the block width, set the bytes it has in flight; four warps
    # a block leave a row's last block at most 127 idle threads.
    multi_wave_threads = 128
    row_blocks = window.rows * -(-groups // multi_wave_threads)
    # A row of fewer groups than a block has threads leaves the rest idle, and an SM holds
    # at most SM_RESIDENT_BLOCKS blocks. Where the row grid has more blocks than the SMs
    # hold at once and those blocks cannot keep threads_per_sm threads busy, the blocks
    # span rows on one grid axis.
    resident = SM_RESIDENT_BLOCKS[call.arch]
    row_idles_sm = (
        row_blocks > call.sm_count * resident
        and resident * min(groups, multi_wave_threads) < call.threads_per_sm
    )
    # One block an SM, of the fewest whole warps that cover the launch, where the row grid
    # puts a second block on some SM and one block of at most MAX_BLOCK_THREADS covers an
    # SM's share: with every block resident at once, no later wave hides a block's dispatch.
    one_block_an_sm = row_blocks > call.sm_count and _one_block_an_sm_covers(warps, call.sm_count)
    # Two runs a thread lie half the launch apart, so they take the one-axis grid.
    flat = runs > 1 or one_block_an_sm or row_idles_sm or window.rows > MAX_GRID_YZ_BLOCKS
    threads = -(-warps // call.sm_count) * WARP_LANES if one_block_an_sm else multi_wave_threads
    return {"threads": threads, "flat": flat, "runs": runs}


class AvgPool1dRegisterKernel(AvgPoolKernelBase, AvgPool1dFwdInterface):
    """Average pooling over rows, a thread's windows read into registers.

    Serves every divisor but ``divisor_override``: a window cut by the row's edge, or by
    a ceil-mode overhang past the padded row, divides by the positions it counts. The
    stride must divide the elements of a 16-byte load, so that one covers whole outputs;
    the outputs must divide into those loads, and the loads into the row. The elements a
    run's windows reach past it on each side are read as one load of their own, of at
    most 4 bytes.
    """

    preferred_over = frozenset({"avg_pool1d_kernel"})
    # The most runs the one-axis grid indexes: its int32 thread index must hold the last
    # run's thread and the rest of its block.
    _FLAT_UNITS_MAX: ClassVar[int] = 2**31 - 1 - MAX_BLOCK_THREADS
    _build = staticmethod(_avg_pool1d_register_kernel)

    @classmethod
    def refusal(cls, call: AvgPoolCall) -> str | None:
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return f"requires float16, bfloat16 or float32, got {call.dtype}"
        if len(call.size) != 1:
            return "serves a 1-d window"
        if call.divisor_override is not None:
            return "does not serve divisor_override"
        (l_in,), (kernel_l,), (stride_l,), (pad_l,) = call.size, call.window, call.stride, call.pad
        run, outputs = _run(call.dtype.itemsize, stride_l)
        if run % stride_l:
            return f"requires a stride dividing the {run}-element load, got {stride_l}"
        (out_l,) = AvgPoolWindow.of(call).out
        # A side's elements are one load of at most 4 bytes, one fp32 element; the run
        # starts on a 16-byte boundary, so a side of two or four bytes starts on its width.
        side_load_bytes = 4
        for side, reach in (("before", pad_l), ("after", kernel_l - stride_l - pad_l)):
            if reach * call.dtype.itemsize > side_load_bytes:
                return (
                    f"reads the elements a window reaches {side} a load as one load of at most"
                    f" {side_load_bytes} bytes; got {reach}"
                )
        if not (
            out_l > 0 and out_l % outputs == 0 and out_l * stride_l <= l_in and l_in % run == 0
        ):
            return f"requires whole {run}-element loads along the row"
        # Only rows past grid y put this many groups on the one-axis grid; the row grid
        # indexes groups within a row.
        rows = call.n * call.c_in
        units = rows * (out_l // outputs)
        if rows > MAX_GRID_YZ_BLOCKS and units > cls._FLAT_UNITS_MAX:
            return f"serves at most {cls._FLAT_UNITS_MAX} groups of outputs"
        return super().refusal(call)

    @property
    def default_config(self) -> dict:
        return _launch(self.call)

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]
