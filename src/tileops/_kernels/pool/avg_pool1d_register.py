"""Average pooling over an NCL row, each thread's windows read into registers.

A thread takes the outputs one 16-byte load of the row covers and reads the few
elements their windows reach past it on either side; no block stages the row in
shared memory.
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
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.pool.call_spec import AvgPool1dFwdInterface, AvgPoolCall
from tileops._kernels.pool.common import ACCUM_DTYPE, AvgPoolWindow
from tileops._utils import WARP_LANES

__all__ = ["AvgPool1dRegisterKernel"]


def _run(itemsize: int, stride_l: int) -> tuple[int, int]:
    """The elements of one 16-byte load, and the outputs whose windows start in it."""
    run = VECTOR_ACCESS_BYTES // itemsize
    return run, run // stride_l


@target_cache(maxsize=32)
def _avg_pool1d_register_kernel(rows, l_in, kernel_l, stride_l, pad_l, out_l, counted, dtype):
    """Build the program for ``rows`` rows of ``l_in`` elements.

    A thread owns ``VECTOR_ACCESS_BYTES`` of the row, ``outputs`` outputs. The windows
    of those outputs start ``pad_l`` elements before the run and end ``after`` past it;
    ``window`` holds them in float32, zero outside the row. ``counted`` is the span
    ``(lo, hi)`` of positions a window counts in its divisor, or ``None`` where every
    window divides by ``kernel_l``. The program takes ``threads`` a block, and ``flat``
    lays every row's groups along one grid axis; otherwise the grid is ``(groups, rows)``.
    """
    run, outputs = _run(getattr(torch, dtype).itemsize, stride_l)
    after = max(kernel_l - stride_l - pad_l, 0)
    groups = out_l // outputs
    units = rows * groups
    # The builder closes over scalars only, so the span is unpacked here.
    whole = counted is None
    lo, hi = (0, 0) if whole else counted
    # An input of 2**31 elements or more takes int64 extents: int32 cannot hold an extent of
    # exactly 2**31, and TileLang widens the offsets of a longer input by itself.
    wide = rows * l_in >= 2**31
    # A position of a row that long, a load's run or the window's width past an element,
    # can pass 2**31 - 1.
    position_dtype = "int64" if l_in + max(kernel_l, VECTOR_ACCESS_BYTES) >= 2**31 else "int32"

    @tilelang.jit(out_idx=[1])
    def build(threads: int, flat: bool):
        x_shape = (T.int64(rows), T.int64(l_in)) if wide else (rows, l_in)
        y_shape = (T.int64(rows), T.int64(out_l)) if wide else (rows, out_l)

        def element(x, row, at):
            """``x[row, at]`` in float32, zero outside the row."""
            inside = T.max(T.min(at, l_in - 1), 0)
            return T.if_then_else(
                inside == at, T.cast(x[row, inside], ACCUM_DTYPE), T.cast(0, ACCUM_DTYPE)
            )

        @T.macro
        def pool(x, row, group, means):
            """The means of ``group``'s outputs of ``row``, into ``means``."""
            values = T.alloc_local([run], dtype)
            window = T.alloc_local([pad_l + run + after], ACCUM_DTYPE)
            totals = T.alloc_local([outputs], ACCUM_DTYPE)
            # Widened at the group, so every position of the row derived from it is.
            start = T.cast(group, position_dtype) * run
            for i in T.vectorized(run):
                values[i] = x[row, start + i]
            for i in T.unroll(run):
                window[pad_l + i] = T.cast(values[i], ACCUM_DTYPE)
            if pad_l:
                for i in T.unroll(pad_l):
                    window[i] = element(x, row, start - pad_l + i)
            if after:
                for i in T.unroll(after):
                    window[pad_l + run + i] = element(x, row, start + run + i)
            for e in T.unroll(outputs):
                totals[e] = T.cast(0, ACCUM_DTYPE)
                for t in T.unroll(kernel_l):
                    totals[e] += window[e * stride_l + t]
                # A division, not a multiply by the reciprocal: torch divides, and the
                # two round apart.
                if whole:
                    means[e] = T.cast(totals[e] / T.cast(kernel_l, ACCUM_DTYPE), dtype)
                else:
                    begin = (T.cast(group, position_dtype) * outputs + e) * stride_l - pad_l
                    divisor = T.min(begin + kernel_l, hi) - T.max(begin, lo)
                    # A whole window divides by the constant; only a window cut by the
                    # row's edge takes the division by a computed divisor.
                    means[e] = T.cast(
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
                with T.Kernel(-(-units // threads), threads=threads) as bx:
                    tx = T.get_thread_binding()
                    unit = bx * threads + tx
                    means = T.alloc_local([outputs], dtype)
                    last = T.min(unit, units - 1)
                    row, group = last // groups, last % groups
                    pool(x, row, group, means)
                    if unit < units:
                        for e in T.vectorized(outputs):
                            y[row, T.cast(group, position_dtype) * outputs + e] = means[e]

        else:

            @T.prim_func
            def main(
                x: T.Tensor[x_shape, dtype],
                y: T.Tensor[y_shape, dtype],
            ):
                with T.Kernel(-(-groups // threads), rows, threads=threads) as (bx, row):
                    tx = T.get_thread_binding()
                    group = bx * threads + tx
                    means = T.alloc_local([outputs], dtype)
                    pool(x, row, T.min(group, groups - 1), means)
                    if group < groups:
                        for e in T.vectorized(outputs):
                            y[row, T.cast(group, position_dtype) * outputs + e] = means[e]

        return main

    return build


class AvgPool1dRegisterKernel(Kernel, AvgPool1dFwdInterface):
    """Average pooling over rows, a thread's windows read into registers.

    Serves every divisor but ``divisor_override``: a window cut by the row's edge, or by
    a ceil-mode overhang past the padded row, divides by the positions it counts. The
    stride must divide the elements of a 16-byte load, so that one covers whole outputs;
    the outputs must divide into those loads, and the loads into the row. A thread's
    windows reach at most one element past its load on either side, each one a load of
    its own.
    """

    supported_archs: ClassVar[list[int]] = [80, 86, 89, 90]
    preferred_over = frozenset({"avg_pool1d_kernel"})

    @staticmethod
    def _window(call: AvgPoolCall) -> AvgPoolWindow:
        return AvgPoolWindow(
            rows=call.n * call.c_in,
            size=call.size,
            kernel=call.window,
            stride=call.stride,
            pad=call.pad,
            ceil_mode=call.ceil_mode,
            count_include_pad=call.count_include_pad,
            divisor_override=call.divisor_override,
        )

    @classmethod
    def refusal(cls, call: AvgPoolCall) -> "str | None":
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
        (out_l,) = cls._window(call).out
        # Each element past the load is a load of its own; at two, AvgPool1dKernel's
        # shared-memory stage is faster.
        max_reach = 1
        if pad_l > max_reach or kernel_l - stride_l - pad_l > max_reach:
            return f"reads at most {max_reach} element past a load"
        if not (
            out_l > 0 and out_l % outputs == 0 and out_l * stride_l <= l_in and l_in % run == 0
        ):
            return f"requires whole {run}-element loads along the row"
        # A thread a group of outputs: the one-axis grid's int32 thread index must hold the
        # last group's thread and the rest of its block.
        if call.n * call.c_in * (out_l // outputs) > 2**31 - 1 - MAX_BLOCK_THREADS:
            return f"serves at most 2**31 - 1 - {MAX_BLOCK_THREADS} groups of outputs"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: AvgPoolCall) -> Entry:
        window = cls._window(call)
        (l_in,), (kernel_l,), (stride_l,), (pad_l,), (out_l,) = (
            window.size,
            window.kernel,
            window.stride,
            window.pad,
            window.out,
        )
        counted = None
        if not window.whole_window_divides:
            ((lo,), (hi,)) = window.overlap
            counted = (lo, hi)
        # One thread a group of outputs, ``groups`` a row.
        _, outputs = _run(call.dtype.itemsize, stride_l)
        groups = out_l // outputs
        units = window.rows * groups
        warps = -(-units // WARP_LANES)
        # The row grid: 128-thread blocks along each row. Past one block an SM, the threads
        # resident on an SM, not the block width, set the bytes it has in flight; four warps
        # a block leave a row's last block at most 127 idle threads.
        multi_wave_threads = 128
        row_blocks = window.rows * -(-groups // multi_wave_threads)
        # A row of fewer groups than a block has threads leaves the rest idle, and an SM
        # holds at most SM_RESIDENT_BLOCKS blocks. Where the row grid has more blocks than
        # the SMs hold at once, that cap binds; where the blocks it lets an SM hold cannot
        # keep threads_per_sm threads busy, the launch runs its blocks spanning rows on one
        # grid axis. Such a row takes one row-grid block, so that launches fewer blocks.
        resident = SM_RESIDENT_BLOCKS[call.arch]
        row_idles_sm = (
            row_blocks > call.sm_count * resident
            and resident * min(groups, multi_wave_threads) < call.threads_per_sm
        )
        # One block an SM, of the fewest whole warps that cover the launch: every block an
        # SM runs past its first is one more to dispatch, and with all of them resident at
        # once no later wave hides the dispatch. It applies where the row grid puts a second
        # block on some SM (more row blocks than SMs), so that it launches fewer blocks, at
        # most one an SM; and where a block of at most MAX_BLOCK_THREADS covers an SM's
        # share, the block limit rather than the SM's residency, since past it an SM needs
        # a second block anyway.
        one_block_threads = -(-warps // call.sm_count) * WARP_LANES
        one_block_an_sm = (
            row_blocks > call.sm_count and warps * WARP_LANES <= call.sm_count * MAX_BLOCK_THREADS
        )
        flat = one_block_an_sm or row_idles_sm or window.rows > MAX_GRID_YZ_BLOCKS
        threads = one_block_threads if one_block_an_sm else multi_wave_threads
        identity = (
            window.rows,
            l_in,
            kernel_l,
            stride_l,
            pad_l,
            out_l,
            counted,
            call.dtype,
            threads,
            flat,
        )
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        rows: int,
        l_in: int,
        kernel_l: int,
        stride_l: int,
        pad_l: int,
        out_l: int,
        counted: "tuple[int, int] | None",
        dtype: torch.dtype,
        threads: int,
        flat: bool,
    ) -> None:
        super().__init__()
        self.rows, self.l_in, self.dtype = rows, l_in, dtype
        self._launch = {"threads": threads, "flat": flat}
        self.kernel = _avg_pool1d_register_kernel(
            rows, l_in, kernel_l, stride_l, pad_l, out_l, counted, self.dtype_str
        )
        self.init_config()

    @property
    def default_config(self) -> dict:
        """The launch ``entry_for`` derived from the call and the device."""
        return dict(self._launch)

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled for the launch in *cfg*."""
        return self.kernel(cfg["threads"], cfg["flat"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._require_devices(x=x)
        x = x.contiguous()
        y = self.program(self._compile)(x.view(self.rows, self.l_in))
        return y.view(*x.shape[:-1], y.shape[-1])
