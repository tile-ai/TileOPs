"""Symmetric INT8 quantization with one scale per row, each row held in registers by one CTA.

``INT8QuantPerChannelFwdKernel`` quantizes the row as it is; ``SmoothQuantFwdKernel`` first
divides each column by its smoothing factor.
"""

import functools
import math
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import MAX_BLOCK_THREADS, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import (
    INT8QuantPerChannelFwdInterface,
    QuantizeCall,
    SmoothQuantFwdInterface,
)
from tileops.kernels.quantization.int8_codes import (
    INV_QMAX,
    SCALE_UP,
    SMALL_SCALE,
    abs_bits,
    quantize,
    widen,
)
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["INT8QuantPerChannelFwdKernel", "SmoothQuantFwdKernel"]


def _quantize_rows(
    rows: int,
    k: int,
    dtype: str,
    threads: int,
    vpt: int,
    pair: bool,
    evict_first: bool,
    min_blocks: int,
    grid: int,
    smoothed: bool,
):
    """The body of CTA ``bx`` of ``grid``: quantize its rows of ``rows`` rows of ``k`` elements.

    The input is addressed as one flat run in 16-byte vectors aligned on the flat index,
    so a vector of ``w`` and the codes it produces are both aligned. CTA ``b`` quantizes
    rows ``b``, ``b + grid``, ...; for each it holds the vectors that overlap the row, and
    when ``k`` is not a whole number of vectors the first and last of them are shared with
    the neighbouring rows, and only their in-row elements count.

    When ``smoothed``, element ``c`` of a row is divided by ``smooth[c]`` first. The CTA
    reads ``smooth`` and takes its reciprocals once for all its rows, and a quotient is the
    product with the reciprocal corrected by two FMAs, which is correctly rounded while no
    intermediate leaves the normal range. A row whose smoothing factors or amax fall
    outside that range is divided again with IEEE division.
    """
    n = rows * k
    vec = VECTOR_ACCESS_BYTES // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    aligned = k % vec == 0
    # The storage may end inside the last vector.
    ragged_end = n % vec != 0
    # A bfloat16 pair is held as one 32-bit word: each half widens to float32 in one bit
    # operation, where extracting a 16-bit element first costs a second.
    words = dtype == "bfloat16"
    held = "uint32" if words else dtype
    width = vec // 2 if words else vec
    # With |smooth| in [2**-60, 2**60], the corrected product equals x / smooth for every
    # finite 16-bit x whose quotient lies in [2**-40, 2**40]; a quotient above that range
    # stays above it or turns non-finite, one below stays below it. A row whose amax is in
    # [2**-30, 2**40] thus has the exact amax, and its quotients below 2**-40 round to code
    # 0 either way. Checked against torch on every finite float16 and bfloat16 value.
    # The bounds as float32 bit patterns, which order as the magnitudes do and put a NaN
    # above every number: a float32 2**e has the pattern (127 + e) << 23.
    smooth_lo = (127 - 60) << 23
    smooth_hi = (127 + 60) << 23
    amax_lo = (127 - 30) << 23
    amax_hi = (127 + 40) << 23
    # The bit pattern above every number, which sends a row to IEEE division.
    unsafe = 2**31 - 1
    # Floats of ``smooth`` in one 16-byte load.
    smooth_vec = VECTOR_ACCESS_BYTES // 4
    rows_per_cta = (rows + grid - 1) // grid
    # A thread holds vpt vectors in groups of ``group`` adjacent ones: group g of thread
    # t starts at window vector (g * threads + t) * group.
    group = 2 if pair else 1
    # Every held vector lies in the row's window only when the window is exactly
    # threads * vpt vectors long.
    exact = aligned and threads * vpt == k // vec
    assert group == 1 or (exact and vpt % group == 0)
    warps = threads // WARP_LANES
    load = "tl::tileops_load16_evict_first" if evict_first else "tl::tileops_load16"

    def slot(j, tx):
        return ((j // group) * threads + tx) * group + j % group

    def element(values, j, c, half):
        """The float32 of element ``2 * c + half`` of vector ``j``, or ``c`` unless words."""
        return widen(values[j, c], half, dtype) if words else T.cast(values[j, c], "float32")

    def scaled(values, xs, j, c, half, num):
        """The dividend of a code, times its row's pre-scale factor: the element, or the
        quotient a smoothed row divided it into."""
        e = 2 * c + half if words else c
        return (xs[j, e] if smoothed else element(values, j, c, half)) * num[0]

    # A smoothed element is divided first, and its quotient kept in ``xs``.
    @T.macro
    def fold_one(acc, xs, div, rcp, j, e, value, v, lo, hi, masked: bool, ieee: bool):
        if smoothed:
            if ieee:
                xs[j, e] = value / div[j, e]
            else:
                q0 = value * rcp[j, e]
                xs[j, e] = T.ieee_fmaf(T.ieee_fmaf(-q0, div[j, e], value), rcp[j, e], q0)
        if masked:
            acc[0] = T.max(
                acc[0],
                T.if_then_else(
                    (v * vec + e >= lo) & (v * vec + e < hi),
                    abs_bits(xs[j, e] if smoothed else value),
                    0,
                ),
            )
        else:
            acc[0] = T.max(acc[0], abs_bits(xs[j, e] if smoothed else value))

    @T.macro
    def fold(acc, values, xs, div, rcp, j, v, lo, hi, masked: bool, ieee: bool):
        for c in T.unroll(width):
            if words:
                fold_one(
                    acc, xs, div, rcp, j, 2 * c, element(values, j, c, 0), v, lo, hi, masked, ieee
                )
                fold_one(
                    acc,
                    xs,
                    div,
                    rcp,
                    j,
                    2 * c + 1,
                    element(values, j, c, 1),
                    v,
                    lo,
                    hi,
                    masked,
                    ieee,
                )
            else:
                fold_one(acc, xs, div, rcp, j, c, element(values, j, c, 0), v, lo, hi, masked, ieee)

    @T.macro
    def codes(out, values, xs, j, at, num, clamp: bool):
        for c in T.unroll(width):
            if words:
                out[at + 2 * c] = quantize(scaled(values, xs, j, c, 0, num), num[1], num[2], clamp)
                out[at + 2 * c + 1] = quantize(
                    scaled(values, xs, j, c, 1, num), num[1], num[2], clamp
                )
            else:
                out[at + c] = quantize(scaled(values, xs, j, c, 0, num), num[1], num[2], clamp)

    @T.macro
    def store_group(q, out, values, xs, g, v, num, clamp: bool):
        for i in T.unroll(group):
            codes(out, values, xs, g * group + i, i * vec, num, clamp)
        for c in T.vectorized(group * vec):
            q[v * vec + c] = out[c]

    @T.macro
    def store_row(q, out, values, xs, v0, count, lo, hi, tx, num, clamp: bool):
        for g in T.unroll(vpt // group):
            if exact:
                store_group(q, out, values, xs, g, v0 + slot(g * group, tx), num, clamp)
            elif slot(g, tx) < count:
                if aligned:
                    store_group(q, out, values, xs, g, v0 + slot(g, tx), num, clamp)
                elif (slot(g, tx) == 0) | (slot(g, tx) == count - 1):
                    codes(out, values, xs, g, 0, num, clamp)
                    for c in T.unroll(vec):
                        if ((v0 + slot(g, tx)) * vec + c >= lo) & (
                            (v0 + slot(g, tx)) * vec + c < hi
                        ):
                            q[(v0 + slot(g, tx)) * vec + c] = out[c]
                else:
                    store_group(q, out, values, xs, g, v0 + slot(g, tx), num, clamp)

    @T.macro
    def load_row(w, values, tx, v0, count):
        if ragged_end:
            for j in T.serial(vpt):
                for c in T.serial(width):
                    values[j, c] = T.cast(0, held)
        for j in T.unroll(vpt):
            if exact or slot(j, tx) < count:
                if ragged_end and (v0 + slot(j, tx) + 1) * vec > n:
                    for c in T.unroll(vec):
                        if (v0 + slot(j, tx)) * vec + c < n:
                            if words:
                                bits = T.cast(
                                    T.reinterpret(w[(v0 + slot(j, tx)) * vec + c], "uint16"),
                                    "uint32",
                                )
                                if c % 2 == 0:
                                    values[j, c // 2] = values[j, c // 2] | bits
                                else:
                                    values[j, c // 2] = values[j, c // 2] | (bits << T.uint32(16))
                            else:
                                values[j, c] = w[(v0 + slot(j, tx)) * vec + c]
                else:
                    T.call_extern(
                        "handle",
                        load,
                        T.address_of(values[j, 0]),
                        T.address_of(w[(v0 + slot(j, tx)) * vec]),
                    )

    @T.macro
    def fold_row(acc, values, xs, div, rcp, tx, v0, count, lo, hi, ieee: bool):
        for j in T.unroll(vpt):
            if exact:
                fold(acc, values, xs, div, rcp, j, v0 + slot(j, tx), lo, hi, False, ieee)
            elif slot(j, tx) < count:
                if aligned:
                    fold(acc, values, xs, div, rcp, j, v0 + slot(j, tx), lo, hi, False, ieee)
                elif (slot(j, tx) == 0) | (slot(j, tx) == count - 1):
                    fold(acc, values, xs, div, rcp, j, v0 + slot(j, tx), lo, hi, True, ieee)
                else:
                    fold(acc, values, xs, div, rcp, j, v0 + slot(j, tx), lo, hi, False, ieee)

    @T.macro
    def reduce_row(acc, warp_max, tx):
        for stage in T.serial(WARP_SHUFFLE_STAGES):
            acc[0] = T.max(
                acc[0],
                T.shfl_xor(acc[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES),
            )
        if tx % WARP_LANES == 0:
            warp_max[tx // WARP_LANES] = acc[0]
        T.sync_threads()
        acc[0] = warp_max[0]
        for i in T.serial(1, warps):
            acc[0] = T.max(acc[0], warp_max[i])

    @T.macro
    def quantize_row(row, w, q, scale, tx, values, xs, div, rcp, flag, acc, out, num, warp_max):
        lo = row * k
        hi = lo + k
        v0 = lo // vec
        count = (hi + vec - 1) // vec - v0

        acc[0] = T.if_then_else(flag[0] != 0, T.int32(unsafe), 0) if smoothed else 0
        fold_row(acc, values, xs, div, rcp, tx, v0, count, lo, hi, False)
        reduce_row(acc, warp_max, tx)
        if smoothed and ((acc[0] < amax_lo) | (acc[0] > amax_hi)):
            load_row(w, values, tx, v0, count)
            acc[0] = 0
            fold_row(acc, values, xs, div, rcp, tx, v0, count, lo, hi, True)
            T.sync_threads()
            reduce_row(acc, warp_max, tx)

        amax = T.reinterpret(acc[0], "float32")
        num[1] = T.if_then_else(amax > 0, amax * T.float32(INV_QMAX), T.float32(1.0))
        if tx == 0:
            scale[row] = num[1]
        num[0] = T.if_then_else(
            num[1] < T.float32(SMALL_SCALE), T.float32(SCALE_UP), T.float32(1.0)
        )
        num[1] = num[1] * num[0]
        num[2] = T.ieee_frcp(num[1])
        # A scale scaled up out of the subnormals may still leave a quotient past 127,
        # which the clamp bounds; a normal scale cannot.
        if num[0] > T.float32(1.0):
            store_row(q, out, values, xs, v0, count, lo, hi, tx, num, True)
        else:
            store_row(q, out, values, xs, v0, count, lo, hi, tx, num, False)

    @T.macro
    def body(bx, w, smooth, q, scale):
        if min_blocks > 1:
            T.annotate_min_blocks_per_sm(min_blocks)
        tx = T.get_thread_binding()
        values = T.alloc_local((vpt, width), held)
        # The quotients of the held elements, their divisors and the divisors' reciprocals.
        xs = T.alloc_local((vpt, vec), "float32") if smoothed else None
        div = T.alloc_local((vpt, vec), "float32") if smoothed else None
        rcp = T.alloc_local((vpt, vec), "float32") if smoothed else None
        # The next row's vectors, loaded while this row is quantized.
        ahead = T.alloc_local((vpt, width), held) if rows_per_cta > 1 else None
        # Nonzero when a smoothing factor this CTA divides by lies outside the fast range.
        flag = T.alloc_local((1,), "int32") if smoothed else None
        acc = T.alloc_local((1,), "int32")
        out = T.alloc_local((group * vec,), "int8")
        num = T.alloc_local((3,), "float32")
        warp_max = T.alloc_shared((warps,), "int32")

        load_row(w, values, tx, bx * k // vec, (bx * k + k + vec - 1) // vec - bx * k // vec)
        if smoothed:
            flag[0] = 0
            # Column of element c of held vector j: slot * vec + c - off, the same in every
            # row of the CTA.
            off = bx * k % vec
            for j in T.unroll(vpt):
                if exact or slot(j, tx) < (bx * k + k + vec - 1) // vec - bx * k // vec:
                    if aligned:
                        for h in T.unroll(vec // smooth_vec):
                            T.call_extern(
                                "handle",
                                "tl::tileops_load16",
                                T.address_of(div[j, h * smooth_vec]),
                                T.address_of(smooth[slot(j, tx) * vec + h * smooth_vec]),
                            )
                    else:
                        for c in T.unroll(vec):
                            div[j, c] = T.if_then_else(
                                (slot(j, tx) * vec + c >= off) & (slot(j, tx) * vec + c < off + k),
                                smooth[slot(j, tx) * vec + c - off],
                                T.float32(1.0),
                            )
                    for c in T.unroll(vec):
                        rcp[j, c] = T.ieee_frcp(div[j, c])
                        bits = abs_bits(div[j, c])
                        if (bits < smooth_lo) | (bits > smooth_hi):
                            flag[0] = 1
        if rows_per_cta == 1:
            quantize_row(bx, w, q, scale, tx, values, xs, div, rcp, flag, acc, out, num, warp_max)
        else:
            for i in T.serial(rows_per_cta):
                if bx + i * grid < rows:
                    if bx + (i + 1) * grid < rows:
                        load_row(
                            w,
                            ahead,
                            tx,
                            (bx + (i + 1) * grid) * k // vec,
                            ((bx + (i + 1) * grid) * k + k + vec - 1) // vec
                            - (bx + (i + 1) * grid) * k // vec,
                        )
                    quantize_row(
                        bx + i * grid,
                        w,
                        q,
                        scale,
                        tx,
                        values,
                        xs,
                        div,
                        rcp,
                        flag,
                        acc,
                        out,
                        num,
                        warp_max,
                    )
                    # The next row rewrites warp_max.
                    T.sync_threads()
                    for j in T.unroll(vpt):
                        for c in T.unroll(width):
                            values[j, c] = ahead[j, c]

    return body


@functools.lru_cache(maxsize=32)
def _int8_quant_per_channel_kernel(rows: int, k: int, dtype: str):
    """Build the quantization of ``rows`` rows of ``k`` contiguous elements."""
    n = rows * k

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _int8_quant_per_channel_func(
        threads: int, vpt: int, pair: bool, evict_first: bool, min_blocks: int
    ):
        body = _quantize_rows(
            rows, k, dtype, threads, vpt, pair, evict_first, min_blocks, rows, False
        )

        @T.prim_func
        def _int8_quant_per_channel_main(
            w: T.Tensor((n,), dtype),
            q: T.Tensor((n,), "int8"),
            scale: T.Tensor((rows,), "float32"),
        ):
            with T.Kernel(rows, threads=threads) as bx:
                body(bx, w, None, q, scale)

        return _int8_quant_per_channel_main

    return _int8_quant_per_channel_func


@functools.lru_cache(maxsize=32)
def _smooth_quant_kernel(rows: int, k: int, dtype: str):
    """Build the smoothing and quantization of ``rows`` rows of ``k`` contiguous elements."""
    n = rows * k
    vec = VECTOR_ACCESS_BYTES // torch.empty((), dtype=getattr(torch, dtype)).element_size()

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _smooth_quant_func(
        threads: int, vpt: int, pair: bool, evict_first: bool, min_blocks: int, ctas: int
    ):
        # Row r starts (r * k) % vec elements into its first vector, which repeats every
        # ``period`` rows; a CTA's rows lie ``grid`` apart, a multiple of it, so each thread
        # holds the same columns in every row and reads their smoothing factors once.
        period = vec // math.gcd(k, vec)
        grid = rows if ctas >= rows else max(period, ctas // period * period)
        body = _quantize_rows(
            rows, k, dtype, threads, vpt, pair, evict_first, min_blocks, grid, True
        )

        @T.prim_func
        def _smooth_quant_main(
            x: T.Tensor((n,), dtype),
            smooth: T.Tensor((k,), "float32"),
            q: T.Tensor((n,), "int8"),
            scale: T.Tensor((rows,), "float32"),
        ):
            with T.Kernel(grid, threads=threads) as bx:
                body(bx, x, smooth, q, scale)

        return _smooth_quant_main

    return _smooth_quant_func


class _INT8QuantPerRowKernel(Kernel):
    """The launch policy of the one-CTA-per-row INT8 quantize program.

    One CTA owns one row and holds it in registers across the row's reduction, so the
    input is read once and ``q`` written once. ``q`` and ``scale`` are bit-equal to the
    torch reference: the scale is the product torch computes, and the quotient is
    correctly rounded.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    # The one integer tensor is ``q``, written before anything reads it.
    autotune_accepts_random_int_inputs = True

    # Launch policy, fitted by timing each manifest row with ``bench_kernel``. Re-fit: time
    # vpt in {2, 4}, min_blocks from 1 to resident threads // threads, pairs on and off and
    # both load policies on every manifest row, and keep the fastest launch per row.
    # Vectors a thread holds, and on a launch with fewer rows than SMs, where more threads
    # per row shorten the one wave ([17, 4099] bfloat16: 2.0 us against 2.2).
    _VPT: ClassVar[int] = 4
    _FEW_ROWS_VPT: ClassVar[int] = 2
    # Rows of at least _WIDE_ROW elements hold _WIDE_ROW_ELEMENTS per thread and fill the SM
    # with threads ([5120, 8192] bfloat16: 32.2 us at 512 threads of 16 elements, 32.7 at
    # 256 of 32).
    _WIDE_ROW: ClassVar[int] = 8192
    _WIDE_ROW_ELEMENTS: ClassVar[int] = 16
    # A CTA wider than _WIDE_THREADS takes MAX_BLOCK_THREADS ([8192, 28672] bfloat16: 896
    # threads spill at the two-CTA register cap and run 240 us, 1024 run 165).
    _WIDE_THREADS: ClassVar[int] = 512
    # Registers a thread may take on a row below _WIDE_ROW, which sets min_blocks; at least
    # two CTAs share an SM ([5120, 8192] bfloat16 at 256 threads: 36.0 us uncapped at 72
    # registers, 32.6 at 64).
    _MAX_REGISTERS: ClassVar[int] = 64
    # Loads are evict-first below this many bytes of w and default-policy above ([5760,
    # 2880] bfloat16: 14.6 us evict-first, 14.8 default; [14336, 4096] 45.3 and 43.8).
    _EVICT_FIRST_BYTES: ClassVar[int] = 96 << 20

    @classmethod
    def refusal(cls, call: QuantizeCall) -> Optional[str]:
        int32_max = 2**31 - 1
        reason = super().refusal(call)
        if reason is None and call.rows * call.cols > int32_max - VECTOR_ACCESS_BYTES:
            return f"indexes elements with int32, and N * K = {call.rows * call.cols}"
        return reason

    def __init__(
        self, call: QuantizeCall, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.dtype = call.dtype
        self.rows = call.rows
        self.cols = call.cols
        self._sm_count = call.sm_count
        props = torch.cuda.get_device_properties(call.device)
        self._resident_threads = props.max_threads_per_multi_processor
        self._register_file = props.regs_per_multiprocessor
        vec = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        # Vectors a row's window spans at most.
        self._span = call.cols // vec if call.cols % vec == 0 else -(-(call.cols + vec - 1) // vec)
        self.kernel = self._builder(call.rows, call.cols, self.dtype_str)
        self.init_config(config, tune)

    def config_for(self, vpt: int, pair: bool, evict_first: bool, min_blocks: int) -> dict:
        """The launch holding about ``vpt`` vectors per thread."""
        threads = -(-self._span // vpt)
        threads = -(-threads // WARP_LANES) * WARP_LANES
        if threads > self._WIDE_THREADS:
            threads = MAX_BLOCK_THREADS
        vpt = -(-self._span // threads)
        vec = VECTOR_ACCESS_BYTES // self.dtype.itemsize
        exact = self.cols % vec == 0 and threads * vpt == self.cols // vec
        return {
            "threads": threads,
            "vpt": vpt,
            "pair": pair and exact and vpt % 2 == 0,
            "evict_first": evict_first,
            "min_blocks": min_blocks,
        }

    @property
    def default_config(self) -> dict:
        if self.cols >= self._WIDE_ROW:
            vpt = self._WIDE_ROW_ELEMENTS * self.dtype.itemsize // VECTOR_ACCESS_BYTES
            threads = self.config_for(vpt, True, True, 1)["threads"]
            min_blocks = self._resident_threads // threads
        else:
            vpt = self._FEW_ROWS_VPT if self.rows < self._sm_count else self._VPT
            threads = self.config_for(vpt, True, True, 1)["threads"]
            min_blocks = max(2, self._register_file // (threads * self._MAX_REGISTERS))
        evict_first = self.rows * self.cols * self.dtype.itemsize < self._EVICT_FIRST_BYTES
        # Pairs of adjacent vectors store 16 bytes of codes at a time: [5120, 8192]
        # bfloat16 runs 32.5 us with them and 33.1 without.
        return self.config_for(vpt, True, evict_first, min_blocks)

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]


class INT8QuantPerChannelFwdKernel(_INT8QuantPerRowKernel, INT8QuantPerChannelFwdInterface):
    """Quantize each row of ``w`` against its own amax, reading ``w`` once.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads``, ``vpt``, ``pair``, ``evict_first`` and
            ``min_blocks``.
        tune: Whether to autotune.
    """

    aligned_inputs = ("w",)

    _builder = staticmethod(_int8_quant_per_channel_kernel)

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_cuda(w=w)
        q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
        scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
        c = self.config
        self.kernel(c["threads"], c["vpt"], c["pair"], c["evict_first"], c["min_blocks"])(
            w.view(-1), q.view(-1), scale
        )
        return q, scale


class SmoothQuantFwdKernel(_INT8QuantPerRowKernel, SmoothQuantFwdInterface):
    """Divide ``x`` by ``smooth`` per column, then quantize each row against its own amax.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads``, ``vpt``, ``pair``, ``evict_first``,
            ``min_blocks`` and ``ctas``.
        tune: Whether to autotune.
    """

    aligned_inputs = ("x", "smooth")

    _builder = staticmethod(_smooth_quant_kernel)

    # Launch policy, fitted by timing each manifest row with ``bench_kernel``. Re-fit: time
    # vpt in {1, 2}, min_blocks in {2, 3} and rows per CTA in {1, 4, 6, 8, 12, 16} on every
    # manifest row, and keep the fastest launch per row.
    # Two vectors a thread on an aligned row of at least _PAIRED_ROW vectors, when the rows
    # fill the SMs ([4096, 4096] bfloat16: 16.0 us at 256 threads of two, 17.8 at 512 of
    # one); one otherwise ([1000, 4099] float16: 9.0 us at 544 threads of one, 10.8 at 288
    # of two).
    _PAIRED_ROW: ClassVar[int] = 512
    # Registers a thread may take, which sets min_blocks and so the CTAs resident at once
    # ([4096, 4096] bfloat16 at 256 threads: 16.0 us at three CTAs per SM, 17.4 at two).
    _MAX_REGISTERS: ClassVar[int] = 80

    def config_for(
        self, vpt: int, pair: bool, evict_first: bool, min_blocks: int, rows_per_cta: int = 1
    ) -> dict:
        """The launch holding about ``vpt`` vectors per thread and ``rows_per_cta`` rows."""
        threads = -(-self._span // vpt)
        threads = min(-(-threads // WARP_LANES) * WARP_LANES, MAX_BLOCK_THREADS)
        vpt = -(-self._span // threads)
        vec = VECTOR_ACCESS_BYTES // self.dtype.itemsize
        exact = self.cols % vec == 0 and threads * vpt == self.cols // vec
        return {
            "threads": threads,
            "vpt": vpt,
            "pair": pair and exact and vpt % 2 == 0,
            "evict_first": evict_first,
            "min_blocks": min_blocks,
            # Rows split evenly over the CTAs.
            "ctas": -(-self.rows // rows_per_cta),
        }

    def _launch(self, rows_per_cta: Optional[int]) -> dict:
        """The fitted launch, with ``rows_per_cta`` rows a CTA or as many as fill the SMs once."""
        vec = VECTOR_ACCESS_BYTES // self.dtype.itemsize
        paired = (
            self.cols % vec == 0 and self._span >= self._PAIRED_ROW and self.rows >= self._sm_count
        )
        threads = self.config_for(2 if paired else 1, True, True, 1)["threads"]
        min_blocks = max(2, self._register_file // (threads * self._MAX_REGISTERS))
        if rows_per_cta is None:
            rows_per_cta = -(-self.rows // (self._sm_count * min_blocks))
        evict_first = self.rows * self.cols * self.dtype.itemsize < self._EVICT_FIRST_BYTES
        return self.config_for(2 if paired else 1, True, evict_first, min_blocks, rows_per_cta)

    @property
    def default_config(self) -> dict:
        return self._launch(None)

    @property
    def autotune_configs(self) -> list[dict]:
        # Fewer rows a CTA than fill the SMs once, which a row count that the fill does not
        # divide can prefer ([3000, 2880] bfloat16: 11.1 us at 8 rows a CTA, 11.7 at 12).
        rows_per_cta = -(-self.rows // self.default_config["ctas"])
        fewer = self._launch(-(-2 * rows_per_cta // 3))
        return [self.default_config] + ([fewer] if fewer != self.default_config else [])

    def forward(self, x: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_cuda(x=x, smooth=smooth)
        q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        scale = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        c = self.config
        self.kernel(
            c["threads"], c["vpt"], c["pair"], c["evict_first"], c["min_blocks"], c["ctas"]
        )(x.view(-1), smooth, q.view(-1), scale)
        return q, scale
