"""GroupNorm forward.

The input is viewed as ``(N * G, D)`` rows for reduction while affine values
remain per channel. Row tails need no host padding copy.
Rows one warp holds take one a warp; rows the registers hold, and wider rows the 16-byte
access divides, are held one a block (`_held_rows`); the rest are staged through shared
memory or cut across blocks by ``GroupNormSplitKernel``.
"""

import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.norm._config import (
    NARROW_ROW,
    make_read_leads,
    pairwise_sum,
    row_padding,
    select_row_config_by_width,
    select_row_configs,
    warp_row_vector,
    widths_for_row,
)
from tileops._kernels.norm.call_spec import GroupNormCall, GroupNormFwdInterface
from tileops._kernels.streaming_load import load16
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = [
    "GroupNormKernel",
    "GroupNormNoAffineKernel",
    "make_corrected_row_reduce",
    "make_warp_row_norm",
]


class _RowNormKernel(Kernel):
    """What both kernels here share: the row's tiling and the config space for it.

    Whether the row is read from global memory straight into register fragments
    decides which reduction the program uses and which block widths are worth
    offering, so both follow from one predicate rather than from either caller.

    Args:
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional tile config dict.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    # Columns one thread carries, for a row held in register fragments. Below the
    # floor it has too few accesses in flight to cover the row's latency; above
    # the ceiling the two fp32 fragments outgrow what it can hold.
    _ELEMENTS_PER_THREAD_BAND = (16, 64)

    # A padded row takes one width rather than a band: every column a thread
    # carries there also pays the per-element bounds guard.
    _GUARDED_ELEMENTS_PER_THREAD = 16

    def __init__(
        self,
        D: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.D = D
        self.eps = eps
        self.dtype = dtype
        self.D_padded = row_padding(D, dtype.itemsize)
        self.init_config(config)

    @classmethod
    def _holds_row_in_registers(cls, D: int, D_padded: int) -> bool:
        """Whether a row of this width is read from global memory into fragments.

        A row whose width the block divides is. A padded one loses the vectorized
        copy to a per-element guard, so it only stays in registers while it is narrow
        enough that staging through shared memory would cost more.

        Args:
            D: Row length.
            D_padded: *D* rounded up to a width the block divides.
        """
        return D_padded == D or D_padded <= NARROW_ROW

    @classmethod
    def _row_widths_for(cls, D: int, D_padded: int) -> tuple:
        """Block widths a row admits, narrowed while it is held in registers."""
        widths = widths_for_row(D_padded)
        if not cls._holds_row_in_registers(D, D_padded):
            return widths
        if D_padded != D:
            low = high = cls._GUARDED_ELEMENTS_PER_THREAD
        else:
            low, high = cls._ELEMENTS_PER_THREAD_BAND
        banded = tuple(t for t in widths if D_padded % t == 0 and low <= D_padded // t <= high)
        return banded or widths

    @property
    def _row_widths(self) -> tuple:
        """Block widths this row admits, narrowed while it is held in registers."""
        return self._row_widths_for(self.D, self.D_padded)

    def _warp_row_vector(self) -> Optional[int]:
        """Vector width of the warp-a-row body for this row, or ``None`` if it does not apply."""
        return self._warp_row_vector_for(self.D, self.dtype)

    @classmethod
    def _warp_row_vector_for(cls, D: int, dtype: torch.dtype) -> Optional[int]:
        """`_warp_row_vector` for a row of *D* elements of *dtype*, before construction."""
        return warp_row_vector(D, dtype.itemsize)

    def _held_row_vector(self) -> Optional[int]:
        """Elements of one access of the block-a-row body, or ``None`` if it does not apply."""
        vec = VECTOR_ACCESS_BYTES // self.dtype.itemsize
        if not self._holds_row_in_registers(self.D, self.D_padded):
            return None if self.D % vec else vec
        while self.D % vec:
            vec //= 2
        return vec

    @property
    def default_config(self) -> dict:
        if self._warp_row_vector() is not None:
            # Eight rows a block, one a warp. On H200 at (1024, 784) it took 3.136 us in
            # float32 against 3.167 at four rows and 3.200 at one; in float16 four and eight
            # tied at 2.464 against 2.496 at one. `_warp_rows` takes fewer for fewer rows.
            warp_row_threads = 256
            return {"block_m": 1, "threads": warp_row_threads}
        return select_row_config_by_width(self.D_padded, self._row_widths)

    @property
    def autotune_configs(self) -> list[dict]:
        if self._warp_row_vector() is not None:
            # Pinned: these rows run in 2.5 to 3.1 us, where the configs differ by about
            # one 32 ns timer tick, which the autotuner cannot rank.
            return [self.default_config]
        if self._held_row_vector() is not None:
            return select_row_configs(
                self.D_padded, self.dtype, widths=self._row_widths, block_ms=(1,)
            )
        return select_row_configs(self.D_padded, self.dtype, widths=self._row_widths)


def make_corrected_row_reduce(block_m, n, n_padded, eps):
    """A macro ``(row, sq, shift, acc, acc_sq, ref, centre, rstd)`` taking corrected two-pass
    statistics of an fp32 row block less its first elements *shift*: it leaves ``row - ref`` in
    *row*, and the normalized row is ``(row - centre) * rstd``."""
    padded = n != n_padded

    def _in_row(j, value):
        return T.if_then_else(j < n, value, T.cast(0.0, "float32")) if padded else value

    @T.macro
    def corrected_row_reduce(row, sq, shift, acc, acc_sq, ref, centre, rstd):
        for i, j in T.Parallel(block_m, n_padded):
            sq[i, j] = _in_row(j, row[i, j] - shift[i])
        T.reduce_sum(sq, acc, dim=1)
        for i in T.Parallel(block_m):
            ref[i] = shift[i] + acc[i] * (1.0 / n)
        for i, j in T.Parallel(block_m, n_padded):
            row[i, j] = _in_row(j, row[i, j] - ref[i])
            sq[i, j] = row[i, j] * row[i, j]
        # One accumulator each: sharing one puts a barrier between the two sums.
        T.reduce_sum(row, acc, dim=1)
        T.reduce_sum(sq, acc_sq, dim=1)
        for i in T.Parallel(block_m):
            centre[i] = acc[i] * (1.0 / n)
            rstd[i] = T.rsqrt(acc_sq[i] * (1.0 / n) - centre[i] * centre[i] + eps)

    return corrected_row_reduce


# fp32's largest finite value.
_FMAX = 3.4028234663852886e38

# The factor a warp's merge takes its lanes' differences in: 2^-k with 32 * 4^-k <= 1.
_LANE_UNIT = 0.125

# A scaled square sum below which a warp keeps its differences unscaled: a warp past it
# holds a spread whose squares' subnormal rounding stays below fp32's precision.
_TINY = 2.0**-64


class ThreadScale:
    """A warp's difference scale: ``scale`` and ``lane``, or one where the warp chose to stay
    unscaled. ``units[at:at + 4]`` holds the thread scale, the lane scale and their reciprocals."""

    def __init__(self, scale, lane):
        self.scaled = scale != 1.0
        self.scale, self.lane = scale, lane

    def pick(self, units, at, scaled, plain):
        """*scaled* where the warp scales, else *plain*."""
        if not self.scaled:
            return plain
        return T.if_then_else(units[at] < 1.0, scaled, plain)

    def inverse(self, units, at=0):
        return units[at + 2] if self.scaled else 1.0

    @property
    def square_unit(self):
        """A scaling warp's unit of a merged square sum's root: both scales' product."""
        return self.scale * self.lane


def make_thread_stats(vec, per_thread, full, stride, vectors, dtype, active=None, span=None):
    """Create ``(thread_stats, warp_merge, units)``: a thread's statistics of the vectors it holds
    (``k * stride + t``), about ``ref``, the mean of its first elements, then merged across its
    warp about lane 0's mean, every difference scaled by the exact powers of two in ``units``
    (`ThreadScale`) so no square sum passes fp32's range before the row's does. Each leaves
    ``ref``, the mean less it and the square sum about it in ``stats[0:3]``."""
    # Elements the reference averages: every thread that holds any holds `full` vectors.
    ref_elements = min(max(4, -(-per_thread * vec // 8)), vec * max(full, 1))
    # Threads below `boundary` hold `per_thread` vectors, the rest `full`; in a warp of
    # fewer `active` lanes, the lanes past them hold nothing.
    boundary = vectors - full * stride
    # Samuelson's bound: a thread's mean square less its squared mean, and a warp's about lane
    # 0's mean, cancel by at most the count's factor, which the scale keeps in range.
    largest = {"float16": 65504.0, "bfloat16": 3.3895e38, "float32": _FMAX}[dtype]
    span = span or WARP_LANES * per_thread * vec
    if per_thread * vec * WARP_LANES * (2.0 * largest) ** 2 <= _FMAX:
        scale, lane = 1.0, 1.0
    else:
        # At least 1/2: a scaled warp is one whose thread scale is below 1.
        scale = 2.0 ** -max(1, (per_thread * vec - 1).bit_length() + 1 >> 1)
        lane = _LANE_UNIT
    bound = 2.0 ** math.floor(math.log2(math.sqrt(_FMAX / (64.0 * span))))
    units = ThreadScale(scale, lane)
    idle = active is not None and active < WARP_LANES

    def _held(t, value):
        return T.if_then_else(t < active, value, 0.0) if idle else value

    def _counted(k, t):
        """Whether thread *t*'s vector *k* is in the row."""
        if idle:
            return t < active
        if per_thread == full:
            return True
        return T.Or(k < full, k * stride + t < vectors)

    def count(t, factor=1.0):
        """The elements thread *t* holds times *factor*, as fp32."""
        if per_thread == full:
            return _held(t, float(vec * full) * factor)
        return T.if_then_else(
            t < boundary, float(vec * per_thread) * factor, float(vec * full) * factor
        )

    def _reciprocal(t):
        if per_thread == full:
            return _held(t, 1.0 / (vec * full))
        short = 1.0 / (vec * full) if full else 0.0
        return T.if_then_else(t < boundary, 1.0 / (vec * per_thread), short)

    def _reference_offset(centred):
        """The mean of the next ``ref_elements - 1`` elements less the first, over all."""
        if ref_elements == 1:
            return T.float32(0.0)
        return pairwise_sum(
            [(centred[i] - centred[0]) * (1.0 / ref_elements) for i in range(1, ref_elements)]
        )

    def _fold(partial):
        """The sum of a thread's ``vec`` partial sums, added pairwise."""
        return pairwise_sum([partial[c] for c in range(vec)])

    @T.macro
    def _sums(centred, t, stats, unit, offset):
        """``(centred - offset) * unit`` in place, and in ``stats[1:4]`` its mean, its square
        sum about the mean and its square sum."""
        sums = T.alloc_local((vec,), "float32")
        squares = T.alloc_local((vec,), "float32")
        for c in T.unroll(vec):
            sums[c] = T.cast(0.0, "float32")
            squares[c] = T.cast(0.0, "float32")
        for k in T.unroll(per_thread):
            for c in T.unroll(vec):
                centred[k * vec + c] = centred[k * vec + c] * unit - offset * unit
            if _counted(k, t):
                for c in T.unroll(vec):
                    sums[c] += centred[k * vec + c]
                    squares[c] += centred[k * vec + c] * centred[k * vec + c]
        stats[1] = _fold(sums) * _reciprocal(t)
        stats[3] = _fold(squares)
        stats[2] = stats[3] - stats[1] * _fold(sums)

    @T.macro
    def thread_stats(centred, t, stats, factors, at=0):
        stats[0] = centred[0] + _reference_offset(centred)
        _sums(centred, t, stats, scale, stats[0])
        if units.scaled:
            # A warp whose squares are all below ``_TINY`` and references within ``bound`` takes its
            # statistics again unscaled, keeping their precision near fp32's smallest normal.
            factors[at] = T.if_then_else(
                T.any_sync(
                    T.if_then_else(
                        T.Or(
                            T.Not(stats[3] < _TINY),
                            T.Not(T.abs(stats[0]) <= bound),
                        ),
                        1,
                        0,
                    )
                )
                != 0,
                scale,
                1.0,
            )
            factors[at + 1] = T.if_then_else(factors[at] < 1.0, lane, 1.0)
            factors[at + 2] = T.if_then_else(factors[at] < 1.0, 1.0 / scale, 1.0)
            factors[at + 3] = T.if_then_else(factors[at] < 1.0, 1.0 / lane, 1.0)

    def _warp_reciprocal(t, factor=1.0):
        """*factor* over the elements thread *t*'s warp holds; a warp holding none takes no
        part, so its value is never read."""

        if idle:
            return factor / (vec * active * full)

        def recip(w):
            held = vec * (WARP_LANES * full + min(max(boundary - w * WARP_LANES, 0), WARP_LANES))
            return factor / held if held else 0.0

        warp, whole = t // WARP_LANES, boundary // WARP_LANES
        before, after = recip(0), recip(stride // WARP_LANES - 1)
        if boundary % WARP_LANES == 0:
            return T.if_then_else(warp < whole, before, after)
        return T.if_then_else(
            warp < whole, before, T.if_then_else(warp == whole, recip(whole), after)
        )

    @T.macro
    def _merge(t, stats, partner, unit, lane_unit):
        # Sums of count * e and of m2 + (count * e) * e, e a lane's mean less lane 0's.
        partner[0] = T.shfl_sync(stats[0], 0)
        partner[2] = (stats[0] - partner[0]) * unit
        partner[1] = T.shfl_sync(stats[1], 0)
        partner[2] = partner[2] + (stats[1] - partner[1])
        if full and not idle:
            stats[2] = stats[2] * (lane_unit * lane_unit) + (count(t, lane_unit) * partner[2]) * (
                partner[2] * lane_unit
            )
            partner[2] = count(t, lane_unit) * partner[2]
        else:
            # A thread holding nothing takes no part; its statistics are never read.
            stats[2] = T.if_then_else(
                count(t) > 0,
                stats[2] * (lane_unit * lane_unit)
                + (count(t, lane_unit) * partner[2]) * (partner[2] * lane_unit),
                0.0,
            )
            partner[2] = T.if_then_else(count(t) > 0, count(t, lane_unit) * partner[2], 0.0)
        for st in T.unroll(WARP_SHUFFLE_STAGES):
            partner[2] += T.shfl_xor(partner[2], T.shift_left(1, st))
            stats[2] += T.shfl_xor(stats[2], T.shift_left(1, st))
        stats[2] = stats[2] - partner[2] * (partner[2] * _warp_reciprocal(t))
        stats[1] = (
            (partner[0] - stats[0]) * unit
            + partner[1]
            + partner[2] * _warp_reciprocal(t, 1.0 / lane_unit)
        )

    @T.macro
    def warp_merge(centred, t, stats, partner, factors, at=0):
        _merge(t, stats, partner, scale, lane)
        if units.scaled and factors[at] == 1.0:
            # Exact: a difference whose unscaled square is normal is normal scaled.
            _sums(centred, t, stats, 1.0 / scale, 0.0)
            _merge(t, stats, partner, 1.0, 1.0)

    return thread_stats, warp_merge, units


def make_warp_row_norm(n, dtype, eps, vec, affine):
    """Create ``(held_n, reduce_row, write_row)`` normalizing a row of *n* elements one warp
    holds (`make_thread_stats`); the caller's program pre-includes ``streaming_load.h``."""
    # Fewer lanes, at least 16, where they hold the row evenly and so fewer elements each.
    lanes = WARP_LANES
    if (n // vec) % WARP_LANES:
        most = -(-(n // vec) // WARP_LANES) * vec
        for access in (vec, vec // 2):
            even = [k for k in range(WARP_LANES, 15, -1) if access and (n // access) % k == 0]
            if n % access == 0 and even and n // even[0] < most:
                vec, lanes = access, even[0]
                break
    vectors = n // vec
    per_lane = -(-vectors // lanes)
    full = vectors // lanes  # vectors every lane holds; the rest are guarded
    held_n = per_lane * vec
    thread_stats, warp_merge, units = make_thread_stats(
        vec, per_lane, full, lanes, vectors, dtype, active=lanes
    )

    def _held(k, lane):
        """Whether vector ``k`` of the lane is in the row."""
        if lanes < WARP_LANES:
            return lane < lanes
        return T.Or(k < full, k * WARP_LANES + lane < vectors)

    @T.macro
    def reduce_row(x, row, held, centred, stats):
        lane = T.get_thread_binding() % WARP_LANES
        partner = T.alloc_local((3,), "float32")
        for k in T.unroll(per_lane):
            if lanes == WARP_LANES and vectors % WARP_LANES == 0:
                load16(held, k * vec, x, (row, (k * WARP_LANES + lane) * vec))
            elif _held(k, lane):
                for c in T.vectorized(vec):
                    held[k * vec + c] = x[row, (k * lanes + lane) * vec + c]
            else:
                for c in T.vectorized(vec):
                    held[k * vec + c] = T.cast(0.0, dtype)
        for i in T.unroll(held_n):
            centred[i] = T.cast(held[i], "float32")
        # stats[4:8] hold the warp's scale choice for the write.
        thread_stats(centred, lane, stats, stats, 4)
        warp_merge(centred, lane, stats, partner, stats, 4)
        stats[1] = stats[1] * units.inverse(stats, 4)
        stats[2] = stats[2] * units.pick(stats, 4, 1.0 / (n * units.square_unit**2), 1.0 / n)
        stats[3] = T.rsqrt(stats[2] + eps)

    @T.macro
    def write_row(y, row, store, centred, stats, scale, shift):
        lane = T.get_thread_binding() % WARP_LANES
        out = T.alloc_local((held_n,), dtype)
        # The held differences carry the factor ``unit``, undone in rstd.
        rstd = T.alloc_local((3,), "float32")
        rstd[0] = stats[3] * units.inverse(stats, 4)
        rstd[1] = -stats[1] * stats[3]
        if affine:
            # rstd folds in the weight where the product stays in fp32's range; else a second pass.
            if not units.scaled:
                rstd[1] = shift[0] + rstd[1] * scale[0]
                rstd[0] = rstd[0] * scale[0]
                for i in T.unroll(held_n):
                    centred[i] = centred[i] * rstd[0] + rstd[1]
            else:
                rstd[2] = T.if_then_else(T.abs(rstd[0] * scale[0]) <= _FMAX, 1.0, 0.0)
                rstd[1] = T.if_then_else(rstd[2] > 0, shift[0] + rstd[1] * scale[0], rstd[1])
                rstd[0] = T.if_then_else(rstd[2] > 0, rstd[0] * scale[0], rstd[0])
                for i in T.unroll(held_n):
                    centred[i] = centred[i] * rstd[0] + rstd[1]
                if rstd[2] == 0:
                    for i in T.unroll(held_n):
                        centred[i] = centred[i] * scale[0] + shift[0]
            for i in T.unroll(held_n):
                out[i] = T.cast(centred[i], dtype)
        else:
            for i in T.unroll(held_n):
                out[i] = T.cast(centred[i] * rstd[0] + rstd[1], dtype)
        if store:
            for k in T.unroll(per_lane):
                if _held(k, lane):
                    for c in T.vectorized(vec):
                        y[row, (k * lanes + lane) * vec + c] = out[k * vec + c]

    return held_n, reduce_row, write_row


def _warp_rows(M, D, eps, dtype, vec, threads, num_groups, affine):
    """Up to ``threads / 32`` rows a block, one a warp; see `make_warp_row_norm`.

    A warp past the last row normalizes the last row and stores nothing.

    Args:
        affine: Whether row ``m`` is scaled and shifted by channel ``m % num_groups``.
    """
    # Fewer rows a block until the grid holds 128 blocks: on H200 at 1024 columns, 256
    # rows took 2.240 us in float32 at eight rows a block against 2.016 at two, while
    # 1024 rows led at eight.
    min_blocks = 128
    warps = threads // WARP_LANES
    while warps > 1 and -(-M // warps) < min_blocks:
        warps //= 2
    threads = warps * WARP_LANES
    held_n, reduce_row, write_row = make_warp_row_norm(D, dtype, eps, vec, affine)

    if affine:

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            weight: T.Tensor[(num_groups,), dtype],
            bias: T.Tensor[(num_groups,), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            with T.Kernel(T.ceildiv(M, warps), threads=threads) as bx:
                m = bx * warps + T.get_thread_binding() // WARP_LANES
                scale = T.alloc_local((1,), "float32")
                shift = T.alloc_local((1,), "float32")
                held = T.alloc_local((held_n,), dtype)
                centred = T.alloc_local((held_n,), "float32")
                stats = T.alloc_local((8,), "float32")
                # Ahead of the row load, so the two reads are in flight while it runs.
                scale[0] = T.cast(weight[T.min(m, M - 1) % num_groups], "float32")
                shift[0] = T.cast(bias[T.min(m, M - 1) % num_groups], "float32")
                reduce_row(x, T.min(m, M - 1), held, centred, stats)
                write_row(y, T.min(m, M - 1), m < M, centred, stats, scale, shift)

        return main

    @T.prim_func
    def main_no_affine(
        x: T.Tensor[(M, D), dtype],
        y: T.Tensor[(M, D), dtype],
    ):
        with T.Kernel(T.ceildiv(M, warps), threads=threads) as bx:
            m = bx * warps + T.get_thread_binding() // WARP_LANES
            held = T.alloc_local((held_n,), dtype)
            centred = T.alloc_local((held_n,), "float32")
            stats = T.alloc_local((8,), "float32")
            reduce_row(x, T.min(m, M - 1), held, centred, stats)
            write_row(y, T.min(m, M - 1), m < M, centred, stats, None, None)

    return main_no_affine


def _held_rows(M, D, eps, dtype, vec, threads, num_groups, channels_per_group, affine):
    """One row a block held in registers: thread ``t`` holds vectors ``k * threads + t``; each
    warp takes its statistics as `make_warp_row_norm` does, and the warps merge by Chan's update."""
    vectors = D // vec
    per_thread = -(-vectors // threads)
    full = vectors // threads
    warps = threads // WARP_LANES
    spatial_size = D // channels_per_group
    C = num_groups * channels_per_group
    # One channel owns every column of an access exactly when channels start on one.
    access_in_channel = spatial_size % vec == 0
    # One channel owns the whole row exactly when a group holds one channel.
    row_constant_affine = affine and channels_per_group == 1

    boundary = vectors - full * threads
    thread_stats, warp_merge, units = make_thread_stats(
        vec, per_thread, full, threads, vectors, dtype, span=D
    )

    def warp_count(w):
        """The elements warp *w* (a Python int) holds."""
        return vec * (WARP_LANES * full + min(max(boundary - w * WARP_LANES, 0), WARP_LANES))

    # The warps merge about a base near the row's mean, so no term rounds at a far warp's
    # distance.
    holding = [w for w in range(warps) if warp_count(w)]

    def row_base(slots):
        """Warp 0's ref plus the unweighted mean of the warps' means less it."""
        return slots[0, 0] + (1.0 / len(holding)) * sum(
            (slots[0, w] - slots[0, 0]) + slots[1, w] for w in holding
        )

    def row_centre(slots, base):
        """The row's mean less *base*."""
        return (1.0 / D) * sum(
            ((slots[0, w] - base) + slots[1, w]) * float(warp_count(w)) for w in holding
        )

    def row_scale(slots):
        """The smallest of the warps' thread scales: below one where any warp scales."""
        terms = [slots[3, w] for w in holding]
        while len(terms) > 1:
            pairs = [T.min(a, b) for a, b in zip(terms[::2], terms[1::2], strict=False)]
            terms = pairs + terms[len(terms) // 2 * 2 :]
        return terms[0]

    def row_square_sum(slots, base, centre, scaled):
        """The row's square sum about its mean (Chan's merge), in the scaling warps' unit where
        *scaled*."""
        if not units.scaled:
            return sum(
                slots[2, w]
                + float(warp_count(w))
                * ((slots[0, w] - base) + slots[1, w] - centre)
                * ((slots[0, w] - base) + slots[1, w] - centre)
                for w in holding
            )
        unit = units.square_unit
        return sum(
            slots[2, w] * T.if_then_else(T.And(scaled < 1.0, slots[3, w] >= 1.0), unit * unit, 1.0)
            + float(warp_count(w))
            * (
                ((slots[0, w] - base) + slots[1, w] - centre)
                * T.if_then_else(scaled < 1.0, unit, 1.0)
            )
            * (
                ((slots[0, w] - base) + slots[1, w] - centre)
                * T.if_then_else(scaled < 1.0, unit, 1.0)
            )
            for w in holding
        )

    @T.macro
    def write(y, row, tx, held, centred, merged, ref, weight, bias, scale, weight_after):
        for k in T.unroll(per_thread):
            col = (k * threads + tx) * vec
            if k < full or k * threads + tx < vectors:
                for i in T.unroll(vec):
                    normed = centred[k * vec + i] * merged[1] + ref[0]
                    if weight_after:
                        normed = normed * scale[0] + scale[1]
                    elif affine and not row_constant_affine:
                        c = (row % num_groups) * channels_per_group + (
                            col // spatial_size if access_in_channel else (col + i) // spatial_size
                        )
                        normed = normed * T.cast(weight[c], "float32") + T.cast(bias[c], "float32")
                    held[k * vec + i] = T.cast(normed, dtype)
                for i in T.vectorized(vec):
                    y[row, col + i] = held[k * vec + i]

    @T.macro
    def body(x, weight, bias, y):
        with T.Kernel(M, threads=threads) as row:
            tx = T.get_thread_binding()
            held = T.alloc_local([per_thread * vec], dtype)
            centred = T.alloc_local([per_thread * vec], "float32")
            stats = T.alloc_local([4], "float32")
            partner = T.alloc_local([3], "float32")
            ref = T.alloc_local([1], "float32")
            merged = T.alloc_local([2], "float32")
            base = T.alloc_local([2], "float32")
            factors = T.alloc_local([4], "float32")
            slots = T.alloc_shared([4, warps], "float32")
            if row_constant_affine:
                # Ahead of the row load, so the two reads are in flight while it runs.
                scale = T.alloc_local([2], "float32")
                scale[0] = T.cast(weight[row % num_groups], "float32")
                scale[1] = T.cast(bias[row % num_groups], "float32")
            for k in T.unroll(per_thread):
                if k < full or k * threads + tx < vectors:
                    for i in T.vectorized(vec):
                        held[k * vec + i] = x[row, (k * threads + tx) * vec + i]
            for i in T.unroll(per_thread * vec):
                centred[i] = T.cast(held[i], "float32")
            thread_stats(centred, tx, stats, factors)
            warp_merge(centred, tx, stats, partner, factors)
            if tx % WARP_LANES == 0:
                slots[0, tx // WARP_LANES] = stats[0]
                slots[1, tx // WARP_LANES] = stats[1] * units.inverse(factors)
                slots[2, tx // WARP_LANES] = stats[2]
                if units.scaled:
                    slots[3, tx // WARP_LANES] = factors[0]
            T.sync_threads()
            ref[0] = stats[0]

            base[0] = row_base(slots)
            if units.scaled:
                base[1] = row_scale(slots)
            merged[0] = row_centre(slots, base[0])
            merged[1] = row_square_sum(slots, base[0], merged[0], base[1])
            merged[1] = T.rsqrt(
                merged[1]
                * (
                    T.if_then_else(base[1] < 1.0, 1.0 / (D * units.square_unit**2), 1.0 / D)
                    if units.scaled
                    else 1.0 / D
                )
                + eps
            )
            ref[0] = -((base[0] - ref[0]) + merged[0]) * merged[1]
            merged[1] = merged[1] * units.inverse(factors)
            if row_constant_affine:
                if not units.scaled or T.abs(merged[1] * scale[0]) <= _FMAX:
                    merged[1] = merged[1] * scale[0]
                    ref[0] = ref[0] * scale[0] + scale[1]
                    write(y, row, tx, held, centred, merged, ref, weight, bias, scale, False)
                else:
                    write(y, row, tx, held, centred, merged, ref, weight, bias, scale, True)
            else:
                write(y, row, tx, held, centred, merged, ref, weight, bias, None, False)

    if affine:

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            weight: T.Tensor[(C,), dtype],
            bias: T.Tensor[(C,), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            body(x, weight, bias, y)

        return main

    @T.prim_func
    def main_no_affine(
        x: T.Tensor[(M, D), dtype],
        y: T.Tensor[(M, D), dtype],
    ):
        body(x, None, None, y)

    return main_no_affine


@target_cache(maxsize=32)
def _group_norm_kernel(M, D, eps, dtype, num_groups, channels_per_group, warp_vec, held_vec):
    """Build a row-wise normalization kernel with a per-channel affine.

    This is the core computation shared by GroupNorm and InstanceNorm. The
    caller is responsible for reshaping the input into (M, D); weight and
    bias stay in their natural per-channel (C,) layout and are gathered by
    the channel each element belongs to.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
        num_groups: Number of groups G.
        channels_per_group: C / G. Row ``m`` covers channels
            ``(m % G) * channels_per_group`` onwards.
        warp_vec: Vector width of the warp-a-row body, or ``None``. Only a group of one
            channel takes it.
        held_vec: Vector width of the block-a-row body, or ``None`` for the staged body.
    """
    D_padded = row_padding(D, 4 if dtype == "float32" else 2)
    spatial_size = D // channels_per_group
    C = num_groups * channels_per_group

    @tilelang.jit(out_idx=[3], compile_flags=csrc_include("streaming_load.h"))
    def _func(block_m, threads):
        if warp_vec is not None:
            return _warp_rows(M, D, eps, dtype, warp_vec, threads, num_groups, True)
        if held_vec is not None:
            return _held_rows(
                M, D, eps, dtype, held_vec, threads, num_groups, channels_per_group, True
            )

        # A non-aligned D would read and write columns >= D unless masked.
        def _channel_of(row, col, num_groups: int, channels_per_group: int, spatial_size: int):
            """Return the channel owning element $[row \\times col]$ of the (M, D) reshape.

            Row ``m`` of the ``(N*G, (C/G)*spatial_size)`` view holds group
            ``m % G``, and column ``d`` holds that group's local channel
            ``d // spatial_size``.

            Args:
                row: Row index into the (M, D) view.
                col: Column index into the (M, D) view.
                num_groups: Number of groups G.
                channels_per_group: C / G.
                spatial_size: Number of spatial elements per channel.

            Returns:
                Index into the length-C weight / bias vectors.
            """
            return (row % num_groups) * channels_per_group + col // spatial_size

        masked = D_padded != D
        # One channel owns the whole row exactly when a group holds one channel.
        row_constant_affine = channels_per_group == 1
        # A tail row block runs past the end unless every index is guarded.
        guarded = masked or M % block_m != 0
        row_reduce = make_corrected_row_reduce(block_m, D, D_padded, eps)
        read_leads = make_read_leads(block_m, M, D)

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            weight: T.Tensor[(C,), dtype],
            bias: T.Tensor[(C,), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                squares = T.alloc_fragment((block_m, D_padded), "float32")
                shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                shift = T.alloc_fragment((block_m,), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                acc_squares = T.alloc_fragment((block_m,), "float32")
                ref = T.alloc_fragment((block_m,), "float32")
                centre = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")
                if row_constant_affine:
                    scale = T.alloc_fragment((block_m,), "float32")
                    bias_row = T.alloc_fragment((block_m,), "float32")

                # Ahead of the row load, so the two reads are in flight while
                # it runs. ``scale`` holds the weight until rstd exists.
                if row_constant_affine:
                    for i in T.Parallel(block_m):
                        c = (pid_m * block_m + i) % num_groups
                        scale[i] = T.cast(weight[c], "float32")
                        bias_row[i] = T.cast(bias[c], "float32")
                read_leads(x, pid_m, shift)

                # A padded wide row is staged through shared memory: the per-element guard
                # costs a register load its vectorized copy.
                for i, j in T.Parallel(block_m, D_padded):
                    shared_buf[i, j] = T.if_then_else(
                        T.And(pid_m * block_m + i < M, j < D),
                        x[pid_m * block_m + i, j],
                        T.cast(0.0, dtype),
                    )
                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                row_reduce(x_f32, squares, shift, acc, acc_squares, ref, centre, rstd)

                if row_constant_affine:
                    # A group of one channel gives the whole row one weight and
                    # one bias, so the two gathers leave the element loop.
                    for i in T.Parallel(block_m):
                        scale[i] = scale[i] * rstd[i]
                        bias_row[i] = bias_row[i] - centre[i] * scale[i]
                    for i, j in T.Parallel(block_m, D_padded):
                        if (not guarded) or T.And(pid_m * block_m + i < M, j < D):
                            y[pid_m * block_m + i, j] = T.cast(
                                x_f32[i, j] * scale[i] + bias_row[i], dtype
                            )
                else:
                    for i, j in T.Parallel(block_m, D_padded):
                        if (not guarded) or T.And(pid_m * block_m + i < M, j < D):
                            c = _channel_of(
                                pid_m * block_m + i,
                                j,
                                num_groups,
                                channels_per_group,
                                spatial_size,
                            )
                            normed = (x_f32[i, j] - centre[i]) * rstd[i]
                            y[pid_m * block_m + i, j] = T.cast(
                                normed * T.cast(weight[c], "float32") + T.cast(bias[c], "float32"),
                                dtype,
                            )

        return main

    return _func


class GroupNormKernel(_RowNormKernel, GroupNormFwdInterface):
    """GroupNorm forward kernel with a per-channel affine.

    Normalizes each group's (C/G, *spatial) slice independently and applies
    ``weight[c]`` / ``bias[c]`` to every element, with *c* derived from the
    element's position in the row. Input is pre-reshaped to (M, D) where
    M = N*G, D = (C/G)*spatial_size; weight and bias keep their (C,) layout.

    InstanceNorm uses this kernel with ``num_groups=C`` and
    ``channels_per_group=1``.

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Single shared buffer reused for input load and output store.

    Args:
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        num_groups: Number of groups G.
        channels_per_group: C / G.
        config: Optional tile config dict.
    """

    @classmethod
    def refusal(cls, call: GroupNormCall) -> "str | None":
        if not call.passes_affine:
            return "requires the affine weight and bias"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: GroupNormCall) -> Entry:
        cpg = call.c // call.num_groups
        identity = (cpg * call.spatial, call.eps, call.dtype, call.num_groups, cpg)
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        D: int,
        eps: float,
        dtype: torch.dtype,
        num_groups: int,
        channels_per_group: int,
        config: Optional[dict] = None,
    ):
        """Build for a row length, dtype and group layout.

        The program for a given row count is compiled on the first launch with that row count.
        """
        self.num_groups = num_groups
        self.channels_per_group = channels_per_group
        super().__init__(D, eps, dtype, config=config)

    def _warp_row_vector(self) -> Optional[int]:
        # The warp-a-row body applies one channel's affine to the whole row.
        return super()._warp_row_vector() if self.channels_per_group == 1 else None

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize ``D``-long rows and apply the per-channel affine.

        Flattening to ``(M, D)`` rows happens here.

        Args:
            x: Input of shape ``(N, C, *spatial)``, contiguous, on a CUDA device.
            weight: Affine scale of shape $[C]$ on the same device.
            bias: Affine shift of shape $[C]$ on the same device.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: An input is not on a CUDA device, or the affine pair is missing.
        """
        self._require_devices(x=x, weight=weight, bias=bias)
        if weight is None or bias is None:
            raise ValueError(
                f"{type(self).__name__} applies a per-channel affine; weight and bias are "
                "required. GroupNormNoAffineKernel serves the affine-free call."
            )

        original_shape = x.shape
        rows = x.reshape(-1, self.D)

        y = self.program(self._compile, rows.shape[0])(rows, weight, bias)
        return y.reshape(original_shape)

    def _compile(self, cfg: dict, m: int):
        """The program for *m* rows."""
        self.kernel = kernel = _group_norm_kernel(
            m,
            self.D,
            self.eps,
            self.dtype_str,
            self.num_groups,
            self.channels_per_group,
            self._warp_row_vector(),
            self._held_row_vector(),
        )
        return kernel(cfg["block_m"], cfg["threads"])


@target_cache(maxsize=32)
def _group_norm_no_affine_kernel(M, D, eps, dtype, warp_vec, held_vec):
    """Build a row-wise normalization kernel for shape (M, D) without affine.

    Same numerics and same boundary handling as `_group_norm_kernel`,
    but omits the trailing weight/bias multiply-add — output is
    ``(x - mean) * rstd``. Used for the no-affine variants of GroupNorm and
    InstanceNorm.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
        warp_vec: Vector width of the warp-a-row body, or ``None``.
        held_vec: Vector width of the block-a-row body, or ``None`` for the staged body.
    """
    D_padded = row_padding(D, 4 if dtype == "float32" else 2)

    @tilelang.jit(out_idx=[1], compile_flags=csrc_include("streaming_load.h"))
    def _func(block_m, threads):
        if warp_vec is not None:
            return _warp_rows(M, D, eps, dtype, warp_vec, threads, 1, False)
        if held_vec is not None:
            return _held_rows(M, D, eps, dtype, held_vec, threads, 1, 1, False)
        # A non-aligned D would read and write columns >= D unless masked.
        masked = D_padded != D
        # A tail row block runs past the end unless every index is guarded.
        guarded = masked or M % block_m != 0
        row_reduce = make_corrected_row_reduce(block_m, D, D_padded, eps)
        read_leads = make_read_leads(block_m, M, D)

        @T.prim_func
        def main(
            x: T.Tensor[(M, D), dtype],
            y: T.Tensor[(M, D), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                squares = T.alloc_fragment((block_m, D_padded), "float32")
                shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                shift = T.alloc_fragment((block_m,), "float32")
                acc = T.alloc_fragment((block_m,), "float32")
                acc_squares = T.alloc_fragment((block_m,), "float32")
                ref = T.alloc_fragment((block_m,), "float32")
                centre = T.alloc_fragment((block_m,), "float32")
                rstd = T.alloc_fragment((block_m,), "float32")
                read_leads(x, pid_m, shift)

                # A padded wide row is staged through shared memory: the per-element guard
                # costs a register load its vectorized copy.
                for i, j in T.Parallel(block_m, D_padded):
                    shared_buf[i, j] = T.if_then_else(
                        T.And(pid_m * block_m + i < M, j < D),
                        x[pid_m * block_m + i, j],
                        T.cast(0.0, dtype),
                    )
                    x_f32[i, j] = T.cast(shared_buf[i, j], "float32")
                row_reduce(x_f32, squares, shift, acc, acc_squares, ref, centre, rstd)

                for i, j in T.Parallel(block_m, D_padded):
                    if (not guarded) or T.And(pid_m * block_m + i < M, j < D):
                        y[pid_m * block_m + i, j] = T.cast(
                            (x_f32[i, j] - centre[i]) * rstd[i], dtype
                        )

        return main

    return _func


class GroupNormNoAffineKernel(_RowNormKernel, GroupNormFwdInterface):
    """GroupNorm forward kernel without affine scale/shift.

    Computes ``y = (x - mean) * rstd`` row-wise for shape $[M \\times D]$ reshaped
    inputs. Shares the build/launch parameters and shared-memory layout of
    `GroupNormKernel`; only the output stage differs (no weight/bias
    multiply-add). Used by the no-affine variants of GroupNorm and
    InstanceNorm.

    Args:
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional tile config dict.
    """

    @classmethod
    def refusal(cls, call: GroupNormCall) -> "str | None":
        if call.passes_affine:
            return "serves a call without the affine weight and bias"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: GroupNormCall) -> Entry:
        identity = (call.c // call.num_groups * call.spatial, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize ``D``-long rows without an affine.

        Flattening to ``(M, D)`` rows happens here.

        Args:
            x: Input of shape ``(N, C, *spatial)``, contiguous, on a CUDA device.
            weight: The op's empty affine slot; this kernel has no affine.
            bias: The op's empty affine slot; this kernel has no affine.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: *x* is not on a CUDA device, or an affine tensor was passed.
        """
        self._require_devices(x=x)
        if weight is not None or bias is not None:
            raise ValueError(
                f"{type(self).__name__} has no affine; GroupNormKernel serves the affine call."
            )

        original_shape = x.shape
        rows = x.reshape(-1, self.D)

        y = self.program(self._compile, rows.shape[0])(rows)
        return y.reshape(original_shape)

    def _compile(self, cfg: dict, m: int):
        """The program for *m* rows."""
        self.kernel = kernel = _group_norm_no_affine_kernel(
            m,
            self.D,
            self.eps,
            self.dtype_str,
            self._warp_row_vector(),
            self._held_row_vector(),
        )
        return kernel(cfg["block_m"], cfg["threads"])
