"""Power-of-two complex-to-complex FFT kernels.

``FFT_PLANS`` maps each served (length, dtype) to a one-CTA plan or a two- or
three-factor four-step plan. Per-slot register code that needs a distinct Python
constant in each slot is written as macro recursion (``_each``), because a traced
``for`` becomes a runtime loop and a runtime index into a register array falls to
local memory.
"""

import dataclasses
import functools
import math
from typing import Any, Callable, Dict, Optional

import tilelang
import tilelang.language as T
import torch

from .call_spec import CallSpec
from .constants import BLOCK_SHARED_BYTES_OPT_IN, MAX_BLOCK_THREADS
from .kernel_base import Kernel

__all__ = ["FFTC2CCall", "FFTC2CDecomposedKernel", "FFTC2COneCTAKernel"]


@dataclasses.dataclass(frozen=True)
class FFTC2CCall(CallSpec):
    """What decides which C2C implementation serves a call.

    The batch extent is not here: every kernel takes it as a symbolic dimension,
    so one compilation serves every batch.
    """

    n: int = 0
    dtype: torch.dtype = torch.complex64


@dataclasses.dataclass(frozen=True)
class FFTPlan:
    """The launch-order factors, layouts, builders, and architectures for one call."""

    factors: tuple
    radix: tuple
    tile: tuple
    pad: tuple
    twiddle_exp: tuple
    builders: tuple
    # Bytes per real: 4 at complex64, 8 at complex128.
    itemsize: int
    # The architectures whose block shared-memory limit the default config fits.
    archs: tuple = ()

    @property
    def decomposed(self) -> bool:
        """Whether this plan takes more than one kernel launch."""
        return len(self.factors) > 1

    @property
    def fixed_pad(self) -> bool:
        """Whether this one-CTA plan sizes its shared memory from the length alone."""
        return self.pad[0] == (0, 0)

    def geometry(self, index: int) -> tuple:
        """``(nf, lanes, extent, twrows, r_last)`` of four-step kernel *index*.

        nf is the factor it transforms and ``lanes = nf // 16`` the threads one such
        transform takes; extent is the grid axis a tile divides; twrows is the
        height of its four-step table, zero for the row kernel; r_last is the radix
        of its final pass.
        """
        nf = self.factors[index]
        column = index < len(self.factors) - 1
        extent = math.prod(self.factors[index + 1 :]) if column else self.factors[0]
        twrows = len(self.twiddle_exp[index]) if column else 0
        return nf, nf // 16, extent, twrows, self.radix[index][-1]

    def four_step_smem_bytes(self, index: int, tw: int, pad: tuple) -> int:
        """Shared memory four-step kernel *index* takes at width *tw* and strides *pad*.

        Its two interleaved value arrays, plus the staged pass tables and the tw
        columns of the four-step table it reads.
        """
        nf, _lanes, _extent, twrows, r_last = self.geometry(index)
        staged = 2 * (nf // 16) + (2 * r_last if _factor_passes(nf) == 3 else 0) + 2 * twrows * tw
        return (2 * _four_step_smem(nf, tw, pad[0], pad[-1]) + staged) * self.itemsize

    def one_cta_smem_bytes(self, row: int, grp: int) -> int:
        """Shared memory the one-CTA kernel takes at strides (row, grp).

        Only the three-pass kernels read (row, grp); the others size their arrays
        from the length, and these figures mirror their builders.
        """
        n = self.factors[0]
        if n <= 32:
            reals = 0
        elif n <= 128:
            lanes = n // 8
            reals = 2 * (64 // lanes) * 8 * (lanes + 1) + 2 * lanes
        elif n == 256:
            reals = 2 * 544
        elif n == 512:
            reals = 2 * 576 * (2 if self.itemsize == 4 else 1) + 2 * 72
        elif len(self.radix[0]) == 4:
            row1, row2 = _four_pass_rows(n)
            reals = 2 * max(16 * row1, 256 * row2) + 2 * (n // 256 + n // 4096)
        else:
            reals = 2 * max(16 * row, 256 * grp) + 2 * (n // 256)
        return reals * self.itemsize

    @property
    def default_smem_bytes(self) -> int:
        """The shared memory the widest kernel of the plan takes at its default config."""
        if not self.decomposed:
            return self.one_cta_smem_bytes(*self.pad[0])
        return max(
            self.four_step_smem_bytes(i, self.tile[i], self.pad[i])
            for i in range(len(self.factors))
        )

    @property
    def smem_cap(self) -> int:
        """The shared memory a config may take and still run on every listed architecture."""
        return min(BLOCK_SHARED_BYTES_OPT_IN[arch] for arch in self.archs)


def _perm(k, r: int):
    """Where output *k* of a radix-*r* pass sits among the register slots.

    A radix-r DFT written as two stages leaves output k at
    ``(r//4)*(k % 4) + k//4`` for r in (4, 8, 16); a radix-2 pass is direct and
    leaves natural order. *k* may be a Python int or a traced loop variable and
    the expression is the same either way, which is what lets the passes that
    write their outputs back run as one loop rather than one statement per slot.
    """
    return k if r == 2 else (r // 4) * (k % 4) + k // 4


def _root(e: int, size: int) -> tuple:
    """``W_size^e`` as a (cos, sin) pair of Python floats, axis roots exact.

    Spelling the axis roots exactly avoids needless FMAs and keeps n = 4 bitwise
    aligned with a float64 reference at the same point.
    """
    angle = -2.0 * math.pi * e / size
    out = []
    for value in (math.cos(angle), math.sin(angle)):
        if abs(value) < 1e-15:
            value = 0.0
        elif abs(abs(value) - 1.0) < 1e-15:
            value = math.copysign(1.0, value)
        out.append(value)
    return tuple(out)


def _bit_reverse(position, bits: int):
    """*position* with its *bits* low bits reversed, as a straight-line expression."""
    total = ((position // 1) % 2) * (1 << (bits - 1))
    for bit in range(1, bits):
        total = total + ((position // (1 << bit)) % 2) * (1 << (bits - 1 - bit))
    return total


def _factor_passes(nf: int) -> int:
    """In-CTA passes one four-step factor runs: one exchange up to 256, two above."""
    return 2 if nf <= 256 else 3


def _factor_radix(nf: int) -> int:
    """The radix of one four-step factor's last pass.

    It comes from the length rather than from the single-CTA plan of that length:
    a factor is not obliged to use it. 512 ships as (8, 8, 8) and runs here as
    (16, 16, 2), which is what ``nf // 256`` says.
    """
    return nf // 16 if nf <= 256 else nf // 256


def _lane_split(nf: int) -> int:
    """s for ``lane = lane % s + s * (lane // s)``, the split of the twiddle base.

    The largest power of two with ``s*s <= lanes``, so the table spends
    ``s + lanes // s`` rows -- its minimum -- on the one index of k_b that is not
    a compile-time constant. lanes = 64 gives 8.
    """
    lanes = nf // 16
    s = 1
    while s * s * 4 <= lanes:
        s *= 2
    return s


@functools.lru_cache(maxsize=32)
def _twiddle_exps(nf: int) -> tuple:
    """The j_a multipliers of one column kernel's four-step twiddle table, one per row.

    For ``k_b = lane + lanes*u``, the factor is a per-lane base times the two
    compile-time digits of ``u``. The returned rows contain those three terms.
    """
    lanes = nf // 16
    s = _lane_split(nf)
    rows = list(range(s)) + [s * h for h in range(lanes // s)]
    rows += [lanes * p for p in (1, 2, 3)]
    rows += [4 * lanes * q for q in (1, 2, 3)]
    return tuple(rows)


def _four_step_smem(nf: int, tw: int, row: int, grp: int) -> int:
    """The reals one four-step kernel's two interleaved value arrays span."""
    stride = 16 * row if _factor_passes(nf) == 2 else max(16 * row, 16 * 16 * grp)
    return stride * tw


def _four_pass_rows(n: int) -> tuple:
    """The (row1, row2) strides of a four-pass kernel: the _smem_pad row rule per exchange.

    The final combine is a shuffle, so there is no third.
    """
    return n // 16 + (n // 256 - n // 16) % 32, n // 256 + (n // 4096 - n // 256) % 32


def _smem_pad(n: int, radix: tuple) -> tuple:
    """The (row, grp) shared-memory strides that make both exchanges conflict-free.

    row is the S1 row stride. Pass 2 reads S1[k1][m1*r3 + m2] with k1 = tx // r3,
    so inside a warp the k1 groups sit row apart and the r3 lanes of one group are
    adjacent; the banks spread exactly when row % 32 == r3.

    grp is the S2 group stride. Pass 3 reads S2[k1'][k1][m2] with k1 = tx % 16, so
    16 lanes sit grp apart; an odd grp is coprime with 32 and covers 16 banks, and
    the next k1' block then starts 16*grp % 32 == 16 banks along.

    (0, 0) for the plans that size their shared memory from the length.
    """
    if n < 1024 or len(radix) == 4:
        return 0, 0
    threads = n // 16
    r3 = radix[2]
    return threads + (r3 - threads) % 32, r3 + 1 - r3 % 2


@T.macro
def _radix4(reg, off: int, step: int):
    """Radix-4 butterfly in place over the four slots off, off+step, ..., off+3*step.

    Decimation in frequency, so the W_4^1 = -i on the second output folds into the
    add and no multiply is needed; outputs stay in natural order.
    """
    t0r = reg[off, 0] + reg[off + 2 * step, 0]
    t0i = reg[off, 1] + reg[off + 2 * step, 1]
    t1r = reg[off + step, 0] + reg[off + 3 * step, 0]
    t1i = reg[off + step, 1] + reg[off + 3 * step, 1]
    t2r = reg[off, 0] - reg[off + 2 * step, 0]
    t2i = reg[off, 1] - reg[off + 2 * step, 1]
    t3r = reg[off + step, 0] - reg[off + 3 * step, 0]
    t3i = reg[off + step, 1] - reg[off + 3 * step, 1]
    reg[off, 0] = t0r + t1r
    reg[off, 1] = t0i + t1i
    reg[off + step, 0] = t2r + t3i
    reg[off + step, 1] = t2i - t3r
    reg[off + 2 * step, 0] = t0r - t1r
    reg[off + 2 * step, 1] = t0i - t1i
    reg[off + 3 * step, 0] = t2r - t3i
    reg[off + 3 * step, 1] = t2i + t3r


@T.macro
def _radix2(reg, off: int, step: int):
    """Radix-2 butterfly in place over the two slots off and off+step."""
    u0r = reg[off, 0] + reg[off + step, 0]
    u0i = reg[off, 1] + reg[off + step, 1]
    u1r = reg[off, 0] - reg[off + step, 0]
    u1i = reg[off, 1] - reg[off + step, 1]
    reg[off, 0] = u0r
    reg[off, 1] = u0i
    reg[off + step, 0] = u1r
    reg[off + step, 1] = u1i


@T.macro
def _dft_pass1(reg, off: int, span: int, j: int):
    """Stage 1: one radix-4 butterfly over each of the *span* stride-*span* groups."""
    if j < span:
        _radix4(reg, off + j, span)
        _dft_pass1(reg, off, span, j + 1)


@T.macro
def _twiddle_slot(reg, slot: int, e: int, size: int):
    """Scale one slot by the constant W_size**e in place.

    A macro of its own so each recursion step binds its temporaries once. The -i
    axis root is a swap and a negate: one fewer rounding step.
    """
    wr, wi = _root(e, size)
    if wr == 0.0 and wi == -1.0:
        sw = reg[slot, 0]
        reg[slot, 0] = reg[slot, 1]
        reg[slot, 1] = -sw
    else:
        cr = reg[slot, 0] * wr - reg[slot, 1] * wi
        ci = reg[slot, 0] * wi + reg[slot, 1] * wr
        reg[slot, 0] = cr
        reg[slot, 1] = ci


@T.macro
def _dft_twiddles(reg, off: int, r: int, k: int):
    """The inter-stage twiddles W_r^(b*c), with b = k % (r//4) and c = k // (r//4)."""
    if k < r:
        if (k % (r // 4)) * (k // (r // 4)):
            _twiddle_slot(reg, off + k, (k % (r // 4)) * (k // (r // 4)), r)
        _dft_twiddles(reg, off, r, k + 1)


@T.macro
def _dft_pass2(reg, off: int, span: int, g: int):
    """Stage 2: one radix-*span* butterfly within each of the four groups of *span*."""
    if g < 4:
        if span == 4:
            _radix4(reg, off + g * span, 1)
        else:
            _radix2(reg, off + g * span, 1)
        _dft_pass2(reg, off, span, g + 1)


@T.macro
def _dft(reg, off: int, r: int):
    """*r*-point DFT in place on the *r* slots from *off*.

    2 and 4 are one butterfly and leave natural order. 8 and 16 are two stages --
    radix-4 over the stride-(r//4) groups, the inter-stage twiddles, then
    radix-(r//4) within each group of four -- which leaves output k at slot
    ``(r//4)*(k % 4) + k//4``. Nothing moves it back; the store index maps absorb
    the permutation, and ``_perm`` is where it is written down.
    """
    if r == 2:
        _radix2(reg, off, 1)
    elif r == 4:
        _radix4(reg, off, 1)
    else:
        _dft_pass1(reg, off, r // 4, 0)
        _dft_twiddles(reg, off, r, 1)
        _dft_pass2(reg, off, r // 4, 0)


@T.macro
def _twiddle_apply(reg, cw, slot: int, w1r, w1i, advance: bool):
    """Scale one slot by the running power of w1 in cw, advancing it first if asked.

    A macro of its own so each recursion step binds its temporaries once.
    """
    if advance:
        nr = cw[0] * w1r - cw[1] * w1i
        ni = cw[0] * w1i + cw[1] * w1r
        cw[0] = nr
        cw[1] = ni
    ar = reg[slot, 0] * cw[0] - reg[slot, 1] * cw[1]
    ai = reg[slot, 0] * cw[1] + reg[slot, 1] * cw[0]
    reg[slot, 0] = ar
    reg[slot, 1] = ai


@T.macro
def _twiddle_step(reg, cw, off: int, r: int, w1r, w1i, k: int):
    """Slot ``off + _perm(k, r)`` onward, each one power of w1 past the last."""
    if k < r:
        _twiddle_apply(reg, cw, off + _perm(k, r), w1r, w1i, k > 1)
        _twiddle_step(reg, cw, off, r, w1r, w1i, k + 1)


@T.macro
def _twiddle_rotor(reg, cw, off: int, r: int, w1r, w1i):
    """Multiply the permuted radix-r result by successive powers of ``w1``."""
    cw[0] = w1r
    cw[1] = w1i
    _twiddle_step(reg, cw, off, r, w1r, w1i, 1)


@T.macro
def _each(op, args, k: int, n: int):
    """Apply a macro recursively where each slot needs a distinct Python constant."""
    if k < n:
        op(*args, k)
        _each(op, args, k + 1, n)


@T.macro
def _read_pairs(x_pair, reg, bb, tx, stride, n: int):
    """Read *n* inputs *stride* apart as vectorized complex pairs."""
    for k in T.unroll(n):
        for v in T.vectorized(2):
            reg[k, v] = x_pair[bb, k * stride + tx, v]


@T.macro
def _read_scratch(s_re, s_im, reg, base, step, n: int):
    """Read *n* values down one scratch row: slot k is s[base + k*step]."""
    for k in T.unroll(n):
        reg[k, 0] = s_re[base + k * step]
        reg[k, 1] = s_im[base + k * step]


@T.macro
def _read_lane(s_re, s_im, reg, base, lane, step: int, n: int):
    """Read the run of *n* one lane owns after a transpose: s[base + lane*step + k]."""
    for k in T.unroll(n):
        reg[k, 0] = s_re[base + lane * step + k]
        reg[k, 1] = s_im[base + lane * step + k]


@T.macro
def _write_perm16(s_re, s_im, reg, base, step):
    """Write the 16 outputs of a pass to base + k*step, undoing the permutation.

    Serves both scratch stages: S1[k1][tx] with base = tx and step = row, and
    S2[k1'][k1][m2] with base = k1*grp + m2 and step = 16*grp.
    """
    for k in T.unroll(16):
        s_re[k * step + base] = reg[_perm(k, 16), 0]
        s_im[k * step + base] = reg[_perm(k, 16), 1]


@T.macro
def _write_lane(s_re, s_im, reg, base, step: int, lane, r: int, n: int):
    """The n outputs of a radix-r pass, transposed into scratch at base + k*step + lane."""
    for k in T.unroll(n):
        s_re[base + k * step + lane] = reg[_perm(k, r), 0]
        s_im[base + k * step + lane] = reg[_perm(k, r), 1]


@T.macro
def _write_out(y_pair, reg, st, bb, lane, span: int, r: int, n: int):
    """The n outputs of a radix-r last pass, to y[bb, lane + span*k] in order."""
    for k in T.unroll(n):
        st[0] = reg[_perm(k, r), 0]
        st[1] = reg[_perm(k, r), 1]
        for v in T.vectorized(2):
            y_pair[bb, lane + span * k, v] = st[v]


@T.macro
def _gather16(s_re, s_im, reg, base, gstep, tw: int, r: int):
    """Read the 16 values of a last pass out of shared memory.

    Slot ``g*r + m`` is group g's m'th value. Every shared index is scaled by
    ``tw`` because a four-step CTA interleaves ``tw`` transforms as
    ``s[idx*tw + col]``; a single-CTA plan stores one per column and passes 1.
    """
    for k in T.unroll(16):
        reg[k, 0] = s_re[base + (k // r) * gstep + (k % r) * tw]
        reg[k, 1] = s_im[base + (k // r) * gstep + (k % r) * tw]


@T.macro
def _dft_group(reg, r: int, g: int):
    """Group g onward of a last pass's 16 // r independent radix-r transforms."""
    if g * r < 16:
        _dft(reg, g * r, r)
        _dft_group(reg, r, g + 1)


@T.macro
def _pass4_gathers(vals, reg, tx, m3, r: int, k: int, j: int):
    """Lane j onward of the r values a k2-group needs, each out of its own lane."""
    if j < r:
        vals[j, 0] = T.tvm_warp_shuffle(
            T.uint32(0xFFFFFFFF), reg[_perm(k, 16), 0], tx - m3 + j, r, 32
        )
        vals[j, 1] = T.tvm_warp_shuffle(
            T.uint32(0xFFFFFFFF), reg[_perm(k, 16), 1], tx - m3 + j, r, 32
        )
        _pass4_gathers(vals, reg, tx, m3, r, k, j + 1)


@T.macro
def _pass4_pick(st, vals, m3, r: int, j: int):
    """Slot j onward of the runtime pick; m3 is a scalar, so this is an if chain."""
    if j < r:
        if m3 == j:
            st[0] = vals[j, 0]
            st[1] = vals[j, 1]
        _pass4_pick(st, vals, m3, r, j + 1)


@T.macro
def _pass4_shuffle_one(y_pair, reg, vals, st, bb, tx, k2, m3, r: int, k: int):
    """Gather and combine one final-pass output without a dynamic register index."""
    _pass4_gathers(vals, reg, tx, m3, r, k, 0)
    _dft(vals, 0, r)
    st[0] = vals[0, 0]
    st[1] = vals[0, 1]
    _pass4_pick(st, vals, m3, r, 1)
    idx = k2 + 256 * k + 4096 * m3
    for v in T.vectorized(2):
        y_pair[bb, idx, v] = st[v]


def _fs_group(pw: int, qw: int, r: int) -> int:
    """Which shared-memory group the u = p + 4q'th output of a radix-r pass sits in."""
    return (pw + 4 * qw) % (16 // r)


def _fs_digit(pw: int, qw: int, r: int) -> int:
    """Which output digit of that group the u = p + 4q'th output is."""
    return (pw + 4 * qw) // (16 // r)


def _fs_slot(pw: int, qw: int, r: int) -> int:
    """Which register slot holds the u = p + 4q'th output of a radix-r last pass."""
    return _fs_group(pw, qw, r) * r + _perm(_fs_digit(pw, qw, r), r)


@T.macro
def _fs_power(dst, src, tmp, s_tw, col, base: int, e: int):
    """dst = src times the e'th tabulated power of one half of u, read whole.

    Exponent 0 is a copy: row base - 1 does not exist, the table starts at 1.
    """
    if e == 0:
        dst[0] = src[0]
        dst[1] = src[1]
    else:
        for v in T.vectorized(2):
            tmp[v] = s_tw[base + e - 1, col, v]
        dst[0] = src[0] * tmp[0] - src[1] * tmp[1]
        dst[1] = src[0] * tmp[1] + src[1] * tmp[0]


@T.macro
def _fs_store(out_pair, st, bb, out0, ogstep, okstep, r: int, pw: int, qw: int):
    """The corner-turned store both last passes share: out0 + g*ogstep + k*okstep."""
    for v in T.vectorized(2):
        out_pair[bb, out0 + _fs_group(pw, qw, r) * ogstep + _fs_digit(pw, qw, r) * okstep, v] = st[
            v
        ]


@T.macro
def _four_step_a_out(
    out_pair,
    s_tw,
    col,
    reg,
    st,
    tmp,
    twb,
    twg,
    twk,
    bb,
    out0,
    ogstep,
    okstep,
    r: int,
    pbase: int,
    i: int,
):
    """Output u = p + 4q of kernel A's last pass, p = i // 4 and q = i % 4.

    Its twiddle is ``twb * P_p * Q_q``; ``twg = twb * P_p`` is formed once per p.
    """
    pw = i // 4
    qw = i % 4
    if qw == 0:
        _fs_power(twg, twb, tmp, s_tw, col, pbase, pw)
    _fs_power(twk, twg, tmp, s_tw, col, pbase + 3, qw)
    st[0] = reg[_fs_slot(pw, qw, r), 0] * twk[0] - reg[_fs_slot(pw, qw, r), 1] * twk[1]
    st[1] = reg[_fs_slot(pw, qw, r), 0] * twk[1] + reg[_fs_slot(pw, qw, r), 1] * twk[0]
    _fs_store(out_pair, st, bb, out0, ogstep, okstep, r, pw, qw)


@T.macro
def _four_step_b_out(out_pair, reg, st, bb, out0, ogstep, okstep, r: int, i: int):
    """Output u = p + 4q of kernel B's last pass, p = i // 4 and q = i % 4; no twiddle."""
    st[0] = reg[_fs_slot(i // 4, i % 4, r), 0]
    st[1] = reg[_fs_slot(i // 4, i % 4, r), 1]
    _fs_store(out_pair, st, bb, out0, ogstep, okstep, r, i // 4, i % 4)


@T.macro
def _tiny_twiddle(w, index, size: int, e: int):
    """Select a stage twiddle into a buffer that survives the macro boundary."""
    if e < size // 2:
        if index % (size // 2) == e:
            w[0] = _root(e, size)[0]
            w[1] = _root(e, size)[1]
        _tiny_twiddle(w, index, size, e + 1)


@T.macro
def _tiny_butterfly(reg, tmp, pair, diff, lane, size: int, lanes: int, q: int):
    """One DIF butterfly of one register slot, against its partner *size* / 2 away.

    The partner is a shuffle while the pair sits in one lane's neighbours and a
    plain register read once it does not. The twiddle is spelled as literals
    selected by a runtime exponent rather than read from a table: these lengths
    stage none. ``pair`` is dead once ``diff`` is formed, so it carries the
    selected twiddle rather than a buffer of its own.
    """
    if size // 2 < lanes:
        pair[0] = T.shfl_xor(tmp[q, 0], size // 2, width=lanes)
        pair[1] = T.shfl_xor(tmp[q, 1], size // 2, width=lanes)
    else:
        pair[0] = tmp[q ^ (size // 2 // lanes), 0]
        pair[1] = tmp[q ^ (size // 2 // lanes), 1]
    if (q * lanes + lane) % size < size // 2:
        reg[q, 0] = tmp[q, 0] + pair[0]
        reg[q, 1] = tmp[q, 1] + pair[1]
    else:
        diff[0] = pair[0] - tmp[q, 0]
        diff[1] = pair[1] - tmp[q, 1]
        pair[0] = 1.0
        pair[1] = 0.0
        _tiny_twiddle(pair, q * lanes + lane, size, 1)
        reg[q, 0] = diff[0] * pair[0] - diff[1] * pair[1]
        reg[q, 1] = diff[0] * pair[1] + diff[1] * pair[0]


@T.macro
def _tiny_network(reg, tmp, pair, diff, lane, n: int, lanes: int, ept: int):
    """Apply the DIF stages recursively with compile-time register indices.

    Each stage copies the register file aside, then combines it in place.
    """
    if n >= 2:
        for k in T.unroll(ept):
            tmp[k, 0] = reg[k, 0]
            tmp[k, 1] = reg[k, 1]
        _each(_tiny_butterfly, (reg, tmp, pair, diff, lane, n, lanes), 0, ept)
        _tiny_network(reg, tmp, pair, diff, lane, n // 2, lanes, ept)


@T.macro
def _tiny_reverse_one(reg, tmp, io, pair, lane, lanes: int, ept: int, bits: int, q: int):
    """Gather slot q's output from the lane and slot the bit reversal names.

    Both halves are shuffled unconditionally and the right one picked by an
    ``if``, because which slot holds it is a runtime value and a runtime index
    into ``tmp`` would fall to local memory.
    """
    for j in T.unroll(2 * ept):
        io[j] = T.tvm_warp_shuffle(
            T.uint32(0xFFFFFFFF),
            tmp[j // 2, j % 2],
            _bit_reverse(q * lanes + lane, bits) % lanes,
            lanes,
            32,
        )
    pair[0] = io[0]
    pair[1] = io[1]
    # Nested, and not a macro: joining the two tests with `and` would build a
    # runtime T.And instead of folding the ept test away, and handing the slot
    # expression to a macro would let-bind it, which is one IR node more than the
    # comparison needs.
    if ept == 2:  # noqa: SIM102
        if _bit_reverse(q * lanes + lane, bits) // lanes == 1:
            pair[0] = io[2]
            pair[1] = io[3]
    reg[q, 0] = pair[0]
    reg[q, 1] = pair[1]


def _build_tiny(n: int, ept: int, real_dtype: str) -> Any:
    """The warp-resident kernel for n <= 32, *ept* values per thread.

    A whole transform sits in one lane group, so every exchange is a shuffle and
    nothing touches shared memory.
    """
    lanes = n // ept

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3"],
    )
    def _func(row: int, grp: int):
        batch = T.dynamic("batch")

        @T.prim_func
        def main(
            x_pair: T.Tensor((batch, n, 2), real_dtype),
            wlut: T.Tensor((n, 2), real_dtype),
            w2lut: T.Tensor((1, 2), real_dtype),
            y_pair: T.Tensor((batch, n, 2), real_dtype),
        ):
            with T.Kernel(T.ceildiv(batch * n, 256 * ept), threads=256) as bx:
                tx = T.get_thread_binding()
                reg = T.alloc_local((ept, 2), real_dtype)
                tmp = T.alloc_local((ept, 2), real_dtype)
                io = T.alloc_local((ept * 2,), real_dtype)
                pair = T.alloc_local((2,), real_dtype)
                diff = T.alloc_local((2,), real_dtype)

                global_lane = bx * 256 + tx
                bb = global_lane // lanes
                lane = global_lane % lanes
                if bb < batch:
                    for v in T.vectorized(2):
                        reg[0, v] = x_pair[bb, 0 * lanes + lane, v]
                    if ept == 2:
                        for v in T.vectorized(2):
                            reg[1, v] = x_pair[bb, 1 * lanes + lane, v]
                    _tiny_network(reg, tmp, pair, diff, lane, n, lanes, ept)
                    # The final bit reversal is what puts the outputs in order.
                    for k in T.unroll(ept):
                        tmp[k, 0] = reg[k, 0]
                        tmp[k, 1] = reg[k, 1]
                    _each(
                        _tiny_reverse_one,
                        (reg, tmp, io, pair, lane, lanes, ept, n.bit_length() - 1),
                        0,
                        ept,
                    )
                    for v in T.vectorized(2):
                        y_pair[bb, 0 * lanes + lane, v] = reg[0, v]
                    if ept == 2:
                        for v in T.vectorized(2):
                            y_pair[bb, 1 * lanes + lane, v] = reg[1, v]

        return main

    return _func


@T.macro
def _warp8_last_128(y_pair, reg, pair, bb, k1, half, real_dtype: str, q: int):
    """One output pair of n = 128's final radix-2 combine.

    The two register-DFT8 halves sit in the same warp, so this is shuffle only
    and touches no shared memory. ``half`` is a runtime value, so which of the
    two a lane owns is an ``if`` rather than an index.
    """
    pair[0] = T.shfl_xor(reg[_perm(q, 8), 0], 1)
    pair[1] = T.shfl_xor(reg[_perm(q, 8), 1], 1)
    er = T.alloc_var(real_dtype)
    ei = T.alloc_var(real_dtype)
    orr = T.alloc_var(real_dtype)
    oi = T.alloc_var(real_dtype)
    if half == 0:
        er = reg[_perm(q, 8), 0]
        ei = reg[_perm(q, 8), 1]
        orr = pair[0]
        oi = pair[1]
    else:
        er = pair[0]
        ei = pair[1]
        orr = reg[_perm(q, 8), 0]
        oi = reg[_perm(q, 8), 1]
    tr = orr * _root(q, 16)[0] - oi * _root(q, 16)[1]
    ti = orr * _root(q, 16)[1] + oi * _root(q, 16)[0]
    if half == 0:
        pair[0] = er + tr
        pair[1] = ei + ti
    else:
        pair[0] = er - tr
        pair[1] = ei - ti
    for v in T.vectorized(2):
        y_pair[bb, k1 + 8 * q + half * 64, v] = pair[v]


def _build_warp8(n: int, real_dtype: str) -> Any:
    """The register-DFT8 warp-exchange kernel for n = 64 or 128.

    block = 64 threads for both dtypes, so a transform's 8 (n=64) or 16 (n=128)
    lanes always sit inside one warp; the mid-kernel transpose barrier is
    T.sync_warp (the one T.sync_threads is only for loading the primitive table).
    The table holds n/8 primitives, not the full W_n table: W_n^k for lane k is
    exactly wlut[k], so the rest of a lane's twiddles come from _twiddle_rotor.
    """
    lanes = n // 8
    transforms = 64 // lanes
    # The transpose row stride; the one added lane is what clears the bank
    # conflict.
    stride = lanes + 1

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3"],
    )
    def _func(row: int, grp: int):
        batch = T.dynamic("batch")

        @T.prim_func
        def main(
            x_pair: T.Tensor((batch, n, 2), real_dtype),
            wlut: T.Tensor((n, 2), real_dtype),
            w2lut: T.Tensor((1, 2), real_dtype),
            y_pair: T.Tensor((batch, n, 2), real_dtype),
        ):
            with T.Kernel(T.ceildiv(batch, transforms), threads=64) as bx:
                tx = T.get_thread_binding()
                reg = T.alloc_local((8, 2), real_dtype)
                cw = T.alloc_local((2,), real_dtype)
                pair = T.alloc_local((2,), real_dtype)
                s_re = T.alloc_shared((transforms * 8 * stride,), real_dtype)
                s_im = T.alloc_shared((transforms * 8 * stride,), real_dtype)
                s_tw = T.alloc_shared((lanes, 2), real_dtype)

                if tx < lanes:
                    for v in T.vectorized(2):
                        s_tw[tx, v] = wlut[tx, v]
                T.sync_threads()
                group = tx // lanes
                lane = tx % lanes
                bb = bx * transforms + group
                group_base = group * (8 * stride)
                if n == 128:
                    k1 = lane // 2
                    half = lane % 2
                # pass 1: DFT8, twiddle, transpose into shared memory
                if bb < batch:
                    _read_pairs(x_pair, reg, bb, lane, lanes, 8)
                    _dft(reg, 0, 8)
                    w1r = s_tw[lane, 0]
                    w1i = s_tw[lane, 1]
                    _twiddle_rotor(reg, cw, 0, 8, w1r, w1i)
                    _write_lane(s_re, s_im, reg, group_base, stride, lane, 8, 8)
                T.sync_warp()
                # pass 2: read the transpose back and DFT8; n = 128 then combines the
                # two interleaved halves by radix 2.
                if bb < batch:
                    if n == 64:
                        _read_lane(s_re, s_im, reg, group_base, lane, stride, 8)
                        _dft(reg, 0, 8)
                        _write_out(y_pair, reg, pair, bb, lane, 8, 8, 8)
                    else:
                        for k in T.unroll(8):
                            reg[k, 0] = s_re[group_base + k1 * stride + 2 * k + half]
                            reg[k, 1] = s_im[group_base + k1 * stride + 2 * k + half]
                        _dft(reg, 0, 8)
                        _each(_warp8_last_128, (y_pair, reg, pair, bb, k1, half, real_dtype), 0, 8)

        return main

    return _func


def _build_packed_256(real_dtype: str) -> Any:
    """The packed n=256 kernel: two 16x16 transforms per 32-thread CTA.

    The shared row stride of 17 is what the 16-wide transpose needs.
    """

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3"],
    )
    def _func(row: int, grp: int):
        batch = T.dynamic("batch")

        @T.prim_func
        def main(
            x_pair: T.Tensor((batch, 256, 2), real_dtype),
            wlut: T.Tensor((256, 2), real_dtype),
            w2lut: T.Tensor((1, 2), real_dtype),
            y_pair: T.Tensor((batch, 256, 2), real_dtype),
        ):
            with T.Kernel(T.ceildiv(batch, 2), threads=32) as bx:
                tx = T.get_thread_binding()
                reg = T.alloc_local((16, 2), real_dtype)
                st = T.alloc_local((2,), real_dtype)
                cw = T.alloc_local((2,), real_dtype)
                s_re = T.alloc_shared((544,), real_dtype)
                s_im = T.alloc_shared((544,), real_dtype)

                tid = tx // 16
                lane = tx % 16
                bb = bx * 2 + tid
                xbase = tid * 272
                # pass 1: DFT16 down the columns into shared memory
                if bb < batch:
                    _read_pairs(x_pair, reg, bb, lane, 16, 16)
                    _dft(reg, 0, 16)
                    for v in T.vectorized(2):
                        st[v] = wlut[lane, v]
                    _twiddle_rotor(reg, cw, 0, 16, st[0], st[1])
                    _write_lane(s_re, s_im, reg, xbase, 17, lane, 16, 16)
                T.sync_threads()
                # pass 2: DFT16 along the rows, stored in order
                if bb < batch:
                    _read_lane(s_re, s_im, reg, xbase, lane, 17, 16)
                    _dft(reg, 0, 16)
                    _write_out(y_pair, reg, st, bb, lane, 16, 16, 16)

        return main

    return _func


def _build_packed_512(table_limit: int, real_dtype: str) -> Any:
    """The packed n=512 kernel: three radix-8 passes over a 9-wide shared row.

    *table_limit* is how many primitives the block reads straight from the table:
    64 at complex64, whose 128 threads pack two transforms per CTA, and 48 at
    complex128, whose 64 threads pack one.
    """
    pack = 2 if table_limit == 64 else 1

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3"],
    )
    def _func(row: int, grp: int):
        batch = T.dynamic("batch")

        @T.prim_func
        def main(
            x_pair: T.Tensor((batch, 512, 2), real_dtype),
            wlut: T.Tensor((512, 2), real_dtype),
            w2lut: T.Tensor((2, 2), real_dtype),
            y_pair: T.Tensor((batch, 512, 2), real_dtype),
        ):
            with T.Kernel(T.ceildiv(batch, pack), threads=64 * pack) as bx:
                tx = T.get_thread_binding()
                reg = T.alloc_local((16, 2), real_dtype)
                st = T.alloc_local((2,), real_dtype)
                cw = T.alloc_local((2,), real_dtype)
                s_re = T.alloc_shared((576 * pack,), real_dtype)
                s_im = T.alloc_shared((576 * pack,), real_dtype)
                tw_r = T.alloc_shared((72,), real_dtype)
                tw_i = T.alloc_shared((72,), real_dtype)

                # Rows 0..63 of the staged table are W_512^k for the pass-1 lane and
                # 64..71 the stride-8 subsample pass 2 reads. At complex128 the block
                # is 64 threads, so the last 16 rows are turned out of the first 48
                # by one multiplication rather than read again.
                if tx < table_limit:
                    for v in T.vectorized(2):
                        st[v] = wlut[tx, v]
                    tw_r[tx] = st[0]
                    tw_i[tx] = st[1]
                    if tx % 8 == 0:
                        tw_r[64 + tx // 8] = st[0]
                        tw_i[64 + tx // 8] = st[1]
                T.sync_threads()
                if table_limit == 48:
                    if tx >= 48 and tx < 64:
                        st[0] = (
                            tw_r[tx - 48] * _root(48, 512)[0] - tw_i[tx - 48] * _root(48, 512)[1]
                        )
                        st[1] = (
                            tw_r[tx - 48] * _root(48, 512)[1] + tw_i[tx - 48] * _root(48, 512)[0]
                        )
                        tw_r[tx] = st[0]
                        tw_i[tx] = st[1]
                        if tx % 8 == 0:
                            tw_r[64 + tx // 8] = st[0]
                            tw_i[64 + tx // 8] = st[1]
                    T.sync_threads()
                tid = tx // 64
                lane = tx % 64
                bb = bx * pack + tid
                xbase = tid * 576
                # pass 1
                if bb < batch:
                    _read_pairs(x_pair, reg, bb, lane, 64, 8)
                    _dft(reg, 0, 8)
                    _twiddle_rotor(reg, cw, 0, 8, tw_r[lane], tw_i[lane])
                    _write_lane(s_re, s_im, reg, xbase, 72, lane, 8, 8)
                T.sync_threads()
                k1 = lane // 8
                q = lane % 8
                # pass 2: a stride-8 read down one pass-1 row, stored as the 9-wide
                # runs pass 3 reads whole
                if bb < batch:
                    for k in T.unroll(8):
                        reg[k, 0] = s_re[xbase + k1 * 72 + q + k * 8]
                        reg[k, 1] = s_im[xbase + k1 * 72 + q + k * 8]
                    _dft(reg, 0, 8)
                    _twiddle_rotor(reg, cw, 0, 8, tw_r[64 + q], tw_i[64 + q])
                    for k in T.unroll(8):
                        s_re[xbase + k * 72 + k1 * 9 + q] = reg[_perm(k, 8), 0]
                        s_im[xbase + k * 72 + k1 * 9 + q] = reg[_perm(k, 8), 1]
                T.sync_threads()
                # pass 3, stored in natural order
                if bb < batch:
                    base3 = xbase + (lane // 8) * 72 + (lane % 8) * 9
                    _read_scratch(s_re, s_im, reg, base3, 1, 8)
                    _dft(reg, 0, 8)
                    _write_out(y_pair, reg, st, bb, lane, 64, 8, 8)

        return main

    return _func


def _build_three_pass(n: int, real_dtype: str) -> Any:
    """The n = 1024, 2048 or 4096 kernel: three radix-16 passes over two exchanges.

    Its last pass runs 16 // r3 independent radix-r3 transforms, r3 = n // 256.
    """
    r3 = n // 256

    # No out_idx on any builder: with a symbolic batch the wrapper would resolve
    # the output shape on every call, so the caller supplies the output buffer.
    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3"],
    )
    def _func(row: int, grp: int):
        """Build for S1 row stride *row* and S2 group stride *grp*, in reals."""
        smem_floats = max(16 * row, 16 * 16 * grp)
        batch = T.dynamic("batch")

        @T.prim_func
        def main(
            x_pair: T.Tensor((batch, n, 2), real_dtype),
            wlut: T.Tensor((n, 2), real_dtype),
            w2lut: T.Tensor((r3, 2), real_dtype),
            y_pair: T.Tensor((batch, n, 2), real_dtype),
        ):
            with T.Kernel(batch, threads=n // 16) as bb:
                tx = T.get_thread_binding()
                reg = T.alloc_local((16, 2), real_dtype)
                st = T.alloc_local((2,), real_dtype)
                cw = T.alloc_local((2,), real_dtype)
                s_re = T.alloc_shared((smem_floats,), real_dtype)
                s_im = T.alloc_shared((smem_floats,), real_dtype)
                # The pass-2 base twiddles are staged once per block: read straight
                # from the table they are r3 apart, which costs a scattered access
                # in every warp instead of one.
                s_w2 = T.alloc_shared((r3, 2), real_dtype)
                if tx < r3:
                    for v in T.vectorized(2):
                        s_w2[tx, v] = w2lut[tx, v]
                T.sync_threads()

                # pass 1: read x[n1*(n/16) + tx], DFT16 over n1, twiddle W_n^(tx*k1), to S1[k1][tx]
                _read_pairs(x_pair, reg, bb, tx, n // 16, 16)
                _dft(reg, 0, 16)
                # Read the interleaved table as a pair to keep it one transaction.
                for v in T.vectorized(2):
                    st[v] = wlut[tx, v]
                _twiddle_rotor(reg, cw, 0, 16, st[0], st[1])
                _write_perm16(s_re, s_im, reg, tx, row)
                T.sync_threads()

                # pass 2: k1 = tx // r3, m2 = tx % r3; DFT16 over m1; twiddle W_(n/16)^(m2*k1')
                k1 = tx // r3
                m2 = tx % r3
                _read_scratch(s_re, s_im, reg, k1 * row + m2, r3, 16)
                T.sync_threads()  # every read of S1 is done before S2 overwrites it
                _dft(reg, 0, 16)
                _twiddle_rotor(reg, cw, 0, 16, s_w2[m2, 0], s_w2[m2, 1])
                _write_perm16(s_re, s_im, reg, k1 * grp + m2, 16 * grp)
                T.sync_threads()

                # pass 3: k1' = tx // 16 plus r3 per group, k1 = tx % 16; 16 // r3
                # DFTs of r3 over m2, stored to (tx + g*n/16) + 256*k.
                _gather16(
                    s_re, s_im, reg, (tx // 16) * (16 * grp) + (tx % 16) * grp, r3 * 16 * grp, 1, r3
                )
                _dft_group(reg, r3, 0)
                for k in T.unroll(16):
                    st[0] = reg[(k // r3) * r3 + _perm(k % r3, r3), 0]
                    st[1] = reg[(k // r3) * r3 + _perm(k % r3, r3), 1]
                    for v in T.vectorized(2):
                        y_pair[bb, tx + (k // r3) * (n // 16) + 256 * (k % r3), v] = st[v]

        return main

    return _func


def _build_four_pass(n: int, real_dtype: str) -> Any:
    """The four-pass n = 8192 or 16384 kernel: three radix-16 exchanges, then a combine.

    The final radix-(n // 4096) pass reads its cross-thread values by warp
    shuffle, so there is no fourth shared-memory round trip.
    """
    a2 = n // 256
    a3 = n // 4096
    row1, row2 = _four_pass_rows(n)

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3"],
    )
    def _func(row: int, grp: int):
        batch = T.dynamic("batch")

        @T.prim_func
        def main(
            x_pair: T.Tensor((batch, n, 2), real_dtype),
            wlut: T.Tensor((n, 2), real_dtype),
            w2lut: T.Tensor((a2, 2), real_dtype),
            y_pair: T.Tensor((batch, n, 2), real_dtype),
        ):
            with T.Kernel(batch, threads=n // 16) as bb:
                tx = T.get_thread_binding()
                reg = T.alloc_local((16, 2), real_dtype)
                st = T.alloc_local((2,), real_dtype)
                cw = T.alloc_local((2,), real_dtype)
                # The final pass's per-k shuffle gather needs up to 4 scratch
                # slots (its radix is 2 or 4), unrelated to reg's 16 pass-1..3
                # slots.
                vals = T.alloc_local((4, 2), real_dtype)
                s_re = T.alloc_shared((max(16 * row1, 256 * row2),), real_dtype)
                s_im = T.alloc_shared((max(16 * row1, 256 * row2),), real_dtype)
                # s_w2 holds the pass-2 base twiddles (a2 = n // 256 of them); s_w3
                # (pass-3's, a3 = n // 4096 = the final radix) is a stride-16
                # subsample of the same table: W_(n/256)^(16 j) == W_(n/4096)^j,
                # so no third input tensor is needed. Unpacked into separate
                # _re/_im arrays, not one (a2, 2) interleaved array: a2 is 32 for
                # every plan in this service region -- exactly one warp -- so an
                # interleaved read (index m2*2) puts 32 *distinct* addresses into
                # only 16 banks (m2 and m2+16 collide, a real conflict, not a
                # broadcast); a stride-1 array puts the 32 distinct m2 in the 32
                # distinct banks they naturally span.
                s_w2_re = T.alloc_shared((a2,), real_dtype)
                s_w2_im = T.alloc_shared((a2,), real_dtype)
                s_w3_re = T.alloc_shared((a3,), real_dtype)
                s_w3_im = T.alloc_shared((a3,), real_dtype)
                if tx < a2:
                    s_w2_re[tx] = w2lut[tx, 0]
                    s_w2_im[tx] = w2lut[tx, 1]
                if tx < a3:
                    s_w3_re[tx] = w2lut[16 * tx, 0]
                    s_w3_im[tx] = w2lut[16 * tx, 1]
                T.sync_threads()

                # pass 1: read x[k*(n/16) + tx], DFT16 over n1, twiddle W_n^(tx*k1), to S1[k1][tx]
                _read_pairs(x_pair, reg, bb, tx, n // 16, 16)
                _dft(reg, 0, 16)
                for v in T.vectorized(2):
                    st[v] = wlut[tx, v]
                _twiddle_rotor(reg, cw, 0, 16, st[0], st[1])
                _write_perm16(s_re, s_im, reg, tx, row1)
                T.sync_threads()

                # pass 2: k1 = tx // a2, m2 = tx % a2; DFT16 over m1; twiddle W_(n/16)^(m2*k1)
                k1 = tx // a2
                m2 = tx % a2
                _read_scratch(s_re, s_im, reg, k1 * row1 + m2, a2, 16)
                T.sync_threads()  # every read of S1 is done before S2 overwrites it
                _dft(reg, 0, 16)
                _twiddle_rotor(reg, cw, 0, 16, s_w2_re[m2], s_w2_im[m2])
                _write_perm16(s_re, s_im, reg, k1 * row2 + m2, 16 * row2)
                T.sync_threads()

                # pass 3: k2 = tx // a3, m3 = tx % a3; DFT16 over m2; twiddle W_a2^(m3*k2)
                k2 = tx // a3
                m3 = tx % a3
                _read_scratch(s_re, s_im, reg, k2 * row2 + m3, a3, 16)
                T.sync_threads()  # every read of S2 is done before this warp's shuffles start
                _dft(reg, 0, 16)
                _twiddle_rotor(reg, cw, 0, 16, s_w3_re[m3], s_w3_im[m3])

                # pass 4: radix-(n // 4096) combine by warp shuffle, no shared memory
                _each(_pass4_shuffle_one, (y_pair, reg, vals, st, bb, tx, k2, m3, n // 4096), 0, 16)

        return main

    return _func


@T.macro
def _four_step_middle(s_re, s_im, reg, cw, s_w2, lane, col, tw: int, row: int, grp: int, nf: int):
    """Pass 2 of a three-pass four-step factor.

    Relabelled as ``k1 = lane % 16, m2 = lane // 16`` against the single-CTA
    plan -- free, since the S2 address is a function of (k1', k1, m2) only -- so
    every pass touches 16 consecutive k1 per warp and the odd ``row``/``grp`` of
    the interleaved layout suffice.
    """
    k1 = lane % 16
    m2 = lane // 16
    _read_scratch(s_re, s_im, reg, (k1 * row + m2) * tw + col, _factor_radix(nf) * tw, 16)
    T.sync_threads()  # every read of S1 is done before S2 overwrites it
    _dft(reg, 0, 16)
    _twiddle_rotor(reg, cw, 0, 16, s_w2[m2, 0], s_w2[m2, 1])
    _write_perm16(s_re, s_im, reg, (k1 * grp + m2) * tw + col, 16 * grp * tw)
    T.sync_threads()


@T.macro
def _four_step_a_body(
    x_pair,
    w1lut,
    w2lut,
    twlut,
    t_pair,
    bx,
    by,
    bb,
    tw: int,
    row: int,
    grp: int,
    rowlen: int,
    n1: int,
    n2: int,
    real_dtype: str,
):
    """Kernel A: the column pass, the four-step twiddle, and the store to T.

    The CTA owns ``tw`` consecutive j_a and all n1 values of j_b, so it runs
    ``tw`` copies of the shipped length-n1 plan. Thread t takes ``col = t % tw``
    (which j_a) and ``lane = t // tw`` (which of the plan's threads); with col in
    the fast position both the pass-1 load and the last store are fully
    coalesced, and the strided axis has moved into the per-thread register index.
    """
    t = T.get_thread_binding()
    col = t % tw
    lane = t // tw
    ja = bx * tw + col
    reg = T.alloc_local((16, 2), real_dtype)
    st = T.alloc_local((2,), real_dtype)
    tmp = T.alloc_local((2,), real_dtype)
    # twb is the per-lane four-step base, twg it times the p power and twk that
    # times the q power.
    cw = T.alloc_local((2,), real_dtype)
    twb = T.alloc_local((2,), real_dtype)
    twg = T.alloc_local((2,), real_dtype)
    twk = T.alloc_local((2,), real_dtype)
    pq = T.alloc_local((2,), real_dtype)
    s_re = T.alloc_shared((_four_step_smem(n1, tw, row, grp),), real_dtype)
    s_im = T.alloc_shared((_four_step_smem(n1, tw, row, grp),), real_dtype)
    # Stage once per CTA because every warp reuses these values.
    s_w1 = T.alloc_shared((n1 // 16, 2), real_dtype)
    if _factor_passes(n1) == 3:
        s_w2 = T.alloc_shared((_factor_radix(n1), 2), real_dtype)
    s_tw = T.alloc_shared((len(_twiddle_exps(n1)), tw, 2), real_dtype)
    if t < n1 // 16:
        for v in T.vectorized(2):
            s_w1[t, v] = w1lut[t, v]
    if _factor_passes(n1) == 3:  # noqa: SIM102
        if t < _factor_radix(n1):
            for v in T.vectorized(2):
                s_w2[t, v] = w2lut[t, v]
    # The CTA has tw * lanes threads and the four-step table is twrows * tw
    # values, so one pass over the threads covers it only while twrows <= lanes.
    # A factor below 256 has fewer lanes than rows and needs a second pass.
    if t < len(_twiddle_exps(n1)) * tw:
        for v in T.vectorized(2):
            s_tw[lane, col, v] = twlut[lane, bx * tw + col, v]
    if len(_twiddle_exps(n1)) > n1 // 16:  # noqa: SIM102
        if lane + n1 // 16 < len(_twiddle_exps(n1)):
            for v in T.vectorized(2):
                s_tw[lane + n1 // 16, col, v] = twlut[lane + n1 // 16, bx * tw + col, v]
    T.sync_threads()

    # pass 1: read x[(j*lanes + lane)*n2 + j_a], DFT16, twiddle W_n1^(lane*k1)
    _read_pairs(x_pair, reg, bb, by * rowlen + lane * n2 + ja, (n1 // 16) * n2, 16)
    _dft(reg, 0, 16)
    for v in T.vectorized(2):
        st[v] = s_w1[lane, v]
    _twiddle_rotor(reg, cw, 0, 16, st[0], st[1])
    _write_perm16(s_re, s_im, reg, lane * tw + col, row * tw)
    T.sync_threads()

    if _factor_passes(n1) == 3:
        _four_step_middle(s_re, s_im, reg, cw, s_w2, lane, col, tw, row, grp, n1)

    # The four-step base W_rowlen^(j_a*lane) = W^(j_a*l) * W^(s*j_a*h), lane = l + s*h;
    # the two halves of u are read per output below.
    for v in T.vectorized(2):
        pq[v] = s_tw[lane % _lane_split(n1), col, v]
    for v in T.vectorized(2):
        st[v] = s_tw[_lane_split(n1) + lane // _lane_split(n1), col, v]
    twb[0] = pq[0] * st[0] - pq[1] * st[1]
    twb[1] = pq[0] * st[1] + pq[1] * st[0]

    # Last pass, W_rowlen^(j_a*k_b), and the store to T[k_b][j_a] at
    # out0 + g*ogstep + k*okstep. A two-pass factor reads S1, a three-pass one the
    # S2 _four_step_middle wrote. The outputs are walked in the order of u, whose
    # twiddle is two table lookups deep for every output.
    if _factor_passes(n1) == 2:
        base = lane * row * tw + col
        gstep = (n1 // 16) * row * tw
    else:
        base = ((lane // 16) * (16 * grp) + (lane % 16) * grp) * tw + col
        gstep = _factor_radix(n1) * 16 * grp * tw
    _gather16(s_re, s_im, reg, base, gstep, tw, _factor_radix(n1))
    _dft_group(reg, _factor_radix(n1), 0)
    _each(
        _four_step_a_out,
        (
            t_pair,
            s_tw,
            col,
            reg,
            st,
            tmp,
            twb,
            twg,
            twk,
            bb,
            by * rowlen + lane * n2 + ja,
            (n1 // 16) * n2,
            (n1 // 16) * 16 // _factor_radix(n1) * n2,
            _factor_radix(n1),
            _lane_split(n1) + (n1 // 16) // _lane_split(n1),
        ),
        0,
        16,
    )


@T.macro
def _four_step_b_body(
    t_pair,
    w1lut,
    w2lut,
    y_pair,
    bx,
    by,
    bb,
    tw: int,
    row: int,
    grp: int,
    nf: int,
    tiled: int,
    row_stride: int,
    out_stride: int,
    real_dtype: str,
):
    """Kernel B: the row pass and the output-side corner turn.

    The mirror of ``_four_step_a_body``: the CTA owns ``tw`` consecutive k_b,
    reads the contiguous rows T[k_b][:], and stores each output digit
    ``out_stride`` apart. Shared memory is index-for-index kernel A's, so the same
    odd strides apply; only the two global index maps differ, and there is no W_n
    here -- the column kernels applied it.
    """
    t = T.get_thread_binding()
    col = t % tw
    lane = t // tw
    kb = bx * tw + col
    reg = T.alloc_local((16, 2), real_dtype)
    st = T.alloc_local((2,), real_dtype)
    cw = T.alloc_local((2,), real_dtype)
    s_re = T.alloc_shared((_four_step_smem(nf, tw, row, grp),), real_dtype)
    s_im = T.alloc_shared((_four_step_smem(nf, tw, row, grp),), real_dtype)
    s_w1 = T.alloc_shared((nf // 16, 2), real_dtype)
    if _factor_passes(nf) == 3:
        s_w2 = T.alloc_shared((_factor_radix(nf), 2), real_dtype)
    if t < nf // 16:
        for v in T.vectorized(2):
            s_w1[t, v] = w1lut[t, v]
    if _factor_passes(nf) == 3:  # noqa: SIM102
        if t < _factor_radix(nf):
            for v in T.vectorized(2):
                s_w2[t, v] = w2lut[t, v]
    T.sync_threads()

    # pass 1: read T[k_b][j*lanes + lane], DFT16, twiddle W_nf^(lane*k1)
    _read_pairs(t_pair, reg, bb, kb * row_stride + by * nf + lane, nf // 16, 16)
    _dft(reg, 0, 16)
    for v in T.vectorized(2):
        st[v] = s_w1[lane, v]
    _twiddle_rotor(reg, cw, 0, 16, st[0], st[1])
    _write_perm16(s_re, s_im, reg, lane * tw + col, row * tw)
    T.sync_threads()

    if _factor_passes(nf) == 3:
        _four_step_middle(s_re, s_im, reg, cw, s_w2, lane, col, tw, row, grp, nf)

    # Last pass and the store to X[k_a*out_stride + k_b]; the reads differ as in
    # _four_step_a_body.
    if _factor_passes(nf) == 2:
        base = lane * row * tw + col
        gstep = (nf // 16) * row * tw
    else:
        base = ((lane // 16) * (16 * grp) + (lane % 16) * grp) * tw + col
        gstep = _factor_radix(nf) * 16 * grp * tw
    _gather16(s_re, s_im, reg, base, gstep, tw, _factor_radix(nf))
    _dft_group(reg, _factor_radix(nf), 0)
    _each(
        _four_step_b_out,
        (
            y_pair,
            reg,
            st,
            bb,
            lane * out_stride + by * tiled + kb,
            (nf // 16) * out_stride,
            (nf // 16) * 16 // _factor_radix(nf) * out_stride,
            _factor_radix(nf),
        ),
        0,
        16,
    )


def _build_four_step_a(factors: tuple, level: int, real_dtype: str) -> Any:
    """Column kernel *level* of a four-step plan, counted from the outermost.

    For every j_a in [0, n2) it transforms the length-n1 stride-n2 sequence over
    j_b, multiplies by the four-step factor W_(n1*n2)^(j_a*k_b), and writes the
    intermediate T[k_b][j_a] -- contiguous in j_a, which is what makes the next
    kernel's read contiguous. Level 2 exists only in a three-factor plan, where
    it is level 1 run again on each row that wrote: grid.y then runs over those
    rows, which are independent transforms, so only the global index maps see it.
    """
    total = math.prod(factors)
    # n2 is the length the kernel strides over (its grid.x extent before tiling),
    # outer how many rows of n1 * n2 one batch element holds.
    n1 = factors[level - 1]
    n2 = math.prod(factors[level:])
    outer = math.prod(factors[: level - 1])
    rowlen = n1 * n2
    lanes = n1 // 16
    passes = _factor_passes(n1)
    twrows = len(_twiddle_exps(n1))
    if twrows > 2 * lanes:
        raise ValueError(
            f"factor {n1} needs {-(-twrows // lanes)} passes to stage {twrows} twiddle rows "
            f"over {lanes} lanes; the staging in _four_step_a_body writes out two"
        )

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3"],
    )
    def _func(tw: int, row: int, grp: int = 0):
        """Build for *tw* columns per CTA and S1/S2 strides *row*, *grp*.

        A two-pass factor has no S2 and ignores grp.
        """
        batch = T.dynamic("batch")
        # Everything the body needs that is not a tensor or a grid index.
        geom = (tw, row, grp, rowlen, n1, n2, real_dtype)

        if passes == 2:

            @T.prim_func
            def main(
                x_pair: T.Tensor((batch, total, 2), real_dtype),
                w1lut: T.Tensor((n1, 2), real_dtype),
                twlut: T.Tensor((twrows, n2, 2), real_dtype),
                t_pair: T.Tensor((batch, total, 2), real_dtype),
            ):
                if outer == 1:
                    with T.Kernel(n2 // tw, batch, threads=tw * lanes) as (bx, bb):
                        _four_step_a_body(x_pair, w1lut, w1lut, twlut, t_pair, bx, 0, bb, *geom)
                else:
                    with T.Kernel(n2 // tw, outer, batch, threads=tw * lanes) as (bx, by, bb):
                        _four_step_a_body(x_pair, w1lut, w1lut, twlut, t_pair, bx, by, bb, *geom)

        else:

            @T.prim_func
            def main(
                x_pair: T.Tensor((batch, total, 2), real_dtype),
                w1lut: T.Tensor((n1, 2), real_dtype),
                w2lut: T.Tensor((_factor_radix(n1), 2), real_dtype),
                twlut: T.Tensor((twrows, n2, 2), real_dtype),
                t_pair: T.Tensor((batch, total, 2), real_dtype),
            ):
                if outer == 1:
                    with T.Kernel(n2 // tw, batch, threads=tw * lanes) as (bx, bb):
                        _four_step_a_body(x_pair, w1lut, w2lut, twlut, t_pair, bx, 0, bb, *geom)
                else:
                    with T.Kernel(n2 // tw, outer, batch, threads=tw * lanes) as (bx, by, bb):
                        _four_step_a_body(x_pair, w1lut, w2lut, twlut, t_pair, bx, by, bb, *geom)

        return main

    return _func


def _build_four_step_b(factors: tuple, real_dtype: str) -> Any:
    """The row kernel of a four-step plan: the row pass and the output corner turn.

    For every k_b in [0, factors[0]) it transforms the contiguous row T[k_b][:]
    over j_a and stores to X[k_a*out_stride + k_b]. No W_n here: the column
    kernels applied it. With three factors the rows a CTA reads are one of ``mid``
    groups per tiled digit, so grid.y runs over those groups and the store map
    carries one more digit.
    """
    total = math.prod(factors)
    # The first factor is tiled because its output digit has stride 1; in a
    # three-factor plan mid indexes the middle digit on a second grid axis.
    nf = factors[-1]
    tiled = factors[0]
    mid = math.prod(factors[1:-1])
    row_stride = math.prod(factors[1:])
    out_stride = math.prod(factors[:-1])
    lanes = nf // 16
    passes = _factor_passes(nf)

    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3"],
    )
    def _func(tw: int, row: int, grp: int = 0):
        """Build for *tw* transforms per CTA and S1/S2 strides *row*, *grp*.

        A two-pass factor has no S2 and ignores grp.
        """
        batch = T.dynamic("batch")
        geom = (tw, row, grp, nf, tiled, row_stride, out_stride, real_dtype)

        if passes == 2:

            @T.prim_func
            def main(
                t_pair: T.Tensor((batch, total, 2), real_dtype),
                w1lut: T.Tensor((nf, 2), real_dtype),
                y_pair: T.Tensor((batch, total, 2), real_dtype),
            ):
                if mid == 1:
                    with T.Kernel(tiled // tw, batch, threads=tw * lanes) as (bx, bb):
                        _four_step_b_body(t_pair, w1lut, w1lut, y_pair, bx, 0, bb, *geom)
                else:
                    with T.Kernel(tiled // tw, mid, batch, threads=tw * lanes) as (bx, by, bb):
                        _four_step_b_body(t_pair, w1lut, w1lut, y_pair, bx, by, bb, *geom)

        else:

            @T.prim_func
            def main(
                t_pair: T.Tensor((batch, total, 2), real_dtype),
                w1lut: T.Tensor((nf, 2), real_dtype),
                w2lut: T.Tensor((_factor_radix(nf), 2), real_dtype),
                y_pair: T.Tensor((batch, total, 2), real_dtype),
            ):
                if mid == 1:
                    with T.Kernel(tiled // tw, batch, threads=tw * lanes) as (bx, bb):
                        _four_step_b_body(t_pair, w1lut, w2lut, y_pair, bx, 0, bb, *geom)
                else:
                    with T.Kernel(tiled // tw, mid, batch, threads=tw * lanes) as (bx, by, bb):
                        _four_step_b_body(t_pair, w1lut, w2lut, y_pair, bx, by, bb, *geom)

        return main

    return _func


# n -> the radices of the in-CTA passes of the one-CTA plan for that length. The
# first two passes of a three- or four-pass plan are radix 16 over the 16
# registers a thread holds; the last does 16 // r3 independent r3-point DFTs.
_RADIX_PLAN = {
    2: (2,),
    4: (4,),
    8: (8,),
    16: (16,),
    32: (32,),
    64: (8, 8),
    128: (8, 8, 2),
    256: (16, 16),
    512: (8, 8, 8),
    1024: (16, 16, 4),
    2048: (16, 16, 8),
    4096: (16, 16, 16),
    8192: (16, 16, 16, 2),
    16384: (16, 16, 16, 4),
}

# (n, dtype) -> four-step factors, outermost first. Two factors launch one
# column and one row kernel; three factors launch two columns and one row.
# Dtypes are keyed separately because factor capacity depends on element size.
_FOUR_STEP_PLAN = {
    (1 << 14, "complex128"): (256, 64),
    (1 << 15, "complex64"): (256, 128),
    (1 << 15, "complex128"): (256, 128),
    (1 << 16, "complex64"): (256, 256),
    (1 << 16, "complex128"): (256, 256),
    (1 << 17, "complex64"): (128, 1024),
    (1 << 17, "complex128"): (128, 1024),
    (1 << 18, "complex64"): (256, 1024),
    (1 << 18, "complex128"): (256, 1024),
    (1 << 19, "complex64"): (256, 2048),
    (1 << 19, "complex128"): (256, 2048),
    (1 << 20, "complex64"): (1024, 1024),
    (1 << 20, "complex128"): (1024, 1024),
    (1 << 21, "complex64"): (1024, 2048),
    (1 << 21, "complex128"): (1024, 2048),
    (1 << 22, "complex64"): (2048, 2048),
    (1 << 22, "complex128"): (2048, 2048),
    (1 << 23, "complex64"): (2048, 4096),
    (1 << 23, "complex128"): (2048, 4096),
    (1 << 24, "complex64"): (4096, 4096),
    (1 << 24, "complex128"): (4096, 4096),
    (1 << 25, "complex64"): (256, 512, 256),
    (1 << 25, "complex128"): (256, 512, 256),
    (1 << 26, "complex64"): (512, 512, 256),
    (1 << 26, "complex128"): (512, 512, 256),
    (1 << 27, "complex64"): (256, 512, 1024),
    (1 << 27, "complex128"): (256, 512, 1024),
    (1 << 28, "complex64"): (512, 512, 1024),
    (1 << 28, "complex128"): (512, 512, 1024),
}

# (n, dtype) -> one tile width per kernel of the plan: transforms per CTA for each
# column kernel and then for the row kernel.
_FOUR_STEP_TILE = {
    (1 << 14, "complex128"): (4, 4),
    (1 << 15, "complex64"): (16, 4),
    (1 << 15, "complex128"): (8, 4),
    (1 << 16, "complex64"): (16, 8),
    (1 << 16, "complex128"): (8, 4),
    (1 << 17, "complex64"): (16, 4),
    (1 << 17, "complex128"): (16, 4),
    (1 << 18, "complex64"): (16, 4),
    (1 << 18, "complex128"): (8, 4),
    (1 << 19, "complex64"): (32, 4),
    (1 << 19, "complex128"): (8, 2),
    (1 << 20, "complex64"): (8, 4),
    (1 << 20, "complex128"): (4, 4),
    (1 << 21, "complex64"): (8, 4),
    (1 << 21, "complex128"): (4, 2),
    (1 << 22, "complex64"): (8, 4),
    (1 << 22, "complex128"): (4, 4),
    (1 << 23, "complex64"): (8, 4),
    (1 << 23, "complex128"): (4, 2),
    (1 << 24, "complex64"): (4, 4),
    (1 << 24, "complex128"): (2, 2),
    (1 << 25, "complex64"): (32, 16, 8),
    (1 << 25, "complex128"): (8, 4, 8),
    (1 << 26, "complex64"): (16, 16, 8),
    (1 << 26, "complex128"): (4, 4, 8),
    (1 << 27, "complex64"): (32, 16, 8),
    (1 << 27, "complex128"): (32, 4, 4),
    (1 << 28, "complex64"): (16, 16, 8),
    (1 << 28, "complex128"): (4, 4, 4),
}


def _plan_table() -> Dict[tuple, FFTPlan]:
    """One record per served (length, dtype), in length order, complex64 first.

    The one-CTA lengths of ``_RADIX_PLAN``, less the pairs ``_FOUR_STEP_PLAN``
    decomposes instead, plus every pair that table names. Every one-CTA builder
    takes ``(row, grp)``; only the three-pass one reads them.
    """
    records = {}
    for n, radix in _RADIX_PLAN.items():
        for dtype in ("complex64", "complex128"):
            if (n, dtype) in _FOUR_STEP_PLAN:
                continue
            if n <= 32:
                builder = functools.partial(_build_tiny, n, 1 if n <= 4 else 2)
            elif n <= 128:
                builder = functools.partial(_build_warp8, n)
            elif n == 256:
                builder = _build_packed_256
            elif n == 512:
                builder = functools.partial(_build_packed_512, 64 if dtype == "complex64" else 48)
            elif n <= 4096:
                builder = functools.partial(_build_three_pass, n)
            else:
                builder = functools.partial(_build_four_pass, n)
            records[n, dtype] = FFTPlan(
                factors=(n,),
                radix=(radix,),
                tile=(),
                pad=(_smem_pad(n, radix),),
                twiddle_exp=(),
                builders=(builder,),
                itemsize=4 if dtype == "complex64" else 8,
            )
    for (n, dtype), factors in _FOUR_STEP_PLAN.items():
        columns = tuple(
            functools.partial(_build_four_step_a, factors, level)
            for level in range(1, len(factors))
        )
        records[n, dtype] = FFTPlan(
            factors=factors,
            radix=tuple(
                (16, _factor_radix(f)) if _factor_passes(f) == 2 else (16, 16, _factor_radix(f))
                for f in factors
            ),
            tile=_FOUR_STEP_TILE[n, dtype],
            # Odd strides at least as wide as what they index: a warp covers 32/tw
            # consecutive values of the strided index of s[idx*tw + col], and an odd
            # stride times tw walks them through all 32 banks.
            pad=tuple(
                (f // 16 + 1,) if _factor_passes(f) == 2 else (f // 16 + 1, f // 256 + 1)
                for f in factors
            ),
            twiddle_exp=tuple(_twiddle_exps(f) for f in factors[:-1]),
            builders=columns + (functools.partial(_build_four_step_b, factors),),
            itemsize=4 if dtype == "complex64" else 8,
        )
    records = {
        key: dataclasses.replace(
            plan,
            archs=tuple(
                arch
                for arch, cap in BLOCK_SHARED_BYTES_OPT_IN.items()
                if plan.default_smem_bytes <= cap
            ),
        )
        for key, plan in records.items()
    }
    return dict(sorted(records.items(), key=lambda kv: (kv[0][0], kv[0][1] != "complex64")))


# A (length, dtype) with a record here is served; one without is refused.
FFT_PLANS: Dict[tuple, FFTPlan] = _plan_table()


# Bounded by FFT_PLANS, one entry per record.
@functools.lru_cache(maxsize=64)
def _fft_builders(n: int, dtype: str) -> tuple:
    """One JIT builder per kernel of the plan for (n, dtype), in launch order."""
    real = "float32" if dtype == "complex64" else "float64"
    return tuple(build(real) for build in FFT_PLANS[n, dtype].builders)


def _pass_tables(n: int, dtype: torch.dtype, device: torch.device) -> tuple:
    """Length n's pass tables, interleaved: the full circle and its pass-2 bases.

    Built in float64 and cast once. Only the three- and four-pass plans read the
    pass-2 bases; the others take a one-row placeholder.
    """
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    ang = -2.0 * math.pi * torch.arange(n, dtype=torch.float64) / n
    ang2 = -2.0 * math.pi * torch.arange(max(1, n // 256), dtype=torch.float64) / max(1, n // 16)
    circle = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    base2 = torch.stack([torch.cos(ang2), torch.sin(ang2)], dim=1)
    return circle.to(real).to(device), base2.to(real).to(device)


class FFTC2COneCTAKernel(Kernel):
    """The (length, dtype) pairs whose FFT_PLANS record names a single factor.

    All but n <= 32 are Stockham-form: the radix permutation is folded into the
    store index maps. n <= 32 is a decimation-in-frequency network that ends
    with a shuffle bit reversal.

    Args:
        n: Transform length; (n, dtype) must have a one-factor plan.
        dtype: complex64 or complex128.
        config: Optional ``{"row", "grp"}`` shared-memory padding override.
        tune: Whether to autotune.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = list(BLOCK_SHARED_BYTES_OPT_IN)
    general = False

    @classmethod
    def applies(cls, call: FFTC2CCall) -> bool:
        """True where the call's plan is one a single CTA runs in one launch."""
        plan = FFT_PLANS.get((call.n, cls.dtype_to_str(call.dtype)))
        return plan is not None and not plan.decomposed and call.arch in plan.archs

    @classmethod
    def entry_for(cls, call: FFTC2CCall):
        """Identity without the batch extent, which every kernel takes symbolically."""
        index = call.device.index if call.device is not None else None
        identity = (call.n, call.dtype, index, call.tune)
        return identity, lambda: cls(call.n, call.dtype, tune=call.tune, device_index=index)

    def __init__(
        self,
        n: int,
        dtype: torch.dtype = torch.complex64,
        config: Optional[Dict[str, Any]] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.dtype = dtype
        self.plan = FFT_PLANS[n, self.dtype_str]
        if config is not None:
            self._check_config(config)
        (self.kernel,) = _fft_builders(n, self.dtype_str)
        self._tables: dict = {}
        self.init_config(config, tune)

    def _check_config(self, config: Dict[str, Any]) -> None:
        """Refuse strides below the record's floor or past the shared-memory limit.

        Raises:
            ValueError: A key is missing or unknown, or a value is out of bounds.
        """
        if set(config) != {"row", "grp"}:
            raise ValueError(f"config keys must be {{'row', 'grp'}}, got {sorted(config)}")
        row, grp = config["row"], config["grp"]
        floor_row, floor_grp = self.plan.pad[0]
        if (row, grp) == (floor_row, floor_grp):
            return
        if self.plan.fixed_pad:
            raise ValueError(
                f"n = {self.n} takes its strides from its length; the only config it runs is "
                f"{{'row': {floor_row}, 'grp': {floor_grp}}}"
            )
        if row < floor_row or grp < floor_grp:
            raise ValueError(
                f"strides must be at or above the record's floor ({floor_row}, {floor_grp}), "
                f"got ({row}, {grp}); below it the exchange reads outside its own rows"
            )
        if self.plan.one_cta_smem_bytes(row, grp) > self.plan.smem_cap:
            raise ValueError(
                f"{self.plan.one_cta_smem_bytes(row, grp)} bytes of shared memory exceeds "
                f"the {self.plan.smem_cap} every architecture the plan serves can give a block"
            )

    @property
    def default_config(self) -> Dict[str, Any]:
        row, grp = self.plan.pad[0]
        return {"row": row, "grp": grp}

    @property
    def autotune_supply_prog(self) -> Callable:
        """Supply a full-device batch for the shared-memory padding sweep."""

        def supply(params: list) -> list:
            real = torch.float32 if self.dtype == torch.complex64 else torch.float64
            device = (
                self.device_index if self.device_index is not None else torch.cuda.current_device()
            )
            x_pair = torch.randn(1024, self.n, 2, dtype=real, device=device)
            return [
                x_pair,
                torch.randn(self.n, 2, dtype=real, device=device),
                torch.randn(max(1, self.n // 256), 2, dtype=real, device=device),
                torch.empty_like(x_pair),
            ]

        return supply

    @property
    def autotune_configs(self) -> list[dict]:
        """Return valid padding pairs, or the fixed layout for non-tunable plans."""
        if self.plan.fixed_pad:
            return [{"row": 0, "grp": 0}]
        row, grp = self.plan.pad[0]
        return [{"row": row + dr, "grp": grp + dg} for dr in (0, 32, 64) for dg in (0, 2)]

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        """Tune padding when the plan has more than one valid configuration."""
        configs = self.autotune_configs
        if len(configs) <= 1:
            self.config = dict(configs[0])
            return
        super().autotune(warmup=warmup, rep=rep)

    def forward(self, x_pair: torch.Tensor) -> torch.Tensor:
        """Transform interleaved input.

        Args:
            x_pair: Input as $[B \\times n \\times 2]$, real and imaginary interleaved.

        Returns:
            The transform, interleaved, same shape as ``x_pair``.
        """
        index = x_pair.device.index
        if index not in self._tables:
            self._tables[index] = _pass_tables(self.n, self.dtype, x_pair.device)
        y_pair = torch.empty_like(x_pair)
        self.kernel(self.config["row"], self.config["grp"])(x_pair, *self._tables[index], y_pair)
        return y_pair


class FFTC2CDecomposedKernel(Kernel):
    """Four-step C2C FFT for the plans of two or three factors, one launch per factor.

    Column kernels perform the strided transforms and cross-factor twiddles; the
    row kernel restores natural output order.

    Args:
        n: Transform length; (n, dtype) must have a decomposed plan.
        dtype: complex64 or complex128.
        config: Optional ``{"tile", "pad"}`` override, one entry per kernel.
        tune: Whether to autotune. Each kernel of the plan is swept separately.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = list(BLOCK_SHARED_BYTES_OPT_IN)
    general = False

    @classmethod
    def applies(cls, call: FFTC2CCall) -> bool:
        """True where the call's plan names more than one factor."""
        plan = FFT_PLANS.get((call.n, cls.dtype_to_str(call.dtype)))
        return plan is not None and plan.decomposed and call.arch in plan.archs

    @classmethod
    def entry_for(cls, call: FFTC2CCall):
        """Identity without the batch extent, which every kernel takes symbolically."""
        index = call.device.index if call.device is not None else None
        identity = (call.n, call.dtype, index, call.tune)
        return identity, lambda: cls(call.n, call.dtype, tune=call.tune, device_index=index)

    def __init__(
        self,
        n: int,
        dtype: torch.dtype = torch.complex64,
        config: Optional[Dict[str, Any]] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.dtype = dtype
        self.plan = FFT_PLANS[n, self.dtype_str]
        if config is not None:
            self._check_config(config)
        self.kernel = _fft_builders(n, self.dtype_str)
        self._tables: dict = {}
        self.init_config(config, tune)

    def _fits(self, index: int, tw: int, strides: tuple) -> bool:
        """Whether kernel *index* at width *tw* and *strides* fits a block."""
        _nf, lanes, extent, _twrows, _r = self.plan.geometry(index)
        if tw < 1 or extent % tw or tw * lanes > MAX_BLOCK_THREADS:
            return False
        return self.plan.four_step_smem_bytes(index, tw, strides) <= self.plan.smem_cap

    def _check_config(self, config: Dict[str, Any]) -> None:
        """Refuse a tile or stride set that breaks a kernel's structural bounds.

        Raises:
            ValueError: A key is missing or unknown, or a value is out of bounds.
        """
        if set(config) != {"tile", "pad"}:
            raise ValueError(f"config keys must be {{'tile', 'pad'}}, got {sorted(config)}")
        tile, pad = config["tile"], config["pad"]
        factors = self.plan.factors
        if len(tile) != len(factors) or len(pad) != len(factors):
            raise ValueError(
                f"config must name one tile and one pad per factor: "
                f"{len(factors)} factors, got {len(tile)} and {len(pad)}"
            )
        for index, (tw, strides) in enumerate(zip(tile, pad, strict=True)):
            floor = self.plan.pad[index]
            if len(strides) != len(floor):
                raise ValueError(
                    f"kernel {index}: expected {len(floor)} shared strides, got {len(strides)}"
                )
            if any(value < minimum for value, minimum in zip(strides, floor, strict=True)):
                raise ValueError(
                    f"kernel {index}: strides must be at or above the record's floor "
                    f"{floor}, got {strides}"
                )
            if not self._fits(index, tw, strides):
                raise ValueError(
                    f"kernel {index}: tile {tw} with strides {strides} breaks the grid, "
                    f"thread or shared-memory bound"
                )

    @property
    def default_config(self) -> Dict[str, Any]:
        return {"tile": self.plan.tile, "pad": self.plan.pad}

    @property
    def autotune_configs(self) -> list[dict]:
        """Return the plan default; sub-kernels are tuned independently."""
        return [self.default_config]

    def _sub_kernel_supply(self, index: int) -> Callable:
        """Supply the tensors accepted by one factor kernel during tuning."""
        plan = self.plan
        nf, _lanes, extent, twrows, r_last = plan.geometry(index)
        passes = len(plan.radix[index])
        real = torch.float32 if self.dtype == torch.complex64 else torch.float64
        batch = max(1, (256 << 20) // (self.n * (8 if real is torch.float32 else 16)))

        def supply(params: list) -> list:
            device = (
                self.device_index if self.device_index is not None else torch.cuda.current_device()
            )
            buf = torch.randn(batch, self.n, 2, dtype=real, device=device)
            tables = [torch.randn(nf, 2, dtype=real, device=device)]
            if passes == 3:
                tables.append(torch.randn(r_last, 2, dtype=real, device=device))
            if twrows:
                tables.append(torch.randn(twrows, extent, 2, dtype=real, device=device))
            return [buf, *tables, torch.empty_like(buf)]

        return supply

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        """Tune each factor kernel over its width and stride neighbours."""
        print(f"Start autotuning {type(self).__name__}...")
        tile: list = []
        pad: list = []
        for index, builder in enumerate(self.kernel):
            width, strides = self.plan.tile[index], self.plan.pad[index]
            seed = {"tw": width, **dict(zip(("row", "grp"), strides, strict=False))}
            # The default width halved and doubled, then each stride widened by 2.
            candidates = [(tw, strides) for tw in (width, width // 2, width * 2)]
            candidates += [
                (width, strides[:i] + (strides[i] + 2,) + strides[i + 1 :])
                for i in range(len(strides))
            ]
            candidates = [
                {"tw": tw, **dict(zip(("row", "grp"), s, strict=False))}
                for tw, s in dict.fromkeys(candidates)
                if self._fits(index, tw, s)
            ]
            won = dict(seed)
            if len(candidates) > 1:
                tuned = self.tune_jit_kernel(
                    builder,
                    candidates,
                    warmup=warmup,
                    rep=rep,
                    seed_config=seed,
                    supply_prog=self._sub_kernel_supply(index),
                )
                won.update(tuned.config)
            tile.append(won["tw"])
            pad.append(tuple(won[name] for name in ("row", "grp") if name in seed))
        self.config = {"tile": tuple(tile), "pad": tuple(pad)}
        print(f"Best config: {self.config}")

    def _four_step_tables(self, device: torch.device) -> tuple:
        """``(w1, w2, twlut)``: per kernel its pass tables, per column kernel its twiddle rows.

        Row r of column kernel *level*'s table is ``W_m^(e_r * j_a)`` over the length
        m that kernel splits, for the exponents ``plan.twiddle_exp[level]`` it was
        built against. The trigonometry runs over one quadrant and is turned by
        ``(-i)**q``, so the rounded angle never exceeds pi/2.
        """
        real = torch.float32 if self.dtype == torch.complex64 else torch.float64
        factors = self.plan.factors
        passes = [_pass_tables(f, self.dtype, device) for f in factors]
        twlut = []
        for level, exps in enumerate(self.plan.twiddle_exp):
            m = math.prod(factors[level:])
            ja = torch.arange(math.prod(factors[level + 1 :]), dtype=torch.int64)
            rows = []
            for e in exps:
                idx = (e * ja) % m
                quadrant = idx // (m // 4)
                ang = -2.0 * math.pi * (idx % (m // 4)).to(torch.float64) / m
                re, im = torch.cos(ang), torch.sin(ang)
                # (re + i*im) * (-i)**q, written as the swap and sign flip it is.
                out_re = torch.where(
                    quadrant == 0,
                    re,
                    torch.where(quadrant == 1, im, torch.where(quadrant == 2, -re, -im)),
                )
                out_im = torch.where(
                    quadrant == 0,
                    im,
                    torch.where(quadrant == 1, -re, torch.where(quadrant == 2, -im, re)),
                )
                rows.append(torch.stack([out_re, out_im], dim=1))
            twlut.append(torch.stack(rows, dim=0).to(real).to(device).contiguous())
        return tuple(p[0] for p in passes), tuple(p[1] for p in passes), tuple(twlut)

    def forward(self, x_pair: torch.Tensor) -> torch.Tensor:
        """Transform interleaved input, one launch per factor.

        Args:
            x_pair: Input as $[B \\times n \\times 2]$, real and imaginary interleaved.

        Returns:
            The transform, interleaved, same shape as ``x_pair``.
        """
        index = x_pair.device.index
        if index not in self._tables:
            self._tables[index] = self._four_step_tables(x_pair.device)
        w1, w2, twlut = self._tables[index]
        # The chain never writes the input nor reads a buffer it is writing:
        # x -> t -> y with two kernels, x -> t -> y -> t with three. Both buffers
        # are per call, so one kernel object is safe to share across streams.
        t_pair = torch.empty_like(x_pair)
        y_pair = torch.empty_like(x_pair)
        chain = [x_pair, t_pair, y_pair, t_pair][: len(self.kernel) + 1]
        tile, pad = self.config["tile"], self.config["pad"]
        for i, build in enumerate(self.kernel):
            # A two-pass factor's kernel takes no pass-2 table.
            tables = (w1[i],) if len(self.plan.radix[i]) == 2 else (w1[i], w2[i])
            if i < len(twlut):
                tables += (twlut[i],)
            build(tile[i], *pad[i])(chain[i], *tables, chain[i + 1])
        return chain[-1]
