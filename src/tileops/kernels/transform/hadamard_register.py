"""Hadamard transforms whose rows stay in registers between the load and the store.

A thread holds slots of eight contiguous elements, read with one 16-byte load each. The eight
elements of a slot are the A fragment of a 16 x 16 x 16 MMA whose B operand is the order-16
Sylvester matrix, so one MMA pair transforms element bits {0, 2, 3, 4} of every slot from the
16-bit input, exactly, accumulating in float32. Every other bit is a float32 add/subtract
butterfly between two registers. ``HadamardWarpRowsKernel`` places whole rows of up to 512
elements in one thread's slots and needs nothing else; ``HadamardExchangeKernel`` moves a row
through shared memory once to reach the bits no thread held; ``HadamardBaseExchangeKernel``
does the same for a row of base_order segments and adds the base product in each thread.
"""

import functools
import itertools
import math
from typing import Any, Dict, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.transform.call_spec import (
    HADAMARD_BASES,
    HadamardTransformCall,
    HadamardTransformFwdInterface,
)

__all__ = ["HadamardBaseExchangeKernel", "HadamardExchangeKernel", "HadamardWarpRowsKernel"]

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    # Every shared-memory hazard is fenced by hand: the inferred fences land inside branches
    # one warp of a pair takes.
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


def _bit(x, b: int):
    return (x >> b) & 1


@T.macro
def _sylvester_fragment(hb, lane, scale: float):
    """The B fragment of the order-16 Sylvester matrix times *scale*.

    MMA k and n index element bits (0, 3, 4, 2) of a slot, so entry (k, n) is the sign of
    popcount(k & n).
    """
    g = lane // 4
    q = lane % 4
    for j in T.unroll(8):
        k = 2 * q + j % 2 + 8 * ((j // 2) % 2)
        c = g + 8 * (j // 4)
        odd = (_bit(k & c, 0) + _bit(k & c, 1) + _bit(k & c, 2) + _bit(k & c, 3)) % 2
        hb[j] = T.if_then_else(odd == 1, -scale, scale)


@T.macro
def _first_bits(xa, hb, acc, slots: int, mma_dtype: str):
    """acc slot s = xa slot s transformed over element bits {0, 2, 3, 4}."""
    for i in T.unroll(slots * 8):
        acc[i] = 0.0
    for s in T.unroll(slots):
        for half in T.unroll(2):
            T.evaluate(
                T.ptx_mma(
                    "float32", "m16n8k16", "row", "col", mma_dtype, mma_dtype, "fp32",
                    xa.data, s * 4, hb.data, half * 4, acc.data, s * 8 + half * 4, False,
                )
            )  # fmt: skip


@T.macro
def _butterflies(acc, strides: tuple, k: int, size: int):
    """Add/subtract butterflies between registers *stride* apart, for each stride in turn."""
    if k < len(strides):
        d = strides[k]
        for i in T.unroll(size // 2):
            lo = (i // d) * (2 * d) + i % d
            u = acc[lo]
            w = acc[lo + d]
            acc[lo] = u + w
            acc[lo + d] = u - w
        _butterflies(acc, strides, k + 1, size)


def _mma_dtype(dtype: str) -> str:
    return "bf16" if dtype == "bfloat16" else "fp16"


@functools.lru_cache(maxsize=32)
def _hadamard_warp_rows_kernel(log_n: int, dtype: str, slots: int, ctas: int):
    """Rows of 2**log_n, 32 to 512 elements, each owned by four lanes of one warp.

    Lanes g = lane // 4 take eight different rows and a thread's slots carry element bits
    5 .. log_n - 1, so after the MMA pair every remaining bit is a register butterfly.
    """

    @tilelang.jit(out_idx=[1], pass_configs=_PASS_CONFIGS)
    def _func():
        n = 1 << log_n
        # A slot spans 32 elements of a row: 8 a lane, 4 lanes.
        slots_per_row = n // 32
        rows_per_warp = 8 * (slots // slots_per_row)
        strides = (2,) + tuple(8 << i for i in range(log_n - 5))
        # The Sylvester fragment carries the power-of-two part of 1 / sqrt(n), an odd log_n's
        # store the rest.
        pow2_scale = 2.0 ** -(log_n // 2)
        odd_scale = math.sqrt(0.5)
        rows = T.dynamic("rows")

        @T.prim_func
        def main(x: T.Tensor((rows, n // 2), "uint32"), y: T.Tensor((rows, n), dtype)):
            blocks = T.ceildiv(rows, rows_per_warp)
            with T.Kernel(T.min(blocks, ctas), threads=32) as bx:
                lane = T.get_thread_binding()
                g = lane // 4
                q = lane % 4
                xa = T.alloc_local((slots * 4,), "uint32")
                hb = T.alloc_local((8,), dtype)
                acc = T.alloc_local((slots * 8,), "float32")
                out = T.alloc_local((8,), dtype)
                grid = T.min(blocks, ctas)
                for s in T.unroll(slots):
                    row = bx * rows_per_warp + s // slots_per_row * 8 + g
                    if row < rows:
                        for v in T.vectorized(4):
                            xa[s * 4 + v] = x[row, s % slots_per_row * 16 + q * 4 + v]
                # Under the first loads' latency.
                _sylvester_fragment(hb, lane, pow2_scale)
                for it in T.serial(T.ceildiv(blocks - bx, grid)):
                    b = bx + it * grid
                    _first_bits(xa, hb, acc, slots, _mma_dtype(dtype))
                    # The next block's input lands in the registers the MMAs just read.
                    for s in T.unroll(slots):
                        row = (b + grid) * rows_per_warp + s // slots_per_row * 8 + g
                        if row < rows:
                            for v in T.vectorized(4):
                                xa[s * 4 + v] = x[row, s % slots_per_row * 16 + q * 4 + v]
                    _butterflies(acc, strides, 0, slots * 8)
                    for s in T.unroll(slots):
                        row = b * rows_per_warp + s // slots_per_row * 8 + g
                        for v in T.vectorized(8):
                            if log_n % 2 == 1:
                                out[v] = T.cast(acc[s * 8 + v] * odd_scale, dtype)
                            else:
                                out[v] = T.cast(acc[s * 8 + v], dtype)
                        if row < rows:
                            for v in T.vectorized(8):
                                y[row, s % slots_per_row * 32 + q * 8 + v] = out[v]

        return main

    return _func


def _exchange_plan(log_n: int, slots: int) -> tuple:
    """Bit assignment of one row of 2**log_n elements, *slots* slots a thread.

    P0, the load: slot s of warp w holds elements w * span + s * 256 + lane * 8 + (0..7), so
    element bits 0-2 are the vector, 3-7 the lane, the next log2(slots) the slot and the rest
    the warp. P1, after the exchange: a thread holds element bits 5-7 and the P0 warp bits, the
    ones no P0 thread held, filled up with bits 0, 1, 2 so the store is a vector of them.

    Returns:
        (warps, p0_slot_bits, p0_warp_bits, p1_local_bits, p1_lane_bits, p1_warp_bits, fill)
    """
    slot_bits = slots.bit_length() - 1
    warps = 1 << (log_n - 8 - slot_bits)
    p0_slot = list(range(8, 8 + slot_bits))
    p0_warp = list(range(8 + slot_bits, log_n))
    capacity = 3 + slot_bits
    need = [5, 6, 7] + p0_warp
    fill = [0, 1, 2][: capacity - len(need)]
    p1_local = fill + need
    spare = [b for b in p0_slot if b not in p1_local]
    while len(p1_local) < capacity:
        p1_local.append(spare.pop())
    rest = [b for b in range(log_n) if b not in p1_local]
    return warps, p0_slot, p0_warp, p1_local, rest[:5], rest[5:], len(fill)


def _exchange_fits(log_n: int, slots: int) -> bool:
    """Whether *slots* slots a thread leave room in P1 for every bit no P0 thread held."""
    slot_bits = slots.bit_length() - 1
    warp_bits = log_n - 8 - slot_bits
    return 0 <= warp_bits <= slot_bits


def _distinct(weights: list, unit: int, groups: int) -> bool:
    """Whether every 0/1 sum of *weights* lands in its own (word // unit) % groups."""
    hit = {(sum(w for i, w in enumerate(weights) if c >> i & 1) // unit) % groups
           for c in range(1 << len(weights))}  # fmt: skip
    return len(hit) == 1 << len(weights)


@functools.lru_cache(maxsize=64)
def _exchange_weights(log_n: int, p1_lane: tuple, vector: int) -> tuple:
    """Word offset of each element bit in the exchange buffer, phys(e) = e + sum c * (e >> s).

    The pad terms keep phys increasing, so each element has its own word, and are the
    smallest pair that puts the lanes of every quarter-warp phase on their own banks: element
    bits 3, 4, 5 for the P0 16-byte writes, the low P1 lane bits for the *vector*-word reads.
    """
    lanes = 3 if vector >= 4 else (4 if vector == 2 else 5)
    unit, groups = (4, 8) if vector >= 4 else ((2, 16) if vector == 2 else (1, 32))
    terms = [(c, s) for s in range(5, log_n) for c in range(4, 32, 4)]
    for count in (1, 2):
        best = None
        for combo in itertools.combinations(terms, count):
            w = [(1 << b) + sum(c << (b - s) for c, s in combo if b >= s) for b in range(log_n)]
            fits = _distinct([w[3], w[4], w[5]], 4, 8) and _distinct(
                [w[b] for b in p1_lane[:lanes]], unit, groups
            )
            if fits and (best is None or sum(w) < sum(best)):
                best = w
        if best is not None:
            return tuple(best)
    raise ValueError(f"no bank-conflict-free exchange for n = 2**{log_n}, P1 lanes {p1_lane}")


def _exchange_bytes(log_n: int, slots: int) -> int:
    """Shared memory one row's exchange takes."""
    *_, p1_lane, _, fill = _exchange_plan(log_n, slots)
    return 4 * (sum(_exchange_weights(log_n, tuple(p1_lane), 1 << fill)) + 1)


@functools.lru_cache(maxsize=32)
def _hadamard_exchange_kernel(log_n: int, dtype: str, slots: int, ctas: int):
    """Rows of 2**log_n, 1024 to 32768 elements, through one shared-memory exchange a row."""

    @tilelang.jit(out_idx=[1], pass_configs=_PASS_CONFIGS)
    def _func():
        n = 1 << log_n
        warps, p0_slot, p0_warp, p1_local, p1_lane, p1_warp, fill = _exchange_plan(log_n, slots)
        size = slots * 8
        vector = 1 << fill
        weights = _exchange_weights(log_n, tuple(p1_lane), vector)
        words = sum(weights) + 1
        strides0 = (2,) + tuple(8 << i for i in range(len(p0_slot)))
        strides1 = tuple(1 << i for i, b in enumerate(p1_local) if b in [5, 6, 7] + p0_warp)
        # The Sylvester fragment carries the power-of-two part of 1 / sqrt(n), an odd log_n's
        # store the rest.
        pow2_scale = 2.0 ** -(log_n // 2)
        odd_scale = math.sqrt(0.5)

        def place(value, bits, weight):
            return sum(((value >> i) & 1) * weight(b) for i, b in enumerate(bits))

        def p0_word(w, s, lane):
            """The 32-bit input word slot s of warp w starts at."""
            half = lambda b: 1 << (b - 1)  # noqa: E731
            return place(w, p0_warp, half) + place(s, p0_slot, half) + lane * 4

        def p0_offset(lane, w, s):
            lane_bits = [3, 4, 5, 6, 7]
            return (
                place(lane, lane_bits, weights.__getitem__)
                + place(w, p0_warp, weights.__getitem__)
                + place(s, p0_slot, weights.__getitem__)
            )

        def p1_offset(lane, w, group):
            return (
                place(lane, p1_lane, weights.__getitem__)
                + place(w, p1_warp, weights.__getitem__)
                + place(group, p1_local[fill:], weights.__getitem__)
            )

        def p1_vector(lane, w, group):
            """The output vector P1 group *group* stores, in units of *vector* elements."""
            unit = lambda b: 1 << (b - fill)  # noqa: E731
            return (
                place(lane, p1_lane, unit)
                + place(w, p1_warp, unit)
                + place(group, p1_local[fill:], unit)
            )

        rows = T.dynamic("rows")

        @T.prim_func
        def main(x: T.Tensor((rows, n // 2), "uint32"), y: T.Tensor((rows, n), dtype)):
            with T.Kernel(T.min(rows, ctas), threads=32 * warps) as bx:
                tx = T.get_thread_binding()
                lane = tx % 32
                w = tx // 32
                ex = T.alloc_shared((words,), "float32")
                xa = T.alloc_local((slots * 4,), "uint32")
                hb = T.alloc_local((8,), dtype)
                acc = T.alloc_local((size,), "float32")
                out = T.alloc_local((vector,), dtype)
                grid = T.min(rows, ctas)
                for s in T.unroll(slots):
                    if bx < rows:
                        for v in T.vectorized(4):
                            xa[s * 4 + v] = x[bx, p0_word(w, s, lane) + v]
                # Under the first loads' latency.
                _sylvester_fragment(hb, lane, pow2_scale)
                for it in T.serial(T.ceildiv(rows - bx, grid)):
                    r = bx + it * grid
                    _first_bits(xa, hb, acc, slots, _mma_dtype(dtype))
                    # The next row's input lands in the registers the MMAs just read.
                    for s in T.unroll(slots):
                        if r + grid < rows:
                            for v in T.vectorized(4):
                                xa[s * 4 + v] = x[r + grid, p0_word(w, s, lane) + v]
                    _butterflies(acc, strides0, 0, size)
                    for s in T.unroll(slots):
                        for c in T.unroll(2):
                            for v in T.vectorized(4):
                                ex[p0_offset(lane, w, s) + c * 4 + v] = acc[s * 8 + c * 4 + v]
                    if warps == 1:
                        T.sync_warp()
                    else:
                        T.sync_threads()
                    for p in T.unroll(size // vector):
                        for v in T.vectorized(vector):
                            acc[p * vector + v] = ex[p1_offset(lane, w, p) + v]
                    # The buffer is written again for the next row, if there is one.
                    if r + grid < rows:
                        if warps == 1:
                            T.sync_warp()
                        else:
                            T.sync_threads()
                    _butterflies(acc, strides1, 0, size)
                    for p in T.unroll(size // vector):
                        for v in T.vectorized(vector):
                            if log_n % 2 == 1:
                                out[v] = T.cast(acc[p * vector + v] * odd_scale, dtype)
                            else:
                                out[v] = T.cast(acc[p * vector + v], dtype)
                        for v in T.vectorized(vector):
                            y[r, p1_vector(lane, w, p) * vector + v] = out[v]

        return main

    return _func


def _rows_of(x: torch.Tensor, n: int) -> "tuple[int, torch.Tensor]":
    """The row count and *x* as rows of 32-bit words, on a 16-byte boundary."""
    rows = x.numel() // n
    return rows, vector_aligned(x.contiguous()).view(rows, n).view(torch.uint32)


class HadamardWarpRowsKernel(Kernel, HadamardTransformFwdInterface):
    """Base order 1, rows of 32 to 512 elements, each in four lanes' registers.

    Args:
        n: Row length, a power of two from 32 to 512.
        dtype: float16 or bfloat16.
        config: Optional ``{"slots", "ctas_per_sm"}`` override.
        sm_count: SMs of the device; the grid is that many times ``ctas_per_sm``.
        device_index: The device the kernel runs on.
    """

    # mma.m16n8k16 on 16-bit operands.
    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"hadamard_shared_rows"})

    @classmethod
    def refusal(cls, call: HadamardTransformCall) -> Optional[str]:
        if call.base_order != 1:
            return f"base order {call.base_order}: rows of a power of two only"
        if not 32 <= call.n <= 512:
            return f"n = {call.n}: a thread's slots hold a row of 32 to 512 elements"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: HadamardTransformCall) -> Entry:
        index = call.device.index if call.device is not None else None
        sms = call.sm_count
        return (call.n, call.dtype, index), lambda: cls(
            call.n, call.dtype, sm_count=sms, device_index=index
        )

    def __init__(
        self,
        n: int,
        dtype: torch.dtype,
        config: Optional[Dict[str, Any]] = None,
        *,
        sm_count: int,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.dtype = dtype
        self.sm_count = sm_count
        self.kernel = functools.partial(self._program, n.bit_length() - 1, self.dtype_str)
        self.init_config(config)

    def _program(self, log_n: int, dtype: str, slots: int, ctas_per_sm: int):
        return _hadamard_warp_rows_kernel(log_n, dtype, slots, ctas_per_sm * self.sm_count)

    @property
    def default_config(self) -> Dict[str, Any]:
        # Eight slots a thread where a row fits them, and a grid of every block up to 64 an
        # SM: the fastest measured launch at 32 to 512 elements a row.
        return {"slots": max(8, self.n // 32), "ctas_per_sm": 64}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows, words = _rows_of(x, self.n)
        if rows == 0:
            return torch.empty_like(x)
        return self.kernel(**self.config)()(words).view(x.shape)


def _fewest_slots(log_n: int) -> int:
    """The fewest slots a thread whose P1 still holds every bit no P0 thread held."""
    return next(s for s in (2, 4, 8, 16) if _exchange_fits(log_n, s))


class HadamardExchangeKernel(Kernel, HadamardTransformFwdInterface):
    """Base order 1, rows of 1024 to 32768 elements, through one shared-memory exchange a row.

    A thread takes the fewest slots the exchange allows, so a row spreads over the most warps
    and each thread has the fewest 16-byte loads in flight; the grid is every CTA the exchange
    buffers let an SM hold.

    Args:
        n: Row length, a power of two from 1024 to 32768.
        dtype: float16 or bfloat16.
        config: Optional ``{"slots"}`` override.
        sm_count: SMs of the device.
        smem_budget: Shared memory one block may take.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"hadamard_shared_rows"})

    @classmethod
    def refusal(cls, call: HadamardTransformCall) -> Optional[str]:
        if call.base_order != 1:
            return f"base order {call.base_order}: rows of a power of two only"
        if not 1024 <= call.n <= 32768:
            return f"n = {call.n}: one exchange serves rows of 1024 to 32768 elements"
        log_n = call.n.bit_length() - 1
        need = _exchange_bytes(log_n, _fewest_slots(log_n))
        if need > call.smem_budget:
            return f"a row's exchange takes {need} bytes, past the {call.smem_budget} a block has"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: HadamardTransformCall) -> Entry:
        index = call.device.index if call.device is not None else None
        sms, budget = call.sm_count, call.smem_budget
        return (call.n, call.dtype, index), lambda: cls(
            call.n, call.dtype, sm_count=sms, smem_budget=budget, device_index=index
        )

    def __init__(
        self,
        n: int,
        dtype: torch.dtype,
        config: Optional[Dict[str, Any]] = None,
        *,
        sm_count: int,
        smem_budget: int,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.dtype = dtype
        self.sm_count = sm_count
        self.smem_budget = smem_budget
        self.log_n = n.bit_length() - 1
        self.kernel = self._program
        self.init_config(config)

    def _program(self, slots: int):
        warps = _exchange_plan(self.log_n, slots)[0]
        # Every CTA an SM holds: 2048 threads, and the exchange buffers its shared memory fits.
        resident = min(2048 // (32 * warps), self.smem_budget // _exchange_bytes(self.log_n, slots))
        return _hadamard_exchange_kernel(
            self.log_n, self.dtype_str, slots, max(1, resident) * self.sm_count
        )

    @property
    def default_config(self) -> Dict[str, Any]:
        return {"slots": _fewest_slots(self.log_n)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows, words = _rows_of(x, self.n)
        if rows == 0:
            return torch.empty_like(x)
        return self.kernel(**self.config)()(words).view(x.shape)


@functools.lru_cache(maxsize=8)
def _base_program(base: int) -> tuple:
    """A straight-line add/subtract program for y[c] = sum over a of B[a][c] * v[a].

    Greedy common-pair elimination over the sign table: while some signed pair of operands
    recurs in two or more outputs, it becomes one temporary. Returns ``(temps, outputs)``:
    temps[k] = (u, v, s) means operand ``base + k`` is u + s * v; outputs[c] is a tuple of
    (operand, sign) to sum. B20 takes 150 adds this way against 380 term by term.
    """
    rows = HADAMARD_BASES[base]
    exprs = [{(a, 1 if rows[a][c] == "+" else -1) for a in range(base)} for c in range(base)]
    temps = []
    while True:
        count: Dict[tuple, int] = {}
        for e in exprs:
            for (u, su), (v, sv) in itertools.combinations(sorted(e), 2):
                count[(u, v, su * sv)] = count.get((u, v, su * sv), 0) + 1
        if not count or max(count.values()) < 2:
            break
        (u, v, rel), _ = max(count.items(), key=lambda kv: kv[1])
        new = base + len(temps)
        temps.append((u, v, rel))
        for e in exprs:
            for su in (1, -1):
                if (u, su) in e and (v, su * rel) in e:
                    e -= {(u, su), (v, su * rel)}
                    e.add((new, su))
                    break
    return tuple(temps), tuple(tuple(sorted(e)) for e in exprs)


# Module level: TileLang rewrites every function defined inside a jit builder.
def _operand(vals, tmp, base: int, cols: int, k: int, j: int):
    """Operand k of the base program at segment position j: an input below base, else a temp."""
    return vals[k * cols + j] if k < base else tmp[k - base]


def _temp_value(vals, tmp, base: int, cols: int, temps: tuple, k: int, j: int):
    u, v, s = temps[k]
    a, b = _operand(vals, tmp, base, cols, u, j), _operand(vals, tmp, base, cols, v, j)
    return a + b if s > 0 else a - b


def _result_value(vals, tmp, base: int, cols: int, terms: tuple, j: int):
    first, sign = terms[0]
    total = _operand(vals, tmp, base, cols, first, j)
    total = total if sign > 0 else -total
    for op, sign in terms[1:]:
        value = _operand(vals, tmp, base, cols, op, j)
        total = total + value if sign > 0 else total - value
    return total


@T.macro
def _base_temps(vals, tmp, base: int, cols: int, temps: tuple, j: int, k: int):
    """tmp[k ..] of the base program at segment position j; four a level keep the recursion
    shallow for the 148 temporaries of base 40."""
    if k < len(temps):
        tmp[k] = _temp_value(vals, tmp, base, cols, temps, k, j)
        if k + 1 < len(temps):
            tmp[k + 1] = _temp_value(vals, tmp, base, cols, temps, k + 1, j)
        if k + 2 < len(temps):
            tmp[k + 2] = _temp_value(vals, tmp, base, cols, temps, k + 2, j)
        if k + 3 < len(temps):
            tmp[k + 3] = _temp_value(vals, tmp, base, cols, temps, k + 3, j)
        _base_temps(vals, tmp, base, cols, temps, j, k + 4)


@T.macro
def _base_outputs(vals, tmp, out, base: int, cols: int, outputs: tuple, j: int, c: int):
    """out[c ..] of the base program at segment position j, once its temporaries are in tmp."""
    if c < base:
        out[c] = _result_value(vals, tmp, base, cols, outputs[c], j)
        _base_outputs(vals, tmp, out, base, cols, outputs, j, c + 1)


@functools.lru_cache(maxsize=16)
def _hadamard_base_exchange_kernel(base: int, log_m: int, dtype: str, ctas: int):
    """Rows of base_order segments of 2**log_m (32 to 128), a warp a row, one exchange a row.

    P0 is the power-of-two kernels' load: slot s holds elements s * 256 + lane * 8 + (0..7),
    the Sylvester MMA pair and one butterfly transform segment bits 0-4. Through shared memory,
    lane t then takes segment index t + 32 j of every segment, so the higher segment bits are
    register butterflies and the base product is a straight-line program in the thread.
    """

    @tilelang.jit(out_idx=[1], pass_configs=_PASS_CONFIGS)
    def _func():
        m = 1 << log_m
        n = base * m
        slots = -(-n // 256)
        # Segment positions a lane holds: t + 32 j for j < cols.
        cols = m // 32
        size = base * cols
        temps, outputs = _base_program(base)
        # vals[a * cols + j]: segment bits 5 and up are j's bits, registers 1 << i apart.
        segment_strides = tuple(1 << i for i in range(log_m - 5))
        scale = 1.0 / math.sqrt(n)
        rows = T.dynamic("rows")

        @T.prim_func
        def main(x: T.Tensor((rows, n // 2), "uint32"), y: T.Tensor((rows, n), dtype)):
            with T.Kernel(T.min(rows, ctas), threads=32) as bx:
                lane = T.get_thread_binding()
                # Four pad words after every 32: the P0 16-byte writes of a quarter-warp land
                # on eight bank groups, the P1 reads are 32 consecutive words.
                ex = T.alloc_shared((n + n // 8,), "float32")
                xa = T.alloc_local((slots * 4,), "uint32")
                hb = T.alloc_local((8,), dtype)
                acc = T.alloc_local((slots * 8,), "float32")
                vals = T.alloc_local((size,), "float32")
                tmp = T.alloc_local((max(1, len(temps)),), "float32")
                out = T.alloc_local((base,), "float32")
                grid = T.min(rows, ctas)
                for s in T.unroll(slots):
                    if bx < rows and s * 256 + lane * 8 < n:
                        for v in T.vectorized(4):
                            xa[s * 4 + v] = x[bx, s * 128 + lane * 4 + v]
                # Under the first loads' latency.
                _sylvester_fragment(hb, lane, 1.0)
                for it in T.serial(T.ceildiv(rows - bx, grid)):
                    r = bx + it * grid
                    _first_bits(xa, hb, acc, slots, _mma_dtype(dtype))
                    # The next row's input lands in the registers the MMAs just read.
                    for s in T.unroll(slots):
                        if r + grid < rows and s * 256 + lane * 8 < n:
                            for v in T.vectorized(4):
                                xa[s * 4 + v] = x[r + grid, s * 128 + lane * 4 + v]
                    _butterflies(acc, (2,), 0, slots * 8)
                    for s in T.unroll(slots):
                        if s * 256 + lane * 8 < n:
                            for c in T.unroll(2):
                                for v in T.vectorized(4):
                                    ex[
                                        (s * 32 + lane) * 8 + (s * 8 + lane // 4) * 4 + c * 4 + v
                                    ] = acc[s * 8 + c * 4 + v]
                    T.sync_warp()
                    for a in T.unroll(base):
                        for j in T.unroll(cols):
                            e = a * m + j * 32 + lane
                            vals[a * cols + j] = ex[e + e // 32 * 4]
                    # The buffer is written again for the next row.
                    T.sync_warp()
                    _butterflies(vals, segment_strides, 0, size)
                    for j in T.unroll(cols):
                        _base_temps(vals, tmp, base, cols, temps, j, 0)
                        _base_outputs(vals, tmp, out, base, cols, outputs, j, 0)
                        for c in T.unroll(base):
                            y[r, c * m + j * 32 + lane] = T.cast(out[c] * scale, dtype)

        return main

    return _func


class HadamardBaseExchangeKernel(Kernel, HadamardTransformFwdInterface):
    """Base order above 1, segments of 32 to 128, a row in one warp's registers.

    Args:
        n: Row length, base_order times a power of two from 32 to 128.
        base_order: 12, 20, 28 or 40.
        dtype: float16 or bfloat16.
        config: Optional ``{"ctas_per_sm"}`` override.
        sm_count: SMs of the device.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"hadamard_shared_rows"})

    @classmethod
    def refusal(cls, call: HadamardTransformCall) -> Optional[str]:
        m = call.n // call.base_order
        if call.base_order == 1:
            return "base order 1 rows take the power-of-two kernels"
        if not 32 <= m <= 128:
            return f"segments of {m}: a lane holds every segment's elements for m of 32 to 128"
        if call.n // 32 > 112:
            return f"n = {call.n}: more than 112 values a lane"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: HadamardTransformCall) -> Entry:
        index = call.device.index if call.device is not None else None
        sms = call.sm_count
        return (call.n, call.base_order, call.dtype, index), lambda: cls(
            call.n, call.base_order, call.dtype, sm_count=sms, device_index=index
        )

    def __init__(
        self,
        n: int,
        base_order: int,
        dtype: torch.dtype,
        config: Optional[Dict[str, Any]] = None,
        *,
        sm_count: int,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.base_order = base_order
        self.dtype = dtype
        self.sm_count = sm_count
        log_m = (n // base_order).bit_length() - 1
        self.kernel = functools.partial(self._program, base_order, log_m, self.dtype_str)
        self.init_config(config)

    def _program(self, base: int, log_m: int, dtype: str, ctas_per_sm: int):
        return _hadamard_base_exchange_kernel(base, log_m, dtype, ctas_per_sm * self.sm_count)

    @property
    def default_config(self) -> Dict[str, Any]:
        return {"ctas_per_sm": 8}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows, words = _rows_of(x, self.n)
        if rows == 0:
            return torch.empty_like(x)
        return self.kernel(**self.config)()(words).view(x.shape)
