"""Hadamard transforms of any served row length, staged through shared or global memory.

``HadamardSharedRowsKernel`` holds whole rows in shared memory as float32 and runs the
transform there in passes of a few bits; ``HadamardMultiPassKernel`` takes a row too long for
shared memory through a float32 copy in global memory, one pass a few bits wide.
"""

import functools
import math
from typing import Any, Dict, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.transform.call_spec import (
    HadamardTransformCall,
    HadamardTransformFwdInterface,
)
from tileops.kernels.transform.hadamard_register import (
    _PASS_CONFIGS,
    _base_outputs,
    _base_program,
    _base_temps,
    _butterflies,
    _first_bits,
    _mma_dtype,
    _sylvester_fragment,
)

__all__ = ["HadamardMultiPassKernel", "HadamardSharedRowsKernel"]


def _strides(bits: int) -> tuple:
    """Butterfly strides over 2**bits values held in order: one stage a bit."""
    return tuple(1 << i for i in range(bits))


@functools.lru_cache(maxsize=64)
def _hadamard_shared_rows_kernel(
    n: int, base: int, dtype: str, out_dtype: str, scale: float, rows_per_cta: int, threads: int
):
    """Transform *rows_per_cta* rows a CTA in shared memory; the batch is the dynamic ``total``.

    The first pass reads the input and, where every 32-element run is one segment's, transforms
    element bits {0, 2, 3, 4} of each 8-element chunk with the Sylvester MMA and bit 1 by
    butterfly. Each later pass takes four more bits of the segment index, then the base
    product mixes the segments, and the last pass scales and stores.
    """

    @tilelang.jit(out_idx=[1], pass_configs=_PASS_CONFIGS)
    def _func():
        m = n // base
        log_m = m.bit_length() - 1
        span = rows_per_cta * n
        chunks = span // 8
        use_mma = m >= 32 and dtype in ("float16", "bfloat16")
        rest = [b for b in range(log_m) if not (use_mma and b < 5)]
        # Four bits a pass: sixteen values a thread between two barriers.
        width = 4
        groups = [rest[i : i + width] for i in range(0, len(rest), width)]
        temps, outputs = _base_program(base) if base > 1 else ((), ())
        total = T.dynamic("total", "int64")

        @T.macro
        def bit_pass(buf, tid, low: int, bits: int):
            tasks = span >> bits
            vals = T.alloc_local((1 << width,), "float32")
            for t0 in T.serial(T.ceildiv(tasks, threads)):
                t = t0 * threads + tid
                if t < tasks:
                    lo = t % (1 << low)
                    above = t >> low
                    hi = above % (m >> (low + bits))
                    segment = above // (m >> (low + bits))
                    idx = segment * m + lo + (hi << (low + bits))
                    for j in T.unroll(1 << bits):
                        vals[j] = buf[idx + (j << low)]
                    _butterflies(vals, _strides(bits), 0, 1 << bits)
                    for j in T.unroll(1 << bits):
                        buf[idx + (j << low)] = vals[j]

        @T.macro
        def passes(buf, tid, i: int):
            if i < len(groups):
                bit_pass(buf, tid, groups[i][0], len(groups[i]))
                T.sync_threads()
                passes(buf, tid, i + 1)

        @T.prim_func
        def main(x: T.Tensor((total,), dtype), y: T.Tensor((total,), out_dtype)):
            with T.Kernel(T.ceildiv(total, span), threads=threads) as bx:
                tid = T.get_thread_binding()
                buf = T.alloc_shared((span,), "float32")
                xa = T.alloc_local((8,), dtype)
                hb = T.alloc_local((8,), dtype)
                acc = T.alloc_local((8,), "float32")
                vin = T.alloc_local((max(base, 1),), "float32")
                vout = T.alloc_local((max(base, 1),), "float32")
                tmp = T.alloc_local((max(1, len(temps)),), "float32")
                out = T.alloc_local((8,), out_dtype)
                first = T.cast(bx, "int64") * span
                if use_mma:
                    _sylvester_fragment(hb, tid % 32, 1.0)
                for c0 in T.serial(T.ceildiv(chunks, threads)):
                    c = c0 * threads + tid
                    f = first + c * 8
                    if c < chunks:
                        if f + 8 <= total:
                            for v in T.vectorized(8):
                                xa[v] = x[f + v]
                        else:
                            for v in T.unroll(8):
                                xa[v] = T.if_then_else(f + v < total, x[f + v], T.cast(0, dtype))
                    else:
                        for v in T.unroll(8):
                            xa[v] = T.cast(0, dtype)
                    if use_mma:
                        _first_bits(xa, hb, acc, 1, _mma_dtype(dtype))
                        # Element bit 1 is the MMA row bit 3: registers 2 apart.
                        _butterflies(acc, (2,), 0, 8)
                    else:
                        for i in T.unroll(8):
                            acc[i] = T.cast(xa[i], "float32")
                    if c < chunks:
                        for v in T.vectorized(8):
                            buf[c * 8 + v] = acc[v]
                T.sync_threads()
                passes(buf, tid, 0)
                if base > 1:
                    for t0 in T.serial(T.ceildiv(rows_per_cta * m, threads)):
                        t = t0 * threads + tid
                        if t < rows_per_cta * m:
                            idx = (t // m) * n + t % m
                            for a in T.unroll(base):
                                vin[a] = buf[idx + a * m]
                            _base_temps(vin, tmp, base, 1, temps, 0, 0)
                            _base_outputs(vin, tmp, vout, base, 1, outputs, 0, 0)
                            for c in T.unroll(base):
                                buf[idx + c * m] = vout[c]
                    T.sync_threads()
                for c0 in T.serial(T.ceildiv(chunks, threads)):
                    c = c0 * threads + tid
                    f = first + c * 8
                    if c < chunks:
                        for v in T.vectorized(8):
                            out[v] = T.cast(buf[c * 8 + v] * scale, out_dtype)
                        if f + 8 <= total:
                            for v in T.vectorized(8):
                                y[f + v] = out[v]
                        else:
                            for v in T.unroll(8):
                                if f + v < total:
                                    y[f + v] = out[v]

        return main

    return _func


def _rows_per_cta(n: int) -> int:
    """Rows a shared-rows CTA holds: 4096 elements' worth, a whole number of 16-byte vectors."""
    rows = max(1, 4096 // n)
    while rows * n % 8:
        rows += 1
    return rows


def _shared_rows_threads(n: int, rows: int) -> int:
    """One thread an 8-element chunk, up to a full block."""
    return min(1024, max(32, -(-(rows * n // 8) // 32) * 32))


class HadamardSharedRowsKernel(Kernel, HadamardTransformFwdInterface):
    """Any base order, rows that fit a block's shared memory as float32.

    Args:
        n: Row length.
        base_order: 1, 12, 20, 28 or 40.
        dtype: float16 or bfloat16.
        config: Optional ``{"rows_per_cta", "threads"}`` override.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    preferred_over = frozenset({"hadamard_multi_pass"})

    @classmethod
    def refusal(cls, call: HadamardTransformCall) -> Optional[str]:
        need = 4 * _rows_per_cta(call.n) * call.n
        if need > call.smem_budget:
            return f"a row takes {need} bytes as float32, past the {call.smem_budget} a block has"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: HadamardTransformCall) -> Entry:
        index = call.device.index if call.device is not None else None
        return (call.n, call.base_order, call.dtype, index), lambda: cls(
            call.n, call.base_order, call.dtype, device_index=index
        )

    def __init__(
        self,
        n: int,
        base_order: int,
        dtype: torch.dtype,
        config: Optional[Dict[str, Any]] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.base_order = base_order
        self.dtype = dtype
        self.kernel = functools.partial(
            _hadamard_shared_rows_kernel, n, base_order, self.dtype_str, self.dtype_str,
            1.0 / math.sqrt(n),
        )  # fmt: skip
        self.init_config(config)

    @property
    def default_config(self) -> Dict[str, Any]:
        rows = _rows_per_cta(self.n)
        return {"rows_per_cta": rows, "threads": _shared_rows_threads(self.n, rows)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.empty_like(x)
        flat = vector_aligned(x.contiguous()).view(-1)
        return self.kernel(**self.config)()(flat).view(x.shape)


@functools.lru_cache(maxsize=64)
def _hadamard_bit_pass_kernel(low: int, bits: int, out_dtype: str, scale: float):
    """One pass over float32 rows: element bits [low, low + bits) of every aligned group.

    Thread t takes group t >> low, offset t mod 2**low: the 2**bits elements 2**low apart, so a
    warp reads 32 consecutive words each time. A float32 pass writes its input in place; a
    pass to another dtype writes a new tensor.
    """
    in_place = out_dtype == "float32"

    @tilelang.jit(out_idx=None if in_place else [1], pass_configs=_PASS_CONFIGS)
    def _func():
        total = T.dynamic("total", "int64")
        # 256 threads, sixteen values each.
        threads = 256

        @T.macro
        def body(x, y):
            with T.Kernel(T.ceildiv(total >> bits, threads), threads=threads) as bx:
                vals = T.alloc_local((1 << bits,), "float32")
                t = T.cast(bx, "int64") * threads + T.get_thread_binding()
                if t < total >> bits:
                    idx = (t >> low << (low + bits)) + (t & ((1 << low) - 1))
                    for j in T.unroll(1 << bits):
                        vals[j] = x[idx + (j << low)]
                    _butterflies(vals, _strides(bits), 0, 1 << bits)
                    for j in T.unroll(1 << bits):
                        y[idx + (j << low)] = T.cast(vals[j] * scale, out_dtype)

        if in_place:

            @T.prim_func
            def main(x: T.Tensor((total,), "float32")):
                body(x, x)

        else:

            @T.prim_func
            def main(x: T.Tensor((total,), "float32"), y: T.Tensor((total,), out_dtype)):
                body(x, y)

        return main

    return _func


@functools.lru_cache(maxsize=16)
def _hadamard_base_pass_kernel(n: int, base: int, out_dtype: str, scale: float):
    """The base product over float32 rows: thread t mixes row t // m's segments at t mod m."""

    @tilelang.jit(out_idx=[1], pass_configs=_PASS_CONFIGS)
    def _func():
        m = n // base
        temps, outputs = _base_program(base)
        total = T.dynamic("total", "int64")
        # 256 threads, one segment position of one row each.
        threads = 256

        @T.prim_func
        def main(x: T.Tensor((total,), "float32"), y: T.Tensor((total,), out_dtype)):
            with T.Kernel(T.ceildiv(total // base, threads), threads=threads) as bx:
                vin = T.alloc_local((base,), "float32")
                vout = T.alloc_local((base,), "float32")
                tmp = T.alloc_local((len(temps),), "float32")
                t = T.cast(bx, "int64") * threads + T.get_thread_binding()
                if t < total // base:
                    idx = t // m * n + t % m
                    for a in T.unroll(base):
                        vin[a] = x[idx + a * m]
                    _base_temps(vin, tmp, base, 1, temps, 0, 0)
                    _base_outputs(vin, tmp, vout, base, 1, outputs, 0, 0)
                    for c in T.unroll(base):
                        y[idx + c * m] = T.cast(vout[c] * scale, out_dtype)

        return main

    return _func


class HadamardMultiPassKernel(Kernel, HadamardTransformFwdInterface):
    """Any served row: shared-memory chunks first, then float32 passes over global memory.

    The first pass transforms contiguous chunks of up to 8192 elements of each segment into a
    float32 copy; each later pass takes four more bits of the segment index, and the base
    product, where there is one, comes last. Each call allocates one float32 buffer the size
    of the input, which every pass but the last writes in place.

    Args:
        n: Row length.
        base_order: 1, 12, 20, 28 or 40.
        dtype: float16 or bfloat16.
        device_index: The device the kernel runs on.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    general = True

    @classmethod
    def entry_for(cls, call: HadamardTransformCall) -> Entry:
        index = call.device.index if call.device is not None else None
        return (call.n, call.base_order, call.dtype, index), lambda: cls(
            call.n, call.base_order, call.dtype, device_index=index
        )

    def __init__(
        self,
        n: int,
        base_order: int,
        dtype: torch.dtype,
        config: Optional[Dict[str, Any]] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.n = n
        self.base_order = base_order
        self.dtype = dtype
        m = n // base_order
        # 8192 float32 values, 32 KB, a chunk: the shared-rows pass at four rows' width.
        chunk = min(m, 8192)
        log_chunk, log_m = chunk.bit_length() - 1, m.bit_length() - 1
        scale = 1.0 / math.sqrt(n)
        stages = [(low, min(4, log_m - low)) for low in range(log_chunk, log_m, 4)]
        # The last stage scales and rounds to the output dtype: the base product, else the last
        # bit pass, else the chunk pass itself.
        last = len(stages) - 1 if base_order == 1 else -1
        chunk_out = (self.dtype_str, scale) if base_order == 1 and not stages else ("float32", 1.0)
        self._chunk = _hadamard_shared_rows_kernel(
            chunk, 1, self.dtype_str, *chunk_out, _rows_per_cta(chunk),
            _shared_rows_threads(chunk, _rows_per_cta(chunk)),
        )  # fmt: skip
        # (program, writes its input in place): every bit pass but a last one that rounds.
        self._passes = [
            (
                _hadamard_bit_pass_kernel(
                    low,
                    bits,
                    self.dtype_str if i == last else "float32",
                    scale if i == last else 1.0,
                ),
                i != last,
            )
            for i, (low, bits) in enumerate(stages)
        ]
        self._base = (
            _hadamard_base_pass_kernel(n, base_order, self.dtype_str, scale)
            if base_order > 1
            else None
        )
        self.kernel = None
        self.init_config(config)

    @property
    def default_config(self) -> Dict[str, Any]:
        return {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.empty_like(x)
        flat = vector_aligned(x.contiguous()).view(-1)
        # One float32 copy of the input lives through every pass; the passes before the last
        # write it in place.
        y = self._chunk()(flat)
        for program, in_place in self._passes:
            if in_place:
                program()(y)
            else:
                y = program()(y)
        if self._base is not None:
            y = self._base()(y)
        return y.view(x.shape)
