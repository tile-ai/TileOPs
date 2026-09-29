"""Per-tensor symmetric INT8 quantization in one cooperative launch."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import (
    BLOCK_SHARED_BYTES_OPT_IN,
    SHARED_BUFFER_ALIGN_BYTES,
    VECTOR_ACCESS_BYTES,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import INT8QuantPerTensorFwdInterface, QuantizeCall
from tileops.kernels.quantization.int8_codes import (
    INV_QMAX,
    SCALE_UP,
    SMALL_SCALE,
    abs_bits,
    quantize,
    widen,
)
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["INT8QuantPerTensorFwdKernel"]


@functools.lru_cache(maxsize=32)
def _int8_quant_per_tensor_kernel(n: int, dtype: str, grid: int, threads: int):
    """Build the quantization of ``n`` contiguous elements on ``grid`` CTAs.

    A tile is one 16-byte vector per thread. CTA ``b`` owns tiles ``j * grid + b``; of its
    whole tiles the first ``reg_tiles`` stay in registers and the next ``smem_tiles`` in
    shared memory across the grid barrier, and the rest are read again after it. ``batch``
    is how many vector loads a thread issues before using any.
    """
    vec = VECTOR_ACCESS_BYTES // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    tile = threads * vec
    tiles = n // tile
    tail = n - tiles * tile
    whole = tiles // grid
    ragged = tiles - whole * grid
    warps = threads // WARP_LANES
    # A bfloat16 pair is held as one 32-bit word: each half widens to float32 in one bit
    # operation, where extracting a 16-bit element first costs a second.
    words = dtype == "bfloat16"
    held = "uint32" if words else dtype
    width = vec // 2 if words else vec

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _int8_quant_per_tensor_func(reg_tiles: int, smem_tiles: int, batch: int):
        reg = min(reg_tiles, whole)
        smem = min(smem_tiles, whole - reg)
        streamed = whole - reg - smem
        rounds = streamed // batch
        rest = streamed - rounds * batch

        def code(value, num, prescale: bool):
            # ``num`` holds the pre-scale factor, then the scale and its reciprocal after
            # that factor.
            rounded = quantize(value * num[0] if prescale else value, num[1], num[2], prescale)
            if not prescale:
                return rounded
            # A scale that underflowed to zero: torch divides by it, so a nonzero value
            # clamps to +-127 and a zero lands on 0.
            signed = T.if_then_else(
                value > 0, T.int8(127), T.if_then_else(value < 0, T.int8(-127), T.int8(0))
            )
            return T.if_then_else(num[1] == T.float32(0), signed, rounded)

        @T.macro
        def load_vector(dst, row, src, offset, evict_first: bool):
            # evict_first for data nothing reads from memory again.
            if evict_first:
                T.call_extern(
                    "handle",
                    "tl::tileops_load16_evict_first",
                    T.address_of(dst[row, 0]),
                    T.address_of(src[offset]),
                )
            else:
                T.call_extern(
                    "handle",
                    "tl::tileops_load16",
                    T.address_of(dst[row, 0]),
                    T.address_of(src[offset]),
                )

        @T.macro
        def block_max(val, warp_max):
            # Leaves the maximum of val[0] over the block in val[0] of every thread.
            tx = T.get_thread_binding()
            for stage in T.serial(WARP_SHUFFLE_STAGES):
                val[0] = T.max(
                    val[0], T.shfl_xor(val[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES)
                )
            if tx % WARP_LANES == 0:
                warp_max[tx // WARP_LANES] = val[0]
            T.sync_threads()
            val[0] = warp_max[0]
            for w in T.serial(1, warps):
                val[0] = T.max(val[0], warp_max[w])

        @T.macro
        def fold(acc, values, row):
            for c in T.serial(width):
                if words:
                    acc[2 * c] = T.max(acc[2 * c], abs_bits(widen(values[row, c], 0, dtype)))
                    acc[2 * c + 1] = T.max(
                        acc[2 * c + 1], abs_bits(widen(values[row, c], 1, dtype))
                    )
                else:
                    acc[c] = T.max(acc[c], abs_bits(T.cast(values[row, c], "float32")))

        @T.macro
        def store(q, out, num, offset, values, row, prescale: bool):
            for c in T.serial(width):
                if words:
                    out[2 * c] = code(widen(values[row, c], 0, dtype), num, prescale)
                    out[2 * c + 1] = code(widen(values[row, c], 1, dtype), num, prescale)
                else:
                    out[c] = code(T.cast(values[row, c], "float32"), num, prescale)
            for c in T.vectorized(vec):
                q[offset + c] = out[c]

        @T.macro
        def quantize_all(x, q, cur, out, in_regs, in_smem, extra, num, bx, tx, prescale: bool):
            # The most recently read tiles first, so that the re-read ones find L2
            # warm; nothing reads them again, hence evict-first.
            if ragged > 0 and bx < ragged:
                store(q, out, num, (whole * grid + bx) * tile + tx * vec, extra, 0, prescale)
            if tail > 0 and bx == grid - 1:
                if tx < tail // vec:
                    store(q, out, num, tiles * tile + tx * vec, extra, 0, prescale)
                elif tail % vec > 0 and tx == tail // vec:
                    # The last, partial vector of the tail, element by element.
                    offset = tiles * tile + tx * vec
                    for c in T.unroll(tail % vec):
                        if words:
                            if c % 2 == 0:
                                q[offset + c] = code(
                                    widen(extra[0, c // 2], 0, dtype), num, prescale
                                )
                            else:
                                q[offset + c] = code(
                                    widen(extra[0, c // 2], 1, dtype), num, prescale
                                )
                        else:
                            q[offset + c] = code(T.cast(extra[0, c], "float32"), num, prescale)
            first = reg + smem + rounds * batch
            for i in T.unroll(rest):
                load_vector(cur, i, x, ((first + i) * grid + bx) * tile + tx * vec, True)
            for i in T.unroll(rest):
                store(q, out, num, ((first + i) * grid + bx) * tile + tx * vec, cur, i, prescale)
            for r in T.serial(rounds):
                for i in T.unroll(batch):
                    load_vector(
                        cur,
                        i,
                        x,
                        ((reg + smem + (rounds - 1 - r) * batch + i) * grid + bx) * tile + tx * vec,
                        True,
                    )
                for i in T.unroll(batch):
                    store(
                        q,
                        out,
                        num,
                        ((reg + smem + (rounds - 1 - r) * batch + i) * grid + bx) * tile + tx * vec,
                        cur,
                        i,
                        prescale,
                    )
            for j in T.unroll(smem, unroll_factor=batch):
                for c in T.vectorized(width):
                    cur[0, c] = in_smem[j, tx * width + c]
                store(q, out, num, ((reg + j) * grid + bx) * tile + tx * vec, cur, 0, prescale)
            for j in T.unroll(reg):
                store(q, out, num, (j * grid + bx) * tile + tx * vec, in_regs, j, prescale)

        @T.prim_func
        def _int8_quant_per_tensor_main(
            x: T.Tensor((n,), dtype),
            partial: T.Tensor((grid,), "int32"),
            q: T.Tensor((n,), "int8"),
            scale: T.Tensor((1,), "float32"),
        ):
            with T.Kernel(grid, threads=threads) as bx:
                tx = T.get_thread_binding()
                in_regs = T.alloc_local((max(reg, 1), width), held)
                in_smem = T.alloc_shared((max(smem, 1), threads * width), held)
                cur = T.alloc_local((batch, width), held)
                extra = T.alloc_local((1, width), held)
                acc = T.alloc_local((vec,), "int32")
                out = T.alloc_local((vec,), "int8")
                val = T.alloc_local((1,), "int32")
                num = T.alloc_local((3,), "float32")
                warp_max = T.alloc_shared((warps,), "int32")

                for c in T.serial(vec):
                    acc[c] = 0
                for c in T.serial(width):
                    extra[0, c] = T.cast(0, held)
                # The ragged tile, or the tail past the last whole tile, which is shorter
                # than a tile and ends in a partial vector when ``n`` is not a whole number
                # of vectors. Only the last CTA has a tail, and it has no ragged tile. Both
                # are read first and held in registers.
                if ragged > 0 and bx < ragged:
                    load_vector(extra, 0, x, (whole * grid + bx) * tile + tx * vec, True)
                if tail > 0 and bx == grid - 1:
                    if tx < tail // vec:
                        load_vector(extra, 0, x, tiles * tile + tx * vec, True)
                    elif tail % vec > 0 and tx == tail // vec:
                        # Element by element; the rest of the vector stays 0.
                        offset = tiles * tile + tx * vec
                        for c in T.unroll(tail % vec):
                            if words:
                                bits = T.cast(T.reinterpret(x[offset + c], "uint16"), "uint32")
                                if c % 2 == 0:
                                    extra[0, c // 2] = extra[0, c // 2] | bits
                                else:
                                    extra[0, c // 2] = extra[0, c // 2] | (bits << T.uint32(16))
                            else:
                                extra[0, c] = x[offset + c]
                # Phase 1 reads every tile once. The held ones are read evict-first, since
                # nothing reads them from memory again; the streamed ones stay cacheable for
                # phase 2.
                for j in T.unroll(reg):
                    load_vector(in_regs, j, x, (j * grid + bx) * tile + tx * vec, True)
                for j in T.unroll(reg):
                    fold(acc, in_regs, j)
                for r in T.serial(T.ceildiv(smem, batch)):
                    for i in T.unroll(batch):
                        if r * batch + i < smem:
                            load_vector(
                                cur,
                                i,
                                x,
                                ((reg + r * batch + i) * grid + bx) * tile + tx * vec,
                                True,
                            )
                    for i in T.unroll(batch):
                        if r * batch + i < smem:
                            fold(acc, cur, i)
                            for c in T.vectorized(width):
                                in_smem[r * batch + i, tx * width + c] = cur[i, c]
                for r in T.serial(rounds):
                    for i in T.unroll(batch):
                        load_vector(
                            cur,
                            i,
                            x,
                            ((reg + smem + r * batch + i) * grid + bx) * tile + tx * vec,
                            False,
                        )
                    for i in T.unroll(batch):
                        fold(acc, cur, i)
                for i in T.unroll(rest):
                    load_vector(
                        cur,
                        i,
                        x,
                        ((reg + smem + rounds * batch + i) * grid + bx) * tile + tx * vec,
                        False,
                    )
                for i in T.unroll(rest):
                    fold(acc, cur, i)
                if ragged > 0 and bx < ragged:
                    fold(acc, extra, 0)
                if tail > 0 and bx == grid - 1:
                    fold(acc, extra, 0)
                val[0] = acc[0]
                for c in T.serial(1, vec):
                    val[0] = T.max(val[0], acc[c])
                block_max(val, warp_max)
                if tx == 0:
                    partial[bx] = val[0]

                T.sync_grid()

                val[0] = 0
                for i in T.serial(T.ceildiv(grid, threads)):
                    if i * threads + tx < grid:
                        val[0] = T.max(val[0], partial[i * threads + tx])
                T.sync_threads()
                block_max(val, warp_max)
                num[1] = T.reinterpret(val[0], "float32") * T.float32(INV_QMAX)
                num[1] = T.if_then_else(
                    T.reinterpret(val[0], "float32") > 0, num[1], T.float32(1.0)
                )
                if bx == 0 and tx == 0:
                    scale[0] = num[1]
                if num[1] < T.float32(SMALL_SCALE):
                    num[0] = T.float32(SCALE_UP)
                    num[1] = num[1] * num[0]
                    num[2] = T.ieee_frcp(num[1])
                    quantize_all(x, q, cur, out, in_regs, in_smem, extra, num, bx, tx, True)
                else:
                    num[2] = T.ieee_frcp(num[1])
                    quantize_all(x, q, cur, out, in_regs, in_smem, extra, num, bx, tx, False)

        return _int8_quant_per_tensor_main

    return _int8_quant_per_tensor_func


class INT8QuantPerTensorFwdKernel(Kernel, INT8QuantPerTensorFwdInterface):
    """Quantize ``x`` against one scale in one cooperative launch, reading ``x`` about once.

    A grid-wide amax has to finish before any value is quantized, so the launch holds as
    much of ``x`` on chip as shared memory and registers take across a grid barrier and
    reads only the rest a second time. ``q`` is bit-equal to the torch reference: the scale is the product torch
    computes, and the quotient is correctly rounded.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``reg_tiles``, ``smem_tiles`` and ``batch``.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [90]
    # The integer tensors are the partials and ``q``, both written before anything reads them.
    autotune_accepts_random_int_inputs = True

    # Threads of a CTA, one CTA per SM since the barrier needs the grid resident. A smaller
    # input halves it, down to _MIN_THREADS, until the whole tiles fill a quarter of the
    # SMs. Fitted with the repo benchmark: 512 is fastest from [1000, 4099] up (256 costs
    # 5-8%, 1024 is no faster), and [17, 2880] runs 3.17 us at 512 and 2.98 at 128. Re-fit
    # by timing the manifest rows at 128, 256, 512 and 1024.
    _THREADS: ClassVar[int] = 512
    _MIN_THREADS: ClassVar[int] = 128
    # Most tiles a thread keeps in registers across the barrier, which hold only what shared
    # memory cannot: a register tile ties up registers the re-read loop needs ([3000, 2880]
    # bfloat16, which shared memory holds whole, takes 10.5 us with no register tile and
    # 11.5 with 8). The cap is the most that ptxas compiles without spilling at
    # _THREADS threads; float16's conversions spill above 8. Re-fit by raising a cap until
    # `ptxas -v` reports spills, then timing the [8192, 5120] rows.
    _REG_TILES: ClassVar[dict[torch.dtype, int]] = {
        torch.float16: 8,
        torch.bfloat16: 16,
        torch.float32: 16,
    }
    # Vector loads a thread keeps in flight, fitted on the manifest rows: 4 measures up to
    # 2.5% slower and 16 no faster. Re-fit by timing 4, 8 and 16.
    _BATCH: ClassVar[int] = 8

    @classmethod
    def refusal(cls, call: QuantizeCall) -> Optional[str]:
        reason = super().refusal(call)
        # Offsets are int32 and run up to one tile past M * K.
        largest_tile = cls._THREADS * VECTOR_ACCESS_BYTES
        if reason is None and call.rows * call.cols > 2**31 - 1 - largest_tile:
            return f"indexes elements with int32, and M * K = {call.rows * call.cols}"
        return reason

    def __init__(
        self, call: QuantizeCall, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.dtype = call.dtype
        self.numel = call.rows * call.cols
        vec = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        threads = self._THREADS
        while threads > self._MIN_THREADS and self.numel // (threads * vec) < call.sm_count // 4:
            threads //= 2
        self._grid = max(1, min(call.sm_count, self.numel // (threads * vec)))
        # The reduction scratch takes one aligned shared buffer of the budget.
        smem_bytes = BLOCK_SHARED_BYTES_OPT_IN[call.arch] - SHARED_BUFFER_ALIGN_BYTES
        self._smem_tiles = smem_bytes // (threads * VECTOR_ACCESS_BYTES)
        whole_tiles = self.numel // (threads * vec) // self._grid
        self._reg_tiles = min(max(whole_tiles - self._smem_tiles, 0), self._REG_TILES[self.dtype])
        self.kernel = _int8_quant_per_tensor_kernel(self.numel, self.dtype_str, self._grid, threads)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"reg_tiles": self._reg_tiles, "smem_tiles": self._smem_tiles, "batch": self._BATCH}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The kernel reads 16-byte vectors from the start of the storage.
        if x.data_ptr() % VECTOR_ACCESS_BYTES:
            x = x.clone()
        partial = torch.empty(self._grid, dtype=torch.int32, device=x.device)
        q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        scale = torch.empty(1, dtype=torch.float32, device=x.device)
        self.kernel(self.config["reg_tiles"], self.config["smem_tiles"], self.config["batch"])(
            x.view(-1), partial, q.view(-1), scale
        )
        return q, scale
