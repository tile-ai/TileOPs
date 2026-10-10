"""Per-block symmetric INT8 quantization: a group of lanes owns each 128-element block."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.constants import QUANT_SCALE_BLOCK, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel, vector_aligned
from tileops.kernels.quantization.call_spec import INT8QuantPerBlockFwdInterface, QuantizeCall
from tileops.kernels.quantization.int8_codes import (
    INV_QMAX,
    SCALE_UP,
    SMALL_SCALE,
    quantize,
    widen,
)
from tileops.utils import WARP_LANES

__all__ = ["INT8QuantPerBlockFwdKernel", "INT8QuantPerBlockShiftedFwdKernel"]


@functools.lru_cache(maxsize=32)
def _int8_quant_per_block_kernel(m: int, k: int, dtype: str):
    """Build the quantization of ``m`` rows of ``k`` elements whose blocks start on a vector.

    A block is its 16-byte vectors laid end to end from its first element, so a vector of
    ``x`` and the codes it produces are both aligned; a partial last block covers fewer
    vectors. ``lanes`` adjacent lanes own a block and reduce its amax by shuffles; a lane
    holds its vectors in runs of ``pack`` adjacent ones and stores their codes at once.
    """
    n = m * k
    nb = -(-k // QUANT_SCALE_BLOCK)
    blocks = m * nb
    itemsize = torch.empty((), dtype=getattr(torch, dtype)).element_size()
    vec = VECTOR_ACCESS_BYTES // itemsize
    # A vector is held as 32-bit words, ``per_word`` elements to a word.
    words = VECTOR_ACCESS_BYTES // 4
    per_word = 4 // itemsize
    whole = k % QUANT_SCALE_BLOCK == 0
    # The bits of a word's elements that order as their magnitudes do; a NaN orders above
    # every number, as torch's amax takes it.
    magnitude = 0x7FFF7FFF if per_word == 2 else 0x7FFFFFFF

    @tilelang.jit(compile_flags=csrc_include("streaming_load.h"))
    def _int8_quant_per_block_func(threads: int, lanes: int, pack: int):
        groups = threads // lanes
        slots = QUANT_SCALE_BLOCK // vec // lanes
        assert slots * lanes * vec == QUANT_SCALE_BLOCK and slots % pack == 0
        assert pack * vec <= VECTOR_ACCESS_BYTES
        # Every slot holds a vector of its block, and every block is in the grid.
        exact = whole and blocks % groups == 0

        def window(s, tx):
            """The vector of its block that slot ``s`` of lane ``tx % lanes`` holds."""
            return (s // pack * lanes + tx % lanes) * pack + s % pack

        def extent(bx, tx):
            """The lane's block, its first element and its count of vectors."""
            b = bx * groups + tx // lanes
            if whole:
                return b, b * QUANT_SCALE_BLOCK, QUANT_SCALE_BLOCK // vec
            j = b % nb
            return (
                b,
                b // nb * k + j * QUANT_SCALE_BLOCK,
                T.min(QUANT_SCALE_BLOCK, k - j * QUANT_SCALE_BLOCK) // vec,
            )

        @T.macro
        def code_one(out, e, value, num, prescale: bool):
            if prescale:
                out[e] = quantize(value * T.float32(SCALE_UP), num[1], num[2], True)
            else:
                out[e] = quantize(value, num[1], num[2], False)

        @T.macro
        def store(q, values, out, bx, tx, num, prescale: bool):
            b, lo, count = extent(bx, tx)
            for g in T.unroll(slots // pack):
                w = window(g * pack, tx)
                if exact or ((b < blocks) & (w < count)):
                    for i in T.unroll(pack):
                        for c in T.unroll(words):
                            word = values[g * pack + i, c]
                            e = i * vec + c * per_word
                            code_one(out, e, widen(word, 0, dtype), num, prescale)
                            if per_word == 2:
                                code_one(out, e + 1, widen(word, 1, dtype), num, prescale)
                    if exact or (w + pack <= count):
                        for e in T.vectorized(pack * vec):
                            q[lo + w * vec + e] = out[e]
                    else:
                        # A partial last block may end inside a lane's run.
                        for i in T.unroll(pack):
                            if w + i < count:
                                for e in T.vectorized(vec):
                                    q[lo + (w + i) * vec + e] = out[i * vec + e]

        @T.prim_func
        def _int8_quant_per_block_main(
            x: T.Tensor((n,), dtype),
            q: T.Tensor((n,), "int8"),
            scale: T.Tensor((blocks,), "float32"),
        ):
            with T.Kernel(T.ceildiv(blocks, groups), threads=threads) as bx:
                tx = T.get_thread_binding()
                values = T.alloc_local((slots, words), "uint32")
                acc = T.alloc_local((1,), "uint32")
                out = T.alloc_local((pack * vec,), "int8")
                num = T.alloc_local((3,), "float32")

                b, lo, count = extent(bx, tx)
                for s in T.unroll(slots):
                    if exact or ((b < blocks) & (window(s, tx) < count)):
                        T.call_extern(
                            "handle",
                            "tl::tileops_load16_evict_first",
                            T.address_of(values[s, 0]),
                            T.address_of(x[lo + window(s, tx) * vec]),
                        )
                acc[0] = T.uint32(0)
                for s in T.unroll(slots):
                    if exact or ((b < blocks) & (window(s, tx) < count)):
                        for c in T.unroll(words):
                            bits = values[s, c] & T.uint32(magnitude)
                            if per_word == 2:
                                acc[0] = T.call_extern("uint32", "__vmaxu2", acc[0], bits)
                            else:
                                acc[0] = T.max(acc[0], bits)
                if per_word == 2:
                    acc[0] = T.max(acc[0] & T.uint32(0xFFFF), acc[0] >> T.uint32(16))
                # The lanes of a group are adjacent and aligned, so XOR offsets below
                # ``lanes`` stay inside it; every lane of the warp takes part.
                for stage in T.unroll(lanes.bit_length() - 1):
                    acc[0] = T.max(
                        acc[0], T.shfl_xor(acc[0], T.int32(lanes // 2) >> stage, width=WARP_LANES)
                    )
                amax = widen(acc[0], 0, dtype)
                num[0] = T.if_then_else(amax > 0, amax * T.float32(INV_QMAX), T.float32(1.0))
                if (tx % lanes == 0) & (exact or (b < blocks)):
                    scale[b] = num[0]
                if num[0] < T.float32(SMALL_SCALE):
                    num[1] = num[0] * T.float32(SCALE_UP)
                    num[2] = T.ieee_frcp(num[1])
                    store(q, values, out, bx, tx, num, True)
                else:
                    num[1] = num[0]
                    num[2] = T.ieee_frcp(num[1])
                    store(q, values, out, bx, tx, num, False)

        return _int8_quant_per_block_main

    return _int8_quant_per_block_func


@functools.lru_cache(maxsize=32)
def _int8_quant_per_block_shifted_kernel(m: int, k: int, dtype: str):
    """Build the quantization of ``m`` rows of ``k`` elements whose blocks start inside a vector.

    ``lanes`` adjacent lanes own a block and each lane a run of it, whole vectors' worth
    long: the lane loads the aligned vectors its run straddles and shifts the run into
    place. A CTA's codes are assembled in shared memory at their flat offsets and stored as
    whole 16-byte chunks; only the run's first and last chunk are stored byte by byte, since
    the CTAs beside it own the rest of them.
    """
    n = m * k
    nb = -(-k // QUANT_SCALE_BLOCK)
    blocks = m * nb
    itemsize = torch.empty((), dtype=getattr(torch, dtype)).element_size()
    vec = VECTOR_ACCESS_BYTES // itemsize
    words = VECTOR_ACCESS_BYTES // 4
    per_word = 4 // itemsize
    ragged_end = n % vec != 0
    magnitude = 0x7FFF7FFF if per_word == 2 else 0x7FFFFFFF

    @tilelang.jit(compile_flags=csrc_include("streaming_load.h"))
    def _int8_quant_per_block_shifted_func(threads: int, lanes: int):
        groups = threads // lanes
        # Vectors' worth of elements in a lane's run, and the words that hold them.
        vpl = QUANT_SCALE_BLOCK // vec // lanes
        span = vpl * vec
        held = vpl * words
        assert vpl * lanes * vec == QUANT_SCALE_BLOCK
        # 16-byte chunks of ``q`` a CTA's run spans at most.
        chunks = groups * QUANT_SCALE_BLOCK // VECTOR_ACCESS_BYTES + 2

        def extent(b):
            """The first element of block ``b`` and one past its last."""
            lo = b // nb * k + b % nb * QUANT_SCALE_BLOCK
            return lo, T.min(lo + QUANT_SCALE_BLOCK, b // nb * k + k)

        def pick(raw, at, skip):
            """Word ``at + skip`` of the vectors ``raw`` holds."""
            word = raw[at]
            for d in range(1, words):
                word = T.if_then_else(skip == d, raw[at + d], word)
            return word

        @T.macro
        def code_one(staged, at, e, count, value, num, prescale: bool, guarded: bool):
            if (not guarded) or (e < count):
                if prescale:
                    staged[at + e] = quantize(value * T.float32(SCALE_UP), num[1], num[2], True)
                else:
                    staged[at + e] = quantize(value, num[1], num[2], False)

        @T.macro
        def codes(staged, own, at, count, num, prescale: bool, guarded: bool):
            for c in T.unroll(held):
                value = widen(own[c], 0, dtype)
                code_one(staged, at, c * per_word, count, value, num, prescale, guarded)
                if per_word == 2:
                    value = widen(own[c], 1, dtype)
                    code_one(staged, at, c * 2 + 1, count, value, num, prescale, guarded)

        @T.macro
        def quantize_run(staged, own, at, count, num, prescale: bool):
            # Only a lane of a partial last block holds fewer than ``span`` elements.
            if count == span:
                codes(staged, own, at, count, num, prescale, False)
            else:
                codes(staged, own, at, count, num, prescale, True)

        @T.prim_func
        def _int8_quant_per_block_shifted_main(
            x: T.Tensor((n,), dtype),
            q: T.Tensor((n,), "int8"),
            scale: T.Tensor((blocks,), "float32"),
        ):
            with T.Kernel(T.ceildiv(blocks, groups), threads=threads) as bx:
                tx = T.get_thread_binding()
                raw = T.alloc_local((held + words,), "uint32")
                own = T.alloc_local((held,), "uint32")
                acc = T.alloc_local((1,), "uint32")
                num = T.alloc_local((3,), "float32")
                staged = T.alloc_shared((chunks * VECTOR_ACCESS_BYTES,), "int8")

                b = bx * groups + tx // lanes
                lo, hi = extent(b)
                first = lo + tx % lanes * span
                run_lo = extent(bx * groups)[0]
                run_hi = extent(T.min(bx * groups + groups, blocks) - 1)[1]
                base = run_lo // VECTOR_ACCESS_BYTES * VECTOR_ACCESS_BYTES

                for c in T.serial(held + words):
                    raw[c] = T.uint32(0)
                for j in T.unroll(vpl + 1):
                    # A run that starts inside a vector ends inside the one after its last.
                    v = first // vec + j
                    if (b < blocks) & (v * vec < hi) & ((j < vpl) | (first % vec != 0)):
                        if ragged_end and (v + 1) * vec > n:
                            for e in T.unroll(vec):
                                if v * vec + e < n:
                                    bits = T.reinterpret(
                                        x[v * vec + e], "uint16" if per_word == 2 else "uint32"
                                    )
                                    shift = T.uint32(32 // per_word * (e % per_word))
                                    raw[j * words + e // per_word] = raw[
                                        j * words + e // per_word
                                    ] | (T.cast(bits, "uint32") << shift)
                        else:
                            T.call_extern(
                                "handle",
                                "tl::tileops_load16_evict_first",
                                T.address_of(raw[j * words]),
                                T.address_of(x[v * vec]),
                            )
                # The run, shifted to the start of ``own``: ``skip`` whole words, then
                # ``part`` elements of a word.
                skip = first % vec // per_word
                part = first % vec % per_word
                for c in T.unroll(held):
                    if per_word == 2:
                        own[c] = T.if_then_else(
                            part == 1,
                            (pick(raw, c, skip) >> T.uint32(16))
                            | (pick(raw, c + 1, skip) << T.uint32(16)),
                            pick(raw, c, skip),
                        )
                    else:
                        own[c] = pick(raw, c, skip)
                count = T.max(T.min(hi - first, span), 0)
                acc[0] = T.uint32(0)
                for c in T.unroll(held):
                    # A partial last block masks the elements past its end.
                    if per_word == 2:
                        keep = T.if_then_else(
                            c * 2 + 1 < count,
                            T.uint32(magnitude),
                            T.if_then_else(c * 2 < count, T.uint32(0x7FFF), T.uint32(0)),
                        )
                        acc[0] = T.call_extern("uint32", "__vmaxu2", acc[0], own[c] & keep)
                    else:
                        keep = T.if_then_else(c < count, T.uint32(magnitude), T.uint32(0))
                        acc[0] = T.max(acc[0], own[c] & keep)
                if per_word == 2:
                    acc[0] = T.max(acc[0] & T.uint32(0xFFFF), acc[0] >> T.uint32(16))
                # Every lane of the warp takes part in the shuffles.
                for stage in T.unroll(lanes.bit_length() - 1):
                    acc[0] = T.max(
                        acc[0], T.shfl_xor(acc[0], T.int32(lanes // 2) >> stage, width=WARP_LANES)
                    )
                amax = widen(acc[0], 0, dtype)
                num[0] = T.if_then_else(amax > 0, amax * T.float32(INV_QMAX), T.float32(1.0))
                if b < blocks:
                    if tx % lanes == 0:
                        scale[b] = num[0]
                    if num[0] < T.float32(SMALL_SCALE):
                        num[1] = num[0] * T.float32(SCALE_UP)
                        num[2] = T.ieee_frcp(num[1])
                        quantize_run(staged, own, first - base, count, num, True)
                    else:
                        num[1] = num[0]
                        num[2] = T.ieee_frcp(num[1])
                        quantize_run(staged, own, first - base, count, num, False)
                T.sync_threads()
                last = (run_hi - base + VECTOR_ACCESS_BYTES - 1) // VECTOR_ACCESS_BYTES
                for i in T.unroll(T.ceildiv(chunks, threads)):
                    c = i * threads + tx
                    if c < last:
                        at = base + c * VECTOR_ACCESS_BYTES
                        # The run's first and last chunk hold codes of the CTAs beside it.
                        if (c == 0) | (c == last - 1):
                            for e in T.unroll(VECTOR_ACCESS_BYTES):
                                if (at + e >= run_lo) & (at + e < run_hi):
                                    q[at + e] = staged[c * VECTOR_ACCESS_BYTES + e]
                        else:
                            for e in T.vectorized(VECTOR_ACCESS_BYTES):
                                q[at + e] = staged[c * VECTOR_ACCESS_BYTES + e]

        return _int8_quant_per_block_shifted_main

    return _int8_quant_per_block_shifted_func


class _INT8QuantPerBlockFwdKernel(Kernel, INT8QuantPerBlockFwdInterface):
    """What the two per-block kernels share: the calls they refuse and how they launch."""

    supported_archs: list[int] = [80, 86, 89, 90]
    # The one integer tensor is ``q``, written before anything reads it.
    autotune_accepts_random_int_inputs = True

    @classmethod
    def refusal(cls, call: QuantizeCall) -> Optional[str]:
        reason = super().refusal(call)
        # Element indices reach one block past M * K.
        if reason is None and call.rows * call.cols > 2**31 - 1 - QUANT_SCALE_BLOCK:
            return f"indexes elements with int32, and M * K = {call.rows * call.cols}"
        return reason

    def __init__(
        self, call: QuantizeCall, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self.kernel = self._builder(call.rows, call.cols, self.dtype_str)
        self.init_config(config, tune)

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_cuda(x=x)
        q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        scale = torch.empty(
            (self.call.rows, -(-self.call.cols // QUANT_SCALE_BLOCK)),
            dtype=torch.float32,
            device=x.device,
        )
        # The kernel reads 16-byte vectors from the start of the storage.
        x = vector_aligned(x)
        self.kernel(**self.config)(x.view(-1), q.view(-1), scale.view(-1))
        return q, scale


class INT8QuantPerBlockFwdKernel(_INT8QuantPerBlockFwdKernel):
    """Quantize each 128-element block of ``x`` against its own amax, reading ``x`` once.

    Serves a K whose blocks start on a 16-byte vector. A group of adjacent lanes owns a
    block: each lane holds some of its vectors in registers, the group reduces the block's
    amax by warp shuffles, and each lane quantizes its vectors from the registers. ``q`` and
    ``scale`` are bit-equal to the torch reference: the scale is the product torch
    computes, and the quotient is correctly rounded.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads``, ``lanes`` and ``pack``.
        tune: Whether to autotune.
    """

    _builder = staticmethod(_int8_quant_per_block_kernel)

    # Launch policy, fitted on the manifest rows with the repo benchmark. Re-fit lanes in
    # {4, 8, 16} and pack in {1, 2} where lanes * pack * 16 bytes fits a block.
    _LANES: ClassVar[int] = 8
    _PACK: ClassVar[int] = 2
    # Re-fit threads in {128, 256, 512} on the decode rows and on the prefill rows.
    _THREADS: ClassVar[int] = 256
    _WIDE_THREADS: ClassVar[int] = 512
    # Grids that fill the device's resident threads this many times over take _WIDE_THREADS.
    _WIDE_WAVES: ClassVar[int] = 2

    @classmethod
    def applies(cls, call: QuantizeCall) -> bool:
        return call.cols * call.dtype.itemsize % VECTOR_ACCESS_BYTES == 0

    @property
    def default_config(self) -> dict:
        blocks = self.call.rows * -(-self.call.cols // QUANT_SCALE_BLOCK)
        per_sm = torch.cuda.get_device_properties(self.call.device).max_threads_per_multi_processor
        wide = blocks * self._LANES >= self._WIDE_WAVES * self.call.sm_count * per_sm
        return {
            "threads": self._WIDE_THREADS if wide else self._THREADS,
            "lanes": self._LANES,
            "pack": self._PACK,
        }


class INT8QuantPerBlockShiftedFwdKernel(_INT8QuantPerBlockFwdKernel):
    """Quantize each 128-element block of ``x`` against its own amax, for any K.

    Each lane loads the aligned vectors its part of a block straddles and shifts them into
    place; the codes of a CTA are assembled in shared memory and stored as whole vectors.
    The register kernel serves the calls whose blocks start on a 16-byte vector; this one
    serves the rest. ``q`` and ``scale`` are bit-equal to the torch reference.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads`` and ``lanes``.
        tune: Whether to autotune.
    """

    _builder = staticmethod(_int8_quant_per_block_shifted_kernel)

    general = True

    # Launch policy, fitted on the ragged-k row with the repo benchmark. Re-fit lanes in
    # {4, 8, 16} and threads in {128, 256, 512}.
    _THREADS: ClassVar[int] = 256
    _LANES: ClassVar[int] = 8

    @property
    def default_config(self) -> dict:
        return {"threads": self._THREADS, "lanes": self._LANES}
