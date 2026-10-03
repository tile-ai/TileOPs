"""Block-scaled FP8 quantization: a CTA owns one 128x128 tile and holds it in registers."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import (
    FP8_E4M3_MAX,
    QUANT_SCALE_BLOCK,
    VECTOR_ACCESS_BYTES,
)
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import FP8QuantPerBlockFwdInterface, QuantizeCall
from tileops.kernels.quantization.int8_codes import SMALL_SCALE, widen
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["FP8QuantPerBlockFwdKernel", "FP8QuantPerBlockUnalignedFwdKernel"]


@functools.lru_cache(maxsize=32)
def _fp8_quant_per_block_kernel(n: int, k: int, dtype: str, aligned: bool):
    """Build the quantization of an ``n x k`` weight, one CTA per 128x128 tile.

    The CTA's threads form ``rows`` rows of ``cols`` lanes, and a thread holds ``slots``
    rows of the tile: tile row ``s * rows + tx // cols`` for slot ``s``, ``vec`` elements of
    it. With ``aligned`` every row of ``w`` starts on a 16-byte vector and the thread's
    elements are one vector, loaded at once; otherwise they lie ``cols`` apart and each is
    loaded alone, so that adjacent lanes still read adjacent elements.
    """
    tn, tk = -(-n // QUANT_SCALE_BLOCK), -(-k // QUANT_SCALE_BLOCK)
    itemsize = torch.empty((), dtype=getattr(torch, dtype)).element_size()
    vec = VECTOR_ACCESS_BYTES // itemsize
    # A thread's elements are held as 32-bit words, ``per_word`` elements to a word.
    words = VECTOR_ACCESS_BYTES // 4
    per_word = 4 // itemsize
    whole = n % QUANT_SCALE_BLOCK == 0 and k % QUANT_SCALE_BLOCK == 0
    # The bits of a word's elements that order as their magnitudes do; a NaN orders above
    # every number, so a tile holding one gets torch's NaN amax.
    magnitude = 0x7FFF7FFF if per_word == 2 else 0x7FFFFFFF
    # torch computes ``amax / 448`` as a product with the float32 reciprocal of 448.
    inv_fp8_max = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(FP8_E4M3_MAX))
    # The reciprocal and two FMAs give the correctly rounded quotient while the residual
    # stays normal. From a scale of SMALL_SCALE up, a residual that underflows belongs to a
    # quotient that casts to a signed zero either way; a smaller, zero or infinite scale
    # takes the IEEE divide, as does a tile holding a NaN, whose scale is 1 but whose
    # infinities are not bounded by it.
    largest = torch.finfo(torch.float32).max

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _fp8_quant_per_block_func(threads: int, min_blocks: Optional[int], evict_first: bool):
        cols = QUANT_SCALE_BLOCK // vec
        rows = threads // cols
        slots = QUANT_SCALE_BLOCK // rows
        warps = threads // WARP_LANES
        assert cols * rows == threads and slots * rows == QUANT_SCALE_BLOCK

        def at(s, e, tx, bx, by):
            """The row and column of element ``e`` of slot ``s``."""
            row = by * QUANT_SCALE_BLOCK + s * rows + tx // cols
            if aligned:
                return row, bx * QUANT_SCALE_BLOCK + tx % cols * vec + e
            return row, bx * QUANT_SCALE_BLOCK + tx % cols + e * cols

        @T.macro
        def code_one(quot, at_, value, num, divide: bool):
            if divide:
                # Only this path meets an infinite quotient, which the saturating cast would
                # turn into NaN: it is clamped here, a NaN kept as torch's clamp keeps it.
                exact = T.ieee_fdiv(value, num[0])
                quot[at_] = T.if_then_else(
                    T.isnan(exact),
                    exact,
                    T.clamp(exact, T.float32(-FP8_E4M3_MAX), T.float32(FP8_E4M3_MAX)),
                )
            else:
                q0 = value * num[1]
                # The residual is negated so that a signed zero stays signed: ``-0 + 0``
                # rounds to ``+0``, ``-0 - 0`` does not.
                quot[at_] = T.ieee_fmaf(-T.ieee_fmaf(q0, num[0], -value), num[1], q0)

        @T.macro
        def quantize(q, values, quot, codes, num, tx, bx, by, divide: bool):
            for s in T.unroll(slots):
                for c in T.unroll(words):
                    code_one(quot, c * per_word, widen(values[s, c], 0, dtype), num, divide)
                    if per_word == 2:
                        code_one(quot, c * per_word + 1, widen(values[s, c], 1, dtype), num, divide)
                # The cast rounds to nearest even and saturates a finite quotient to +-448.
                for e in T.vectorized(vec):
                    codes[e] = T.cast(quot[e], "float8_e4m3fn")
                if aligned:
                    row, col = at(s, 0, tx, bx, by)
                    if whole or ((row < n) & (col < k)):
                        for e in T.vectorized(vec):
                            q[row * k + col + e] = codes[e]
                else:
                    for e in T.unroll(vec):
                        row, col = at(s, e, tx, bx, by)
                        if whole or ((row < n) & (col < k)):
                            q[row * k + col] = codes[e]

        @T.prim_func
        def _fp8_quant_per_block_main(
            w: T.Tensor((n * k,), dtype),
            q: T.Tensor((n * k,), "float8_e4m3fn"),
            scale: T.Tensor((tn * tk,), "float32"),
        ):
            with T.Kernel(tk, tn, threads=threads) as (bx, by):
                if min_blocks is not None:
                    T.annotate_min_blocks_per_sm(min_blocks)
                tx = T.get_thread_binding()
                values = T.alloc_local((slots, words), "uint32")
                quot = T.alloc_local((vec,), "float32")
                codes = T.alloc_local((vec,), "float8_e4m3fn")
                acc = T.alloc_local((1,), "uint32")
                num = T.alloc_local((2,), "float32")
                partial = T.alloc_shared((warps,), "uint32")

                for s in T.unroll(slots):
                    for c in T.unroll(words):
                        values[s, c] = T.uint32(0)
                    if aligned:
                        row, col = at(s, 0, tx, bx, by)
                        if whole or ((row < n) & (col < k)):
                            T.call_extern(
                                "handle",
                                "tl::tileops_load16_evict_first"
                                if evict_first
                                else "tl::tileops_load16",
                                T.address_of(values[s, 0]),
                                T.address_of(w[row * k + col]),
                            )
                    else:
                        for e in T.unroll(vec):
                            row, col = at(s, e, tx, bx, by)
                            if whole or ((row < n) & (col < k)):
                                bits = T.cast(
                                    T.reinterpret(
                                        w[row * k + col], "uint16" if per_word == 2 else "uint32"
                                    ),
                                    "uint32",
                                )
                                shift = T.uint32(32 // per_word * (e % per_word))
                                values[s, e // per_word] = values[s, e // per_word] | (
                                    bits << shift
                                )
                acc[0] = T.uint32(0)
                for s in T.unroll(slots):
                    for c in T.unroll(words):
                        bits = values[s, c] & T.uint32(magnitude)
                        if per_word == 2:
                            acc[0] = T.call_extern("uint32", "__vmaxu2", acc[0], bits)
                        else:
                            acc[0] = T.max(acc[0], bits)
                if per_word == 2:
                    acc[0] = T.max(acc[0] & T.uint32(0xFFFF), acc[0] >> T.uint32(16))
                for stage in T.unroll(WARP_SHUFFLE_STAGES):
                    acc[0] = T.max(acc[0], T.shfl_xor(acc[0], 1 << stage, width=WARP_LANES))
                if tx % WARP_LANES == 0:
                    partial[tx // WARP_LANES] = acc[0]
                T.sync_threads()
                for i in T.unroll(warps):
                    acc[0] = T.max(acc[0], partial[i])
                amax = widen(acc[0], 0, dtype)
                num[0] = T.if_then_else(amax > 0, amax * T.float32(inv_fp8_max), T.float32(1.0))
                if tx == 0:
                    scale[by * tk + bx] = num[0]
                if (
                    T.isnan(amax)
                    | (num[0] < T.float32(SMALL_SCALE))
                    | (num[0] > T.float32(largest))
                ):
                    quantize(q, values, quot, codes, num, tx, bx, by, True)
                else:
                    num[1] = T.ieee_frcp(num[0])
                    quantize(q, values, quot, codes, num, tx, bx, by, False)

        return _fp8_quant_per_block_main

    return _fp8_quant_per_block_func


class _FP8QuantPerBlockFwdKernel(Kernel, FP8QuantPerBlockFwdInterface):
    """What the two tile kernels share: the calls they refuse and how they launch."""

    supported_archs: list[int] = [89, 90]

    _aligned: ClassVar[bool]

    # Launch policy, fitted on the manifest rows with the repo benchmark (device_busy, L2
    # flushed). Re-fit each value by sweeping it with the others held.
    # 16-byte vectors a thread holds: threads = tile bytes / (16 * _SLOTS). Re-fit in {4, 8, 16}.
    _SLOTS: ClassVar[int] = 8
    # Registers a thread is capped at, which sets the CTAs an SM holds; ``None`` leaves the
    # count to the compiler. Re-fit in {None, 48, 64, 85}.
    _REGISTERS: ClassVar[Optional[int]] = 64
    # Inputs below this share of the L2 are read evict-first; larger ones with the default
    # policy. Re-fit on inputs from a quarter of the L2 to twice it.
    _EVICT_FIRST_L2_SHARE: ClassVar[float] = 0.5

    @classmethod
    def refusal(cls, call: QuantizeCall) -> Optional[str]:
        reason = super().refusal(call)
        if reason is None and call.rows * call.cols > 2**31 - 1:
            return f"indexes elements with int32, and N * K = {call.rows * call.cols}"
        return reason

    def __init__(
        self, call: QuantizeCall, config: Optional[dict] = None, tune: bool = False
    ) -> None:
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self.kernel = _fp8_quant_per_block_kernel(
            call.rows, call.cols, self.dtype_str, self._aligned
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        props = torch.cuda.get_device_properties(self.call.device)
        itemsize = self.call.dtype.itemsize
        threads = (
            QUANT_SCALE_BLOCK * QUANT_SCALE_BLOCK * itemsize // (VECTOR_ACCESS_BYTES * self._SLOTS)
        )
        cap = self._REGISTERS
        min_blocks = None if cap is None else props.regs_per_multiprocessor // (threads * cap)
        size = self.call.rows * self.call.cols * itemsize
        return {
            "threads": threads,
            "min_blocks": min_blocks,
            "evict_first": size < self._EVICT_FIRST_L2_SHARE * props.L2_cache_size,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_cuda(w=w)
        n, k = self.call.rows, self.call.cols
        q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
        scale = torch.empty(
            (-(-n // QUANT_SCALE_BLOCK), -(-k // QUANT_SCALE_BLOCK)),
            dtype=torch.float32,
            device=w.device,
        )
        # The aligned kernel reads 16-byte vectors from the start of the storage.
        if self._aligned and w.data_ptr() % VECTOR_ACCESS_BYTES:
            w = w.clone()
        self.kernel(**self.config)(w.view(-1), q.view(-1), scale.view(-1))
        return q, scale


class FP8QuantPerBlockFwdKernel(_FP8QuantPerBlockFwdKernel):
    """Quantize each 128x128 tile of ``w`` against its own amax, reading ``w`` once.

    Serves a K whose rows start on a 16-byte vector. A CTA owns a tile: each thread holds
    16-byte vectors of it in registers, the CTA reduces the tile's amax by warp shuffles and
    one exchange through shared memory, and each thread quantizes its vectors from the
    registers. ``q`` and ``scale`` are bit-equal to the torch reference: the scale is the
    product torch computes, and the quotient is correctly rounded.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads``, ``min_blocks`` and ``evict_first``.
        tune: Whether to autotune.
    """

    _aligned = True

    @classmethod
    def applies(cls, call: QuantizeCall) -> bool:
        return call.cols * call.dtype.itemsize % VECTOR_ACCESS_BYTES == 0


class FP8QuantPerBlockUnalignedFwdKernel(_FP8QuantPerBlockFwdKernel):
    """Quantize each 128x128 tile of ``w`` against its own amax, for any K.

    The tile kernel with each element loaded and stored alone, adjacent lanes on adjacent
    elements; it serves the calls whose rows do not all start on a 16-byte vector. ``q``
    and ``scale`` are bit-equal to the torch reference.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads``, ``min_blocks`` and ``evict_first``.
        tune: Whether to autotune.
    """

    _aligned = False

    general = True

    # Fitted on the ragged-k row: caps of 64 and 85 registers measured 1% slower than none.
    _REGISTERS = None
