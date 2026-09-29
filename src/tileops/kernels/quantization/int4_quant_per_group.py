"""Per-group asymmetric INT4 quantization into ``GemmW4A16FwdOp``'s weight operands."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.gemm.w4a16 import W4A16_LAYOUT
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import INT4QuantPerGroupFwdInterface, QuantizeCall
from tileops.utils import WARP_LANES

__all__ = ["INT4QuantPerGroupFwdKernel", "INT4QuantPerGroupRowFwdKernel"]

# Elements of a chunk, the K run whose codes are exactly the words (lane, j) of the packing:
# chunk c of a row is K step c // lanes and packing index j = c % lanes, and word
# lane * lanes + j of that step holds its codes of pair p of elements 32 j + 8 p + 2 lane
# (bits 4 p) and 32 j + 8 p + 2 lane + 1 (bits 16 + 4 p).
_CHUNK = W4A16_LAYOUT.mma_step_k // W4A16_LAYOUT.lanes


@functools.lru_cache(maxsize=32)
def _int4_quant_per_group_kernel(n: int, k: int, group_size: int, per_cta: bool):
    """Build the quantization of ``n`` rows of ``k`` float16 elements in groups of ``group_size``.

    A thread holds whole chunks in registers. With ``per_cta`` a CTA owns one group, thread
    ``tx`` holding its chunks ``u * threads + tx``, and reduces the range across the CTA;
    otherwise thread ``tx`` of CTA ``bx`` holds chunk ``bx * threads + tx`` and the lanes
    of a group reduce its range by shuffles.
    """
    abi_lanes = W4A16_LAYOUT.lanes
    vec = VECTOR_ACCESS_BYTES // 2
    vpc = _CHUNK // vec
    # A chunk is held as 32-bit words of two float16 elements, VECTOR_ACCESS_BYTES // 4 to
    # a vector.
    words = _CHUNK // 2
    vwords = VECTOR_ACCESS_BYTES // 4
    total = n * k
    chunks = total // _CHUNK
    groups = total // group_size
    lanes = group_size // _CHUNK
    # torch computes (hi - lo) / 15 as a product with the float32 reciprocal of 15.
    inv15 = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(15.0))
    tiny = torch.finfo(torch.float16).tiny
    # A float16 over a normal float16 scale is a half-integer exactly or at least 2**-13 from
    # one. Its product by the reciprocal plus snap lands on the 2**-14 grid, which puts an
    # exact half-integer on it and keeps every other quotient on its side; adding
    # round_magic - snap then rounds half to even to an integer, which the bits of the sum
    # hold above magic_bits.
    snap = 768.0
    round_magic = 12582912.0
    magic_bits = 0x4B400000

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _int4_quant_per_group_func(threads: int, cpt: int, evict_first: bool, min_blocks: int):
        assert threads % WARP_LANES == 0
        assert threads * cpt >= lanes if per_cta else (threads % lanes == 0 and cpt == 1)
        warps = threads // WARP_LANES
        load = "tl::tileops_load16_evict_first" if evict_first else "tl::tileops_load16"

        def widen(word, half):
            bits = T.cast(word >> T.cast(16 * half, "uint32"), "uint16")
            return T.cast(T.reinterpret(bits, "float16"), "float32")

        def chunk_of(bx, u, tx):
            if per_cta:
                return bx * lanes + u * threads + tx
            return bx * threads + tx

        def held(u, tx):
            """Whether slot ``u`` of the thread holds a chunk of its group."""
            if per_cta:
                return u * threads + tx < lanes
            return True

        @T.prim_func
        def _int4_quant_per_group_main(
            w: T.Tensor((total,), "float16"),
            packed: T.Tensor((total // 8,), "uint32"),
            scale: T.Tensor((groups,), "float16"),
            zero: T.Tensor((groups,), "uint8"),
        ):
            with T.Kernel(groups if per_cta else T.ceildiv(chunks, threads), threads=threads) as bx:
                if min_blocks > 1:
                    T.annotate_min_blocks_per_sm(min_blocks)
                tx = T.get_thread_binding()
                vals = T.alloc_local((cpt, words), "uint32")
                acc = T.alloc_local((2,), "float16x2")
                num = T.alloc_local((3,), "float32")
                code_base = T.alloc_local((1,), "int32")
                out = T.alloc_local((abi_lanes,), "uint32")
                recv = T.alloc_local((1,), "uint32")
                warp_range = T.alloc_shared((2, warps), "float32")

                # min and max are exact in float16; the range starts at [0, 0].
                acc[0] = T.reinterpret(T.uint32(0), "float16x2")
                acc[1] = T.reinterpret(T.uint32(0), "float16x2")
                for u in T.unroll(cpt):
                    c = chunk_of(bx, u, tx)
                    if held(u, tx) & (c < chunks):
                        for p in T.unroll(vpc):
                            T.call_extern(
                                "handle",
                                load,
                                T.address_of(vals[u, p * vwords]),
                                T.address_of(w[c * _CHUNK + p * vec]),
                            )
                    else:
                        # A zero leaves the range, which includes 0, unchanged.
                        for e in T.serial(words):
                            vals[u, e] = T.uint32(0)
                for u in T.unroll(cpt):
                    for e in T.unroll(words):
                        acc[0] = T.min2(acc[0], T.reinterpret(vals[u, e], "float16x2"))
                        acc[1] = T.max2(acc[1], T.reinterpret(vals[u, e], "float16x2"))
                num[0] = T.min(
                    widen(T.reinterpret(acc[0], "uint32"), 0),
                    widen(T.reinterpret(acc[0], "uint32"), 1),
                )
                num[1] = T.max(
                    widen(T.reinterpret(acc[1], "uint32"), 0),
                    widen(T.reinterpret(acc[1], "uint32"), 1),
                )
                reach = WARP_LANES if per_cta else lanes
                for st in T.unroll(reach.bit_length() - 1):
                    num[0] = T.min(
                        num[0], T.shfl_xor(num[0], T.int32(reach // 2) >> st, width=WARP_LANES)
                    )
                    num[1] = T.max(
                        num[1], T.shfl_xor(num[1], T.int32(reach // 2) >> st, width=WARP_LANES)
                    )
                if per_cta and warps > 1:
                    if tx % WARP_LANES == 0:
                        warp_range[0, tx // WARP_LANES] = num[0]
                        warp_range[1, tx // WARP_LANES] = num[1]
                    T.sync_threads()
                    for i in T.serial(warps):
                        num[0] = T.min(num[0], warp_range[0, i])
                        num[1] = T.max(num[1], warp_range[1, i])
                s16 = T.cast(
                    T.if_then_else(
                        num[1] > num[0],
                        T.max((num[1] - num[0]) * T.float32(inv15), T.float32(tiny)),
                        T.float32(1.0),
                    ),
                    "float16",
                )
                num[2] = T.ieee_frcp(T.cast(s16, "float32"))
                code_base[0] = T.reinterpret(
                    T.ieee_add(
                        T.ieee_fmaf(-num[0], num[2], T.float32(snap)),
                        T.float32(round_magic - snap),
                    ),
                    "int32",
                ) - T.int32(magic_bits)
                first = (tx == 0) if per_cta else (bx * threads + tx) % lanes == 0
                g = bx if per_cta else (bx * threads + tx) // lanes
                if first & (g < groups):
                    scale[g] = s16
                    zero[g] = T.cast(code_base[0], "uint8")
                # code = bits + code_base is zero + the rounded quotient.
                code_base[0] = code_base[0] - T.int32(magic_bits)
                for u in T.unroll(cpt):
                    c = chunk_of(bx, u, tx)
                    for lane in T.unroll(abi_lanes):
                        out[lane] = T.uint32(0)
                        for p in T.unroll(vpc):
                            for e in T.unroll(2):
                                bits = T.reinterpret(
                                    T.ieee_add(
                                        T.ieee_fmaf(
                                            widen(vals[u, p * vwords + lane], e),
                                            num[2],
                                            T.float32(snap),
                                        ),
                                        T.float32(round_magic - snap),
                                    ),
                                    "int32",
                                )
                                # A clamp at 0 is never reached: the rounding is monotone
                                # and odd, and w >= lo.
                                code = T.min(bits + code_base[0], T.int32(15))
                                # The nibbles are disjoint, so adding a shifted code sets it.
                                out[lane] = out[lane] + (
                                    T.cast(code, "uint32") << T.uint32(4 * p + 16 * e)
                                )
                    # Transpose the 4 x 4 words of the quad's chunks, one step's, so thread
                    # j holds the words of packing lane j, 16 contiguous bytes: each stage
                    # swaps one bit of the thread index with that bit of the word index.
                    for st in T.unroll(2):
                        m = 1 << st
                        upper = (tx >> st) & 1
                        for y0 in T.unroll(abi_lanes):
                            if y0 & m == 0:
                                recv[0] = T.shfl_xor(
                                    T.if_then_else(upper == 1, out[y0], out[y0 | m]),
                                    m,
                                    width=WARP_LANES,
                                )
                                if upper == 1:
                                    out[y0] = recv[0]
                                else:
                                    out[y0 | m] = recv[0]
                    if held(u, tx) & (c < chunks):
                        for x in T.vectorized(abi_lanes):
                            packed[c * abi_lanes + x] = out[x]

        return _int4_quant_per_group_main

    return _int4_quant_per_group_func


class _INT4QuantPerGroupFwdKernel(Kernel, INT4QuantPerGroupFwdInterface):
    """What the two per-group kernels share: the calls they refuse and how they launch."""

    supported_archs: list[int] = [90]
    # The integer tensors are outputs, written before anything reads them.
    autotune_accepts_random_int_inputs = True

    _PER_CTA: ClassVar[bool]

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
        self.kernel = _int4_quant_per_group_kernel(
            call.rows, call.cols, call.group_size, self._PER_CTA
        )
        self.init_config(config, tune)

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_cuda(w=w)
        n, k = self.call.rows, self.call.cols
        groups = (n, k // self.call.group_size)
        packed = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
        scale = torch.empty(groups, dtype=torch.float16, device=w.device)
        zero = torch.empty(groups, dtype=torch.uint8, device=w.device)
        # The kernel reads 16-byte vectors from the start of the storage.
        if w.data_ptr() % VECTOR_ACCESS_BYTES:
            w = w.clone()
        self.kernel(**self.config)(
            w.view(-1), packed.view(-1).view(torch.uint32), scale.view(-1), zero.view(-1)
        )
        return packed, scale, zero


class INT4QuantPerGroupFwdKernel(_INT4QuantPerGroupFwdKernel):
    """Quantize each group of ``w`` against its own range, adjacent lanes of a warp to a group.

    Serves a ``group_size`` of 32 to ``32 * WARP_LANES`` elements, a power of two. A lane
    holds 32 contiguous elements in registers, the lanes of a group reduce its range by
    shuffles, and each lane packs its codes into the four words they fill. The outputs are
    bit-equal to the torch reference.

    Args:
        call: The call's shape, dtype, group size and device facts.
        config: Optional dict with ``threads``, ``cpt`` (always 1), ``evict_first`` and
            ``min_blocks``.
        tune: Whether to autotune.
    """

    _PER_CTA = False

    # Launch policy, fitted on the manifest rows with the repo benchmark: below
    # _DEFAULT_LOAD_BYTES of w, _THREADS threads with evict-first loads; from it,
    # _WIDE_THREADS with default-policy loads. Re-fit threads in {128, 256, 512} and both
    # load policies on sizes of w from 32 MB to 470 MB, and move the crossover.
    _THREADS: ClassVar[int] = 512
    _WIDE_THREADS: ClassVar[int] = 256
    _DEFAULT_LOAD_BYTES: ClassVar[int] = 176 << 20

    @classmethod
    def applies(cls, call: QuantizeCall) -> bool:
        lanes = call.group_size // _CHUNK
        return call.group_size % _CHUNK == 0 and lanes & (lanes - 1) == 0 and lanes <= WARP_LANES

    @property
    def default_config(self) -> dict:
        wide = self.call.rows * self.call.cols * 2 >= self._DEFAULT_LOAD_BYTES
        return {
            "threads": self._WIDE_THREADS if wide else self._THREADS,
            "cpt": 1,
            "evict_first": not wide,
            "min_blocks": 1,
        }


class INT4QuantPerGroupRowFwdKernel(_INT4QuantPerGroupFwdKernel):
    """Quantize each group of ``w`` against its own range, one CTA to a group.

    Serves a ``group_size`` that is a multiple of 128, up to ``_MAX_THREADS`` threads holding
    ``_MAX_CPT`` chunks of 32 elements each (``group_size == K`` is per-channel
    quantization). The CTA holds its group in registers, reduces the range by shuffles and
    across warps in shared memory, and packs the codes from the registers. The outputs are
    bit-equal to the torch reference.

    Args:
        call: The call's shape, dtype, group size and device facts.
        config: Optional dict with ``threads``, ``cpt``, ``evict_first`` and ``min_blocks``.
        tune: Whether to autotune.
    """

    _PER_CTA = True
    general = True

    # Design bounds on the groups served: at most _MAX_CPT chunks (64 registers of input) in
    # each of at most _MAX_THREADS threads, so that a thread keeps 128 registers of a 64 K
    # register file and holds its chunks without spilling. Raising either needs a ptxas -v
    # check that the largest group does not spill.
    _MAX_CPT: ClassVar[int] = 4
    _MAX_THREADS: ClassVar[int] = 512
    # Launch policy, fitted with the repo benchmark on per-channel K of 4096, 11008, 18944
    # and 28672: _CPT chunks per thread, and as many CTAs held on an SM as fit when each
    # thread takes _REGISTERS registers per _CPT chunks it holds. Re-fit by timing cpt in
    # 1..4 against min_blocks from 1 to 12 on those K.
    _CPT: ClassVar[int] = 2
    _REGISTERS: ClassVar[int] = 64

    @classmethod
    def applies(cls, call: QuantizeCall) -> bool:
        return (
            call.group_size % W4A16_LAYOUT.mma_step_k == 0
            and call.group_size <= cls._MAX_THREADS * cls._MAX_CPT * _CHUNK
        )

    @property
    def default_config(self) -> dict:
        lanes = self.call.group_size // _CHUNK
        cpt = max(self._CPT, -(-lanes // self._MAX_THREADS))
        threads = -(-lanes // cpt // WARP_LANES) * WARP_LANES
        registers = torch.cuda.get_device_properties(self.call.device).regs_per_multiprocessor
        return {
            "threads": threads,
            "cpt": cpt,
            "evict_first": True,
            "min_blocks": max(1, registers * self._CPT // (threads * self._REGISTERS * cpt)),
        }
