"""Per-channel symmetric INT8 quantization: one CTA per row, the row held in registers."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import MAX_BLOCK_THREADS, VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.quantization.call_spec import INT8QuantPerChannelFwdInterface, QuantizeCall
from tileops.utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["INT8QuantPerChannelFwdKernel"]


@functools.lru_cache(maxsize=32)
def _int8_quant_per_channel_kernel(rows: int, k: int, dtype: str):
    """Build the quantization of ``rows`` rows of ``k`` contiguous elements.

    The input is addressed as one flat run in 16-byte vectors aligned on the flat index,
    so a vector of ``w`` and the codes it produces are both aligned. CTA ``r`` holds the
    vectors that overlap row ``r``; when ``k`` is not a whole number of vectors the first
    and last of them are shared with the neighbouring rows, and only their in-row elements
    count.
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
    # torch's ``amax / 127`` multiplies by this float32 reciprocal, since the divisor is a
    # CPU scalar; the scale is that product so that ``q`` divides by the reference's scale.
    inv_qmax = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(127.0))
    # Adding 1.5 * 2**23 to a float of magnitude below 2**22 rounds it half to even to an
    # integer, which the low byte of the sum's bit pattern then holds in two's complement.
    round_magic = 12582912.0
    # The quotient is correctly rounded while its residual stays normal, which holds for a
    # scale of at least 2**-100. A scale below 2**-60 is multiplied, with every element, by
    # 2**64 first, which is exact and lifts any nonzero float32 scale to at least 2**-85.
    small_scale = 2.0**-60
    scale_up = 2.0**64

    @tilelang.jit(compile_flags=["-include", csrc_path("streaming_load.h")])
    def _int8_quant_per_channel_func(
        threads: int, vpt: int, pair: bool, evict_first: bool, min_blocks: int
    ):
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

        def quantize(value, num):
            """``value / scale`` of a float32, rounded half to even, as int8.

            ``num`` holds the pre-scale factor (1 or ``scale_up``), then the scale and its
            reciprocal after that factor.
            """
            x = value * num[0]
            q0 = x * num[2]
            residual = T.ieee_fmaf(-q0, num[1], x)
            quotient = T.ieee_fmaf(residual, num[2], q0)
            return T.cast(T.reinterpret(quotient + T.float32(round_magic), "int32"), "int8")

        # The bit pattern of ``|value|`` as int32 orders as the magnitude does and puts a
        # NaN above every number; a bfloat16 widens to float32 by moving its bits into the
        # high half of a word.
        @T.macro
        def fold_one(acc, e, value, v, lo, hi, masked: bool):
            if masked:
                acc[0] = T.max(
                    acc[0],
                    T.if_then_else(
                        (v * vec + e >= lo) & (v * vec + e < hi),
                        T.reinterpret(T.abs(value), "int32"),
                        0,
                    ),
                )
            else:
                acc[0] = T.max(acc[0], T.reinterpret(T.abs(value), "int32"))

        @T.macro
        def fold(acc, values, j, v, lo, hi, masked: bool):
            for c in T.unroll(width):
                if words:
                    fold_one(
                        acc,
                        2 * c,
                        T.reinterpret(values[j, c] << T.uint32(16), "float32"),
                        v,
                        lo,
                        hi,
                        masked,
                    )
                    fold_one(
                        acc,
                        2 * c + 1,
                        T.reinterpret(values[j, c] & T.uint32(0xFFFF0000), "float32"),
                        v,
                        lo,
                        hi,
                        masked,
                    )
                else:
                    fold_one(acc, c, T.cast(values[j, c], "float32"), v, lo, hi, masked)

        @T.macro
        def codes(out, values, j, at, num):
            for c in T.unroll(width):
                if words:
                    out[at + 2 * c] = quantize(
                        T.reinterpret(values[j, c] << T.uint32(16), "float32"), num
                    )
                    out[at + 2 * c + 1] = quantize(
                        T.reinterpret(values[j, c] & T.uint32(0xFFFF0000), "float32"), num
                    )
                else:
                    out[at + c] = quantize(T.cast(values[j, c], "float32"), num)

        @T.macro
        def store_group(q, out, values, g, v, num):
            for i in T.unroll(group):
                codes(out, values, g * group + i, i * vec, num)
            for c in T.vectorized(group * vec):
                q[v * vec + c] = out[c]

        @T.prim_func
        def _int8_quant_per_channel_main(
            w: T.Tensor((n,), dtype),
            q: T.Tensor((n,), "int8"),
            scale: T.Tensor((rows,), "float32"),
        ):
            with T.Kernel(rows, threads=threads) as bx:
                if min_blocks > 1:
                    T.annotate_min_blocks_per_sm(min_blocks)
                tx = T.get_thread_binding()
                values = T.alloc_local((vpt, width), held)
                acc = T.alloc_local((1,), "int32")
                out = T.alloc_local((group * vec,), "int8")
                num = T.alloc_local((3,), "float32")
                warp_max = T.alloc_shared((warps,), "int32")

                lo = bx * k
                hi = lo + k
                v0 = lo // vec
                count = (hi + vec - 1) // vec - v0

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
                                            T.reinterpret(
                                                w[(v0 + slot(j, tx)) * vec + c], "uint16"
                                            ),
                                            "uint32",
                                        )
                                        if c % 2 == 0:
                                            values[j, c // 2] = values[j, c // 2] | bits
                                        else:
                                            values[j, c // 2] = values[j, c // 2] | (
                                                bits << T.uint32(16)
                                            )
                                    else:
                                        values[j, c] = w[(v0 + slot(j, tx)) * vec + c]
                        else:
                            T.call_extern(
                                "handle",
                                load,
                                T.address_of(values[j, 0]),
                                T.address_of(w[(v0 + slot(j, tx)) * vec]),
                            )
                acc[0] = 0
                for j in T.unroll(vpt):
                    if exact:
                        fold(acc, values, j, v0 + slot(j, tx), lo, hi, False)
                    elif slot(j, tx) < count:
                        if aligned:
                            fold(acc, values, j, v0 + slot(j, tx), lo, hi, False)
                        elif (slot(j, tx) == 0) | (slot(j, tx) == count - 1):
                            fold(acc, values, j, v0 + slot(j, tx), lo, hi, True)
                        else:
                            fold(acc, values, j, v0 + slot(j, tx), lo, hi, False)
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

                amax = T.reinterpret(acc[0], "float32")
                num[1] = T.if_then_else(amax > 0, amax * T.float32(inv_qmax), T.float32(1.0))
                if tx == 0:
                    scale[bx] = num[1]
                num[0] = T.if_then_else(
                    num[1] < T.float32(small_scale), T.float32(scale_up), T.float32(1.0)
                )
                num[1] = num[1] * num[0]
                num[2] = T.ieee_frcp(num[1])
                for g in T.unroll(vpt // group):
                    if exact:
                        store_group(q, out, values, g, v0 + slot(g * group, tx), num)
                    elif slot(g, tx) < count:
                        if aligned:
                            store_group(q, out, values, g, v0 + slot(g, tx), num)
                        elif (slot(g, tx) == 0) | (slot(g, tx) == count - 1):
                            codes(out, values, g, 0, num)
                            for c in T.unroll(vec):
                                if ((v0 + slot(g, tx)) * vec + c >= lo) & (
                                    (v0 + slot(g, tx)) * vec + c < hi
                                ):
                                    q[(v0 + slot(g, tx)) * vec + c] = out[c]
                        else:
                            store_group(q, out, values, g, v0 + slot(g, tx), num)

        return _int8_quant_per_channel_main

    return _int8_quant_per_channel_func


class INT8QuantPerChannelFwdKernel(Kernel, INT8QuantPerChannelFwdInterface):
    """Quantize each row of ``w`` against its own amax, reading ``w`` once.

    One CTA owns one row and holds it in registers across the row's reduction, so ``w``
    is read once and ``q`` written once. ``q`` and ``scale`` are bit-equal to the torch
    reference: the scale is the product torch computes, and the quotient is correctly
    rounded.

    Args:
        call: The call's shape, dtype and device facts.
        config: Optional dict with ``threads``, ``vpt``, ``pair``, ``evict_first`` and
            ``min_blocks``.
        tune: Whether to autotune.
    """

    supported_archs: list[int] = [90]
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
        self.kernel = _int8_quant_per_channel_kernel(call.rows, call.cols, self.dtype_str)
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

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_cuda(w=w)
        # The kernel reads 16-byte vectors from the start of the storage.
        if w.data_ptr() % VECTOR_ACCESS_BYTES:
            w = w.clone()
        q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
        scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
        c = self.config
        self.kernel(c["threads"], c["vpt"], c["pair"], c["evict_first"], c["min_blocks"])(
            w.view(-1), q.view(-1), scale
        )
        return q, scale
