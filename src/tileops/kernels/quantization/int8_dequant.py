"""INT8 dequantize kernels."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_path
from tileops.kernels.constants import (
    QUANT_SCALE_BLOCK,
    SM_RESIDENT_BLOCKS,
    VECTOR_ACCESS_BYTES,
)
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.quantization.call_spec import (
    DequantizeCall,
    INT8DequantPerBlockFwdInterface,
    INT8DequantPerChannelFwdInterface,
    INT8DequantPerTensorFwdInterface,
)
from tileops.utils import get_sm_version

__all__ = [
    "INT8DequantPerBlockFwdKernel",
    "INT8DequantPerBlockSmallFwdKernel",
    "INT8DequantPerChannelFwdKernel",
    "INT8DequantPerTensorFwdKernel",
    "INT8DequantPerTensorSmallFwdKernel",
]

_INT32_MAX = 2**31 - 1

# The float32 bit pattern of 2^23 + 128. Adding an int8 code c to it gives the pattern of
# 2^23 + 128 + c exactly, so subtracting 2^23 + 128 recovers float(c) with an integer add
# and an FADD instead of I2F, which issues 16 results per clock per SM against FADD's 128.
_CODE_BIAS_BITS = 0x4B000080
_CODE_BIAS = 8388736.0


@functools.lru_cache(maxsize=32)
def _int8_dequant_per_channel_kernel(m: int, k: int, out_dtype: str, npt: int):
    n = m * k

    @tilelang.jit(out_idx=[2])
    def _int8_dequant_per_channel_func(threads, steps):
        chunk = threads * npt
        block = chunk * steps
        # Blocks that take the vector path: every whole block, when a thread's npt codes
        # span at most two rows.
        vector_blocks = n // block if k >= npt else 0

        @T.prim_func
        def _int8_dequant_per_channel_main(
            q: T.Tensor((n,), T.int8),
            scale: T.Tensor((m,), T.float32),
            x: T.Tensor((n,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(n, block), threads=threads) as bx:
                tx = T.get_thread_binding()
                q_local = T.alloc_local((steps * npt,), T.int8)
                x_local = T.alloc_local((steps * npt,), out_dtype)
                if bx < vector_blocks:
                    # Every load of the block issues before the first scale is used.
                    for r in T.unroll(steps):
                        for j in T.vectorized(npt):
                            q_local[r * npt + j] = q[bx * block + r * chunk + tx * npt + j]
                    for r in T.unroll(steps):
                        base = bx * block + r * chunk + tx * npt
                        row = base // k
                        # This thread's codes before the next row starts.
                        split = (row + 1) * k - base
                        lo = scale[row]
                        hi = scale[T.min(row + 1, m - 1)]
                        for j in T.unroll(npt):
                            x_local[r * npt + j] = T.Cast(
                                out_dtype,
                                T.Cast(T.float32, q_local[r * npt + j])
                                * T.if_then_else(j < split, lo, hi),
                            )
                    for r in T.unroll(steps):
                        for j in T.vectorized(npt):
                            x[bx * block + r * chunk + tx * npt + j] = x_local[r * npt + j]
                else:
                    for r in T.unroll(steps):
                        for j in T.unroll(npt):
                            idx = bx * block + r * chunk + tx * npt + j
                            if idx < n:
                                x[idx] = T.Cast(
                                    out_dtype, T.Cast(T.float32, q[idx]) * scale[idx // k]
                                )

        return _int8_dequant_per_channel_main

    return _int8_dequant_per_channel_func


class INT8DequantPerChannelFwdKernel(Kernel, INT8DequantPerChannelFwdInterface):
    """``x = (q.float() * scale[:, None]).to(out_dtype)`` for one scale per row of ``q``.

    The matrix is read as one flat run of ``m * k`` codes. A thread converts ``npt``
    contiguous codes per step, one 16-byte store of ``x``, and each block runs ``steps``
    such chunks with every load issued first; a chunk that crosses a row boundary selects
    between the two rows' scales. The last block, and every block when ``k < npt``,
    converts code by code against the run's end.

    Args:
        m: Rows of ``q``.
        k: Columns of ``q``.
        out_dtype: Torch dtype of ``x``.
        config: Optional dict with "threads" and "steps".
        tune: Whether to autotune.
        device_index: The device the kernel is built for.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    aligned_inputs = ("q",)

    general: bool = True

    # ``q`` is data: random codes run the same instructions as real ones.
    autotune_accepts_random_int_inputs: bool = True

    # Launch policy by the byte width of ``x``, fitted on H200 by timing the manifest rows
    # with the repo benchmark; re-fit by timing ``threads`` in {128, 256, 512} and
    # ``steps`` in {1, 2, 4}. Two steps put two loads per thread in flight.
    _CONFIGS: ClassVar[dict[int, dict]] = {
        2: {"threads": 128, "steps": 2},
        4: {"threads": 512, "steps": 2},
    }
    # Below _SMALL_N codes, blocks of 1024 codes measured fastest: the matrix spreads
    # over more SMs.
    _SMALL_CONFIGS: ClassVar[dict[int, dict]] = {
        2: {"threads": 128, "steps": 1},
        4: {"threads": 128, "steps": 2},
    }
    _SMALL_N: ClassVar[int] = 1 << 17
    _TUNE_THREADS: ClassVar[tuple[int, ...]] = (128, 256, 512)
    _TUNE_STEPS: ClassVar[tuple[int, ...]] = (1, 2, 4)

    @classmethod
    def refusal(cls, call: DequantizeCall) -> Optional[str]:
        reason = super().refusal(call)
        # The last block's indices run up to one block past M * K; the widest block holds
        # 16-bit codes, VECTOR_ACCESS_BYTES // 2 per thread per step.
        largest_block = max(cls._TUNE_THREADS) * max(cls._TUNE_STEPS) * VECTOR_ACCESS_BYTES // 2
        if reason is None and call.m * call.k > _INT32_MAX - largest_block:
            return f"indexes elements with int32, and M * K = {call.m * call.k}"
        return reason

    @classmethod
    def entry_for(cls, call: DequantizeCall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (call.m, call.k, call.out_dtype, index)
        return identity, lambda: cls(*identity[:3], device_index=index)

    def __init__(
        self,
        m: int,
        k: int,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        self.m = m
        self.k = k
        self.dtype = out_dtype
        # Codes per thread per step: one 16-byte vector of ``x``.
        npt = VECTOR_ACCESS_BYTES // out_dtype.itemsize
        self.kernel = _int8_dequant_per_channel_kernel(m, k, self.dtype_str, npt)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        small = self.m * self.k < self._SMALL_N
        return dict((self._SMALL_CONFIGS if small else self._CONFIGS)[self.dtype.itemsize])

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t, "steps": s} for t in self._TUNE_THREADS for s in self._TUNE_STEPS]

    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        self._require_cuda(q=q, scale=scale)
        x = self.kernel(self.config["threads"], self.config["steps"])(q.view(-1), scale)
        return x.view(q.shape)


@functools.lru_cache(maxsize=32)
def _int8_dequant_per_tensor_kernel(
    n: int, out_dtype: str, vec: int, staged: bool, resident_threads: int, resident_blocks: int
):
    @tilelang.jit(out_idx=[2], compile_flags=["-include", csrc_path("streaming_load.h")])
    def _int8_dequant_per_tensor_func(threads, steps):
        chunk = threads * vec
        block = chunk * steps
        full_blocks = n // block

        def code_to_float(code):
            return T.reinterpret(T.Cast(T.int32, code) + _CODE_BIAS_BITS, T.float32) - _CODE_BIAS

        @T.prim_func
        def _int8_dequant_per_tensor_main(
            q: T.Tensor((n,), T.int8),
            scale: T.Tensor((1,), T.float32),
            x: T.Tensor((n,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(n, block), threads=threads) as bx:
                # Registers for a full SM of blocks: the last block's path must not cost
                # every other block its occupancy.
                T.annotate_min_blocks_per_sm(min(resident_blocks, resident_threads // threads))
                tx = T.get_thread_binding()
                q_local = T.alloc_local((steps * vec,), T.int8)
                x_local = T.alloc_local((steps * vec,), out_dtype)
                x_shared = T.alloc_shared((block,), out_dtype)
                s = T.alloc_var(T.float32)
                # Every block reads the one scale: kept in L1, later blocks on an SM hit it.
                s = T.call_extern(
                    T.float32, "tl::tileops_load_f32_evict_last", T.address_of(scale[0])
                )
                if bx < full_blocks:
                    # Every load of the block issues before the first conversion.
                    for r in T.unroll(steps):
                        for j in T.vectorized(vec):
                            q_local[r * vec + j] = q[bx * block + r * chunk + tx * vec + j]
                    if staged:
                        for r in T.unroll(steps):
                            for j in T.unroll(vec):
                                x_shared[r * chunk + tx * vec + j] = T.Cast(
                                    out_dtype, code_to_float(q_local[r * vec + j]) * s
                                )
                        # One bulk store of the block's output.
                        T.copy(x_shared, x[bx * block : (bx + 1) * block])
                    else:
                        for r in T.unroll(steps):
                            for j in T.unroll(vec):
                                x_local[r * vec + j] = T.Cast(
                                    out_dtype, code_to_float(q_local[r * vec + j]) * s
                                )
                        for r in T.unroll(steps):
                            for j in T.vectorized(vec):
                                x[bx * block + r * chunk + tx * vec + j] = x_local[r * vec + j]
                else:
                    # The last block moves whole vectors where they fit, and code by code,
                    # with loads clamped to the last code, where the run ends.
                    for r in T.unroll(steps):
                        base = bx * block + r * chunk + tx * vec
                        if base + vec <= n:
                            for j in T.vectorized(vec):
                                q_local[r * vec + j] = q[base + j]
                        else:
                            for j in T.unroll(vec):
                                q_local[r * vec + j] = q[T.min(base + j, n - 1)]
                    for r in T.unroll(steps):
                        for j in T.unroll(vec):
                            x_local[r * vec + j] = T.Cast(
                                out_dtype, code_to_float(q_local[r * vec + j]) * s
                            )
                    for r in T.unroll(steps):
                        base = bx * block + r * chunk + tx * vec
                        if base + vec <= n:
                            for j in T.vectorized(vec):
                                x[base + j] = x_local[r * vec + j]
                        else:
                            for j in T.unroll(vec):
                                if base + j < n:
                                    x[base + j] = x_local[r * vec + j]

        return _int8_dequant_per_tensor_main

    return _int8_dequant_per_tensor_func


class INT8DequantPerTensorFwdKernel(Kernel, INT8DequantPerTensorFwdInterface):
    """``x = (q.float() * scale).to(out_dtype)`` for one scale over a whole INT8 matrix.

    The matrix is read as one flat run of ``n`` codes. A thread converts one 16-byte
    vector of ``x`` per step, ``steps`` steps per block with every load issued first, and
    the block stages its output in shared memory and writes it with one bulk copy. The
    last block converts code by code against ``n``.

    Args:
        n: Codes in ``q``, ``M * K``.
        out_dtype: Torch dtype of ``x``.
        config: Optional dict with "threads" and "steps".
        tune: Whether to autotune.
        device_index: The device the kernel is built for.
    """

    # The bulk copy from shared memory lowers to cp.async.bulk on SM90, to plain stores
    # before it.
    supported_archs: list[int] = [80, 86, 89, 90]
    aligned_inputs = ("q",)

    general: bool = True

    # ``q`` is data: random codes run the same instructions as real ones.
    autotune_accepts_random_int_inputs: bool = True

    _STAGED: ClassVar[bool] = True

    # Launch policy, fitted by timing the manifest rows with the repo benchmark; re-fit by
    # timing ``threads`` in _TUNE_THREADS and ``steps`` in _TUNE_STEPS.
    _THREADS: ClassVar[int] = 64
    _STEPS: ClassVar[int] = 4
    _TUNE_THREADS: ClassVar[tuple[int, ...]] = (32, 64, 128)
    _TUNE_STEPS: ClassVar[tuple[int, ...]] = (2, 4, 8)

    @classmethod
    def refusal(cls, call: DequantizeCall) -> Optional[str]:
        reason = super().refusal(call)
        # The last block's indices run up to one block past M * K; the widest block holds
        # 16-bit codes, VECTOR_ACCESS_BYTES // 2 per thread per step.
        largest_block = max(cls._TUNE_THREADS) * max(cls._TUNE_STEPS) * VECTOR_ACCESS_BYTES // 2
        if reason is None and call.m * call.k > _INT32_MAX - largest_block:
            return f"indexes elements with int32, and M * K = {call.m * call.k}"
        return reason

    @classmethod
    def entry_for(cls, call: DequantizeCall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (call.m * call.k, call.out_dtype, index)
        return identity, lambda: cls(*identity[:2], device_index=index)

    def __init__(
        self,
        n: int,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        self.n = n
        self.dtype = out_dtype
        # Codes per thread per step: one 16-byte vector of ``x``.
        vec = VECTOR_ACCESS_BYTES // out_dtype.itemsize
        device = torch.device("cuda", device_index) if device_index is not None else None
        resident_threads = torch.cuda.get_device_properties(device).max_threads_per_multi_processor
        resident_blocks = SM_RESIDENT_BLOCKS[get_sm_version(device_index)]
        self.kernel = _int8_dequant_per_tensor_kernel(
            n, self.dtype_str, vec, self._STAGED, resident_threads, resident_blocks
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": self._THREADS, "steps": self._STEPS}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t, "steps": s} for t in self._TUNE_THREADS for s in self._TUNE_STEPS]

    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        self._require_cuda(q=q, scale=scale)
        x = self.kernel(self.config["threads"], self.config["steps"])(q.view(-1), scale)
        return x.view(q.shape)


class INT8DequantPerTensorSmallFwdKernel(INT8DequantPerTensorFwdKernel):
    """The same conversion for a matrix of fewer than ``_SMALL_N`` codes, stored from registers.

    A matrix that fills few blocks is bound by latency, and staging its output in shared
    memory adds a barrier and a copy to every block's path.
    """

    general: bool = False

    _STAGED: ClassVar[bool] = False

    # Fitted like the parent's policy.
    _THREADS: ClassVar[int] = 512
    _STEPS: ClassVar[int] = 1
    _TUNE_THREADS: ClassVar[tuple[int, ...]] = (128, 256, 512)
    _TUNE_STEPS: ClassVar[tuple[int, ...]] = (1, 2)
    _SMALL_N: ClassVar[int] = 1 << 19

    @classmethod
    def applies(cls, call: DequantizeCall) -> bool:
        return call.m * call.k < cls._SMALL_N


@functools.lru_cache(maxsize=32)
def _int8_dequant_per_block_kernel(
    m: int,
    k: int,
    out_dtype: str,
    vec: int,
    staged: bool,
    resident_threads: int,
    resident_blocks: int,
):
    n = m * k

    @tilelang.jit(out_idx=[2])
    def _int8_dequant_per_block_func(threads, steps):
        chunk = threads * vec
        block = chunk * steps
        full_blocks = n // block
        # One scale per block of a row, the last one covering the codes left over; the
        # scales of all rows form one flat run in the order of the codes they cover.
        per_row = -(-k // QUANT_SCALE_BLOCK)
        scales = m * per_row
        # A vector crosses a scale boundary only when k is not a multiple of vec, and two
        # only when the short last group of a row fits inside it with a code on either side.
        straddles = k % vec != 0
        three_scales = straddles and 0 < k % QUANT_SCALE_BLOCK <= vec - 2
        # Scales one vector may need.
        spans = 3 if three_scales else 2 if straddles else 1
        # Scales one block's codes may need: a group starts at every block boundary of a row
        # and at every row start.
        block_scales = block // QUANT_SCALE_BLOCK + block // k + 3

        def code_to_float(code):
            return T.reinterpret(T.Cast(T.int32, code) + _CODE_BIAS_BITS, T.float32) - _CODE_BIAS

        def scaled(q_local, s_local, cross, r, j):
            # The scale of code j of step r's vector: the lanes before each boundary take
            # the earlier scale.
            s = s_local[r * spans]
            if three_scales:
                s = T.if_then_else(
                    j < cross[r * 2],
                    s,
                    T.if_then_else(
                        j < cross[r * 2 + 1], s_local[r * spans + 1], s_local[r * spans + 2]
                    ),
                )
            elif straddles:
                s = T.if_then_else(j < cross[r * 2], s, s_local[r * spans + 1])
            return T.Cast(out_dtype, code_to_float(q_local[r * vec + j]) * s)

        @T.macro
        def load_scales(src, first, last, s_local, cross, r, base):
            # Step r's scales from src, which holds scale[first:] up to index last.
            row = base // k
            col = base - row * k
            g = row * per_row + col // QUANT_SCALE_BLOCK - first
            for i in T.unroll(spans):
                # Past the scales a vector needs, the index is clamped and the value unused.
                s_local[r * spans + i] = src[T.min(g + i, last)]
            if straddles:
                # The first scale boundary after base.
                cross[r * 2] = (
                    row * k + T.min(k, (col // QUANT_SCALE_BLOCK + 1) * QUANT_SCALE_BLOCK) - base
                )
            if three_scales:
                # The group starting there is the next row's first, or this row's short last.
                cross[r * 2 + 1] = cross[r * 2] + T.min(
                    QUANT_SCALE_BLOCK, k - (base + cross[r * 2]) % k
                )

        @T.macro
        def load_block_scales(scale, s_shared, s_local, cross, bx, tx):
            if staged:
                # The block's scales go to shared memory once; each vector then reads its
                # one to three scales there.
                row = bx * block // k
                first = row * per_row + (bx * block - row * k) // QUANT_SCALE_BLOCK
                for i in T.unroll(T.ceildiv(block_scales, threads)):
                    if i * threads + tx < block_scales:
                        s_shared[i * threads + tx] = scale[
                            T.min(first + i * threads + tx, scales - 1)
                        ]
                T.sync_threads()
                for r in T.unroll(steps):
                    load_scales(
                        s_shared,
                        first,
                        block_scales - 1,
                        s_local,
                        cross,
                        r,
                        bx * block + r * chunk + tx * vec,
                    )
            else:
                for r in T.unroll(steps):
                    load_scales(
                        scale, 0, scales - 1, s_local, cross, r, bx * block + r * chunk + tx * vec
                    )

        @T.prim_func
        def _int8_dequant_per_block_main(
            q: T.Tensor((n,), T.int8),
            scale: T.Tensor((scales,), T.float32),
            x: T.Tensor((n,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(n, block), threads=threads) as bx:
                # Registers for a full SM of blocks: the last block's path must not cost
                # every other block its occupancy.
                T.annotate_min_blocks_per_sm(min(resident_blocks, resident_threads // threads))
                tx = T.get_thread_binding()
                q_local = T.alloc_local((steps * vec,), T.int8)
                x_local = T.alloc_local((steps * vec,), out_dtype)
                x_shared = T.alloc_shared((block,), out_dtype)
                s_shared = T.alloc_shared((block_scales,), T.float32)
                s_local = T.alloc_local((steps * spans,), T.float32)
                cross = T.alloc_local((steps * 2,), T.int32)
                if k < vec:
                    # A vector spans more rows than its scales cover: code by code.
                    for r in T.unroll(steps):
                        for j in T.unroll(vec):
                            idx = bx * block + r * chunk + tx * vec + j
                            if idx < n:
                                x[idx] = T.Cast(
                                    out_dtype,
                                    code_to_float(q[idx])
                                    * scale[idx // k * per_row + idx % k // QUANT_SCALE_BLOCK],
                                )
                elif bx < full_blocks:
                    # Every load of the block issues before the first conversion.
                    for r in T.unroll(steps):
                        for j in T.vectorized(vec):
                            q_local[r * vec + j] = q[bx * block + r * chunk + tx * vec + j]
                    load_block_scales(scale, s_shared, s_local, cross, bx, tx)
                    if staged:
                        for r in T.unroll(steps):
                            for j in T.unroll(vec):
                                x_shared[r * chunk + tx * vec + j] = scaled(
                                    q_local, s_local, cross, r, j
                                )
                        # One bulk store of the block's output.
                        T.copy(x_shared, x[bx * block : (bx + 1) * block])
                    else:
                        for r in T.unroll(steps):
                            for j in T.unroll(vec):
                                x_local[r * vec + j] = scaled(q_local, s_local, cross, r, j)
                        for r in T.unroll(steps):
                            for j in T.vectorized(vec):
                                x[bx * block + r * chunk + tx * vec + j] = x_local[r * vec + j]
                else:
                    # The last block moves whole vectors where they fit, and code by code,
                    # with loads clamped to the last code, where the run ends.
                    for r in T.unroll(steps):
                        base = bx * block + r * chunk + tx * vec
                        if base + vec <= n:
                            for j in T.vectorized(vec):
                                q_local[r * vec + j] = q[base + j]
                        else:
                            for j in T.unroll(vec):
                                q_local[r * vec + j] = q[T.min(base + j, n - 1)]
                    load_block_scales(scale, s_shared, s_local, cross, bx, tx)
                    for r in T.unroll(steps):
                        for j in T.unroll(vec):
                            x_local[r * vec + j] = scaled(q_local, s_local, cross, r, j)
                    for r in T.unroll(steps):
                        base = bx * block + r * chunk + tx * vec
                        if base + vec <= n:
                            for j in T.vectorized(vec):
                                x[base + j] = x_local[r * vec + j]
                        else:
                            for j in T.unroll(vec):
                                if base + j < n:
                                    x[base + j] = x_local[r * vec + j]

        return _int8_dequant_per_block_main

    return _int8_dequant_per_block_func


class INT8DequantPerBlockFwdKernel(Kernel, INT8DequantPerBlockFwdInterface):
    """``x[m, k] = (q[m, k].float() * scale[m, k // 128]).to(out_dtype)``.

    The matrix is read as one flat run of ``m * k`` codes, with the per-tensor kernel's
    schedule: a thread converts one 16-byte vector of ``x`` per step, ``steps`` steps per
    block with every load issued first, and the block stages its output in shared memory
    and writes it with one bulk copy. A block reads the scales its codes need into shared
    memory once, and each vector takes the scale of the 128 codes it lies in; when ``k``
    is not a multiple of the vector, a vector crossing a group or row boundary selects
    among the scales on either side. The last block moves whole vectors where they fit
    and converts code by code where the run ends; every block converts code by code when
    ``k`` is shorter than a vector.

    Args:
        m: Rows of ``q``.
        k: Columns of ``q``.
        out_dtype: Torch dtype of ``x``.
        config: Optional dict with "threads" and "steps".
        tune: Whether to autotune.
        device_index: The device the kernel is built for.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    aligned_inputs = ("q",)

    general: bool = True

    # ``q`` is data: random codes run the same instructions as real ones.
    autotune_accepts_random_int_inputs: bool = True

    _STAGED: ClassVar[bool] = True

    # Launch policy, fitted by timing the manifest rows with the repo benchmark; re-fit by
    # timing ``threads`` in _TUNE_THREADS and ``steps`` in _TUNE_STEPS.
    _THREADS: ClassVar[int] = 64
    _STEPS: ClassVar[int] = 4
    _TUNE_THREADS: ClassVar[tuple[int, ...]] = (32, 64, 128)
    _TUNE_STEPS: ClassVar[tuple[int, ...]] = (2, 4, 8)

    @classmethod
    def refusal(cls, call: DequantizeCall) -> Optional[str]:
        reason = super().refusal(call)
        # The last block's indices run up to one block past M * K; the widest block holds
        # 16-bit codes, VECTOR_ACCESS_BYTES // 2 per thread per step.
        largest_block = max(cls._TUNE_THREADS) * max(cls._TUNE_STEPS) * VECTOR_ACCESS_BYTES // 2
        if reason is None and call.m * call.k > _INT32_MAX - largest_block:
            return f"indexes elements with int32, and M * K = {call.m * call.k}"
        return reason

    @classmethod
    def entry_for(cls, call: DequantizeCall) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (call.m, call.k, call.out_dtype, index)
        return identity, lambda: cls(*identity[:3], device_index=index)

    def __init__(
        self,
        m: int,
        k: int,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: "int | None" = None,
    ):
        super().__init__(device_index=device_index)
        self.m = m
        self.k = k
        self.dtype = out_dtype
        # Codes per thread per step: one 16-byte vector of ``x``.
        vec = VECTOR_ACCESS_BYTES // out_dtype.itemsize
        device = torch.device("cuda", device_index) if device_index is not None else None
        resident_threads = torch.cuda.get_device_properties(device).max_threads_per_multi_processor
        resident_blocks = SM_RESIDENT_BLOCKS[get_sm_version(device_index)]
        self.kernel = _int8_dequant_per_block_kernel(
            m, k, self.dtype_str, vec, self._STAGED, resident_threads, resident_blocks
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": self._THREADS, "steps": self._STEPS}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t, "steps": s} for t in self._TUNE_THREADS for s in self._TUNE_STEPS]

    def forward(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        self._require_cuda(q=q, scale=scale)
        x = self.kernel(self.config["threads"], self.config["steps"])(q.view(-1), scale.view(-1))
        return x.view(q.shape)


class INT8DequantPerBlockSmallFwdKernel(INT8DequantPerBlockFwdKernel):
    """The same conversion for a matrix of fewer than ``_SMALL_N`` codes, stored from registers.

    A matrix that fills few blocks is bound by latency, and staging its output in shared
    memory adds a barrier and a copy to every block's path.
    """

    general: bool = False

    _STAGED: ClassVar[bool] = False

    # Fitted like the parent's policy.
    _THREADS: ClassVar[int] = 128
    _STEPS: ClassVar[int] = 1
    _TUNE_THREADS: ClassVar[tuple[int, ...]] = (128, 256, 512)
    _TUNE_STEPS: ClassVar[tuple[int, ...]] = (1, 2)
    _SMALL_N: ClassVar[int] = 1 << 19

    @classmethod
    def applies(cls, call: DequantizeCall) -> bool:
        return call.m * call.k < cls._SMALL_N
