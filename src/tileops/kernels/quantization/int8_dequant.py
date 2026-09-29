"""INT8 dequantize kernels."""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel

from .dequant_call import DequantizeCall

__all__ = ["INT8DequantPerChannelKernel"]

_INT32_MAX = 2**31 - 1


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


class INT8DequantPerChannelKernel(Kernel):
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
    def applies(cls, call: DequantizeCall) -> bool:
        return call.granularity == "channel"

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
        return identity, lambda: cls(*identity[:3], tune=call.tune, device_index=index)

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
        # The vector loads need a storage start on a vector boundary.
        q = q.clone() if q.data_ptr() % VECTOR_ACCESS_BYTES else q
        x = self.kernel(self.config["threads"], self.config["steps"])(q.view(-1), scale)
        return x.view(q.shape)
