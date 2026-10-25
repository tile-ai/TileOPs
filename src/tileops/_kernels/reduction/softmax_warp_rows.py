"""Softmax / log-softmax of rows of up to 2 KB, one or several rows a warp.

A row is split across a group of adjacent lanes of one warp, each lane holding a few
16-byte vectors of it in registers, read evict-last in L2. The group's lanes exchange
the row's maximum and sum by shuffles, so a row needs no shared memory and no block
barrier.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.reduction._primitives import exp_shifted, restore_same_shape
from tileops._kernels.reduction.call_spec import SoftmaxCall
from tileops._kernels.reduction.softmax import _SoftmaxKernelBase
from tileops._kernels.streaming_load import load16_evict_last
from tileops._utils import WARP_LANES

__all__ = ["SoftmaxWarpRowsKernel"]


@functools.lru_cache(maxsize=32)
def _softmax_warp_rows_kernel(M, N, op_kind, dtype, out_dtype, lanes, per_lane, warps):
    """Build the program for ``(M, N)`` rows, ``lanes`` lanes a row.

    A lane holds ``per_lane`` 16-byte vectors of its row, vector ``k * lanes + lane``,
    so the group's lanes read each stretch of the row in one coalesced access. A lane
    past the last row reads the last row and stores nothing. The loads are evict-last in
    L2: at (32768, 512) float16 on H200 that took 17.34 us against 18.18 with default
    loads.
    """
    vec = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    held_elems = per_lane * vec
    rows_per_cta = WARP_LANES // lanes * warps
    # Shuffle distances within a group: 1, 2, ..., lanes / 2.
    steps = lanes.bit_length() - 1

    @tilelang.jit(out_idx=[1], compile_flags=csrc_include("streaming_load.h"))
    def build():
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M, N), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, rows_per_cta), threads=warps * WARP_LANES) as bx:
                tx = T.get_thread_binding()
                lane = tx % lanes
                row = T.min(bx * rows_per_cta + tx // lanes, M - 1)
                held = T.alloc_local([held_elems], dtype)
                values = T.alloc_local([held_elems], "float32")
                result = T.alloc_local([held_elems], out_dtype)
                stat = T.alloc_local([2], "float32")  # the row's (max, sum)
                for k in T.unroll(per_lane):
                    load16_evict_last(held, k * vec, x, (row, (k * lanes + lane) * vec))
                stat[0] = -T.infinity("float32")
                for i in T.unroll(held_elems):
                    values[i] = T.cast(held[i], "float32")
                    stat[0] = T.max(stat[0], values[i])
                for s in T.unroll(steps):
                    stat[0] = T.max(stat[0], T.shfl_xor(stat[0], T.shift_left(1, s)))
                stat[1] = 0.0
                if op_kind == "softmax":
                    for i in T.unroll(held_elems):
                        values[i] = exp_shifted(values[i], stat[0])
                        stat[1] += values[i]
                else:
                    for i in T.unroll(held_elems):
                        stat[1] += exp_shifted(values[i], stat[0])
                for s in T.unroll(steps):
                    stat[1] += T.shfl_xor(stat[1], T.shift_left(1, s))
                if op_kind == "softmax":
                    # One reciprocal a row, then a multiply an element, as SoftmaxKernel does;
                    # a division an element measured up to 9% slower.
                    stat[1] = 1.0 / stat[1]
                    for i in T.unroll(held_elems):
                        result[i] = T.cast(values[i] * stat[1], out_dtype)
                else:
                    stat[1] = T.log(stat[1])
                    for i in T.unroll(held_elems):
                        result[i] = T.cast(values[i] - stat[0] - stat[1], out_dtype)
                if bx * rows_per_cta + tx // lanes < M:
                    for k in T.unroll(per_lane):
                        for i in T.vectorized(vec):
                            y[row, (k * lanes + lane) * vec + i] = result[k * vec + i]

        return main

    return build


class SoftmaxWarpRowsKernel(_SoftmaxKernelBase):
    """Softmax / log-softmax of rows of up to 2 KB, a group of a warp's lanes a row.

    A row of ``N`` elements is ``N`` over the 16-byte vector width vectors, at most 128.
    A lane holds one to four of them, and a power of two of adjacent lanes holds the row:
    up to 16 lanes for rows of up to 64 vectors, a whole warp past that. Rows whose vectors
    do not split that way, and wider rows, stay on :class:`SoftmaxKernel`.
    """

    preferred_over = frozenset({"softmax_fwd"})

    @classmethod
    def _plan(cls, call: SoftmaxCall) -> Optional[tuple]:
        """``(lanes, per_lane)`` for *call*'s rows, or ``None``."""
        vec = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        vectors = call.n // vec
        # A 2 KB row on a whole warp took 31.97 us at (32768, 512) float32 against 33.3 on
        # SoftmaxKernel; 4 KB rows run SoftmaxKernel's register-held row.
        if call.n % vec or vectors > 128:
            return None
        # Two vectors a lane measured faster than one at 128 and 256 16-bit columns; a
        # 128-vector row took 31.97 us on 32 lanes of four against 32.03 on 16 of eight.
        lane_cap = WARP_LANES // 2 if vectors <= 64 else WARP_LANES
        for per_lane in range(max(min(2, vectors), -(-vectors // lane_cap)), 5):
            lanes = vectors // per_lane
            if vectors % per_lane == 0 and lanes & (lanes - 1) == 0:
                return lanes, per_lane
        return None

    @classmethod
    def refusal(cls, call: SoftmaxCall) -> "str | None":
        if cls._plan(call) is None:
            return f"no plan of this kernel holds a row of {call.n}"
        return super().refusal(call)

    def __init__(self, call: SoftmaxCall):
        super().__init__(device_index=call.device.index)
        self.call = call
        self.dtype = call.dtype
        lanes, per_lane = self._plan(call)
        self.kernel = _softmax_warp_rows_kernel(
            call.m,
            call.n,
            call.op_kind,
            self.dtype_str,
            self.dtype_to_str(call.out_dtype),
            lanes=lanes,
            per_lane=per_lane,
            # Four warps a CTA measured as fast as eight or more at every width.
            warps=4,
        )
        self.init_config(None)

    @property
    def default_config(self) -> dict:
        return {}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``call.axis`` of the contiguous input *x*."""
        y = self.program(self._compile)(self._rows(x))
        return restore_same_shape(y, self.call.shape, (self.call.axis,))

    def _compile(self, cfg: dict):
        """The program."""
        return self.kernel()
