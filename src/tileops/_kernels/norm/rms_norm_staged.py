"""RMS normalization with two asynchronously staged rows per CTA.

Separate copy groups allow the first row's stores to overlap the second row's loads.
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._csrc import csrc_include
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.norm.call_spec import LayerNormCall
from tileops._kernels.norm.rms_norm import _RMSNormRowsKernel
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["RMSNormStagedKernel"]

_THREADS = 128
_ROWS = 2
# Eight CTAs fit the supported SMs' thread and register limits at 64 registers/thread.
# Dispatch also checks the shared-memory footprint.
_CTAS_PER_SM = 8


@target_cache(maxsize=32)
def _rms_norm_staged_kernel(M, N, eps, dtype, has_weight):
    vec = VECTOR_ACCESS_BYTES // getattr(torch, dtype).itemsize
    steps = N // (_THREADS * vec)
    warps = _THREADS // WARP_LANES

    @tilelang.jit(out_idx=[2], compile_flags=csrc_include("streaming_load.h"))
    def build():
        @T.macro
        def normalize(weight, y, tile, sums, piece, scale, total, tx, row, g, pending):
            """Scale row *g* of the tile once all but *pending* copy groups have landed."""
            T.ptx_wait_group(pending)
            total[0] = T.cast(0.0, "float32")
            for k in T.unroll(steps):
                for c in T.vectorized(vec):
                    piece[c] = tile[g, (k * _THREADS + tx) * vec + c]
                for c in T.unroll(vec):
                    total[0] += T.cast(piece[c], "float32") * T.cast(piece[c], "float32")
            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                total[0] += T.shfl_xor(
                    total[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                )
            if tx % WARP_LANES == 0:
                sums[g, tx // WARP_LANES] = total[0]
            T.sync_threads()
            total[0] = sums[g, 0]
            for w in T.unroll(1, warps):
                total[0] += sums[g, w]
            total[0] = T.rsqrt(total[0] / float(N) + eps)
            for k in T.unroll(steps):
                at = (k * _THREADS + tx) * vec
                for c in T.vectorized(vec):
                    piece[c] = tile[g, at + c]
                if has_weight:
                    for c in T.vectorized(vec):
                        scale[c] = weight[at + c]
                for c in T.unroll(vec):
                    if has_weight:
                        piece[c] = T.cast(
                            T.cast(piece[c], "float32") * total[0] * T.cast(scale[c], "float32"),
                            dtype,
                        )
                    else:
                        piece[c] = T.cast(T.cast(piece[c], "float32") * total[0], dtype)
                if row < M:
                    for c in T.vectorized(vec):
                        y[row, at + c] = piece[c]

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N if has_weight else 1,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(T.ceildiv(M, _ROWS), threads=_THREADS) as cta:
                T.annotate_min_blocks_per_sm(_CTAS_PER_SM)
                tx = T.get_thread_binding()
                tile = T.alloc_shared((_ROWS, N), dtype)
                sums = T.alloc_shared((_ROWS, warps), "float32")
                piece = T.alloc_local((vec,), dtype)
                scale = T.alloc_local((vec,), dtype)
                total = T.alloc_local((1,), "float32")
                for g in T.unroll(_ROWS):
                    for k in T.unroll(steps):
                        T.call_extern(
                            "handle",
                            "tileops::cp_async16",
                            T.address_of(tile[g, (k * _THREADS + tx) * vec]),
                            # A row past the last reads the last row and stores nothing.
                            T.address_of(
                                x[T.min(cta * _ROWS + g, M - 1), (k * _THREADS + tx) * vec]
                            ),
                        )
                    T.ptx_commit_group()
                normalize(weight, y, tile, sums, piece, scale, total, tx, cta * _ROWS, 0, 1)
                normalize(weight, y, tile, sums, piece, scale, total, tx, cta * _ROWS + 1, 1, 0)

        return main

    return build


class RMSNormStagedKernel(_RMSNormRowsKernel):
    """RMS normalization of a launch whose rows, two a CTA, fill one wave.

    Serves launches that leave less than a sixteenth of one wave of two rows a CTA free and do
    not exceed it, of rows of at least three whole vectors a thread.
    """

    # SM86 and SM89 hold 100 KB of shared memory an SM: eight CTAs of the shortest row this
    # kernel takes do not fit there, so the shared-memory refusal turns down every call.
    supported_archs: list[int] = [80, 90]
    _builder = staticmethod(_rms_norm_staged_kernel)

    @classmethod
    def refusal(cls, call: LayerNormCall) -> Optional[str]:
        vec = VECTOR_ACCESS_BYTES // call.dtype.itemsize
        wave = _CTAS_PER_SM * call.sm_count
        # At least three vectors a thread amortize the shared-memory staging and its barriers.
        min_vectors = 3
        if call.n % (_THREADS * vec) or call.n // (_THREADS * vec) < min_vectors:
            return f"a row of {call.n} elements is not {min_vectors} or more whole thread chunks"
        rows = _ROWS * wave
        # Staging serves a nearly full launch within the eight-CTA budget an SM; the free-slot
        # fraction is a conservative performance cutoff.
        if not rows - rows // 16 < call.rows <= rows:
            return f"{call.rows} rows do not fill one wave of {_ROWS} rows a CTA"
        # The SM keeps the difference from the per-block opt-in limit for each resident CTA.
        reserved = call.smem_per_sm - call.smem_budget
        footprint = _ROWS * (call.n * call.dtype.itemsize + _THREADS // WARP_LANES * 4) + reserved
        if _CTAS_PER_SM * footprint > call.smem_per_sm:
            return (
                f"{_CTAS_PER_SM} CTAs of {_ROWS} rows of {call.n} do not fit an SM's shared memory"
            )
        return super().refusal(call)
