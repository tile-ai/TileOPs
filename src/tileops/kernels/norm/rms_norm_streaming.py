"""RMS normalization for rows exceeding shared memory: a pass to reduce, a pass to write.

A resident CTA walks rows in turn. It reads its row once to sum the squares, then again
to normalize and write it, the second time from the far end: the tiles it read last are
the likeliest still in L2.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tvm import DataType

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm.call_spec import LayerNormCall, RMSNormFwdInterface
from tileops.kernels.tiling import ALIGNMENT, align_up
from tileops.utils import WARP_LANES, get_sm_count

__all__ = ["RMSNormStreamingKernel"]


@functools.lru_cache(maxsize=32)
def _rms_norm_streaming_kernel(M, N, eps, dtype, has_weight, ctas):
    # The widest access that divides the row, so every row starts on one.
    vec = VECTOR_ACCESS_BYTES // (DataType(dtype).bits // 8)
    while N % vec:
        vec //= 2

    @tilelang.jit(out_idx=[2])
    def build(threads, accesses):
        tile = threads * accesses * vec  # elements one pass reads at a time
        tiles = -(-N // tile)
        warps = threads // WARP_LANES

        def normalize(value, rrms, scale):
            """*value* scaled by the row's reciprocal RMS and, where there is one, the weight."""
            normed = T.cast(value, "float32") * rrms
            return normed * T.cast(scale, "float32") if has_weight else normed

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N if has_weight else 1,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(ctas, threads=threads) as cta:
                tx = T.get_thread_binding()
                held = T.alloc_local([accesses * vec], dtype)
                scale = T.alloc_local([vec], dtype)
                acc = T.alloc_local([1], "float32")
                total = T.alloc_local([1], "float32")
                # Two slots, so a row's partials never overwrite the ones still being read.
                warp_sums = T.alloc_shared([2, warps], "float32")

                for step in T.serial(T.ceildiv(M - cta, ctas)):
                    row = cta + step * ctas
                    acc[0] = T.cast(0, "float32")
                    for t in T.serial(tiles):
                        for a in T.unroll(accesses):
                            col = t * tile + (a * threads + tx) * vec
                            if col < N:
                                for i in T.vectorized(vec):
                                    held[a * vec + i] = x[row, col + i]
                            else:
                                for i in T.vectorized(vec):
                                    held[a * vec + i] = T.cast(0, dtype)
                        for j in T.unroll(accesses * vec):
                            v = T.cast(held[j], "float32")
                            acc[0] += v * v
                    for k in T.unroll(WARP_LANES.bit_length() - 1):
                        acc[0] += T.shfl_xor(acc[0], T.shift_left(1, k))
                    if tx % WARP_LANES == 0:
                        warp_sums[step % 2, tx // WARP_LANES] = acc[0]
                    T.sync_threads()
                    total[0] = T.cast(0, "float32")
                    for w in T.unroll(warps):
                        total[0] += warp_sums[step % 2, w]
                    rrms = T.rsqrt(total[0] / float(N) + eps)

                    for t_back in T.serial(tiles):
                        t = tiles - 1 - t_back
                        for a in T.unroll(accesses):
                            col = t * tile + (a * threads + tx) * vec
                            if col < N:
                                for i in T.vectorized(vec):
                                    held[a * vec + i] = x[row, col + i]
                                if has_weight:
                                    for i in T.vectorized(vec):
                                        scale[i] = weight[col + i]
                                for i in T.unroll(vec):
                                    held[a * vec + i] = T.cast(
                                        normalize(held[a * vec + i], rrms, scale[i]), dtype
                                    )
                                for i in T.vectorized(vec):
                                    y[row, col + i] = held[a * vec + i]

        return main

    return build


class RMSNormStreamingKernel(Kernel, RMSNormFwdInterface):
    """Reduce a row in tiles, then reread it from the far end to normalize it."""

    supported_archs = [80, 86, 89, 90]
    preferred_over = frozenset({"rms_norm"})

    @classmethod
    def applies(cls, call: LayerNormCall) -> bool:
        element_bytes = call.dtype.itemsize
        budget = torch.cuda.get_device_properties(call.device).shared_memory_per_block_optin
        return align_up(call.n, ALIGNMENT) * element_bytes > budget

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)

    def __init__(self, n: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.n, self.eps, self.dtype = n, eps, dtype
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {"threads": 512, "accesses": 8}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        rows = x.reshape(-1, self.n)
        has_weight = weight is not None
        weight = weight.reshape(self.n) if has_weight else rows.new_empty(1)
        m = rows.shape[0]
        # The device the input is on, not whichever is current.
        # Two CTAs an SM, each walking its share of the rows.
        ctas = min(m, 2 * get_sm_count(x.device.index))
        self.kernel = _rms_norm_streaming_kernel(
            m, self.n, self.eps, self.dtype_str, has_weight, ctas
        )
        return self.kernel(**self.config)(rows, weight).reshape_as(x)
