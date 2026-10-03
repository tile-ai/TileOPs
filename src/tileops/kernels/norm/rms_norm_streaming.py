"""RMS normalization with bounded tile storage for rows exceeding shared memory."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.norm.call_spec import LayerNormCall, RMSNormFwdInterface
from tileops.kernels.tiling import ALIGNMENT, align_up

__all__ = ["RMSNormStreamingKernel"]


@functools.lru_cache(maxsize=32)
def _rms_norm_streaming_kernel(M, N, eps, dtype, has_weight):
    @tilelang.jit(out_idx=[2])
    def build(block_n, threads):
        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N if has_weight else 1,), dtype],
            y: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                partial = T.alloc_fragment((block_n,), "float32")
                total = T.alloc_fragment((1,), "float32")
                T.clear(partial)
                for tile in T.serial(T.ceildiv(N, block_n)):
                    for col in T.Parallel(block_n):
                        value = T.if_then_else(
                            tile * block_n + col < N,
                            T.cast(x[row, tile * block_n + col], "float32"),
                            0.0,
                        )
                        partial[col] += value * value
                T.reduce_sum(partial, total, dim=0)
                total[0] = T.rsqrt(total[0] / N + eps)
                for tile in T.serial(T.ceildiv(N, block_n)):
                    for col in T.Parallel(block_n):
                        if tile * block_n + col < N:
                            if has_weight:
                                y[row, tile * block_n + col] = (
                                    T.cast(x[row, tile * block_n + col], "float32")
                                    * total[0]
                                    * T.cast(weight[tile * block_n + col], "float32")
                                )
                            else:
                                y[row, tile * block_n + col] = (
                                    T.cast(x[row, tile * block_n + col], "float32") * total[0]
                                )

        return main

    return build


class RMSNormStreamingKernel(Kernel, RMSNormFwdInterface):
    """Reduce a row in tiles, then reread it to normalize with bounded storage."""

    supported_archs = [80, 86, 89, 90]
    preferred_over = frozenset({"rms_norm"})

    @classmethod
    def applies(cls, call: LayerNormCall) -> bool:
        element_bytes = torch.empty((), dtype=call.dtype).element_size()
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
        return {"block_n": 4096, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, x: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        rows = x.reshape(-1, self.n)
        has_weight = weight is not None
        weight = weight.reshape(self.n) if has_weight else rows.new_empty(1)
        self.kernel = _rms_norm_streaming_kernel(
            rows.shape[0], self.n, self.eps, self.dtype_str, has_weight
        )
        return self.kernel(**self.config)(rows, weight).reshape_as(x)
