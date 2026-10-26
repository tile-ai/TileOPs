"""Block-scaled masked FP8 grouped GEMM using SM89 MMA tiles."""

import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.moe.call_spec import MGroupedGEMMFP8Call
from tileops._kernels.moe.moe_grouped_gemm_fp8 import MoEGroupedGEMMFP8Kernel

__all__ = ["MoEGroupedGEMMFP8MMAKernel"]


@functools.lru_cache(maxsize=32)
def _moe_grouped_gemm_fp8_mma_kernel(groups: int, rows: int, n: int, k: int):
    scale_block = 128

    @tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
    def factory(block_m: int, block_n: int):
        @T.prim_func
        def main(
            a: T.Tensor((groups, rows, k), "float8_e4m3fn"),
            sa: T.Tensor((groups, rows, k // scale_block), "float32"),
            b: T.Tensor((groups, n, k), "float8_e4m3fn"),
            sb: T.Tensor((groups, n // scale_block, k // scale_block), "float32"),
            counts: T.Tensor((groups,), "int32"),
            out: T.Tensor((groups, rows, n), "bfloat16"),
        ):
            with T.Kernel(T.ceildiv(rows, block_m), T.ceildiv(n, block_n), groups, threads=128) as (
                bm,
                bn,
                g,
            ):
                aa = T.alloc_shared((block_m, scale_block), "float8_e4m3fn")
                bb = T.alloc_shared((block_n, scale_block), "float8_e4m3fn")
                acc = T.alloc_fragment((block_m, block_n), "float32")
                part = T.alloc_fragment((block_m, block_n), "float32")
                T.clear(acc)
                if bm * block_m < T.min(counts[g], rows):
                    for ki in T.serial(k // scale_block):
                        T.copy(
                            a[
                                g,
                                bm * block_m : bm * block_m + block_m,
                                ki * scale_block : ki * scale_block + scale_block,
                            ],
                            aa,
                        )
                        T.copy(
                            b[
                                g,
                                bn * block_n : bn * block_n + block_n,
                                ki * scale_block : ki * scale_block + scale_block,
                            ],
                            bb,
                        )
                        T.gemm(aa, bb, part, transpose_B=True, clear_accum=True)
                        for i, j in T.Parallel(block_m, block_n):
                            if bm * block_m + i < rows:
                                acc[i, j] += (
                                    part[i, j]
                                    * sa[g, bm * block_m + i, ki]
                                    * sb[g, (bn * block_n + j) // scale_block, ki]
                                )
                    for i, j in T.Parallel(block_m, block_n):
                        if bm * block_m + i < rows and bn * block_n + j < n:
                            out[g, bm * block_m + i, bn * block_n + j] = acc[i, j]

        return main

    return factory


class MoEGroupedGEMMFP8MMAKernel(MoEGroupedGEMMFP8Kernel):
    """The masked FP8 contract with one MMA contraction per 128-K scale block."""

    supported_archs: list[int] = [89]

    def __init__(self, call: MGroupedGEMMFP8Call) -> None:
        device_index = call.device.index if call.device is not None else None
        Kernel.__init__(self, device_index=device_index)
        self.call = call
        self.scale_k = call.k // 128
        self.kernel = _moe_grouped_gemm_fp8_mma_kernel(call.num_groups, call.max_m, call.n, call.k)
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {"block_m": 32, "block_n": 64}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"block_m": m, "block_n": n} for m, n in itertools.product((32, 64), (64, 128))]

    def _check_config(self, config: dict) -> None:
        if set(config) != {"block_m", "block_n"}:
            raise ValueError("FP8 MMA takes only block_m and block_n")
        if not all(type(value) is int for value in config.values()):
            raise ValueError("FP8 MMA tile sizes must be integers")
        if config["block_m"] not in (32, 64) or config["block_n"] not in (64, 128):
            raise ValueError("FP8 MMA needs block_m in (32, 64), block_n in (64, 128)")

    def init_config(self, config: Optional[dict] = None) -> None:
        if config is not None and not set(config) <= {"block_m", "block_n"}:
            raise ValueError("FP8 MMA takes only block_m and block_n")
        Kernel.init_config(self, config)
        self._check_config(self.config)
