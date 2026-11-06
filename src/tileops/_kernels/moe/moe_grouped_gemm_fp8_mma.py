"""Block-scaled FP8 grouped GEMM on masked slabs or aligned SM89 MMA rows."""

import itertools
from typing import Optional

import tilelang
import tilelang.language as T

from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.moe.call_spec import MGroupedGEMMFP8Call
from tileops._kernels.moe.layout import aligned_group
from tileops._kernels.moe.moe_grouped_gemm_fp8 import MoEGroupedGEMMFP8Kernel

__all__ = ["MoEGroupedGEMMFP8MMAKernel"]


@target_cache(maxsize=32)
def _moe_grouped_gemm_fp8_mma_kernel(groups: int, rows: Optional[int], n: int, k: int):
    scale_block = 128
    contiguous = rows is None

    @tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
    def factory(block_m: int, block_n: int):
        m = T.dynamic("m") if contiguous else rows
        a_shape = (m, k) if contiguous else (groups, m, k)
        sa_shape = (m, k // scale_block) if contiguous else (groups, m, k // scale_block)
        out_shape = (m, n) if contiguous else (groups, m, n)
        layout_shape = (m,) if contiguous else (groups,)

        @T.prim_func
        def main(
            a: T.Tensor(a_shape, "float8_e4m3fn"),
            sa: T.Tensor(sa_shape, "float32"),
            b: T.Tensor((groups, n, k), "float8_e4m3fn"),
            sb: T.Tensor((groups, n // scale_block, k // scale_block), "float32"),
            counts: T.Tensor(layout_shape, "int32"),
            out: T.Tensor(out_shape, "bfloat16"),
        ):
            with T.Kernel(
                T.ceildiv(m, block_m),
                T.ceildiv(n, block_n),
                1 if contiguous else groups,
                threads=128,
            ) as (
                bm,
                bn,
                slab,
            ):
                aa = T.alloc_shared((block_m, scale_block), "float8_e4m3fn")
                bb = T.alloc_shared((block_n, scale_block), "float8_e4m3fn")
                acc = T.alloc_fragment((block_m, block_n), "float32")
                part = T.alloc_fragment((block_m, block_n), "float32")
                g = aligned_group(counts, bm * block_m, groups) if contiguous else slab
                T.clear(acc)
                active = g >= 0 if contiguous else bm * block_m < T.min(counts[g], m)
                if active:
                    for ki in T.serial(k // scale_block):
                        if contiguous:
                            T.copy(
                                a[
                                    bm * block_m : bm * block_m + block_m,
                                    ki * scale_block : ki * scale_block + scale_block,
                                ],
                                aa,
                            )
                        else:
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
                            if bm * block_m + i < m:
                                acc[i, j] += (
                                    part[i, j]
                                    * (
                                        sa[bm * block_m + i, ki]
                                        if contiguous
                                        else sa[g, bm * block_m + i, ki]
                                    )
                                    * sb[g, (bn * block_n + j) // scale_block, ki]
                                )
                    for i, j in T.Parallel(block_m, block_n):
                        if bm * block_m + i < m and bn * block_n + j < n:
                            if contiguous:
                                out[bm * block_m + i, bn * block_n + j] = acc[i, j]
                            else:
                                out[g, bm * block_m + i, bn * block_n + j] = acc[i, j]

        return main

    return factory


class MoEGroupedGEMMFP8MMAKernel(MoEGroupedGEMMFP8Kernel):
    """Masked slabs and aligned per-row FP8 layouts, one MMA per 128-K scale block."""

    supported_archs: list[int] = [89]
    _BLOCK_M = (32, 64)

    @classmethod
    def refusal(cls, call: MGroupedGEMMFP8Call) -> Optional[str]:
        """Masked slabs, or aligned rows whose metadata names each row's expert."""
        if call.kind != "masked" and (call.kind, call.packing, call.metadata_kind) != (
            "contiguous",
            "aligned",
            "per_row",
        ):
            return "serves masked slabs or aligned per-row layouts"
        return super().refusal(call)

    def __init__(self, call: MGroupedGEMMFP8Call) -> None:
        device_index = call.device.index if call.device is not None else None
        Kernel.__init__(self, device_index=device_index)
        self.call = call
        self.scale_k = call.k // 128
        self.alignment = call.alignment if call.packing == "aligned" else 0
        self.kernel = _moe_grouped_gemm_fp8_mma_kernel(call.num_groups, call.max_m, call.n, call.k)
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {"block_m": 32, "block_n": 64}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": m, "block_n": n}
            for m, n in itertools.product(self._BLOCK_M, (64, 128))
            if self.alignment % m == 0
        ]

    def _check_config(self, config: dict) -> None:
        if set(config) != {"block_m", "block_n"}:
            raise ValueError("FP8 MMA takes only block_m and block_n")
        if not all(type(value) is int for value in config.values()):
            raise ValueError("FP8 MMA tile sizes must be integers")
        if config["block_m"] not in self._BLOCK_M or config["block_n"] not in (64, 128):
            raise ValueError("FP8 MMA needs block_m in (32, 64), block_n in (64, 128)")
        if self.alignment % config["block_m"]:
            raise ValueError("FP8 MMA block_m must divide the aligned layout's alignment")

    def init_config(self, config: Optional[dict] = None) -> None:
        if config is not None and not set(config) <= {"block_m", "block_n"}:
            raise ValueError("FP8 MMA takes only block_m and block_n")
        Kernel.init_config(self, config)
        self._check_config(self.config)
