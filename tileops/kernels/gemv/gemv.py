from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.utils import get_sm_version, str2dtype

__all__ = [
    'GemvKernel',
]


def _gemv_kernel(n: int, k: int, dtype: str = "float16") -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemv_func(
        block_n: int,
        reduce_threads: int,
        num_stages: int,
    ) -> Callable:

        max_transaction_size_in_bits = 128
        tile_k = max_transaction_size_in_bits // (str2dtype[dtype].itemsize * 8)
        block_k = reduce_threads * tile_k

        @T.prim_func
        def _gemv_main(
                a: T.Buffer((k,), dtype),
                b: T.Buffer((n, k), dtype),
                c: T.Buffer((n,), dtype),
        ):
            # threads=(reduce_threads, block_n): tk=threadIdx.x is the fast-varying
            # dimension so consecutive warp threads access consecutive columns of B
            # (same row, stride-1) → coalesced 128-bit loads.
            with T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n)) as bn:
                tk = T.get_thread_binding(0)  # threadIdx.x — varies within a warp
                tn = T.get_thread_binding(1)  # threadIdx.y — one row per warp (when reduce_threads=32)
                c_accum = T.alloc_local((1,), accum_dtype)

                T.clear(c_accum)

                # O3: pipeline B loads through shared memory using T.Pipelined.
                # T.copy issues cp.async for the next tile while the current tile is
                # being consumed, hiding HBM3e latency.
                # num_stages=1 → sequential (no overlap), num_stages>=2 → actual pipeline.
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                a_local = T.alloc_local((tile_k,), dtype)

                for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    # disable_tma=True: use cp.async instead of TMA to avoid
                    # mbarrier requirements that TileLang cannot infer for
                    # manually-indexed b_shared in a non-wgmma kernel.
                    T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)
                    # a is tiny (fits in L1), load directly to registers
                    for _k in T.vectorized(tile_k):
                        a_local[_k] = a[bk * block_k + tk * tile_k + _k]
                    # FMA
                    for _k in T.serial(tile_k):
                        c_accum[0] += a_local[_k].astype(accum_dtype) * b_shared[
                            tn, tk * tile_k + _k].astype(accum_dtype)

                c_reduced = T.alloc_local((1,), accum_dtype)
                with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            c_accum[0],
                            True,
                            c_reduced[0],
                            tk,
                            dtype="handle",
                        ))

                c[bn * block_n + tn] = c_reduced[0]

        return _gemv_main

    return _gemv_func


@torch.library.custom_op("top::gemv_wrapped_kernel", mutates_args=())
def _gemv_wrapped_kernel(
    n: int,
    k: int,
    dtype: str,
    block_n: int,
    reduce_threads: int,
    num_stages: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _gemv_kernel(n, k, dtype)(block_n, reduce_threads, num_stages)(a, b)


@_gemv_wrapped_kernel.register_fake
def _(n: int, k: int,  # noqa: U100
      dtype: str, block_n: int, reduce_threads: int, num_stages: int,  # noqa: U100
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:  # noqa: U100
    return torch.empty((n,), dtype=inputs[0].dtype, device=inputs[0].device)


class GemvKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 n: int,
                 k: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self.dtype = dtype

        self.kernel = _gemv_kernel(n, k, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        sm_version = get_sm_version()

        if sm_version in {90}:
            # reduce_threads=32: full warp per row → coalesced B access + warp shuffle reduce
            # block_n=8: 256 threads/block, 448 blocks for n=7168 → ~3.4 blocks/SM on H200
            # num_stages=2: double-buffer B tile to hide HBM3e latency
            return {
                "block_n": 8,
                "reduce_threads": 32,
                "num_stages": 2,
            }

        return {
            "block_n": 32,
            "reduce_threads": 32,
            "num_stages": 1,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # num_stages=1: sequential shared-memory path (no overlap, baseline for comparison)
        # num_stages>=2: actual pipeline with cp.async prefetch to hide HBM latency
        return [
            {'block_n': bn, 'reduce_threads': rt, 'num_stages': ns}
            for bn in [1, 2, 4, 8, 16]
            for rt in [32]
            for ns in [1, 2, 3]
        ]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.flatten().contiguous()
        # Call the JIT-compiled kernel directly to avoid Python overhead from
        # closure recreation + JIT cache lookup in _gemv_wrapped_kernel on every
        # forward pass. _gemv_wrapped_kernel is kept for torch.compile compatibility.
        return self.kernel(
            self.config["block_n"],
            self.config["reduce_threads"],
            self.config["num_stages"],
        )(a, b)
