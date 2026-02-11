import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel
from top.utils import get_sm_version

__all__ = [
    'GemvKernel',
]


def _gemv_kernel(n: int, k: int, dtype: str = "float16") -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemv_func(
        block_n: int,
        reduce_threads: int,
        tile_k: int = 8,
    ) -> Callable:

        # MAX_TRANSACTION_SIZE_IN_BITS = 128
        # tile_k = MAX_TRANSACTION_SIZE_IN_BITS // 16
        block_k = reduce_threads * tile_k

        @T.prim_func
        def _gemv_main(
                a: T.Buffer((k,), dtype),
                b: T.Buffer((n, k), dtype),
                c: T.Buffer((n,), dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_n), threads=(block_n, reduce_threads)) as bn:
                tn = T.get_thread_binding(0)
                tk = T.get_thread_binding(1)
                a_local = T.alloc_local((tile_k,), dtype)
                b_local = T.alloc_local((tile_k,), dtype)
                c_accum = T.alloc_local((1,), accum_dtype)

                T.clear(c_accum)
                for bk in T.serial(T.ceildiv(k, block_k)):
                    for _k in T.vectorized(tile_k):
                        a_local[_k] = a[bk * block_k + tk * tile_k + _k]
                        b_local[_k] = b[bn * block_n + tn, bk * block_k + tk * tile_k + _k]
                    for _k in T.serial(tile_k):
                        c_accum[0] += a_local[_k].astype(accum_dtype) * b_local[_k].astype(
                            accum_dtype)
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
    tile_k: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _gemv_kernel(n, k, dtype)(block_n, reduce_threads, tile_k)(a, b)


@_gemv_wrapped_kernel.register_fake
def _(n: int, k: int,  # noqa: U100
      dtype: str, block_n: int, reduce_threads: int, tile_k: int,  # noqa: U100
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
        # From tilelang/examples/gemm/example_gemm_autotune.py
        sm_version = get_sm_version()

        if sm_version in {90}:
            return {
                "block_n": 32,
                "reduce_threads": 1,
                "tile_k": 8,
            }

        return {
            "block_n": 128,
            "reduce_threads": 32,
            "tile_k": 8,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        block_n = [64, 128, 256]
        reduce_threads = [16, 32]
        tile_k = [8, 16]
        _configs = list(itertools.product(block_n, reduce_threads, tile_k))

        return [{
            'block_n': c[0],
            'reduce_threads': c[1],
            'tile_k': c[2],
        } for c in _configs]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.flatten().contiguous()
        return _gemv_wrapped_kernel(self.n, self.k, self.dtype_str, self.config["block_n"],
                                    self.config["reduce_threads"], self.config["tile_k"], a, b)
