import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel
from top.utils import get_sm_version

__all__ = [
    'GemmKernel',
]


def _gemm_kernel(m: int,
                 n: int,
                 k: int,
                 trans_a: bool,
                 trans_b: bool,
                 dtype: str = 'float16') -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_func(block_m: int, block_n: int, block_k: int, threads: int, num_stages: int,
                   enable_rasteration: bool) -> Callable:

        a_shape = (k, m) if trans_a else (m, k)
        b_shape = (n, k) if trans_b else (k, n)
        a_shared_shape = (block_k, block_m) if trans_a else (block_m, block_k)
        b_shared_shape = (block_n, block_k) if trans_b else (block_k, block_n)

        @T.prim_func
        def _gemm_main(
                a: T.Tensor(a_shape, dtype),  # type: ignore
                b: T.Tensor(b_shape, dtype),  # type: ignore
                c: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                    T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
                a_shared = T.alloc_shared(a_shared_shape, dtype)
                b_shared = T.alloc_shared(b_shared_shape, dtype)
                c_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                c_shared = T.alloc_shared((block_m, block_n), dtype)

                T.annotate_layout({
                    c_shared: tilelang.layout.make_swizzled_layout(c_shared),
                })
                T.use_swizzle(10, enable=enable_rasteration)

                T.clear(c_local)

                for _k in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    if not trans_a:
                        # a: (m, k)
                        T.copy(a[by * block_m, _k * block_k], a_shared)  # [block_m, block_k]
                    else:
                        # a: (k, m)
                        T.copy(a[_k * block_k, by * block_m], a_shared)  # [block_k, block_m]

                    if not trans_b:
                        # b: (k, n)
                        T.copy(b[_k * block_k, bx * block_n], b_shared)  # [block_k, block_n]
                    else:
                        # b: (n, k)
                        T.copy(b[bx * block_n, _k * block_k], b_shared)  # [block_n, block_k]
                    T.gemm(a_shared, b_shared, c_local, trans_a, trans_b)

                T.copy(c_local, c_shared)
                T.copy(c_shared, c[by * block_m, bx * block_n])

        return _gemm_main

    return _gemm_func


@torch.library.custom_op("top::gemm_wrapped_kernel", mutates_args=())
def _gemm_wrapped_kernel(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    enable_rasteration: bool,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return _gemm_kernel(m, n, k, trans_a, trans_b, dtype)(block_m, block_n, block_k, threads,
                                                          num_stages, enable_rasteration)(a, b)


@_gemm_wrapped_kernel.register_fake
def _(m: int, n: int, k: int,
      trans_a: bool, trans_b: bool,
      dtype: str, block_m: int, block_n: int, block_k: int,
      num_stages: int, threads: int, enable_rasteration: bool,
      *inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.empty((m, n), dtype=inputs[0].dtype, device=inputs[0].device)


class GemmKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False,
                 trans_a: bool = False,
                 trans_b: bool = False) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        sm_version = get_sm_version()

        if sm_version in {80}:
            return {
                "block_m": 128,
                "block_n": 256,
                "block_k": 32,
                "num_stages": 2,
                "threads": 128,
                "enable_rasteration": True
            }
        if sm_version in {90}:
            return {
                "block_m": 128,
                "block_n": 256,
                "block_k": 64,
                "num_stages": 3,
                "threads": 256,
                "enable_rasteration": True
            }

        return {
            "block_m": 128,
            "block_n": 256,
            "block_k": 32,
            "num_stages": 0,
            "threads": 128,
            "enable_rasteration": True
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        block_m = [64, 128, 256]
        block_n = [64, 128, 256]
        block_k = [32, 64]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        enable_rasteration = [True, False]
        _configs = list(
            itertools.product(block_m, block_n, block_k, num_stages, threads, enable_rasteration))

        return [{
            'block_m': c[0],
            'block_n': c[1],
            'block_k': c[2],
            'num_stages': c[3],
            'threads': c[4],
            'enable_rasteration': c[5]
        } for c in _configs]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _gemm_wrapped_kernel(self.m, self.n, self.k, self.trans_a, self.trans_b,
                                    self.dtype_str, self.config["block_m"], self.config["block_n"],
                                    self.config["block_k"], self.config["num_stages"],
                                    self.config["threads"], self.config["enable_rasteration"], a, b)


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
                "reduce_threads": 32,
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


# TODO: add persistent, split-k, steam-k...
