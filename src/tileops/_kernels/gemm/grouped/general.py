import itertools
import math
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.gemm.grouped.call_spec import (
    GroupedGEMMCall,
    GroupedGEMMFwdInterface,
    grouped_gemm_entry,
)
from tileops._kernels.grouped_tiling import GroupTiling
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = [
    "GroupedGEMMKernel",
]


@target_cache(maxsize=32)
def _grouped_gemm_kernel(batch_sum, batch_count, N, K, transpose_a, transpose_b, dtype="float16"):
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _grouped_gemm_func(block_m, block_n, block_k, num_stages, threads):
        if not transpose_a:
            # NT / NN: groups split the rows of A.
            A_shape = (batch_sum, K)
            C_shape = (batch_sum, N)
            A_shared_shape = (block_m, block_k)
            if transpose_b:
                B_shape = (batch_count, N, K)
                B_shared_shape = (block_n, block_k)
            else:
                B_shape = (batch_count, K, N)
                B_shared_shape = (block_k, block_n)

            # Tiles enumerated per group; CTAs past the real count exit.
            _tiling = GroupTiling(batch_count, block_m)
            _num_pid_m = _tiling.tile_upper_bound(batch_sum)
            _num_pid_n = math.ceil(N / block_n)

            @T.prim_func
            def _grouped_gemm_main(
                A: T.Tensor(A_shape, dtype),  # type: ignore
                B: T.Tensor(B_shape, dtype),  # type: ignore
                C: T.Tensor(C_shape, dtype),  # type: ignore
                batch_sizes: T.Tensor([batch_count], "int32"),
                batch_offsets: T.Tensor([batch_count], "int32"),
            ):
                with T.Kernel(_num_pid_m * _num_pid_n, threads=threads) as (pid,):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_m, block_n], accum_dtype)
                    s_cum = T.alloc_shared([batch_count + 1], "int32")
                    lo = T.alloc_local([1], "int32")
                    hi = T.alloc_local([1], "int32")
                    row = T.alloc_local([1], "int32")
                    cur_batch_idx = T.alloc_local([1], "int32")

                    _tiling.cumsum(batch_sizes, s_cum)

                    # Consecutive pids walk the N tiles of one M tile.
                    m_tile = pid // _num_pid_n
                    n_start = (pid % _num_pid_n) * block_n

                    if m_tile < s_cum[batch_count]:
                        _tiling.decode(m_tile, s_cum, lo, hi, cur_batch_idx, row)
                        m_start = batch_offsets[cur_batch_idx[0]] + row[0]
                        actual_rows = T.min(block_m, batch_sizes[cur_batch_idx[0]] - row[0])
                        actual_cols = T.min(block_n, N - n_start)
                        T.clear(C_local)

                        for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                            for i, j in T.Parallel(block_m, block_k):
                                A_shared[i, j] = T.if_then_else(
                                    i < actual_rows and j < K - k * block_k,
                                    A[m_start + i, k * block_k + j],
                                    0,
                                )
                            if transpose_b:
                                for i, j in T.Parallel(block_n, block_k):
                                    B_shared[i, j] = T.if_then_else(
                                        j < K - k * block_k and i < actual_cols,
                                        B[cur_batch_idx[0], n_start + i, k * block_k + j],
                                        0,
                                    )
                            else:
                                for i, j in T.Parallel(block_k, block_n):
                                    B_shared[i, j] = T.if_then_else(
                                        i < K - k * block_k and j < actual_cols,
                                        B[cur_batch_idx[0], k * block_k + i, n_start + j],
                                        0,
                                    )
                            T.gemm(A_shared, B_shared, C_local, transpose_B=transpose_b)
                        for i, j in T.Parallel(block_m, block_n):
                            if i < actual_rows and j < actual_cols:
                                C[m_start + i, n_start + j] = C_local[i, j]

        else:
            # TN / TT: groups split the contraction; one output matrix per group.
            A_shape = (batch_sum, N)
            C_shape = (batch_count, N, K)
            A_shared_shape = (block_m, block_n)
            if transpose_b:
                B_shape = (K, batch_sum)
                B_shared_shape = (block_k, block_m)
            else:
                B_shape = (batch_sum, K)
                B_shared_shape = (block_m, block_k)

            @T.prim_func
            def _grouped_gemm_main(
                A: T.Tensor(A_shape, dtype),  # type: ignore
                B: T.Tensor(B_shape, dtype),  # type: ignore
                C: T.Tensor(C_shape, dtype),  # type: ignore
                batch_sizes: T.Tensor([batch_count], "int32"),
                batch_offsets: T.Tensor([batch_count], "int32"),
            ):
                with T.Kernel(
                    batch_count, T.ceildiv(N, block_n) * T.ceildiv(K, block_k), threads=threads
                ) as (bx, by):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_n, block_k], accum_dtype)

                    n_block_idx = by // T.ceildiv(K, block_k)
                    k_block_idx = by % T.ceildiv(K, block_k)
                    n_start = n_block_idx * block_n
                    k_start = k_block_idx * block_k
                    actual_N = T.min(block_n, N - n_start)
                    actual_K = T.min(block_k, K - k_start)
                    T.clear(C_local)

                    batch_start = batch_offsets[bx]
                    batch_size = batch_sizes[bx]

                    for m in T.Pipelined(T.ceildiv(batch_size, block_m), num_stages=num_stages):
                        m_start = batch_start + m * block_m
                        actual_rows = T.min(block_m, batch_size - m * block_m)
                        for i, j in T.Parallel(block_m, block_n):
                            A_shared[i, j] = T.if_then_else(
                                i < actual_rows and j < actual_N, A[m_start + i, n_start + j], 0
                            )
                        if transpose_b:
                            for i, j in T.Parallel(block_k, block_m):
                                B_shared[i, j] = T.if_then_else(
                                    i < actual_K and j < actual_rows, B[k_start + i, m_start + j], 0
                                )
                        else:
                            for i, j in T.Parallel(block_m, block_k):
                                B_shared[i, j] = T.if_then_else(
                                    i < actual_rows and j < actual_K, B[m_start + i, k_start + j], 0
                                )
                        T.gemm(
                            A_shared, B_shared, C_local, transpose_A=True, transpose_B=transpose_b
                        )
                    for i, j in T.Parallel(block_n, block_k):
                        if i < actual_N and j < actual_K:
                            C[bx, n_start + i, k_start + j] = C_local[i, j]

        return _grouped_gemm_main

    return _grouped_gemm_func


class GroupedGEMMKernel(Kernel, GroupedGEMMFwdInterface):
    supported_archs: list[int] = [80, 86, 89, 90]
    general: bool = True
    # (block_m, block_n, block_k) by layout (transpose_a, transpose_b); two stages, 128 threads.
    _DEFAULT_TILES = {
        (False, True): (64, 256, 64),
        (False, False): (128, 128, 32),
        (True, False): (32, 128, 128),
        (True, True): (64, 256, 64),
    }

    @classmethod
    def entry_for(cls, call: GroupedGEMMCall) -> Entry:
        return grouped_gemm_entry(cls, call)

    def __init__(
        self,
        batch_sum,
        batch_count,
        N,
        K,
        dtype,
        transpose_a: bool = False,
        transpose_b: bool = True,
        config: dict | None = None,
        *,
        device_index: int | None = None,
    ):
        super().__init__(device_index=device_index)
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

        self.kernel = _grouped_gemm_kernel(
            self.batch_sum,
            self.batch_count,
            self.N,
            self.K,
            self.transpose_a,
            self.transpose_b,
            self.dtype_str,
        )

        self.init_config(config)

    @property
    def autotune_supply_prog(self):
        """Supply autotuning with even ``batch_sizes`` and their tight prefix-sum offsets.

        Tight offsets end the last group at ``batch_sum``, so every tile stays inside ``A``.
        """
        from tilelang.utils.device import get_current_device
        from tilelang.utils.tensor import get_tensor_supply

        default_supply = get_tensor_supply(tilelang.TensorSupplyType.Auto)
        batch_count = self.batch_count
        base, extra = divmod(self.batch_sum, batch_count)

        def supply_prog(params):
            device = get_current_device()
            sizes = torch.full((batch_count,), base, dtype=torch.int32, device=device)
            sizes[:extra] += 1
            offsets = torch.zeros(batch_count, dtype=torch.int32, device=device)
            offsets[1:] = torch.cumsum(sizes[:-1], dim=0)

            # Matched by position among themselves: the prim_func takes
            # batch_sizes, then batch_offsets.
            is_metadata = [
                str(p.dtype) == "int32" and list(p.shape) == [batch_count] for p in params
            ]
            if sum(is_metadata) != 2:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects 2 int32 [{batch_count}] "
                    f"parameters (batch_sizes, batch_offsets), got {sum(is_metadata)}"
                )

            seen = 0
            inputs = []
            for param, metadata in zip(params, is_metadata, strict=True):
                if metadata:
                    inputs.append(sizes if seen == 0 else offsets)
                    seen += 1
                else:
                    inputs.append(default_supply(param))
            return inputs

        return supply_prog

    @property
    def default_config(self) -> dict:
        block_m, block_n, block_k = self._DEFAULT_TILES[(self.transpose_a, self.transpose_b)]
        return {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "num_stages": 2,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_m": bm, "block_n": bn, "block_k": bk, "num_stages": ns, "threads": threads}
            for bm, bn, bk, ns, threads in itertools.product(
                (32, 64, 128, 256), (32, 64, 128, 256), (32, 64, 128, 256), (0, 1, 2, 3), (128, 256)
            )
        ]

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self.program(self._compile)(a, b, batch_sizes, batch_offsets)

    def _compile(self, cfg: dict) -> Callable:
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(
            cfg["block_m"], cfg["block_n"], cfg["block_k"], cfg["num_stages"], cfg["threads"]
        )
