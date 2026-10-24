"""Produce causal Mamba-2 chunk coupling matrices ``C @ B.T``.

The kernel reads sequence-fused group-owned inputs and writes the lower
triangle of each ``(chunk, group)`` matrix directly in its storage dtype.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.mamba.call_spec import SSDChunkCouplingCall, SSDChunkCouplingFwdInterface

__all__ = ["SSDChunkCouplingKernel"]


@functools.lru_cache(maxsize=32)
def _ssd_chunk_coupling_kernel(
    batch: int,
    num_chunks: int,
    n_groups: int,
    chunk_len: int,
    d_state: int,
    dtype: str,
) -> Callable:
    """Build the lower-triangular ``C @ B.T`` kernel for each chunk and group."""
    accum_dtype = "float"

    B = batch
    C = num_chunks
    G = n_groups
    Q = chunk_len
    N = d_state
    S = C * Q  # Total sequence length

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_l: int,
        block_s: int,
        block_n: int,
        threads: int,
    ):
        C_shape = (B, S, G, N)  # Accept contiguous (B, S, G, N) shape
        B_shape = (B, S, G, N)  # Accept contiguous (B, S, G, N) shape
        cb_shape = (B, C, G, Q, Q)

        @T.prim_func
        def main(
            C_mat: T.Tensor(C_shape, dtype),  # type: ignore
            B_mat: T.Tensor(B_shape, dtype),  # type: ignore
            cb: T.Tensor(cb_shape, dtype),  # type: ignore  OUTPUT in dtype, not accum_dtype
        ):
            with T.Kernel(
                T.ceildiv(Q, block_l),  # l tiles
                T.ceildiv(Q, block_s),  # s tiles
                B * C * G,  # batch * chunks * groups
                threads=threads,
            ) as (bl, bs, bcg):
                bcg_tmp = bcg
                bb = bcg_tmp // (C * G)
                cg = bcg_tmp % (C * G)
                bc = cg // G
                bg = cg % G

                l0 = bl * block_l
                s0 = bs * block_s

                # Check causality: only compute if s0 < l0 + block_l
                # This prunes upper-triangular tiles
                if s0 < l0 + block_l:
                    C_tile = T.alloc_shared((block_l, block_n), dtype)
                    B_tile = T.alloc_shared((block_s, block_n), dtype)

                    # Swizzled layout for better memory coalescing
                    T.annotate_layout(
                        {
                            C_tile: tilelang.layout.make_swizzled_layout(C_tile),
                            B_tile: tilelang.layout.make_swizzled_layout(B_tile),
                        }
                    )

                    acc = T.alloc_fragment((block_l, block_s), accum_dtype)
                    T.clear(acc)

                    for n_blk in T.serial(T.ceildiv(N, block_n)):
                        n0 = n_blk * block_n

                        for ll, nn in T.Parallel(block_l, block_n):
                            l_abs = l0 + ll
                            n_abs = n0 + nn
                            if l_abs < Q and n_abs < N:
                                C_tile[ll, nn] = C_mat[bb, bc * Q + l_abs, bg, n_abs]
                            else:
                                C_tile[ll, nn] = T.cast(T.float32(0.0), dtype)

                        for ss, nn in T.Parallel(block_s, block_n):
                            s_abs = s0 + ss
                            n_abs = n0 + nn
                            if s_abs < Q and n_abs < N:
                                B_tile[ss, nn] = B_mat[bb, bc * Q + s_abs, bg, n_abs]
                            else:
                                B_tile[ss, nn] = T.cast(T.float32(0.0), dtype)

                        T.sync_threads()

                        # GEMM: acc[l, s] += C[l, n] @ B[s, n]^T
                        T.gemm(
                            C_tile,
                            B_tile,
                            acc,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                        T.sync_threads()

                    for ll, ss in T.Parallel(block_l, block_s):
                        l_abs = l0 + ll
                        s_abs = s0 + ss
                        if l_abs < Q and s_abs < Q:
                            if s_abs <= l_abs:
                                cb[bb, bc, bg, l_abs, s_abs] = T.cast(acc[ll, ss], dtype)
                            else:
                                cb[bb, bc, bg, l_abs, s_abs] = T.cast(T.float32(0.0), dtype)
                else:
                    # Upper-triangular tile: write zeros explicitly
                    for ll, ss in T.Parallel(block_l, block_s):
                        l_abs = l0 + ll
                        s_abs = s0 + ss
                        if l_abs < Q and s_abs < Q:
                            cb[bb, bc, bg, l_abs, s_abs] = T.cast(T.float32(0.0), dtype)

        return main

    return kernel_func


class SSDChunkCouplingKernel(Kernel, SSDChunkCouplingFwdInterface):
    """CB (C@B) matrix producer kernel.

    Computes cb[b,c,g,l,s] = sum_n C[b,c,g,l,n] * B[b,c,g,s,n]
    with causal masking (cb[l,s] = 0 if s > l).

    Args:
        batch: Batch size
        num_chunks: Number of chunks
        n_groups: Number of groups
        chunk_len: Chunk length (Q)
        d_state: State dimension (N)
        dtype: Data type (fp16/bf16)
        config: Tile configuration
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: SSDChunkCouplingCall) -> Entry:
        return call, lambda: cls(
            call.batch,
            call.seq_len // call.chunk_len,
            call.n_groups,
            call.chunk_len,
            call.d_state,
            call.dtype,
        )

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_groups: int,
        chunk_len: int,
        d_state: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_groups = n_groups
        self.chunk_len = chunk_len
        self.d_state = d_state
        self.dtype = dtype

        self.kernel = _ssd_chunk_coupling_kernel(
            batch, num_chunks, n_groups, chunk_len, d_state, self.dtype_str
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default tile configuration optimized for typical Mamba workloads."""
        return {
            "block_l": 64,
            "block_s": 64,
            "block_n": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        """Autotune search space."""
        configs = []
        for block_l in [32, 64, 128]:
            for block_s in [32, 64, 128]:
                for block_n in [32, 64, 128]:
                    for threads in [128, 256]:
                        if block_l <= self.chunk_len and block_s <= self.chunk_len:
                            configs.append(
                                {
                                    "block_l": block_l,
                                    "block_s": block_s,
                                    "block_n": block_n,
                                    "threads": threads,
                                }
                            )
        return configs

    def forward(
        self,
        C_mat: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            C_mat: [B, S, G, N]  dtype (contiguous)
            B_mat: [B, S, G, N]  dtype (contiguous)

        Returns:
            cb: [B, C, G, Q, Q]  dtype
        """
        C_mat = C_mat.contiguous()
        B_mat = B_mat.contiguous()
        return self.kernel(
            self.config["block_l"],
            self.config["block_s"],
            self.config["block_n"],
            self.config["threads"],
        )(
            C_mat,
            B_mat,
        )
