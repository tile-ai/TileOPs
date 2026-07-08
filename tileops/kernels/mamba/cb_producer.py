"""
Mamba-2 CB (C@B) matrix producer kernel.

Computes: cb[b,c,g,l,s] = sum_n C[b,c,g,l,n] * B[b,c,g,s,n]
with causal mask: cb[b,c,g,l,s] = 0 if s > l

Replaces torch.compile(einsum) with specialized TileOps kernel that:
1. Exploits causal structure (only compute lower triangle)
2. Fuses mask application
3. Uses optimal tile sizes for QxQxN GEMM

Input:
  C: [B, C, G, Q, N]  dtype (fp16/bf16)
  B: [B, C, G, Q, N]  dtype (fp16/bf16)

Output:
  cb: [B, C, G, Q, Q]  dtype (direct output to avoid standalone cast)
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["CBProducerKernel"]


@functools.lru_cache(maxsize=32)
def _cb_producer_kernel(
    batch: int,
    num_chunks: int,
    n_groups: int,
    chunk_len: int,
    d_state: int,
    dtype: str,
) -> Callable:
    """
    Compute CB[b,c,g,l,s] = sum_n C[b,c,g,l,n] * B[b,c,g,s,n]
    with causal mask (s <= l)

    Grid: (num_l_tiles, num_s_tiles, B*C*G)
    Each CTA computes a [block_l, block_s] tile of the output
    """
    accum_dtype = "float"

    B = batch
    C = num_chunks
    G = n_groups
    Q = chunk_len
    N = d_state

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_l: int,
        block_s: int,
        block_n: int,
        threads: int,
    ):
        C_shape = (B, C, G, Q, N)
        B_shape = (B, C, G, Q, N)
        cb_shape = (B, C, G, Q, Q)

        @T.prim_func
        def main(
            C_mat: T.Tensor(C_shape, dtype),      # type: ignore
            B_mat: T.Tensor(B_shape, dtype),      # type: ignore
            cb: T.Tensor(cb_shape, dtype),        # type: ignore  OUTPUT in dtype, not accum_dtype
        ):
            with T.Kernel(
                T.ceildiv(Q, block_l),  # l tiles
                T.ceildiv(Q, block_s),  # s tiles
                B * C * G,              # batch * chunks * groups
                threads=threads,
            ) as (bl, bs, bcg):

                # Decode indices
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
                    # Allocate shared memory for C and B tiles
                    C_tile = T.alloc_shared((block_l, block_n), dtype)
                    B_tile = T.alloc_shared((block_s, block_n), dtype)

                    # Swizzled layout for better memory coalescing
                    T.annotate_layout({
                        C_tile: tilelang.layout.make_swizzled_layout(C_tile),
                        B_tile: tilelang.layout.make_swizzled_layout(B_tile),
                    })

                    # Accumulator for this tile
                    acc = T.alloc_fragment((block_l, block_s), accum_dtype)
                    T.clear(acc)

                    # Blocked GEMM over N dimension
                    for n_blk in range(T.ceildiv(N, block_n)):
                        n0 = n_blk * block_n

                        # Load C tile [block_l, block_n]
                        for ll, nn in T.Parallel(block_l, block_n):
                            l_abs = l0 + ll
                            n_abs = n0 + nn
                            if l_abs < Q and n_abs < N:
                                C_tile[ll, nn] = C_mat[bb, bc, bg, l_abs, n_abs]
                            else:
                                C_tile[ll, nn] = T.cast(T.float32(0.0), dtype)

                        # Load B tile [block_s, block_n]
                        for ss, nn in T.Parallel(block_s, block_n):
                            s_abs = s0 + ss
                            n_abs = n0 + nn
                            if s_abs < Q and n_abs < N:
                                B_tile[ss, nn] = B_mat[bb, bc, bg, s_abs, n_abs]
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

                    # Write output with causal mask: cb[l, s] = acc if s <= l else 0
                    # Cast from float32 accumulator to dtype for storage
                    for ll, ss in T.Parallel(block_l, block_s):
                        l_abs = l0 + ll
                        s_abs = s0 + ss
                        if l_abs < Q and s_abs < Q:
                            # Causal mask: only write if s <= l
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


# ========================================================================
# PyTorch custom op registration
# ========================================================================

@torch.library.custom_op("top::cb_producer", mutates_args=())
def _cb_producer_wrapped(
    batch: int,
    num_chunks: int,
    n_groups: int,
    chunk_len: int,
    d_state: int,
    dtype: str,
    block_l: int,
    block_s: int,
    block_n: int,
    threads: int,
    C_mat: torch.Tensor,
    B_mat: torch.Tensor,
) -> torch.Tensor:
    kernel_func = _cb_producer_kernel(
        batch, num_chunks, n_groups, chunk_len, d_state, dtype
    )
    kernel = kernel_func(block_l, block_s, block_n, threads)

    # TileLang with out_idx=[-1] returns the output directly
    return kernel(C_mat, B_mat)


@_cb_producer_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    n_groups: int,
    chunk_len: int,
    d_state: int,
    dtype: str,
    block_l: int,
    block_s: int,
    block_n: int,
    threads: int,
    C_mat: torch.Tensor,
    B_mat: torch.Tensor,
) -> torch.Tensor:
    return C_mat.new_empty(
        (batch, num_chunks, n_groups, chunk_len, chunk_len), dtype=C_mat.dtype  # Return in dtype, not float32
    )


# ========================================================================
# High-level Kernel wrapper
# ========================================================================

class CBProducerKernel(Kernel):
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
        tune: Whether to autotune
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_groups: int,
        chunk_len: int,
        d_state: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_groups = n_groups
        self.chunk_len = chunk_len
        self.d_state = d_state
        self.dtype = dtype

        self.kernel_func = _cb_producer_kernel(
            batch, num_chunks, n_groups, chunk_len, d_state, self.dtype_str
        )
        self.init_config(config, tune)

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
                        # Reasonable constraints
                        if block_l <= self.chunk_len and block_s <= self.chunk_len:
                            configs.append({
                                "block_l": block_l,
                                "block_s": block_s,
                                "block_n": block_n,
                                "threads": threads,
                            })
        return configs

    def forward(
        self,
        C_mat: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            C_mat: [B, C, G, Q, N]  dtype
            B_mat: [B, C, G, Q, N]  dtype

        Returns:
            cb: [B, C, G, Q, Q]  dtype
        """
        return _cb_producer_wrapped(
            self.batch, self.num_chunks, self.n_groups, self.chunk_len, self.d_state,
            self.dtype_str,
            self.config["block_l"], self.config["block_s"],
            self.config["block_n"], self.config["threads"],
            C_mat.contiguous(), B_mat.contiguous(),
        )
