"""Build Mamba-2 SSD chunk states as a weighted ``x.T @ B`` tile.

The input is sequence-fused while the float32 output is ``(B, C, H, P,
N)``. Packed sequences mask tokens outside the final token's sequence. The
weight uses the chunk-local decay ending at that token; its exponent is
clamped to zero from above for numerical safety.
"""

import functools
import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.mamba.call_spec import SSDChunkStateCall, SSDChunkStateFwdInterface

__all__ = ["SSDChunkStateFwdKernel"]


@functools.lru_cache(maxsize=32)
def _ssd_chunk_state_fwd_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    has_seq_idx: bool = False,
    dtype: str = "float16",
    dt_dtype: str = "float32",
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    Q = chunk_len
    H = n_heads
    P = d_head
    N = d_state
    G = n_groups
    S = C * Q
    HEADS_PER_GROUP = H // G

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_n: int,
        block_p: int,
        block_l: int,
        threads: int,
        num_stages: int,
    ):
        @T.prim_func
        def main(
            x: T.Tensor((B, S, H, P), dtype),  # type: ignore
            Bmat: T.Tensor((B, S, G, N), dtype),  # type: ignore
            dt: T.Tensor((B, H, C, Q), dt_dtype),  # type: ignore
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),  # type: ignore
            seq_idx: T.Tensor((B, S), "int32"),  # type: ignore
            out: T.Tensor((B, C, H, P, N), accum_dtype),  # type: ignore
        ):
            with T.Kernel(
                B * H * C,
                T.ceildiv(P, block_p),
                T.ceildiv(N, block_n),
                threads=threads,
            ) as (bhc, bp, bn):
                bz = bhc // (C * H)
                bc = (bhc % (C * H)) // H
                bh = bhc % H

                n0 = bn * block_n
                p0 = bp * block_p

                bg = bh // HEADS_PER_GROUP

                chunk_start = bc * Q

                acc = T.alloc_fragment((block_p, block_n), accum_dtype)
                T.clear(acc)

                dA_end = dA_cumsum[bz, bh, bc, Q - 1]
                seq_end = seq_idx[bz, chunk_start + Q - 1] if has_seq_idx else T.int32(0)

                # Share one fp32 weight per chunk position across output columns; the barrier
                # makes it visible before scaling x. Only the scaled tile rounds to storage dtype.
                w_tile = T.alloc_shared((block_l,), accum_dtype)
                x_scaled = T.alloc_shared((block_l, block_p), dtype)
                b_tile = T.alloc_shared((block_l, block_n), dtype)

                for l_blk in T.Pipelined(
                    T.ceildiv(Q, block_l),
                    num_stages=num_stages,
                ):
                    l0 = l_blk * block_l

                    for ll in T.Parallel(block_l):
                        l_idx = l0 + ll
                        dA_l = T.if_then_else(
                            l_idx < Q,
                            dA_cumsum[bz, bh, bc, l_idx],
                            T.float32(0.0),
                        )
                        dt_l = T.if_then_else(
                            l_idx < Q,
                            T.cast(dt[bz, bh, bc, l_idx], accum_dtype),
                            T.float32(0.0),
                        )
                        if has_seq_idx:
                            same_seq = T.if_then_else(
                                l_idx < Q,
                                (seq_end >= T.int32(0))
                                and (seq_idx[bz, chunk_start + l_idx] == seq_end),
                                T.bool(False),
                            )
                            w_tile[ll] = T.if_then_else(
                                same_seq,
                                T.exp(T.min(dA_end - dA_l, T.float32(0.0))) * dt_l,
                                T.float32(0.0),
                            )
                        else:
                            w_tile[ll] = T.exp(T.min(dA_end - dA_l, T.float32(0.0))) * dt_l
                    T.sync_threads()

                    for ll, pp in T.Parallel(block_l, block_p):
                        l_idx = l0 + ll
                        p_idx = p0 + pp
                        x_val = T.if_then_else(
                            (l_idx < Q) and (p_idx < P),
                            T.cast(x[bz, chunk_start + l_idx, bh, p_idx], accum_dtype),
                            T.float32(0.0),
                        )
                        x_scaled[ll, pp] = T.cast(x_val * w_tile[ll], dtype)

                    for ll, nn in T.Parallel(block_l, block_n):
                        l_idx = l0 + ll
                        n_idx = n0 + nn
                        b_tile[ll, nn] = T.if_then_else(
                            (l_idx < Q) and (n_idx < N),
                            Bmat[bz, chunk_start + l_idx, bg, n_idx],
                            T.cast(T.float32(0.0), dtype),
                        )

                    T.gemm(x_scaled, b_tile, acc, transpose_A=True)

                for pp, nn in T.Parallel(block_p, block_n):
                    p_idx = p0 + pp
                    n_idx = n0 + nn
                    if p_idx < P and n_idx < N:
                        out[bz, bc, bh, p_idx, n_idx] = acc[pp, nn]

        return main

    return kernel_func


class SSDChunkStateFwdKernel(Kernel, SSDChunkStateFwdInterface):
    """Mamba-2 SSD chunk state forward kernel.

    Computes the chunk-end SSM state for each chunk:

      out[b, c, h, p, n] =
          sum_{l=0}^{Q-1}
              x[b, c*Q+l, h, p]
              * B[b, c*Q+l, g(h), n]
              * exp(dA_cumsum[b,h,c,Q-1] - dA_cumsum[b,h,c,l])
              * dt[b, h, c, l]
              * (1 if not has_seq_idx else (seq_idx[b,c*Q+Q-1] >= 0 and seq_idx[b,c*Q+l] == seq_idx[b,c*Q+Q-1]))

    Inputs:
        x:         (batch, seqlen, n_heads, d_head)           dtype
        Bmat:      (batch, seqlen, n_groups, d_state)         dtype
        dt:        (batch, n_heads, num_chunks, chunk_len)    dtype — accepts target dtype,
                   will be cast to fp32 internally for accumulation
        dA_cumsum: (batch, n_heads, num_chunks, chunk_len)    float32
        seq_idx:   (batch, seqlen)                            int32 (optional)

    Output:
        out:       (batch, num_chunks, n_heads, d_head, d_state) float32

    """

    supported_archs: list[int] = [80, 86, 89, 90]

    # ``seq_idx`` masks contributions; the trip count comes from the chunk
    # length.
    autotune_accepts_random_int_inputs: bool = True

    @classmethod
    def entry_for(cls, call: SSDChunkStateCall) -> Entry:
        return call, lambda: cls(
            call.batch,
            call.num_chunks,
            call.chunk_len,
            call.n_heads,
            call.d_head,
            call.d_state,
            call.n_groups,
            call.dtype,
            has_seq_idx=call.has_seq_idx,
            dt_dtype=call.dt_dtype,
        )

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        has_seq_idx: bool = False,
        config: Optional[dict] = None,
        dt_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.dt_dtype = dt_dtype or dtype
        self.has_seq_idx = has_seq_idx
        self.kernel = _ssd_chunk_state_fwd_kernel(
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            d_head,
            d_state,
            n_groups,
            has_seq_idx,
            self.dtype_str,
            self.dtype_to_str(self.dt_dtype),
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # threads=128 (4 warps) reduces register pressure and improves occupancy over 256.
        # The primary Mamba geometry benefits from fewer reduction-loop syncs.
        # Small grids split N for more CTAs; saturated grids avoid reloading X.
        grid_size = self.batch * self.num_chunks * self.n_heads
        primary_mamba_geometry = self.chunk_len == 256 and self.d_head == 64 and self.d_state == 128
        if primary_mamba_geometry:
            small_grid = grid_size <= 640
            launch_shape = (self.batch, self.num_chunks, self.n_heads)
            if small_grid and not self.has_seq_idx and self.dt_dtype == torch.float32:
                if self.dtype == torch.float16 and launch_shape == (1, 8, 64):
                    return {
                        "block_n": 128,
                        "block_p": 64,
                        "block_l": 128,
                        "threads": 256,
                        "num_stages": 2,
                    }
                if self.dtype == torch.bfloat16 and launch_shape == (1, 8, 80):
                    return {
                        "block_n": 64,
                        "block_p": 64,
                        "block_l": 64,
                        "threads": 128,
                        "num_stages": 2,
                    }
            return {
                "block_n": 64 if small_grid else 128,
                "block_p": 64,
                "block_l": 128 if small_grid else 64,
                "threads": 128,
                "num_stages": 2,
            }
        # The GEMM's N extent splits across warps in 16-column steps; a d_state
        # that is not a whole number of them rounds up, and the tail is masked.
        n_step = 16
        return {
            "block_n": min(128, -(-self.d_state // n_step) * n_step),
            "block_p": 64,
            "block_l": 32,
            "threads": 128,
            "num_stages": 2,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # Grid rationale:
        #   block_n in {64, 128}: 128 covers the full d_state=128 in one tile
        #     and is the default; 64 is retained for d_state<128 shapes.
        #   block_p in {16, 32, 64}: 16 is the minimum MMA M-atom.
        #   block_l in {32, 64, 128}: larger K improves GEMM arithmetic
        #     intensity; 32 keeps shared-memory pressure low for small d_head.
        #   threads in {128, 256}: 128 warps = 4, 256 warps = 8; higher thread
        #     count hides latency but increases register/shared pressure.
        block_n = [64, 128]
        block_p = [16, 32, 64]
        block_l = [32, 64, 128]
        threads = [128, 256]
        _configs = list(itertools.product(block_n, block_p, block_l, threads))
        return [
            {
                "block_n": c[0],
                "block_p": c[1],
                "block_l": c[2],
                "threads": c[3],
                "num_stages": 2,
            }
            for c in _configs
        ]

    def forward(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        seq_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chunk-end SSM states.

        Args:
            x: (batch, seqlen, n_heads, d_head) dtype
            Bmat: (batch, seqlen, n_groups, d_state) dtype
            dt: (batch, n_heads, num_chunks, chunk_len) float32 or dtype
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32
            seq_idx: (batch, seqlen) int32

        Returns:
            out: (batch, num_chunks, n_heads, d_head, d_state) float32

        The compiled kernel reads dt in its input dtype and converts values to
        float32 during accumulation.
        """
        return self.program(self._compile)(x, Bmat, dt, dA_cumsum, seq_idx)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(
            cfg["block_n"],
            cfg["block_p"],
            cfg["block_l"],
            cfg["threads"],
            cfg.get("num_stages", 2),
        )
