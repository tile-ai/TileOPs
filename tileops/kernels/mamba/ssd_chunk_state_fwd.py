"""
Mamba-2 SSD chunk state forward kernel.

Inputs (pre-reshaped to chunked view):
  x:          (batch, seq_len, n_heads, d_head)
              -- input features, seq_len = num_chunks * chunk_len
  Bmat:       (batch, seq_len, n_groups, d_state)
              -- SSM B matrix, grouped over heads
  dt:         (batch, n_heads, num_chunks, chunk_len)
              -- per-position discretization factor (float32)
  dA_cumsum:  (batch, n_heads, num_chunks, chunk_len)
              -- chunk-local prefix sums of dA = dt * A (float32)

Output:
  out:        (batch, num_chunks, n_heads, d_state, d_head)  float32

For each (b, c, h, n, p), the kernel computes:

  out[b, c, h, n, p] =
      sum_{l in [0, chunk_len)}
          x[b, s, h, p]
          * B[b, s, g(h), n]
          * exp(dA_cumsum[b, h, c, Q-1] - dA_cumsum[b, h, c, l])
          * dt[b, h, c, l]

where:
  s = c * chunk_len + l
  g(h) = h // heads_per_group  (head -> group mapping)
  Q    = chunk_len

The output shape (B, C, H, N, P) matches the prev_states layout consumed
by ssd_chunk_scan_fwd: (batch, num_chunks, n_heads, d_state, d_head).

The decay term exp(dA_end - dA_l) with dA_end = dA_cumsum[..., Q-1]
is equivalent to exp(min(dA_end - dA_l, 0)) because dA_cumsum is
monotonically non-decreasing within a chunk (dA <= 0 everywhere in
Mamba-2), so dA_end - dA_l >= 0 always.  We keep the min-clamp for
numerical safety.

Notation:
  B = batch, S = seq_len, H = n_heads, P = d_head, G = n_groups,
  N = d_state, C = num_chunks, Q = chunk_len
"""

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdChunkStateFwdKernel"]


def _ssd_chunk_state_fwd_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    # Derived constants
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
    ):
        @T.prim_func
        def main(
            x: T.Tensor((B, S, H, P), dtype),                # type: ignore
            Bmat: T.Tensor((B, S, G, N), dtype),              # type: ignore
            dt: T.Tensor((B, H, C, Q), accum_dtype),          # type: ignore
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),   # type: ignore
            out: T.Tensor((B, C, H, N, P), accum_dtype),      # type: ignore
        ):
            # Grid layout:
            #   axis-0: fused (batch, head, chunk)  -> B*H*C blocks
            #   axis-1: tile over N = d_state
            #   axis-2: tile over P = d_head
            with T.Kernel(
                B * H * C,
                T.ceildiv(N, block_n),
                T.ceildiv(P, block_p),
                threads=threads,
            ) as (bhc, bn, bp):

                # --------------------------------------------------------
                # 1. Decode fused axis
                # --------------------------------------------------------
                bz = bhc // (H * C)
                bh = (bhc % (H * C)) // C
                bc = bhc % C

                n0 = bn * block_n
                p0 = bp * block_p

                # head -> group mapping
                bg = bh // HEADS_PER_GROUP

                # starting token index of this chunk in the full sequence
                chunk_start = bc * Q

                # --------------------------------------------------------
                # 2. Allocate accumulator for one output tile (N x P)
                # --------------------------------------------------------
                acc = T.alloc_fragment((block_n, block_p), accum_dtype)
                T.clear(acc)

                # --------------------------------------------------------
                # 3. Load chunk-end cumulative decay scalar
                #    dA_end = dA_cumsum[bz, bh, bc, Q-1]
                # --------------------------------------------------------
                dA_end = dA_cumsum[bz, bh, bc, Q - 1]

                # --------------------------------------------------------
                # 4. Allocate tiles once outside the reduction loop
                # --------------------------------------------------------
                x_tile = T.alloc_shared((block_l, block_p), dtype)
                b_tile = T.alloc_shared((block_l, block_n), dtype)
                decay_tile = T.alloc_fragment((block_l,), accum_dtype)

                # --------------------------------------------------------
                # 5. Reduce over chunk positions in L-tiles
                # --------------------------------------------------------
                for l_blk in T.Serial(T.ceildiv(Q, block_l)):
                    l0 = l_blk * block_l

                    # 5.1 Cooperative load: x
                    #     x_tile[ll, pp] = x[bz, chunk_start + l0 + ll, bh, p0 + pp]
                    for ll, pp in T.Parallel(block_l, block_p):
                        l_idx = l0 + ll
                        p_idx = p0 + pp
                        x_tile[ll, pp] = T.if_then_else(
                            (l_idx < Q) and (p_idx < P),
                            x[bz, chunk_start + l_idx, bh, p_idx],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # 5.2 Cooperative load: B
                    #     b_tile[ll, nn] = Bmat[bz, chunk_start + l0 + ll, bg, n0 + nn]
                    for ll, nn in T.Parallel(block_l, block_n):
                        l_idx = l0 + ll
                        n_idx = n0 + nn
                        b_tile[ll, nn] = T.if_then_else(
                            (l_idx < Q) and (n_idx < N),
                            Bmat[bz, chunk_start + l_idx, bg, n_idx],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # 5.3 Compute per-position decay * dt
                    #     decay_tile[ll] = exp(min(dA_end - dA_cumsum[l], 0)) * dt[l]
                    for ll in T.Parallel(block_l):
                        l_idx = l0 + ll
                        dA_l = T.if_then_else(
                            l_idx < Q,
                            dA_cumsum[bz, bh, bc, l_idx],
                            T.float32(0.0),
                        )
                        dt_l = T.if_then_else(
                            l_idx < Q,
                            dt[bz, bh, bc, l_idx],
                            T.float32(0.0),
                        )
                        decay_tile[ll] = T.exp(T.min(dA_end - dA_l, T.float32(0.0))) * dt_l

                    # 5.4 Local outer-product reduction:
                    #     acc[n, p] += sum_ll  B[ll, n] * x[ll, p] * decay[ll]
                    for ll in T.Serial(block_l):
                        for nn, pp in T.Parallel(block_n, block_p):
                            acc[nn, pp] += (
                                T.cast(b_tile[ll, nn], accum_dtype)
                                * T.cast(x_tile[ll, pp], accum_dtype)
                                * decay_tile[ll]
                            )

                # --------------------------------------------------------
                # 6. Write back output tile: out[bz, bc, bh, n0:n0+block_n, p0:p0+block_p]
                # --------------------------------------------------------
                for nn, pp in T.Parallel(block_n, block_p):
                    n_idx = n0 + nn
                    p_idx = p0 + pp
                    if n_idx < N and p_idx < P:
                        out[bz, bc, bh, n_idx, p_idx] = acc[nn, pp]

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_chunk_state_fwd", mutates_args=())
def _ssd_chunk_state_fwd_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_n: int,
    block_p: int,
    block_l: int,
    threads: int,
    x: torch.Tensor,
    Bmat: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
) -> torch.Tensor:
    return _ssd_chunk_state_fwd_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype)(
        block_n, block_p, block_l, threads,
    )(x, Bmat, dt, dA_cumsum)


@_ssd_chunk_state_fwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_n: int,
    block_p: int,
    block_l: int,
    threads: int,
    x: torch.Tensor,
    Bmat: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
) -> torch.Tensor:
    # Output shape matches prev_states layout: (B, C, H, N, P)
    return x.new_empty((batch, num_chunks, n_heads, d_state, d_head), dtype=torch.float32)


class SsdChunkStateFwdKernel(Kernel):
    """Mamba-2 SSD chunk state forward kernel.

    Computes the chunk-end SSM state for each chunk:

      out[b, c, h, n, p] =
          sum_{l=0}^{Q-1}
              x[b, c*Q+l, h, p]
              * B[b, c*Q+l, g(h), n]
              * exp(dA_cumsum[b,h,c,Q-1] - dA_cumsum[b,h,c,l])
              * dt[b, h, c, l]

    Inputs:  x, Bmat, dt, dA_cumsum
    Output:  out  (batch, num_chunks, n_heads, d_state, d_head), float32
    """

    supported_archs: list[int] = [80, 86, 89, 90]

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
        config: Optional[dict] = None,
        tune: bool = False,
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
        self.kernel = _ssd_chunk_state_fwd_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_n": 32,
            "block_p": 64,
            "block_l": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_n = [16, 32]
        block_p = [32, 64]
        block_l = [32, 64]
        threads = [128, 256]
        _configs = list(itertools.product(block_n, block_p, block_l, threads))
        return [{
            "block_n": c[0],
            "block_p": c[1],
            "block_l": c[2],
            "threads": c[3],
        } for c in _configs]

    def forward(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
    ) -> torch.Tensor:
        return _ssd_chunk_state_fwd_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.d_head,
            self.d_state, self.n_groups, self.dtype_str,
            self.config["block_n"], self.config["block_p"], self.config["block_l"],
            self.config["threads"],
            x, Bmat, dt, dA_cumsum,
        )
