"""
Mamba-2 State-Space Dual (SSD) chunk state forward kernel.

Inputs (pre-reshaped to chunked view):
  x:          (batch, seq_len, n_heads, d_head)
              -- input features, seq_len = num_chunks * chunk_len
  Bmat:       (batch, seq_len, n_groups, d_state)
              -- State Space Model (SSM) B matrix, grouped over heads
  dt:         (batch, n_heads, num_chunks, chunk_len)
              -- per-position discretization factor (float32)
  dA_cumsum:  (batch, n_heads, num_chunks, chunk_len)
              -- chunk-local prefix sums of dA = dt * A (float32)
  seq_idx:    (batch, seq_len)  int32, optional
              -- sequence index per token for packed/variable-length inputs;
                 positions where seq_idx[l] != seq_idx[Q-1] are masked to zero

Output:
  out:        (batch, num_chunks, n_heads, d_head, d_state)  float32

For each (b, c, h, p, n), the kernel computes:

  out[b, c, h, p, n] =
      sum_{l in [0, chunk_len)}
          x[b, s, h, p]
          * B[b, s, g(h), n]
          * exp(dA_cumsum[b, h, c, Q-1] - dA_cumsum[b, h, c, l])
          * dt[b, h, c, l]
          * (1 if seq_idx is None else (seq_idx[b, s] == seq_idx[b, c*Q+Q-1]))

where:
  s = c * chunk_len + l
  g(h) = h // heads_per_group  (head -> group mapping)
  Q    = chunk_len

The output shape (B, C, H, P, N) matches the ssd_minimal reference:
  states = einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

The decay term uses dA_end - dA_l, where dA_end = dA_cumsum[..., Q-1].
Since dA <= 0 everywhere in Mamba-2, dA_cumsum is monotonically
non-increasing within a chunk, so dA_end - dA_l <= 0 for all l.
Therefore exp(dA_end - dA_l) is already a decaying factor in (0, 1],
and exp(min(dA_end - dA_l, 0)) is equivalent. We keep the min-clamp
for numerical safety.

GEMM reformulation
------------------
Define the per-position scalar weight:

  w[l] = exp(min(dA_end - dA_cumsum[l], 0)) * dt[l]
          * (1 if no seq_idx else (seq_idx[l] == seq_idx[Q-1]))

Then the output tile can be written as a matrix product:

  out[p, n] = sum_l  x[l, p] * w[l] * B[l, n]
            = (x_scaled)^T @ B_tile

where x_scaled[l, p] = x[l, p] * w[l]  (scale each row of x by a
per-row scalar before the GEMM).  This is a (block_p × block_l)^T @
(block_l × block_n) tensor-core GEMM with transpose_A=True, replacing
the previous serialised outer-product loop.

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

__all__ = ["SSDChunkStateFwdKernel"]


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
            seq_idx: T.Tensor((B, S), "int32"),               # type: ignore
            out: T.Tensor((B, C, H, P, N), accum_dtype),      # type: ignore
        ):
            # Grid layout:
            #   axis-0: fused (batch, head, chunk)  -> B*H*C blocks
            #   axis-1: tile over P = d_head
            #   axis-2: tile over N = d_state
            with T.Kernel(
                B * H * C,
                T.ceildiv(P, block_p),
                T.ceildiv(N, block_n),
                threads=threads,
            ) as (bhc, bp, bn):

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
                # 2. Allocate accumulator for one output tile (P x N)
                # --------------------------------------------------------
                acc = T.alloc_fragment((block_p, block_n), accum_dtype)
                T.clear(acc)

                # --------------------------------------------------------
                # 3. Load chunk-end cumulative decay scalar and seq_idx
                #    dA_end = dA_cumsum[bz, bh, bc, Q-1]
                # --------------------------------------------------------
                dA_end = dA_cumsum[bz, bh, bc, Q - 1]
                seq_end = seq_idx[bz, chunk_start + Q - 1] if has_seq_idx else T.int32(0)

                # --------------------------------------------------------
                # 4. Allocate tiles once outside the reduction loop
                #
                #    x_scaled_tile[l, p] = x[l, p] * w[l]   (row-scaled x)
                #    b_tile[l, n]         = B[l, n]           (unscaled)
                #
                #    x_scaled_f32  -- float32 scratch for the row-scale step
                #    x_scaled      -- dtype version passed to T.gemm
                #    b_tile        -- dtype shared tile for B
                #
                #    GEMM:  acc[p, n] += x_scaled^T @ b_tile
                #           i.e. (block_l x block_p)^T @ (block_l x block_n)
                #                = (block_p x block_l) @ (block_l x block_n)
                # --------------------------------------------------------
                x_scaled_f32 = T.alloc_fragment((block_l, block_p), accum_dtype)
                x_scaled     = T.alloc_shared((block_l, block_p), dtype)
                b_tile       = T.alloc_shared((block_l, block_n), dtype)

                # --------------------------------------------------------
                # 5. Reduce over chunk positions in L-tiles
                # --------------------------------------------------------
                for l_blk in T.Serial(T.ceildiv(Q, block_l)):
                    l0 = l_blk * block_l

                    # 5.1 Compute x_scaled[ll, pp] = x[ll, pp] * w[ll]
                    #     where w[ll] = exp(min(dA_end - dA_cumsum[l], 0)) * dt[l]
                    #                   * seq_mask[l]
                    #     Done in float32, then cast to dtype for GEMM.
                    for ll, pp in T.Parallel(block_l, block_p):
                        l_idx = l0 + ll
                        p_idx = p0 + pp
                        in_bounds = (l_idx < Q) and (p_idx < P)
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
                        x_val = T.if_then_else(
                            in_bounds,
                            T.cast(x[bz, chunk_start + l_idx, bh, p_idx], accum_dtype),
                            T.float32(0.0),
                        )
                        if has_seq_idx:
                            same_seq = T.if_then_else(
                                l_idx < Q,
                                seq_idx[bz, chunk_start + l_idx] == seq_end,
                                T.bool(False),
                            )
                            w = T.if_then_else(
                                same_seq,
                                T.exp(T.min(dA_end - dA_l, T.float32(0.0))) * dt_l,
                                T.float32(0.0),
                            )
                        else:
                            w = T.exp(T.min(dA_end - dA_l, T.float32(0.0))) * dt_l
                        x_scaled_f32[ll, pp] = x_val * w

                    # Cast scaled-x to kernel dtype for tensor-core GEMM
                    T.copy(x_scaled_f32, x_scaled)

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

                    # 5.3 Tensor-core GEMM:
                    #     acc[p, n] += x_scaled^T @ b_tile
                    #     shapes: (block_l x block_p)^T @ (block_l x block_n)
                    T.gemm(x_scaled, b_tile, acc, transpose_A=True)

                # --------------------------------------------------------
                # 6. Write back output tile: out[bz, bc, bh, p0:p0+block_p, n0:n0+block_n]
                # --------------------------------------------------------
                for pp, nn in T.Parallel(block_p, block_n):
                    p_idx = p0 + pp
                    n_idx = n0 + nn
                    if p_idx < P and n_idx < N:
                        out[bz, bc, bh, p_idx, n_idx] = acc[pp, nn]

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
    has_seq_idx: bool,
    dtype: str,
    block_n: int,
    block_p: int,
    block_l: int,
    threads: int,
    x: torch.Tensor,
    Bmat: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    seq_idx: torch.Tensor,
) -> torch.Tensor:
    return _ssd_chunk_state_fwd_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, has_seq_idx, dtype)(
        block_n, block_p, block_l, threads,
    )(x, Bmat, dt, dA_cumsum, seq_idx)


@_ssd_chunk_state_fwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    has_seq_idx: bool,
    dtype: str,
    block_n: int,
    block_p: int,
    block_l: int,
    threads: int,
    x: torch.Tensor,
    Bmat: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    seq_idx: torch.Tensor,
) -> torch.Tensor:
    # Output shape: (B, C, H, P, N) matching ssd_minimal bchpn convention
    return x.new_empty((batch, num_chunks, n_heads, d_head, d_state), dtype=torch.float32)


class SSDChunkStateFwdKernel(Kernel):
    """Mamba-2 SSD chunk state forward kernel.

    Computes the chunk-end SSM state for each chunk:

      out[b, c, h, p, n] =
          sum_{l=0}^{Q-1}
              x[b, c*Q+l, h, p]
              * B[b, c*Q+l, g(h), n]
              * exp(dA_cumsum[b,h,c,Q-1] - dA_cumsum[b,h,c,l])
              * dt[b, h, c, l]
              * (1 if not has_seq_idx else (seq_idx[b,c*Q+l] == seq_idx[b,c*Q+Q-1]))

    Inputs:  x, Bmat, dt, dA_cumsum[, seq_idx]
    Output:  out  (batch, num_chunks, n_heads, d_head, d_state), float32
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
        has_seq_idx: bool = False,
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
        self.has_seq_idx = has_seq_idx
        self.kernel = _ssd_chunk_state_fwd_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups,
            has_seq_idx, self.dtype_str,
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
        seq_idx: torch.Tensor,
    ) -> torch.Tensor:
        return _ssd_chunk_state_fwd_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.d_head,
            self.d_state, self.n_groups, self.has_seq_idx, self.dtype_str,
            self.config["block_n"], self.config["block_p"], self.config["block_l"],
            self.config["threads"],
            x, Bmat, dt, dA_cumsum, seq_idx,
        )
