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
          * (1 if seq_idx is None else (seq_idx[b, c*Q+Q-1] >= 0 and seq_idx[b, s] == seq_idx[b, c*Q+Q-1]))

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

import functools
import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["SSDChunkStateFwdKernel", "SSDChunkStateFwdKernelMH"]


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
                # 1. Decode fused axis  (b, c, h — h is fastest-changing)
                #
                # Consecutive CTAs share the same (b, c), so they cover the
                # same chunk rows in Bmat.  When HEADS_PER_GROUP > 1, the
                # HEADS_PER_GROUP consecutive h values that belong to the same
                # group map to the same bg and therefore load identical b_tile
                # data.  Those loads are served from L2 after the first CTA
                # warms the cache, reducing effective Bmat bandwidth by up to
                # HEADS_PER_GROUP×.  The alternative b,h,c order (c fastest)
                # shifts chunk_start on every CTA step so no Bmat rows are
                # reused between consecutive CTAs.
                # --------------------------------------------------------
                bz = bhc // (C * H)
                bc = (bhc % (C * H)) // H
                bh = bhc % H

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
                #    x_scaled[l, p] = x[l, p] * w(l)   (row-scaled x, dtype)
                #    b_tile[l, n]   = B[l, n]            (unscaled, dtype)
                #
                #    w_tile[block_l] holds the per-position scalar weight in
                #    shared memory.  It is filled by T.Parallel(block_l) — one
                #    load of dA_cumsum[l] and dt[l] per l — then
                #    T.sync_threads() makes it visible to the subsequent
                #    T.Parallel(block_l, block_p) loop, avoiding block_p
                #    redundant global loads per l.
                #
                #    x_scaled is written directly as dtype (cast in-place):
                #      x_scaled[ll, pp] = cast(float(x[ll, pp]) * w_tile[ll])
                #    This eliminates the x_scaled_f32 register fragment
                #    (block_l * block_p / 32 fp32 regs per thread) and the
                #    separate T.copy cast step, freeing register budget for
                #    larger output tiles without changing the shared-memory
                #    footprint or numerical behavior (the multiply is still
                #    done in fp32 before truncation to dtype).
                #
                #    GEMM:  acc[p, n] += x_scaled^T @ b_tile
                #           i.e. (block_l x block_p)^T @ (block_l x block_n)
                #                = (block_p x block_l) @ (block_l x block_n)
                # --------------------------------------------------------
                w_tile   = T.alloc_shared((block_l,), accum_dtype)
                x_scaled = T.alloc_shared((block_l, block_p), dtype)
                b_tile   = T.alloc_shared((block_l, block_n), dtype)

                # --------------------------------------------------------
                # 5. Reduce over chunk positions in L-tiles
                # --------------------------------------------------------
                for l_blk in T.Serial(T.ceildiv(Q, block_l)):
                    l0 = l_blk * block_l

                    # 5.0 Fill w_tile[ll] = exp(min(dA_end - dA_cumsum[l], 0)) * dt[l]
                    #     One global load of dA_cumsum[l] and dt[l] per l.
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
                        if has_seq_idx:
                            same_seq = T.if_then_else(
                                l_idx < Q,
                                (seq_end >= T.int32(0)) and (seq_idx[bz, chunk_start + l_idx] == seq_end),
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

                    # 5.1 Compute x_scaled[ll, pp] = cast(float(x[ll,pp]) * w_tile[ll])
                    #     Written directly to shared memory as dtype, bypassing
                    #     the intermediate fp32 register fragment.
                    for ll, pp in T.Parallel(block_l, block_p):
                        l_idx = l0 + ll
                        p_idx = p0 + pp
                        x_val = T.if_then_else(
                            (l_idx < Q) and (p_idx < P),
                            T.cast(x[bz, chunk_start + l_idx, bh, p_idx], accum_dtype),
                            T.float32(0.0),
                        )
                        x_scaled[ll, pp] = T.cast(x_val * w_tile[ll], dtype)

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
              * (1 if not has_seq_idx else (seq_idx[b,c*Q+Q-1] >= 0 and seq_idx[b,c*Q+l] == seq_idx[b,c*Q+Q-1]))

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
        # block_n=128 covers the full d_state=128 in one N-tile, halving the
        # grid from 2*B*H*C to B*H*C CTAs and doubling the GEMM N-dimension.
        # This improves L2 hit rate (Bmat reuse within a CTA) and reduces
        # grid-launch overhead on large shapes.  block_l=64 keeps K-depth
        # high for MMA pipeline efficiency without exceeding the register
        # budget (94 regs × 128 threads = 12032 regs/block).
        return {
            "block_n": 128,
            "block_p": 64,
            "block_l": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # Grid rationale:
        #   block_n in {64, 128}: 128 covers the full d_state=128 in one tile
        #     and is the default; 64 is retained for d_state<128 shapes.
        #   block_p in {16, 32, 64}: 16 is the minimum MMA M-atom; 64 only
        #     viable after the x_scaled_f32 fragment was removed.
        #   block_l in {32, 64, 128}: larger K improves GEMM arithmetic
        #     intensity; 32 keeps shared-memory pressure low for small d_head.
        #   threads in {128, 256}: 128 warps = 4, 256 warps = 8; higher thread
        #     count hides latency but increases register/shared pressure.
        block_n = [64, 128]
        block_p = [16, 32, 64]
        block_l = [32, 64, 128]
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


# ---------------------------------------------------------------------------
# Multi-head-per-CTA variant (SSDChunkStateFwdKernelMH)
# ---------------------------------------------------------------------------
#
# Each CTA processes `heads_per_group` (HPG) consecutive heads for one
# (batch, chunk) pair.  The Bmat tile for a given (b, c, group) is loaded
# once per L-block and reused across all HPG GEMMs, reducing Bmat HBM
# traffic by HPG×.
#
# Grid:  B * (H // HPG) * C  ×  ceil(P / block_p)  ×  ceil(N / block_n)
#
# With HPG=4 and the standard Mamba-2 configs (H=32/64/80, C=8/32):
#   b4/L8k cases: 256–640 CTAs (1.9–4.8 waves on H200) — good utilisation
#   b1 cases:     64–160 CTAs (0.5–1.2 waves) — still launch-limited but
#                 4× Bmat reuse improves effective bandwidth
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _ssd_chunk_state_fwd_mh_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    heads_per_group: int,
    has_seq_idx: bool = False,
    dtype: str = "float16",
) -> Callable:
    """Multi-head-per-CTA kernel builder.

    Identical math to ``_ssd_chunk_state_fwd_kernel`` but each CTA covers
    ``heads_per_group`` heads, loading the shared Bmat tile once per L-block
    and reusing it across all head GEMMs.
    """
    accum_dtype = "float"

    B = batch
    C = num_chunks
    Q = chunk_len
    H = n_heads
    P = d_head
    N = d_state
    G = n_groups
    S = C * Q
    HPG = heads_per_group          # heads processed per CTA
    N_HEAD_GROUPS = H // HPG       # number of CTA groups along the head axis
    HEADS_PER_GROUP = H // G       # head-to-Bmat-group mapping (unchanged)

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_n: int,
        block_p: int,
        block_l: int,
        threads: int,
    ):
        @T.prim_func
        def main(
            x: T.Tensor((B, S, H, P), dtype),               # type: ignore
            Bmat: T.Tensor((B, S, G, N), dtype),             # type: ignore
            dt: T.Tensor((B, H, C, Q), accum_dtype),         # type: ignore
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),  # type: ignore
            seq_idx: T.Tensor((B, S), "int32"),              # type: ignore
            out: T.Tensor((B, C, H, P, N), accum_dtype),     # type: ignore
        ):
            # Grid layout:
            #   axis-0: fused (batch, head_group, chunk) -> B*(H//HPG)*C blocks
            #           chunk is fastest-changing so consecutive CTAs in axis-0
            #           share the same (b, head_group) and therefore the same
            #           Bmat group, maximising L2 reuse of b_tile across CTAs.
            #   axis-1: tile over P = d_head
            #   axis-2: tile over N = d_state
            with T.Kernel(
                B * N_HEAD_GROUPS * C,
                T.ceildiv(P, block_p),
                T.ceildiv(N, block_n),
                threads=threads,
            ) as (bhgc, bp, bn):

                # --------------------------------------------------------
                # 1. Decode fused axis (b, head_group, chunk)
                #    chunk is fastest-changing for Bmat L2 reuse.
                # --------------------------------------------------------
                bz  = bhgc // (N_HEAD_GROUPS * C)
                bhg = (bhgc % (N_HEAD_GROUPS * C)) // C
                bc  = bhgc % C

                bh_start    = bhg * HPG          # first head index for this CTA
                chunk_start = bc * Q

                n0 = bn * block_n
                p0 = bp * block_p

                # Bmat group for the first head in this CTA.
                # All HPG heads in this CTA belong to the same Bmat group
                # because HPG divides HEADS_PER_GROUP (enforced in __init__).
                bg = bh_start // HEADS_PER_GROUP

                # --------------------------------------------------------
                # 2. Allocate shared tiles (reused across head loop)
                #
                #   b_tile[block_l, block_n]   — loaded once per l_blk,
                #                                shared across all HPG heads
                #   w_tile[block_l]            — per-head scalar weights
                #   x_scaled[block_l, block_p] — per-head scaled input
                # --------------------------------------------------------
                b_tile   = T.alloc_shared((block_l, block_n), dtype)
                w_tile   = T.alloc_shared((block_l,), accum_dtype)
                x_scaled = T.alloc_shared((block_l, block_p), dtype)

                # --------------------------------------------------------
                # 3. Reduce over L-blocks
                # --------------------------------------------------------
                for l_blk in T.Serial(T.ceildiv(Q, block_l)):
                    l0 = l_blk * block_l

                    # 3.0 Load b_tile ONCE for this l_blk — shared across HPG heads.
                    #     b_tile[ll, nn] = Bmat[bz, chunk_start + l0 + ll, bg, n0 + nn]
                    for ll, nn in T.Parallel(block_l, block_n):
                        l_idx = l0 + ll
                        n_idx = n0 + nn
                        b_tile[ll, nn] = T.if_then_else(
                            (l_idx < Q) and (n_idx < N),
                            Bmat[bz, chunk_start + l_idx, bg, n_idx],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # 3.1 Head loop: HPG iterations, b_tile reused each time.
                    for h_off in T.Serial(HPG):
                        bh = bh_start + h_off

                        # Per-head accumulator (register fragment, reset each head)
                        acc = T.alloc_fragment((block_p, block_n), accum_dtype)
                        T.clear(acc)

                        # Load chunk-end decay scalar and seq_idx for this head
                        dA_end = dA_cumsum[bz, bh, bc, Q - 1]
                        seq_end = seq_idx[bz, chunk_start + Q - 1] if has_seq_idx else T.int32(0)

                        # Fill w_tile: exp(min(dA_end - dA_cumsum[l], 0)) * dt[l]
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
                            if has_seq_idx:
                                same_seq = T.if_then_else(
                                    l_idx < Q,
                                    (seq_end >= T.int32(0)) and (seq_idx[bz, chunk_start + l_idx] == seq_end),
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

                        # Compute x_scaled[ll, pp] = cast(x[ll, pp] * w_tile[ll])
                        for ll, pp in T.Parallel(block_l, block_p):
                            l_idx = l0 + ll
                            p_idx = p0 + pp
                            x_val = T.if_then_else(
                                (l_idx < Q) and (p_idx < P),
                                T.cast(x[bz, chunk_start + l_idx, bh, p_idx], accum_dtype),
                                T.float32(0.0),
                            )
                            x_scaled[ll, pp] = T.cast(x_val * w_tile[ll], dtype)

                        # GEMM: acc[p, n] += x_scaled^T @ b_tile
                        T.gemm(x_scaled, b_tile, acc, transpose_A=True)

                        # Write output for this head
                        for pp, nn in T.Parallel(block_p, block_n):
                            p_idx = p0 + pp
                            n_idx = n0 + nn
                            if p_idx < P and n_idx < N:
                                out[bz, bc, bh, p_idx, n_idx] = acc[pp, nn]

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_chunk_state_fwd_mh", mutates_args=())
def _ssd_chunk_state_fwd_mh_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    heads_per_group: int,
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
    return _ssd_chunk_state_fwd_mh_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups,
        heads_per_group, has_seq_idx, dtype,
    )(block_n, block_p, block_l, threads)(x, Bmat, dt, dA_cumsum, seq_idx)


@_ssd_chunk_state_fwd_mh_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    heads_per_group: int,
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
    return x.new_empty((batch, num_chunks, n_heads, d_head, d_state), dtype=torch.float32)


class SSDChunkStateFwdKernelMH(Kernel):
    """Multi-head-per-CTA variant of the Mamba-2 SSD chunk state forward kernel.

    Each CTA processes ``heads_per_group`` consecutive heads for one
    (batch, chunk) pair.  The Bmat tile is loaded once per L-block and
    reused across all ``heads_per_group`` GEMMs, reducing Bmat HBM traffic
    by ``heads_per_group``×.

    Computes the same formula as ``SSDChunkStateFwdKernel``:

      out[b, c, h, p, n] =
          sum_{l=0}^{Q-1}
              x[b, c*Q+l, h, p]
              * B[b, c*Q+l, g(h), n]
              * exp(dA_cumsum[b,h,c,Q-1] - dA_cumsum[b,h,c,l])
              * dt[b, h, c, l]
              * (1 if not has_seq_idx else (seq_idx[b,c*Q+Q-1] >= 0 and seq_idx[b,c*Q+l] == seq_idx[b,c*Q+Q-1]))

    Args:
        heads_per_group: Number of heads each CTA processes (default 4).
            Must divide ``n_heads // n_groups``.  Larger values reduce Bmat
            traffic but shrink the grid, reducing SM occupancy.

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
        self.init_config(config, tune)
        heads_per_group = self.config["heads_per_group"]
        heads_per_bmat_group = n_heads // n_groups
        if heads_per_bmat_group % heads_per_group != 0:
            raise ValueError(
                f"heads_per_group={heads_per_group} must divide "
                f"n_heads // n_groups = {heads_per_bmat_group}"
            )
        self.kernel = _ssd_chunk_state_fwd_mh_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups,
            heads_per_group, has_seq_idx, self.dtype_str,
        )

    @property
    def default_config(self) -> dict:
        # heads_per_group=4: 4× Bmat reuse with enough CTAs for b4/L8k shapes
        # (256–640 CTAs = 1.9–4.8 waves on H200 with 132 SMs).
        # block_n=128 covers full d_state=128 in one N-tile.
        return {
            "block_n": 128,
            "block_p": 64,
            "block_l": 64,
            "threads": 128,
            "heads_per_group": 4,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # heads_per_group in {1, 2, 4, 8}: larger values reduce Bmat traffic
        #   but shrink the grid; 1 degenerates to the single-head kernel.
        # Other tile params follow the same rationale as SSDChunkStateFwdKernel.
        block_n = [64, 128]
        block_p = [16, 32, 64]
        block_l = [32, 64, 128]
        threads = [128, 256]
        hpg     = [1, 2, 4, 8]
        _configs = list(itertools.product(block_n, block_p, block_l, threads, hpg))
        return [{
            "block_n": c[0],
            "block_p": c[1],
            "block_l": c[2],
            "threads": c[3],
            "heads_per_group": c[4],
        } for c in _configs]

    def forward(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        seq_idx: torch.Tensor,
    ) -> torch.Tensor:
        return _ssd_chunk_state_fwd_mh_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.d_head,
            self.d_state, self.n_groups, self.config["heads_per_group"],
            self.has_seq_idx, self.dtype_str,
            self.config["block_n"], self.config["block_p"], self.config["block_l"],
            self.config["threads"],
            x, Bmat, dt, dA_cumsum, seq_idx,
        )
