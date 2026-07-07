"""
Mamba-2 State-Space Dual (SSD) fused chunk output forward kernel (history + intra-chunk paths).

Official-aligned interface (matches _chunk_scan_fwd in mamba_ssm):

  x:            (batch, seqlen, n_heads, d_head)              dtype
  cb:           (batch, num_chunks, n_groups, chunk_len, chunk_len)  dtype
                -- precomputed C@B coupling; group-owned (not head-owned)
  dA_cumsum:    (batch, n_heads, num_chunks, chunk_len)        float32
  C:            (batch, seqlen, n_groups, d_state)             dtype
                -- readout matrix; seqlen-fused, group-owned
  prev_states:  (batch, num_chunks, n_heads, d_head, d_state)  float32
                -- state entering each chunk; P before N (official convention)
  dt:           (batch, n_heads, num_chunks, chunk_len)        dtype

Output:
  out:          (batch, seqlen, n_heads, d_head)               float32
                -- seqlen-fused output

For each (b, c, l, h, p), the kernel computes:

  out[b, c*Q+l, h, p] = Y_off[b,c,l,h,p] + Y_diag[b,c,l,h,p]

where the history / prev-state contribution is:

  Y_off[b,c,l,h,p]
    = exp(dA_cumsum[b,h,c,l]) * sum_n C[b,c*Q+l,g(h),n] * prev_states[b,c,h,p,n]

and the intra-chunk / causal-diagonal contribution is:

  Y_diag[b,c,l,h,p]
    = sum_{s <= l}
        cb[b,c,g(h),l,s]
        * exp(dA_cumsum[b,h,c,l] - dA_cumsum[b,h,c,s])
        * dt[b,h,c,s]
        * x[b,c*Q+s,h,p]

where g(h) = h // heads_per_group.

Layout changes vs old TileOPs version
--------------------------------------
  old                              new (official)
  x:          [B,C,L,H,P]         [B,S,H,P]          seqlen-fused
  cb:         [B,C,H,L,L]         [B,C,G,L,L]        group-owned
  C:          [B,C,L,H,N]         [B,S,G,N]           seqlen-fused, group-owned
  prev_states:[B,C,H,N,P]         [B,C,H,P,N]        P before N
  dt:         [B,C,L,H]           [B,H,C,L]          H before C
  out:        [B,C,L,H,P]         [B,S,H,P]          seqlen-fused

Notation:
  B = batch, S = seqlen = C*Q, H = n_heads, P = d_head, G = n_groups,
  N = d_state, C = num_chunks, Q = chunk_len
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["SSDChunkScanFwdKernel"]


@functools.lru_cache(maxsize=32)
def _ssd_chunk_scan_fwd_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str = "float16",
    diagonal_microtile_size: int = 32,  # 32=M32 (only for block_l=block_s=64)
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
        block_l: int,
        block_p: int,
        block_n: int,
        block_s: int,
        threads: int,
        num_stages: int,
    ):
        # Official layouts
        x_shape          = (B, S, H, P)        # [B, S, H, P]
        cb_shape         = (B, C, G, Q, Q)     # [B, C, G, L, L]  group-owned
        dA_shape         = (B, H, C, Q)        # [B, H, C, L]
        c_shape          = (B, S, G, N)        # [B, S, G, N]     group-owned
        states_shape     = (B, C, H, P, N)     # [B, C, H, P, N]  P before N
        dt_shape         = (B, H, C, Q)        # [B, H, C, L]
        out_shape        = (B, S, H, P)        # [B, S, H, P]

        @T.prim_func
        def main(
            x:           T.Tensor(x_shape, dtype),           # type: ignore
            cb:          T.Tensor(cb_shape, dtype),           # type: ignore
            dA_cumsum:   T.Tensor(dA_shape, accum_dtype),    # type: ignore
            C_mat:       T.Tensor(c_shape, dtype),            # type: ignore
            prev_states: T.Tensor(states_shape, accum_dtype),  # type: ignore
            dt:          T.Tensor(dt_shape, dtype),           # type: ignore
            out:         T.Tensor(out_shape, accum_dtype),   # type: ignore
        ):
            # Grid redesign: separate H dimension for better load balancing
            # New layout: (L_tiles × P_tiles, B × C, H)
            with T.Kernel(
                T.ceildiv(Q, block_l) * T.ceildiv(P, block_p),
                B * C,
                H,
                threads=threads,
            ) as (blp, bc, bh):
                bl = blp // T.ceildiv(P, block_p)
                bp = blp % T.ceildiv(P, block_p)
                bz = bc // C
                bc_idx = bc % C

                bg = bh // HEADS_PER_GROUP
                chunk_start = bc_idx * Q

                l0 = bl * block_l
                p0 = bp * block_p

                # output accumulator [block_l, block_p]
                acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(acc)

                # =====================================================
                # PART 1: history path
                #   acc[l,p] += exp(dA_l[l]) * sum_n C[l,g,n] * prev_states[h,p,n]
                # =====================================================

                hist_acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(hist_acc)

                c_tile     = T.alloc_shared((block_l, block_n), dtype)
                state_tile = T.alloc_shared((block_n, block_p), dtype)

                # Swizzled layouts eliminate bank conflicts and enable tensor-core GEMMs.
                T.annotate_layout({
                    c_tile:     tilelang.layout.make_swizzled_layout(c_tile),
                    state_tile: tilelang.layout.make_swizzled_layout(state_tile),
                })

                #trace c_tile loading
                for n_blk in T.Pipelined(T.ceildiv(N, block_n), num_stages=num_stages):
                    n0 = n_blk * block_n

                    # C_mat[b, chunk_start+l, g, n]  layout: [B, S, G, N]
                    for ll, nn in T.Parallel(block_l, block_n):
                        l_abs = l0 + ll
                        n_abs = n0 + nn
                        safe_l = T.min(l_abs, Q - 1)
                        safe_n_c = T.min(n_abs, N - 1)
                        c_tile[ll, nn] = T.if_then_else(
                            (l_abs < Q) and (n_abs < N),
                            C_mat[bz, chunk_start + safe_l, bg, safe_n_c],
                            T.cast(T.float32(0.0), dtype),
                        )

                    #trace state_tile loading
                    # prev_states[b, c, h, p, n]  layout: [B, C, H, P, N]  float32
                    # Iterate (block_p, block_n) so consecutive threads vary nn (the contiguous N
                    # dim), giving coalesced 128-byte loads instead of strided-by-N accesses.
                    for pp, nn in T.Parallel(block_p, block_n):
                        n_abs = n0 + nn
                        p_abs = p0 + pp
                        safe_n = T.min(n_abs, N - 1)
                        safe_p = T.min(p_abs, P - 1)
                        state_tile[nn, pp] = T.if_then_else(
                            (n_abs < N) and (p_abs < P),
                            T.cast(prev_states[bz, bc_idx, bh, safe_p, safe_n], dtype),
                            T.cast(T.float32(0.0), dtype),
                        )

                    # trace gemm
                    # hist_acc += c_tile @ state_tile
                    T.gemm(c_tile, state_tile, hist_acc)



                # =====================================================
                # Cache dA_cumsum and dt for this chunk in shared memory.
                # Eliminates repeated L2 round-trips in the exp_l/exp_s and
                # diagonal-path loops.  Q fp32 scalars = Q*4 bytes (e.g. 1 KB
                # for Q=256).  Loaded once, reused by every s-block.
                # =====================================================
                dA_smem = T.alloc_shared((Q,), accum_dtype)
                dt_smem  = T.alloc_shared((Q,), accum_dtype)
                for q in T.Parallel(Q):
                    dA_smem[q] = dA_cumsum[bz, bh, bc_idx, q]
                    dt_smem[q]  = T.cast(dt[bz, bh, bc_idx, q], accum_dtype)
                T.sync_threads()
                # Ensure all threads have finished loading dA_cumsum and dt into shared memory

                # Precompute exp(dA_l[ll]) once per l-tile for history path scaling.
                # Now reads from dA_smem (smem hit) instead of global.
                exp_dA_l = T.alloc_shared((block_l,), accum_dtype)
                for ll in T.Parallel(block_l):
                    l_abs = l0 + ll
                    safe_l_da = T.min(l_abs, Q - 1)
                    exp_dA_l[ll] = T.if_then_else(
                        l_abs < Q,
                        T.exp(dA_smem[safe_l_da]),
                        T.float32(1.0),
                    )
                T.sync_threads()

                # scale by exp(dA_l[l]) and accumulate into acc
                for ll, pp in T.Parallel(block_l, block_p):
                    l_abs = l0 + ll
                    p_abs = p0 + pp
                    if (l_abs < Q) and (p_abs < P):
                        acc[ll, pp] += hist_acc[ll, pp] * exp_dA_l[ll]

                # =====================================================
                # PART 2: intra-chunk causal path
                #   acc[l,p] += sum_{s<=l} cb[c,g,l,s]
                #               * exp(dA_l[l] - dA_s[s]) * dt[h,c,s] * x[s,h,p]
                #
                # L-side anchor factored form (full-lower blocks only):
                #   exp(dA_l - dA_s) = exp(dA_l - anchor) * exp(anchor - dA_s)
                # where anchor = dA_cumsum[bz, bh, bc_idx, l0] (the largest value in the
                # l-tile, since dA_cumsum is non-increasing).
                #
                # Both arguments are non-positive for all valid causal pairs:
                #   dA_l[ll] - anchor <= 0  because dA_l[ll] <= dA_l[l0] = anchor
                #   anchor - dA_s[ss] <= 0  because anchor = dA_l[l0] <= dA_s[ss]
                #                           for any s >= l0 (full-lower condition)
                # Non-positive exponents cannot overflow; underflow to 0 is numerically
                # correct (the true value is also ~0).  No clamp required.
                # MUFU count: block_l + block_s per s-block.
                #
                # Diagonal blocks (s0 <= l0 < s0 + block_s): s is close to l so the
                # difference dA_l - dA_s is bounded by at most block_s decay steps
                # (a small value); compute exp(dA_l - dA_s) directly per element.
                # This path already has per-element guard branches, so the extra MUFU
                # cost (block_l * block_s in the worst case) is acceptable.
                #
                # Upper-triangle blocks are skipped entirely by the loop bound.
                # =====================================================
                cb_tile    = T.alloc_shared((block_l, block_s), dtype)
                x_tile     = T.alloc_shared((block_s, block_p), dtype)
                # Full-lower buffers (l-side anchor).
                # exp_l[ll]  = exp(dA_l[ll] - anchor),  anchor = dA_l[l0]
                # exp_s[ss]  = exp(anchor - dA_s[ss]) * dt[ss]
                exp_l = T.alloc_shared((block_l,), accum_dtype)
                exp_s = T.alloc_shared((block_s,), accum_dtype)
                lcb_cast = T.alloc_fragment((block_l, block_s), dtype)

                # Swizzled layouts for causal GEMM operands.
                T.annotate_layout({
                    cb_tile:  tilelang.layout.make_swizzled_layout(cb_tile),
                    x_tile:   tilelang.layout.make_swizzled_layout(x_tile),
                })

                # anchor = dA_smem at l0, the largest value in this l-tile.
                safe_l0 = T.min(l0, Q - 1)
                anchor = T.if_then_else(
                    l0 < Q,
                    dA_smem[safe_l0],
                    T.float32(0.0),
                )

                # Precompute exp_l once before the s-loop; reused by every
                # full-lower s-block without re-fetching dA_cumsum from global.
                for ll in T.Parallel(block_l):
                    l_abs = l0 + ll
                    safe_l_exp = T.min(l_abs, Q - 1)
                    dA_l_val = T.if_then_else(
                        l_abs < Q,
                        dA_smem[safe_l_exp],
                        anchor,
                    )
                    # dA_l_val - anchor <= 0 always, so exp <= 1 (no overflow).
                    exp_l[ll] = T.exp(dA_l_val - anchor)
                T.sync_threads()

                # Only iterate over s-blocks that have at least one s <= l_max.
                for s_blk in T.Pipelined(T.ceildiv(l0 + block_l, block_s), num_stages=num_stages):
                    s0 = s_blk * block_s

                    # cb[b, c, g, l, s]  layout: [B, C, G, L, L]
                    for ll, ss in T.Parallel(block_l, block_s):
                        l_abs = l0 + ll
                        s_abs = s0 + ss
                        safe_l_cb = T.min(l_abs, Q - 1)
                        safe_s_cb = T.min(s_abs, Q - 1)
                        cb_tile[ll, ss] = T.if_then_else(
                            (l_abs < Q) and (s_abs < Q),
                            cb[bz, bc_idx, bg, safe_l_cb, safe_s_cb],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # x[b, chunk_start+s, h, p]  layout: [B, S, H, P]
                    for ss, pp in T.Parallel(block_s, block_p):
                        s_abs = s0 + ss
                        p_abs = p0 + pp
                        safe_s_x = T.min(s_abs, Q - 1)
                        safe_p_x = T.min(p_abs, P - 1)
                        x_tile[ss, pp] = T.if_then_else(
                            (s_abs < Q) and (p_abs < P),
                            x[bz, chunk_start + safe_s_x, bh, safe_p_x],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # full-lower path: s0 + block_s <= l0 means every (ll,ss) has
                    # s_abs < l0 <= l_abs, so s_abs < l_abs (causal mask always true)
                    # and anchor <= dA_s[ss] (anchor - dA_s[ss] <= 0, no overflow).
                    if s0 + block_s <= l0:
                        for ss in T.Parallel(block_s):
                            s_abs = s0 + ss
                            safe_s_full = T.min(s_abs, Q - 1)
                            dA_s_val = T.if_then_else(
                                s_abs < Q,
                                dA_smem[safe_s_full],
                                anchor,
                            )
                            dt_val = T.if_then_else(
                                s_abs < Q,
                                dt_smem[safe_s_full],
                                T.float32(0.0),
                            )
                            exp_s[ss] = T.exp(anchor - dA_s_val) * dt_val
                        T.sync_threads()

                        # Scale cb_tile by exp_l * exp_s
                        for ll, ss in T.Parallel(block_l, block_s):
                            lcb_cast[ll, ss] = T.cast(
                                T.cast(cb_tile[ll, ss], accum_dtype)
                                * exp_l[ll]
                                * exp_s[ss],
                                dtype,
                            )
                    else:
                        T.sync_threads()
                        # Diagonal path: s is within block_s steps of l, so
                        # dA_l - dA_s is bounded and safe to compute directly.
                        # Reads from dA_smem/dt_smem (smem) instead of global.

                        # Micro-block factorization: only for block_l=block_s=64, M=32
                        if diagonal_microtile_size == 32 and block_l == 64 and block_s == 64:
                            # Python static unrolling for 2×2 micro-blocks
                            M = 32
                            R = 2  # block_l // M

                            # Allocate shared memory for factorized exponentials
                            exp_l_micro = T.alloc_shared((block_l,), accum_dtype)
                            exp_s_micro = T.alloc_shared((R, block_s), accum_dtype)

                            # Precompute exp_l_micro[ll] = exp(dA[ll] - anchor[r])
                            for ll in T.Parallel(block_l):
                                r = ll // M
                                safe_l = T.min(l0 + ll, Q - 1)
                                anchor_idx = T.min(l0 + r * M, Q - 1)
                                exp_l_micro[ll] = T.exp(dA_smem[safe_l] - dA_smem[anchor_idx])

                            # Precompute exp_s_micro[r, ss] = exp(anchor[r] - dA[ss]) * dt[ss]
                            for r, ss in T.Parallel(R, block_s):
                                safe_s = T.min(s0 + ss, Q - 1)
                                anchor_idx = T.min(l0 + r * M, Q - 1)
                                exp_s_micro[r, ss] = T.exp(dA_smem[anchor_idx] - dA_smem[safe_s]) * dt_smem[safe_s]

                            T.sync_threads()

                            # Python static unrolling: 4 micro-blocks
                            # mr=0, mc=0: diagonal block
                            for ll, ss in T.Parallel(M, M):
                                l_abs = l0 + ll
                                s_abs = s0 + ss
                                valid = (l_abs < Q) and (s_abs < Q) and (s_abs <= l_abs)
                                safe_l = T.min(l_abs, Q - 1)
                                safe_s = T.min(s_abs, Q - 1)
                                exp_factor = T.exp(dA_smem[safe_l] - dA_smem[safe_s])
                                value = T.cast(cb_tile[ll, ss], accum_dtype) * exp_factor * dt_smem[safe_s]
                                lcb_cast[ll, ss] = T.if_then_else(valid, T.cast(value, dtype), T.cast(T.float32(0.0), dtype))

                            # mr=0, mc=1: upper block (all zeros)
                            for ll, ss in T.Parallel(M, M):
                                lcb_cast[ll, M + ss] = T.cast(T.float32(0.0), dtype)

                            # mr=1, mc=0: lower block (factorization, NO direct exp)
                            for ll, ss in T.Parallel(M, M):
                                l_abs = l0 + M + ll
                                s_abs = s0 + ss
                                valid = (l_abs < Q) and (s_abs < Q) and (s_abs <= l_abs)
                                # r=1 for second micro-row, inline directly
                                value = T.cast(cb_tile[M + ll, ss], accum_dtype) * exp_l_micro[M + ll] * exp_s_micro[1, ss]
                                lcb_cast[M + ll, ss] = T.if_then_else(valid, T.cast(value, dtype), T.cast(T.float32(0.0), dtype))

                            # mr=1, mc=1: diagonal block
                            for ll, ss in T.Parallel(M, M):
                                l_abs = l0 + M + ll
                                s_abs = s0 + M + ss
                                valid = (l_abs < Q) and (s_abs < Q) and (s_abs <= l_abs)
                                safe_l = T.min(l_abs, Q - 1)
                                safe_s = T.min(s_abs, Q - 1)
                                exp_factor = T.exp(dA_smem[safe_l] - dA_smem[safe_s])
                                value = T.cast(cb_tile[M + ll, M + ss], accum_dtype) * exp_factor * dt_smem[safe_s]
                                lcb_cast[M + ll, M + ss] = T.if_then_else(valid, T.cast(value, dtype), T.cast(T.float32(0.0), dtype))
                        else:
                            # Original diagonal path (fallback)
                            for ll, ss in T.Parallel(block_l, block_s):
                                l_abs = l0 + ll
                                s_abs = s0 + ss
                                valid = (l_abs < Q) and (s_abs < Q) and (s_abs <= l_abs)
                                safe_l_diag = T.min(l_abs, Q - 1)
                                safe_s_diag = T.min(s_abs, Q - 1)
                                dA_l_val = T.if_then_else(
                                    l_abs < Q,
                                    dA_smem[safe_l_diag],
                                    T.float32(0.0),
                                )
                                dA_s_val = T.if_then_else(
                                    s_abs < Q,
                                    dA_smem[safe_s_diag],
                                    T.float32(0.0),
                                )
                                dt_val = T.if_then_else(
                                    s_abs < Q,
                                    dt_smem[safe_s_diag],
                                    T.float32(0.0),
                                )
                                lcb_cast[ll, ss] = T.if_then_else(
                                    valid,
                                    T.cast(
                                        T.cast(cb_tile[ll, ss], accum_dtype)
                                        * T.exp(dA_l_val - dA_s_val)
                                        * dt_val,
                                        dtype,
                                    ),
                                    T.cast(T.float32(0.0), dtype),
                                )

                    # acc += lcb_cast @ x_tile
                    T.gemm(lcb_cast, x_tile, acc)



                # write back: out[b, chunk_start+l, h, p]  layout: [B, S, H, P]
                for ll, pp in T.Parallel(block_l, block_p):
                    l_abs = l0 + ll
                    p_abs = p0 + pp
                    if (l_abs < Q) and (p_abs < P):
                        out[bz, chunk_start + l_abs, bh, p_abs] = acc[ll, pp]

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_chunk_scan_fwd", mutates_args=())
def _ssd_chunk_scan_fwd_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_l: int,
    block_p: int,
    block_n: int,
    block_s: int,
    threads: int,
    num_stages: int,
    x: torch.Tensor,
    cb: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    return _ssd_chunk_scan_fwd_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype)(
        block_l, block_p, block_n, block_s, threads, num_stages,
    )(x, cb, dA_cumsum, C, prev_states, dt)


@_ssd_chunk_scan_fwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_l: int,
    block_p: int,
    block_n: int,
    block_s: int,
    threads: int,
    num_stages: int,
    x: torch.Tensor,
    cb: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    # output: [B, S, H, P]
    return x.new_empty(
        (batch, num_chunks * chunk_len, n_heads, d_head), dtype=torch.float32,
    )


class SSDChunkScanFwdKernel(Kernel):
    """Mamba-2 SSD fused chunk output forward kernel.

    Official-aligned interface (matches _chunk_scan_fwd in mamba_ssm):

    Inputs:
      x:           [B, S, H, P]        dtype       seqlen-fused
      cb:          [B, C, G, L, L]     dtype       group-owned
      dA_cumsum:   [B, H, C, L]        float32
      C:           [B, S, G, N]        dtype       seqlen-fused, group-owned
      prev_states: [B, C, H, P, N]     float32     P before N
      dt:          [B, H, C, L]        dtype

    Output:
      out:         [B, S, H, P]        float32     seqlen-fused
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
        self.kernel = _ssd_chunk_scan_fwd_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # threads=128 (4 warps) balances parallelism with register pressure.
        # block_n=64 keeps occupancy high for typical d_state sizes (64-128).
        # num_stages=3 optimal based on full-kernel benchmarking.
        return {
            "block_l": 64,
            "block_p": 64,
            "block_n": min(64, self.d_state),
            "block_s": 64,
            "threads": 128,
            "num_stages": 3,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # Focused search around the known-good default (block_l=64, block_p=64,
        # block_n=128, block_s=64, threads=128, num_stages=3).
        #
        # NCU evidence:
        #   - block_l=64, block_p=64 anchors GEMM tile efficiency; smaller tiles
        #     hurt more than they help (tested: shape-aware default was slower).
        #   - block_n only affects the history-path loop count; vary minimally.
        #   - block_s and threads are the primary levers: block_s controls causal
        #     GEMM tile size and s-loop iteration count; threads controls warps/block
        #     and latency-hiding capacity.
        #   - num_stages=3 is optimal (tested: stages=5 is 13% slower)
        #
        # 6–8 configs total (2 block_n entries dropped when d_state <= 32 or 64).
        block_n = min(128, self.d_state)
        return [
            {"block_l": 64, "block_p": 64, "block_n": block_n, "block_s": bs, "threads": t, "num_stages": 3}
            for bs in [64, 128]
            for t  in [128, 256]
        ] + [
            # block_n sweep at fixed block_s=64, threads=128
            {"block_l": 64, "block_p": 64, "block_n": bn, "block_s": 64, "threads": 128, "num_stages": 3}
            for bn in [32, 64]
            if bn <= self.d_state
        ] + [
            # threads=64 (2 warps/block) — more blocks/SM at cost of less ILP
            # Include block_n=64 to cover the optimal config found via AKO tuning
            {"block_l": 64, "block_p": 64, "block_n": bn, "block_s": bs, "threads": 64, "num_stages": 3}
            for bs in [64, 128]
            for bn in [64, 128]
            if bn <= self.d_state
        ]

    def forward(
        self,
        x: torch.Tensor,
        cb: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:           [B, S, H, P]        dtype
            cb:          [B, C, G, L, L]     dtype
            dA_cumsum:   [B, H, C, L]        float32
            C:           [B, S, G, N]        dtype
            prev_states: [B, C, H, P, N]     float32     P before N
            dt:          [B, H, C, L]        dtype

        Returns:
            out: [B, S, H, P]  float32
        """
        return _ssd_chunk_scan_fwd_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads,
            self.d_head, self.d_state, self.n_groups, self.dtype_str,
            self.config["block_l"], self.config["block_p"], self.config["block_n"],
            self.config["block_s"], self.config["threads"], self.config["num_stages"],
            x.contiguous(), cb.contiguous(), dA_cumsum.contiguous(),
            C.contiguous(), prev_states.contiguous(), dt.contiguous(),
        )
