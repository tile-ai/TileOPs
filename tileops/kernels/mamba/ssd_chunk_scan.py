"""
Mamba-2 State-Space Dual (SSD) fused chunk output forward kernel (history + intra-chunk paths).

Official-aligned interface (matches _chunk_scan_fwd in mamba_ssm):

  x:            (batch, seqlen, n_heads, d_head)              dtype
  cb:           (batch, num_chunks, n_groups, chunk_len, chunk_len)  dtype
                -- precomputed C@B coupling; group-owned (not head-owned)
  dA_cumsum:    (batch, n_heads, num_chunks, chunk_len)        float32
  C:            (batch, seqlen, n_groups, d_state)             dtype
                -- readout matrix; seqlen-fused, group-owned
  prev_states:  (batch, num_chunks, n_heads, d_head, d_state)  dtype
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

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["SSDChunkScanFwdKernel"]


def _ssd_chunk_scan_fwd_kernel(
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
            prev_states: T.Tensor(states_shape, dtype),       # type: ignore
            dt:          T.Tensor(dt_shape, dtype),           # type: ignore
            out:         T.Tensor(out_shape, accum_dtype),   # type: ignore
        ):
            # Grid: fuse (B, H, C) into axis-0; tile L and P
            with T.Kernel(
                B * H * C,
                T.ceildiv(Q, block_l),
                T.ceildiv(P, block_p),
                threads=threads,
            ) as (bhc, bl, bp):

                bz = bhc // (H * C)
                bh = (bhc % (H * C)) // C
                bc = bhc % C

                bg = bh // HEADS_PER_GROUP   # head -> group
                chunk_start = bc * Q         # first token index of this chunk

                l0 = bl * block_l
                p0 = bp * block_p

                # output accumulator [block_l, block_p]
                acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(acc)

                # load target-side dA_cumsum[b,h,c,l] for this l-tile
                dA_l = T.alloc_fragment((block_l,), accum_dtype)
                for ll in T.Parallel(block_l):
                    l_abs = l0 + ll
                    dA_l[ll] = T.if_then_else(
                        l_abs < Q,
                        dA_cumsum[bz, bh, bc, l_abs],
                        T.float32(0.0),
                    )

                # =====================================================
                # PART 1: history path
                #   acc[l,p] += exp(dA_l[l]) * sum_n C[l,g,n] * prev_states[h,p,n]
                # =====================================================
                hist_acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(hist_acc)

                c_tile     = T.alloc_shared((block_l, block_n), dtype)
                state_tile = T.alloc_shared((block_n, block_p), dtype)

                for n_blk in T.serial(T.ceildiv(N, block_n)):
                    n0 = n_blk * block_n

                    # C_mat[b, chunk_start+l, g, n]  layout: [B, S, G, N]
                    for ll, nn in T.Parallel(block_l, block_n):
                        l_abs = l0 + ll
                        n_abs = n0 + nn
                        c_tile[ll, nn] = T.if_then_else(
                            (l_abs < Q) and (n_abs < N),
                            C_mat[bz, chunk_start + l_abs, bg, n_abs],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # prev_states[b, c, h, p, n]  layout: [B, C, H, P, N]
                    for nn, pp in T.Parallel(block_n, block_p):
                        n_abs = n0 + nn
                        p_abs = p0 + pp
                        state_tile[nn, pp] = T.if_then_else(
                            (n_abs < N) and (p_abs < P),
                            prev_states[bz, bc, bh, p_abs, n_abs],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # hist_acc += c_tile @ state_tile
                    T.gemm(c_tile, state_tile, hist_acc)

                # scale by exp(dA_l[l]) and accumulate into acc
                for ll, pp in T.Parallel(block_l, block_p):
                    l_abs = l0 + ll
                    p_abs = p0 + pp
                    if (l_abs < Q) and (p_abs < P):
                        acc[ll, pp] += hist_acc[ll, pp] * T.exp(dA_l[ll])

                # =====================================================
                # PART 2: intra-chunk causal path
                #   acc[l,p] += sum_{s<=l} cb[c,g,l,s]
                #               * exp(dA_l[l] - dA_s[s]) * dt[h,c,s] * x[s,h,p]
                # =====================================================
                cb_tile  = T.alloc_shared((block_l, block_s), dtype)
                x_tile   = T.alloc_shared((block_s, block_p), dtype)
                dA_s     = T.alloc_fragment((block_s,), accum_dtype)
                dt_s     = T.alloc_fragment((block_s,), accum_dtype)
                lcb      = T.alloc_fragment((block_l, block_s), accum_dtype)
                lcb_cast = T.alloc_fragment((block_l, block_s), dtype)

                for s_blk in T.serial(T.ceildiv(Q, block_s)):
                    s0 = s_blk * block_s

                    # cb[b, c, g, l, s]  layout: [B, C, G, L, L]
                    for ll, ss in T.Parallel(block_l, block_s):
                        l_abs = l0 + ll
                        s_abs = s0 + ss
                        cb_tile[ll, ss] = T.if_then_else(
                            (l_abs < Q) and (s_abs < Q),
                            cb[bz, bc, bg, l_abs, s_abs],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # x[b, chunk_start+s, h, p]  layout: [B, S, H, P]
                    for ss, pp in T.Parallel(block_s, block_p):
                        s_abs = s0 + ss
                        p_abs = p0 + pp
                        x_tile[ss, pp] = T.if_then_else(
                            (s_abs < Q) and (p_abs < P),
                            x[bz, chunk_start + s_abs, bh, p_abs],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # dA_cumsum[b,h,c,s] and dt[b,h,c,s]  layout: [B,H,C,L]
                    for ss in T.Parallel(block_s):
                        s_abs = s0 + ss
                        dA_s[ss] = T.if_then_else(
                            s_abs < Q,
                            dA_cumsum[bz, bh, bc, s_abs],
                            T.float32(0.0),
                        )
                        dt_s[ss] = T.if_then_else(
                            s_abs < Q,
                            T.cast(dt[bz, bh, bc, s_abs], accum_dtype),
                            T.float32(0.0),
                        )

                    # lcb[l,s] = cb[l,s] * exp(dA_l[l] - dA_s[s]) * dt[s]  if s<=l else 0
                    for ll, ss in T.Parallel(block_l, block_s):
                        l_abs = l0 + ll
                        s_abs = s0 + ss
                        valid = (l_abs < Q) and (s_abs < Q) and (s_abs <= l_abs)
                        lcb[ll, ss] = T.if_then_else(
                            valid,
                            T.cast(cb_tile[ll, ss], accum_dtype)
                            * T.exp(dA_l[ll] - dA_s[ss])
                            * dt_s[ss],
                            T.float32(0.0),
                        )

                    # cast to dtype for gemm (both operands must share dtype)
                    T.copy(lcb, lcb_cast)

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
    x: torch.Tensor,
    cb: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    return _ssd_chunk_scan_fwd_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype)(
        block_l, block_p, block_n, block_s, threads,
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
      prev_states: [B, C, H, P, N]     dtype       P before N
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
        return {
            "block_l": 64,
            "block_p": 64,
            "block_n": 32,
            "block_s": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_l = [32, 64]
        block_p = [32, 64]
        block_n = [16, 32]
        block_s = [32, 64]
        threads = [128, 256]
        return [
            {"block_l": c[0], "block_p": c[1], "block_n": c[2], "block_s": c[3], "threads": c[4]}
            for c in itertools.product(block_l, block_p, block_n, block_s, threads)
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
            prev_states: [B, C, H, P, N]     dtype
            dt:          [B, H, C, L]        dtype

        Returns:
            out: [B, S, H, P]  float32
        """
        return _ssd_chunk_scan_fwd_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads,
            self.d_head, self.d_state, self.n_groups, self.dtype_str,
            self.config["block_l"], self.config["block_p"], self.config["block_n"],
            self.config["block_s"], self.config["threads"],
            x.contiguous(), cb.contiguous(), dA_cumsum.contiguous(),
            C.contiguous(), prev_states.contiguous(), dt.contiguous(),
        )
