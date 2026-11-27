"""Fuse Mamba-2 SSD history and causal intra-chunk output paths.

Inputs follow the Mamba-2 sequence-fused layouts: ``x`` and ``C`` are
sequence-fused, ``cb`` and ``C`` are group-owned, and ``prev_states`` stores
``P`` before ``N``. The float32 output adds the decayed incoming state to the
causal chunk contribution.
"""

from collections.abc import Callable
from typing import ClassVar

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.mamba.call_spec import SSDChunkScanCall, SSDChunkScanFwdInterface

__all__ = ["SSDChunkScanFwdKernel"]


@target_cache(maxsize=32)
def _ssd_chunk_scan_fwd_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    diagonal_microtiles: bool,
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
        scan_annotations = {"pragma_unroll_explicit": True}
        if num_stages > 0:
            scan_annotations["num_stages"] = T.int32(num_stages)

        @T.prim_func
        def main(
            x: T.Tensor((B, S, H, P), dtype),  # type: ignore
            cb: T.Tensor((B, C, G, Q, Q), dtype),  # type: ignore
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),  # type: ignore
            C_mat: T.Tensor((B, S, G, N), dtype),  # type: ignore
            prev_states: T.Tensor((B, C, H, P, N), accum_dtype),  # type: ignore
            dt: T.Tensor((B, H, C, Q), dtype),  # type: ignore
            out: T.Tensor((B, S, H, P), accum_dtype),  # type: ignore
        ):
            # Grid: (L_tiles × P_tiles, B × C, H).
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

                acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(acc)

                # PART 1: history path
                #   acc[l,p] += exp(dA_l[l]) * sum_n C[l,g,n] * prev_states[h,p,n]

                hist_acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(hist_acc)

                c_tile = T.alloc_shared((block_l, block_n), dtype)
                state_tile = T.alloc_shared((block_n, block_p), dtype)

                T.annotate_layout(
                    {
                        c_tile: tilelang.layout.make_swizzled_layout(c_tile),
                        state_tile: tilelang.layout.make_swizzled_layout(state_tile),
                    }
                )

                for n_blk in T.Pipelined(T.ceildiv(N, block_n), num_stages=num_stages):
                    n0 = n_blk * block_n

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

                    # (block_p, block_n) order: consecutive threads vary the contiguous N dim.
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

                    T.gemm(c_tile, state_tile, hist_acc)

                # dA_cumsum and dt of this chunk, read by every s-block.
                dA_smem = T.alloc_shared((Q,), accum_dtype)
                dt_smem = T.alloc_shared((Q,), accum_dtype)
                for q in T.Parallel(Q):
                    dA_smem[q] = dA_cumsum[bz, bh, bc_idx, q]
                    dt_smem[q] = T.cast(dt[bz, bh, bc_idx, q], accum_dtype)
                T.sync_threads()

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

                for ll, pp in T.Parallel(block_l, block_p):
                    l_abs = l0 + ll
                    p_abs = p0 + pp
                    if (l_abs < Q) and (p_abs < P):
                        acc[ll, pp] += hist_acc[ll, pp] * exp_dA_l[ll]

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
                # Non-positive exponents cannot overflow, so no clamp is needed.
                #
                # Diagonal blocks (s0 <= l0 < s0 + block_s) compute exp(dA_l - dA_s) per element.
                # The loop bound skips upper-triangle blocks.
                cb_tile = T.alloc_shared((block_l, block_s), dtype)
                x_tile = T.alloc_shared((block_s, block_p), dtype)
                # exp_l[ll]  = exp(dA_l[ll] - anchor),  anchor = dA_l[l0]
                # exp_s[ss]  = exp(anchor - dA_s[ss]) * dt[ss]
                exp_l = T.alloc_shared((block_l,), accum_dtype)
                exp_s = T.alloc_shared((block_s,), accum_dtype)
                lcb_cast = T.alloc_fragment((block_l, block_s), dtype)

                T.annotate_layout(
                    {
                        cb_tile: tilelang.layout.make_swizzled_layout(cb_tile),
                        x_tile: tilelang.layout.make_swizzled_layout(x_tile),
                    }
                )

                safe_l0 = T.min(l0, Q - 1)
                anchor = T.if_then_else(
                    l0 < Q,
                    dA_smem[safe_l0],
                    T.float32(0.0),
                )

                for ll in T.Parallel(block_l):
                    l_abs = l0 + ll
                    safe_l_exp = T.min(l_abs, Q - 1)
                    dA_l_val = T.if_then_else(
                        l_abs < Q,
                        dA_smem[safe_l_exp],
                        anchor,
                    )
                    exp_l[ll] = T.exp(dA_l_val - anchor)
                T.sync_threads()

                # Only iterate over s-blocks that have at least one s <= l_max.
                for s_blk in T.serial(
                    T.ceildiv(l0 + block_l, block_s),
                    annotations=scan_annotations,
                ):
                    s0 = s_blk * block_s

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

                    # Full-lower: every (ll, ss) is causal and anchor - dA_s[ss] <= 0.
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

                        for ll, ss in T.Parallel(block_l, block_s):
                            lcb_cast[ll, ss] = T.cast(
                                T.cast(cb_tile[ll, ss], accum_dtype) * exp_l[ll] * exp_s[ss],
                                dtype,
                            )
                    else:
                        T.sync_threads()
                        if diagonal_microtiles and block_l == 64 and block_s == 64:
                            # 2×2 micro-blocks; only the strict-lower block (1,0) is factored.
                            M = 32
                            row_factor = T.alloc_shared((M,), accum_dtype)
                            col_factor = T.alloc_shared((M,), accum_dtype)

                            anchor_idx = T.min(l0 + M, Q - 1)
                            anchor = dA_smem[anchor_idx]

                            for i in T.Parallel(M):
                                safe_l = T.min(l0 + M + i, Q - 1)
                                row_factor[i] = T.exp(dA_smem[safe_l] - anchor)

                                safe_s = T.min(s0 + i, Q - 1)
                                col_factor[i] = T.exp(anchor - dA_smem[safe_s]) * dt_smem[safe_s]

                            T.sync_threads()

                            # mr=0, mc=0: diagonal block (direct exp)
                            for ll, ss in T.Parallel(M, M):
                                l_abs = l0 + ll
                                s_abs = s0 + ss
                                valid = (l_abs < Q) and (s_abs < Q) and (s_abs <= l_abs)
                                safe_l = T.min(l_abs, Q - 1)
                                safe_s = T.min(s_abs, Q - 1)
                                exp_factor = T.exp(dA_smem[safe_l] - dA_smem[safe_s])
                                value = (
                                    T.cast(cb_tile[ll, ss], accum_dtype)
                                    * exp_factor
                                    * dt_smem[safe_s]
                                )
                                lcb_cast[ll, ss] = T.if_then_else(
                                    valid, T.cast(value, dtype), T.cast(T.float32(0.0), dtype)
                                )

                            # mr=0, mc=1: upper block, zero
                            for ll, ss in T.Parallel(M, M):
                                lcb_cast[ll, M + ss] = T.cast(T.float32(0.0), dtype)

                            # mr=1, mc=0: lower block, factored
                            for ll, ss in T.Parallel(M, M):
                                l_abs = l0 + M + ll
                                s_abs = s0 + ss
                                valid = (l_abs < Q) and (s_abs < Q) and (s_abs <= l_abs)
                                value = (
                                    T.cast(cb_tile[M + ll, ss], accum_dtype)
                                    * row_factor[ll]
                                    * col_factor[ss]
                                )
                                lcb_cast[M + ll, ss] = T.if_then_else(
                                    valid, T.cast(value, dtype), T.cast(T.float32(0.0), dtype)
                                )

                            # mr=1, mc=1: diagonal block (direct exp)
                            for ll, ss in T.Parallel(M, M):
                                l_abs = l0 + M + ll
                                s_abs = s0 + M + ss
                                valid = (l_abs < Q) and (s_abs < Q) and (s_abs <= l_abs)
                                safe_l = T.min(l_abs, Q - 1)
                                safe_s = T.min(s_abs, Q - 1)
                                exp_factor = T.exp(dA_smem[safe_l] - dA_smem[safe_s])
                                value = (
                                    T.cast(cb_tile[M + ll, M + ss], accum_dtype)
                                    * exp_factor
                                    * dt_smem[safe_s]
                                )
                                lcb_cast[M + ll, M + ss] = T.if_then_else(
                                    valid, T.cast(value, dtype), T.cast(T.float32(0.0), dtype)
                                )
                        else:
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

                    T.gemm(lcb_cast, x_tile, acc)

                for ll, pp in T.Parallel(block_l, block_p):
                    l_abs = l0 + ll
                    p_abs = p0 + pp
                    if (l_abs < Q) and (p_abs < P):
                        out[bz, chunk_start + l_abs, bh, p_abs] = acc[ll, pp]

        return main

    return kernel_func


class SSDChunkScanFwdKernel(Kernel, SSDChunkScanFwdInterface):
    """Mamba-2 SSD fused chunk output forward kernel, matching ``_chunk_scan_fwd`` in
    mamba_ssm."""

    supported_archs: list[int] = [80, 86, 89, 90]
    # The tensor-core instruction takes the history GEMM's K extent, block_n, in 16-element
    # steps; default_config and autotune_configs both round d_state up to it.
    _gemm_k_step: ClassVar[int] = 16

    @classmethod
    def entry_for(cls, call: SSDChunkScanCall) -> Entry:
        return call, lambda: cls(
            call.batch,
            call.num_chunks,
            call.chunk_len,
            call.n_heads,
            call.d_head,
            call.d_state,
            call.n_groups,
            call.dtype,
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
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        self.dtype = dtype
        self.arch = self.build_target.arch
        self.kernel = _ssd_chunk_scan_fwd_kernel(
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            d_head,
            d_state,
            n_groups,
            self.dtype_str,
            # The diagonal micro-blocks write quadrants of an MMA operand fragment, whose loop
            # layout TileLang infers only for the SM90 fragment.
            self.arch == 90,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # A d_state that is not a whole number of _gemm_k_step rounds up, and the loads mask
        # the padded columns.
        return {
            "block_l": 64,
            "block_p": 64,
            "block_n": min(64, -(-self.d_state // self._gemm_k_step) * self._gemm_k_step),
            "block_s": 64,
            "threads": 128,
            "num_stages": 2 if self.arch == 89 else 3,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # Varies block_s and threads around the default, plus a block_n sweep.
        block_n = min(128, -(-self.d_state // self._gemm_k_step) * self._gemm_k_step)
        configs = (
            [
                {
                    "block_l": 64,
                    "block_p": 64,
                    "block_n": block_n,
                    "block_s": bs,
                    "threads": t,
                    "num_stages": 3,
                }
                for bs in [64, 128]
                for t in [128, 256]
            ]
            + [
                {
                    "block_l": 64,
                    "block_p": 64,
                    "block_n": bn,
                    "block_s": 64,
                    "threads": 128,
                    "num_stages": 3,
                }
                for bn in [32, 64]
                if bn <= self.d_state
            ]
            + [
                {
                    "block_l": 64,
                    "block_p": 64,
                    "block_n": bn,
                    "block_s": bs,
                    "threads": 64,
                    "num_stages": 3,
                }
                for bs in [64, 128]
                for bn in [64, 128]
                if bn <= self.d_state
            ]
        )
        # The block_n sweep repeats the first group's entry where block_n rounds to 64.
        return [dict(items) for items in dict.fromkeys(tuple(c.items()) for c in configs)]

    def forward(
        self,
        x: torch.Tensor,
        cb: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        return self.program(self._compile)(x, cb, dA_cumsum, C, prev_states, dt)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(
            cfg["block_l"],
            cfg["block_p"],
            cfg["block_n"],
            cfg["block_s"],
            cfg["threads"],
            cfg["num_stages"],
        )
