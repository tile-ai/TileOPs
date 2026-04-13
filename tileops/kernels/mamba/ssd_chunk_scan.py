"""
Mamba-2 State-Space Dual (SSD) fused chunk output forward kernel (history + intra-chunk paths).

Inputs (pre-reshaped to chunked view):
  x:            (batch, num_chunks, chunk_len, n_heads, d_head)
  cb:           (batch, num_chunks, n_heads, chunk_len, chunk_len)
                -- precomputed source-target coupling term
                   cb[l, s] = sum_n C[l, n] * B[s, n]
  dA_cumsum:    (batch, n_heads, num_chunks, chunk_len)
                -- chunk-local prefix sums of dA, where dA = dt * A
  C:            (batch, num_chunks, chunk_len, n_heads, d_state)
  prev_states:  (batch, num_chunks, n_heads, d_state, d_head)
                -- state entering each chunk from previous chunks
  dt:           (batch, num_chunks, chunk_len, n_heads)
                -- per-position discretization factor

Output:
  out:          (batch, num_chunks, chunk_len, n_heads, d_head)

For each (b, c, l, h, p), the kernel computes:

  out[l, p] = Y_off[l, p] + Y_diag[l, p]

where the history / Step-4 contribution is

  Y_off[l, p]
    = exp(dA_cumsum[l]) * sum_n C[l, n] * prev_states[n, p]

and the intra-chunk / Step-1 contribution is

  Y_diag[l, p]
    = sum_{s <= l}
        cb[l, s]
        * exp(dA_cumsum[l] - dA_cumsum[s])
        * dt[s]
        * x[s, p]

Thus the kernel fuses:
  - Step 4: state -> output conversion from chunk-entry states
  - Step 1: intra-chunk diagonal contribution

Notation:
  b = batch, T = sequence length, h = n_heads, p = d_head, n = d_state
  Q = chunk_len, c = T / Q = num_chunks

Note:
  dA_cumsum uses layout (batch, head, chunk, position),
  while x/C/out use layout (batch, chunk, position, head, ...).
  The tensor cb uses layout (batch, chunk, head, target_position, source_position).
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
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_l: int,
        block_p: int,
        block_n: int,
        block_s: int,
        threads: int,
    ):
        x_shape = (batch, num_chunks, chunk_len, n_heads, d_head)
        cb_shape = (batch, num_chunks, n_heads, chunk_len, chunk_len)
        d_a_shape = (batch, n_heads, num_chunks, chunk_len)
        c_shape = (batch, num_chunks, chunk_len, n_heads, d_state)
        state_shape = (batch, num_chunks, n_heads, d_state, d_head)
        d_t_shape = (batch, num_chunks, chunk_len, n_heads)
        out_shape = (batch, num_chunks, chunk_len, n_heads, d_head)

        @T.prim_func
        def main(
            x: T.Tensor(x_shape, dtype),                   # type: ignore
            cb: T.Tensor(cb_shape, dtype),                  # type: ignore
            dA_cumsum: T.Tensor(d_a_shape, accum_dtype),    # type: ignore
            C: T.Tensor(c_shape, dtype),                    # type: ignore
            prev_states: T.Tensor(state_shape, dtype),      # type: ignore
            d_t: T.Tensor(d_t_shape, dtype),                # type: ignore
            out: T.Tensor(out_shape, accum_dtype),          # type: ignore
        ):
            # grid = (b*h*c, l_tile, p_tile) -- CUDA supports at most 3D grids;
            # recover (bz, bh, bc) from the fused first dimension.
            with T.Kernel(
                batch * n_heads * num_chunks,
                T.ceildiv(chunk_len, block_l),
                T.ceildiv(d_head, block_p),
                threads=threads,
            ) as (b_h_c, bl, bp):
                bz = b_h_c // (n_heads * num_chunks)
                bh = (b_h_c % (n_heads * num_chunks)) // num_chunks
                bc = b_h_c % num_chunks

                # ----------------------------
                # output tile accumulator
                # ----------------------------
                acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(acc)

                l0 = bl * block_l
                p0 = bp * block_p

                # ----------------------------
                # load target-side dA_cumsum[l]
                # used in both step4 and step1
                # ----------------------------
                dA_l = T.alloc_fragment((block_l,), accum_dtype)

                for l in T.Parallel(block_l):
                    l_abs = l0 + l
                    dA_l[l] = T.if_then_else(
                        l_abs < chunk_len,
                        dA_cumsum[bz, bh, bc, l_abs],
                        T.float32(0.0),
                    )

                # =========================================================
                # PART 1: history / Step-4 path
                #   acc[l,p] += sum_n C[l,n] * prev_states[n,p] * exp(dA_l[l])
                # =========================================================

                # local accumulator before multiplying target decay
                hist_acc = T.alloc_fragment((block_l, block_p), accum_dtype)
                T.clear(hist_acc)

                c_tile = T.alloc_shared((block_l, block_n), dtype)
                state_tile = T.alloc_shared((block_n, block_p), dtype)

                for n0 in T.serial(T.ceildiv(d_state, block_n)):
                    # load C tile: (l_tile, n_tile)
                    # C: (b, c, l, h, n)
                    T.copy(
                        C[bz, bc, l0:l0 + block_l, bh, n0 * block_n:(n0 + 1) * block_n],
                        c_tile,
                    )

                    # load prev_states tile: (n_tile, p_tile)
                    # prev_states: (b, c, h, n, p)
                    T.copy(
                        prev_states[bz, bc, bh,
                                    n0 * block_n:(n0 + 1) * block_n,
                                    p0:p0 + block_p],
                        state_tile,
                    )

                    # hist_acc += C_tile @ state_tile
                    T.gemm(c_tile, state_tile, hist_acc)

                # multiply by exp(dA_l[l]) row-wise
                for l, p in T.Parallel(block_l, block_p):
                    l_abs = l0 + l
                    p_abs = p0 + p
                    valid = (l_abs < chunk_len) and (p_abs < d_head)
                    hist_scale = T.exp(dA_l[l])
                    acc[l, p] += T.if_then_else(
                        valid,
                        hist_acc[l, p] * hist_scale,
                        T.float32(0.0),
                    )

                # =========================================================
                # PART 2: intra-chunk / Step-1 path
                #   acc[l,p] += sum_s cb[l,s] * exp(dA_l[l] - dA_s[s]) * dt[s] * x[s,p]
                # =========================================================

                cb_tile = T.alloc_shared((block_l, block_s), dtype)
                x_tile = T.alloc_shared((block_s, block_p), dtype)
                dA_s = T.alloc_fragment((block_s,), accum_dtype)
                lcb = T.alloc_fragment((block_l, block_s), accum_dtype)
                # lcb_cast holds lcb downcast to dtype for gemm.
                # TileLang requires A.dtype == B.dtype for T.gemm, and x_tile is dtype
                # (fp16/bf16).  Both Part 1 and Part 2 share the same `acc` fragment;
                # changing the input dtype of the Part-2 gemm (e.g. to float32) would
                # alter the fragment layout inferred for `acc`, conflicting with the
                # layout from Part-1's gemm_ss (fp16/bf16 × fp16/bf16 → fp32).
                # The precision loss here is bounded: cb_tile is already fp16/bf16,
                # so lcb is limited by that quantization anyway.
                # TODO: if TileLang adds mixed-precision gemm or a way to pin fragment
                # layouts, remove this cast and use lcb directly.
                lcb_cast = T.alloc_fragment((block_l, block_s), dtype)
                dt_tile = T.alloc_fragment((block_s,), accum_dtype)

                for s0 in T.serial(T.ceildiv(chunk_len, block_s)):
                    # load cb tile: (l_tile, s_tile)
                    # cb: (b, c, h, l, s)
                    T.copy(
                        cb[bz, bc, bh, l0:l0 + block_l, s0 * block_s:(s0 + 1) * block_s],
                        cb_tile,
                    )

                    # load x tile: (s_tile, p_tile)
                    # x: (b, c, s, h, p)
                    T.copy(
                        x[bz, bc, s0 * block_s:(s0 + 1) * block_s, bh, p0:p0 + block_p],
                        x_tile,
                    )

                    # load source-side dA_cumsum[s] and dt[s]
                    for s in T.Parallel(block_s):
                        s_abs = s0 * block_s + s
                        dt_tile[s] = T.if_then_else(
                            s_abs < chunk_len,
                            d_t[bz, bc, s_abs, bh],
                            T.float32(0.0),
                        )
                        dA_s[s] = T.if_then_else(
                            s_abs < chunk_len,
                            dA_cumsum[bz, bh, bc, s_abs],
                            T.float32(0.0),
                        )

                    # build lcb[l,s] = cb[l,s] * exp(dA_l[l] - dA_s[s]) * dt[s] if s <= l else 0
                    for l, s in T.Parallel(block_l, block_s):
                        l_abs = l0 + l
                        s_abs = s0 * block_s + s
                        valid = (l_abs < chunk_len) and (s_abs < chunk_len) and (s_abs <= l_abs)
                        lcb[l, s] = T.if_then_else(
                            valid,
                            cb_tile[l, s] * T.exp(dA_l[l] - dA_s[s]) * dt_tile[s],
                            T.float32(0.0),
                        )

                    # cast lcb float32 -> dtype so both operands match for gemm
                    T.copy(lcb, lcb_cast)

                    # acc += lcb_cast @ x_tile
                    T.gemm(lcb_cast, x_tile, acc)

                # ----------------------------
                # write back once
                # ----------------------------
                T.copy(
                    acc,
                    out[bz, bc, l0:l0 + block_l, bh, p0:p0 + block_p],
                )

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
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype)(
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
    return x.new_empty((batch, num_chunks, chunk_len, n_heads, d_head), dtype=torch.float32)


class SSDChunkScanFwdKernel(Kernel):
    """Mamba-2 SSD fused chunk output forward kernel.

    Fuses the history (prev_states) contribution and intra-chunk causal decay
    into a single pass, computing:

      out[l, p] = exp(dA_cumsum[l]) * (C[l] @ prev_states)
                + sum_{s <= l} cb[l, s] * exp(dA_cumsum[l] - dA_cumsum[s]) * dt[s] * x[s, p]

    Inputs:  x, cb, dA_cumsum, C, prev_states, dt
    Output:  out  (batch, num_chunks, chunk_len, n_heads, d_head), float32
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
        self.dtype = dtype
        self.kernel = _ssd_chunk_scan_fwd_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, self.dtype_str,
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
        _configs = list(itertools.product(block_l, block_p, block_n, block_s, threads))
        return [{
            "block_l": c[0],
            "block_p": c[1],
            "block_n": c[2],
            "block_s": c[3],
            "threads": c[4],
        } for c in _configs]

    def forward(
        self,
        x: torch.Tensor,
        cb: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        return _ssd_chunk_scan_fwd_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.d_head, self.d_state,
            self.dtype_str, self.config["block_l"], self.config["block_p"], self.config["block_n"],
            self.config["block_s"], self.config["threads"],
            x, cb, dA_cumsum, C, prev_states, dt,
        )
