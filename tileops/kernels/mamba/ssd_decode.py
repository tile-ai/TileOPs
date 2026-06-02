"""
Mamba-2 State-Space Dual (SSD) recurrent decode (step) kernel.

Performs a single decode step of the Mamba-2 State Space Model (SSM) core, updating the
persistent state in-place and computing the output y for the current token.

This corresponds to the step() path in the official Mamba-2 implementation:
  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py

Inputs:
  A:      (n_heads, d_head, d_state)           float32  -- SSM decay parameter (A <= 0)
  dt:     (batch, n_heads, d_head)             float32  -- discretization step (post-softplus)
  x:      (batch, n_heads, d_head)             dtype    -- input features per head
  B_in:   (batch, n_groups, d_state)           dtype    -- SSM B matrix (per group)
  C_in:   (batch, n_groups, d_state)           dtype    -- SSM C matrix (per group)
  state:  (batch, n_heads, d_head, d_state)    float32  -- recurrent state (updated in-place)

Output:
  y_out:  (batch, n_heads, d_head)             float32

Math (for each b, h, p, n):
  g                = h // (n_heads // n_groups)
  dA[b, h, p, n]   = exp(dt[b, h, p] * A[h, p, n])
  dBx[b, h, p, n]  = dt[b, h, p] * B_in[b, g, n] * x[b, h, p]
  state[b,h,p,n]  <- dA[b,h,p,n] * state[b,h,p,n] + dBx[b,h,p,n]   (in-place)
  y_out[b, h, p]   = sum_n  state[b, h, p, n] * C_in[b, g, n]

Notes:
  - dt is assumed post-softplus (positive). A is negative (Mamba-2 decay parameter).
    In Mamba-2, A = -exp(A_log) repeated to (nheads, headdim, d_state), and
    dt = softplus(dt_raw + dt_bias) repeated to (batch, nheads, headdim). Because
    dt > 0 and A < 0, dA = exp(dt * A) is always in (0, 1) (a decaying factor).
  - ngroups divides n_heads; with ngroups=1 all heads share the same B/C.
    With ngroups=n_heads each head has its own B/C.
  - The skip connection (D * x) and output gate (z * silu) are NOT fused here;
    they should be applied by the caller after this kernel.

Notation:
  B = batch, H = n_heads, P = d_head, N = d_state, G = n_groups
"""

import functools
import math
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["SSDDecodeKernel"]

# =============================================================================
# Differences vs. the official Mamba-2 step() in mamba_ssm/modules/mamba2.py
# (https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)
#
# This kernel implements ONLY the SSM state-update + output-accumulation core.
# The surrounding steps that belong to a complete decode pass are NOT included:
#
# 1. in_proj / input splitting
#    Official: zxbcdt = self.in_proj(hidden_states.squeeze(1))
#              z0, x0, z, xBC, dt = torch.split(zxbcdt, [...])
#    Here:     x, B_in, C_in, dt are assumed pre-split and passed in directly.
#
# 2. Causal-conv1d update (produces xBC from conv_state)
#    Official: conv_state is rolled and convolved with self.conv1d.weight, then
#              SiLU-activated: xBC = act(conv(conv_state, weight) + bias)
#              xBC is then split into x, B, C.
#    Here:     x, B_in, C_in are post-conv inputs; conv_state handling is absent.
#
# 3. dt discretization (softplus + dt_bias)
#    Official: dt = F.softplus(dt + self.dt_bias)   — applied inside step()
#    Here:     dt is assumed already post-softplus; caller must apply softplus
#              and add dt_bias before passing dt into this kernel.
#
# 4. A parameter derivation
#    Official: A = -torch.exp(self.A_log.float())   — derived from a log param, then
#              repeated to (nheads, headdim, d_state) before passing to the triton kernel.
#    Here:     A is passed directly as a pre-computed (negative) float32 tensor with
#              shape (n_heads, d_head, d_state), matching the triton path exactly.
#
# 5. D skip connection
#    Official: y = y + rearrange(self.D, "h -> h 1") * x
#    Here:     NOT applied. Caller must add D * x to y_out after this kernel.
#
# 6. Output gate  (two variants in official code)
#    a) Without RMSNorm:  y = y * self.act(z)   (element-wise SiLU gate)
#    b) With RMSNorm:     y = self.norm(y, z)   (RMSNormGated)
#    Here:     Neither variant is applied. Caller must handle gating/norm.
#
# 7. Optional MLP branch (d_mlp > 0)
#    Official: y = torch.cat([F.silu(z0) * x0, y], dim=-1)
#    Here:     Not present; this kernel has no notion of z0/x0.
#
# 8. Output projection
#    Official: out = self.out_proj(y)   — linear projection back to model dim
#    Here:     NOT applied. y_out has shape (batch, n_heads, d_head) = (B, H, P).
#              Official step() returns shape (batch, 1, d_model) after out_proj.
#
# 9. Output shape / layout
#    Official (fallback path): rearrange(y, "b h p -> b (h p)") before out_proj,
#              final output is (batch, 1, d_model).
#    Here:     y_out is (batch, n_heads, d_head) in float32, not yet flattened.
#
# 10. ngroups == 1 restriction in official fallback
#     Official fallback (no selective_state_update): asserts ngroups == 1.
#     Here:     Supports arbitrary ngroups via g = h // HEADS_PER_GROUP indexing,
#               matching the behaviour of the optimised selective_state_update path.
# =============================================================================

@functools.lru_cache(maxsize=32)
def _ssd_decode_kernel(
    batch: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    B = batch
    H = n_heads
    P = d_head
    N = d_state
    G = n_groups
    assert H % G == 0, f"n_heads ({H}) must be divisible by n_groups ({G})"
    HEADS_PER_GROUP = H // G

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_p: int,
        block_n: int,
        threads: int,
    ):
        # block_n must be a power of 2 for the log-depth tree reduction.
        if not (block_n > 0 and (block_n & (block_n - 1)) == 0):
            raise ValueError(
                f"block_n must be a power of 2 for the log-depth tree reduction; got {block_n}"
            )
        _log_n = int(math.log2(block_n))

        @T.prim_func
        def main(
            A: T.Tensor((H, P, N), accum_dtype),                # type: ignore
            dt: T.Tensor((B, H, P), accum_dtype),               # type: ignore
            x: T.Tensor((B, H, P), dtype),                      # type: ignore
            B_in: T.Tensor((B, G, N), dtype),                   # type: ignore
            C_in: T.Tensor((B, G, N), dtype),                   # type: ignore
            state: T.Tensor((B, H, P, N), accum_dtype),         # type: ignore  in-place
            y_out: T.Tensor((B, H, P), accum_dtype),            # type: ignore
        ):
            # ----------------------------------------------------------------
            # Grid: axis-0 fuses (batch, head); axis-1 tiles d_head.
            # threads = block_p * block_n.
            #
            # Thread layout: 2D T.Parallel(block_p, block_n).
            # Thread (pp, nn) owns p=p0+pp and n=n0+nn for each n-chunk.
            #
            # Coalescing: for a fixed pp row (one warp when block_n=32), all
            # block_n threads access consecutive n-positions in A, state, B, C
            # → stride-1 loads, perfectly coalesced within each warp.
            #
            # x and dt only depend on pp (not nn or n_blk) — loaded once into
            # x_frag/dt_frag before the n_blk loop to avoid redundant HBM reads.
            #
            # y reduction: each thread accumulates a partial sum in a register
            # (y_frag), then a log-depth tree reduction in shared memory
            # collapses the block_n partial sums for each pp into y_out[p].
            # ----------------------------------------------------------------
            with T.Kernel(B * H, T.ceildiv(P, block_p), threads=threads) as (bh, bp):
                b = bh // H
                h = bh % H
                g = h // HEADS_PER_GROUP
                p0 = bp * block_p

                # Per-thread y accumulator.  Thread (pp, nn) owns y_frag[pp, nn].
                y_frag = T.alloc_fragment((block_p, block_n), accum_dtype)
                T.clear(y_frag)

                # --------------------------------------------------------
                # Hoist x and dt loads out of the n_blk loop.
                #   x[b,h,p] and dt[b,h,p] depend only on pp, not on n_blk,
                #   so reading them once before the loop eliminates
                #   ceil(N/block_n) redundant global loads per thread.
                #   Allocated as (block_p,) — one scalar per pp — so each
                #   thread stores exactly one value, not block_n redundant
                #   copies.  Loaded by T.Parallel(block_p) only.
                # --------------------------------------------------------
                x_frag  = T.alloc_fragment((block_p,), accum_dtype)
                dt_frag = T.alloc_fragment((block_p,), accum_dtype)
                for pp in T.Parallel(block_p):
                    p_idx = p0 + pp
                    valid_p = p_idx < P
                    x_frag[pp] = T.if_then_else(
                        valid_p,
                        T.cast(x[b, h, p_idx], accum_dtype),
                        T.float32(0.0),
                    )
                    dt_frag[pp] = T.if_then_else(
                        valid_p,
                        dt[b, h, p_idx],
                        T.float32(0.0),
                    )

                # --------------------------------------------------------
                # Main loop: 2D parallel state update.
                #   T.Parallel(block_p, block_n) maps (pp, nn) → thread index.
                #   Adjacent threads in the same pp-row access consecutive n
                #   positions → coalesced loads for A, state, B_in, C_in.
                # --------------------------------------------------------
                for n_blk in T.serial(T.ceildiv(N, block_n)):
                    n0 = n_blk * block_n
                    for pp, nn in T.Parallel(block_p, block_n):
                        p_idx = p0 + pp
                        n_idx = n0 + nn
                        valid = (p_idx < P) and (n_idx < N)

                        A_val = T.if_then_else(
                            valid,
                            A[h, p_idx, n_idx],
                            T.float32(0.0),
                        )
                        B_val = T.if_then_else(
                            valid,
                            T.cast(B_in[b, g, n_idx], accum_dtype),
                            T.float32(0.0),
                        )
                        C_val = T.if_then_else(
                            valid,
                            T.cast(C_in[b, g, n_idx], accum_dtype),
                            T.float32(0.0),
                        )
                        old_s = T.if_then_else(
                            valid,
                            state[b, h, p_idx, n_idx],
                            T.float32(0.0),
                        )

                        dA_val = T.exp(dt_frag[pp] * A_val)
                        new_s = dA_val * old_s + dt_frag[pp] * x_frag[pp] * B_val
                        if valid:
                            state[b, h, p_idx, n_idx] = new_s

                        y_frag[pp, nn] = y_frag[pp, nn] + new_s * C_val

                # --------------------------------------------------------
                # y reduction: store fragments to shared memory, then
                # perform a log-depth tree reduction over the nn dimension.
                # Round r: threads with nn < block_n>>(r+1) add the value
                # at nn + block_n>>(r+1) into their cell.  Unrolled at
                # Python trace-time; each _stride is a compile-time int.
                # --------------------------------------------------------
                y_smem = T.alloc_shared((block_p, block_n), accum_dtype)
                for pp, nn in T.Parallel(block_p, block_n):
                    y_smem[pp, nn] = y_frag[pp, nn]

                T.sync_threads()

                for _d in range(_log_n):
                    _stride = block_n >> (_d + 1)
                    for pp, nn in T.Parallel(block_p, block_n):
                        if nn < _stride:
                            y_smem[pp, nn] = y_smem[pp, nn] + y_smem[pp, nn + _stride]
                    T.sync_threads()

                # --------------------------------------------------------
                # Write y_out from the reduced y_smem[:, 0].
                # --------------------------------------------------------
                for pp, nn in T.Parallel(block_p, block_n):
                    if nn == 0:
                        p_idx = p0 + pp
                        if p_idx < P:
                            y_out[b, h, p_idx] = y_smem[pp, 0]

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_decode", mutates_args=("state",))
def _ssd_decode_wrapped(
    batch: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_p: int,
    block_n: int,
    threads: int,
    A: torch.Tensor,
    dt: torch.Tensor,
    x: torch.Tensor,
    B_in: torch.Tensor,
    C_in: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    return _ssd_decode_kernel(batch, n_heads, d_head, d_state, n_groups, dtype)(
        block_p, block_n, threads,
    )(A, dt, x, B_in, C_in, state)


@_ssd_decode_wrapped.register_fake
def _(
    batch: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_p: int,
    block_n: int,
    threads: int,
    A: torch.Tensor,
    dt: torch.Tensor,
    x: torch.Tensor,
    B_in: torch.Tensor,
    C_in: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    return dt.new_empty((batch, n_heads, d_head), dtype=torch.float32)


class SSDDecodeKernel(Kernel):
    """Mamba-2 SSD recurrent decode (step) kernel.

    Performs a single decode step: updates the SSM state in-place and
    returns the output for the current token:

      g                = h // (n_heads // n_groups)
      dA[b, h, p, n]   = exp(dt[b, h, p] * A[h, p, n])
      state[b,h,p,n]  <- dA[b,h,p,n] * state[b,h,p,n]
                         + dt[b,h,p] * B_in[b,g,n] * x[b,h,p]
      y_out[b, h, p]   = sum_n  state[b, h, p, n] * C_in[b, g, n]

    Inputs:  A (n_heads,d_head,d_state), dt (batch,n_heads,d_head),
             x, B_in, C_in, state (mutated in-place)
    Output:  y_out  (batch, n_heads, d_head), float32

    Matches the interface of the official Mamba-2 selective_state_update
    triton kernel (selective_state_update.py in mamba_ssm).
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int = 1,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.kernel = _ssd_decode_kernel(
            batch, n_heads, d_head, d_state, n_groups, self.dtype_str,
        )
        self.init_config(config, tune)
        cfg = self.config
        bp, bn, threads = cfg.get("block_p"), cfg.get("block_n"), cfg.get("threads")
        _bn_is_pow2 = bn is not None and bn > 0 and (bn & (bn - 1)) == 0
        if not _bn_is_pow2:
            raise ValueError(
                f"SSDDecodeKernel requires block_n to be a power of 2 for the "
                f"log-depth tree reduction, but got block_n={bn}."
            )
        if bp is not None and bn is not None and threads != bp * bn:
            raise ValueError(
                f"SSDDecodeKernel requires threads == block_p * block_n "
                f"({bp} * {bn} = {bp * bn}) for the 2D T.Parallel layout, "
                f"but got threads={threads}."
            )

    @property
    def default_config(self) -> dict:
        # threads = block_p * block_n (2D layout).
        # block_n=32 gives one warp per pp-row → perfect coalescing.
        return {
            "block_p": 4,
            "block_n": 32,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # threads = block_p * block_n.  block_n must be a power of 2.
        return [
            {"block_p": bp, "block_n": bn, "threads": bp * bn}
            for bn in [32, 64, 128]
            for bp in [1, 2, 4]
        ]

    def forward(
        self,
        A: torch.Tensor,
        dt: torch.Tensor,
        x: torch.Tensor,
        B_in: torch.Tensor,
        C_in: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return _ssd_decode_wrapped(
            self.batch, self.n_heads, self.d_head, self.d_state, self.n_groups,
            self.dtype_str,
            self.config["block_p"], self.config["block_n"], self.config["threads"],
            A, dt, x, B_in, C_in, state,
        )
