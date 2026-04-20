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

from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

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
            # d_state (N) is swept serially inside the block.
            #
            # Thread layout: T.Parallel(block_p) is the only parallel axis.
            # Each thread "owns" one p position (pp) and handles ALL n positions
            # for that p serially. This avoids any cross-thread shared-memory
            # access and follows the same pattern as ssd_state_passing_fwd.
            # ----------------------------------------------------------------
            with T.Kernel(B * H, T.ceildiv(P, block_p), threads=threads) as (bh, bp):
                b = bh // H
                h = bh % H
                g = h // HEADS_PER_GROUP
                p0 = bp * block_p

                # --------------------------------------------------------
                # 1. Load x tile into a fragment.
                #    Each thread owns x_tile[pp] for its pp index and never
                #    needs to share it with other threads.
                # --------------------------------------------------------
                x_tile = T.alloc_fragment((block_p,), accum_dtype)
                for pp in T.Parallel(block_p):
                    p_idx = p0 + pp
                    x_tile[pp] = T.if_then_else(
                        p_idx < P,
                        T.cast(x[b, h, p_idx], accum_dtype),
                        T.float32(0.0),
                    )

                # --------------------------------------------------------
                # 2. y accumulator — explicitly zeroed via T.Parallel.
                # --------------------------------------------------------
                y_acc = T.alloc_fragment((block_p,), accum_dtype)
                for pp in T.Parallel(block_p):
                    y_acc[pp] = T.float32(0.0)

                # --------------------------------------------------------
                # 3. Sweep d_state (N) in tiles of block_n.
                #    Each thread processes its own pp for all nn serially —
                #    no shared memory required, no cross-thread sync needed.
                #
                #    dt[b, h, p_idx] and A[h, p_idx, n_idx] are read per
                #    (pp, nn), matching the official selective_state_update
                #    triton kernel's per-element access pattern.
                # --------------------------------------------------------
                for n_blk in T.serial(T.ceildiv(N, block_n)):
                    n0 = n_blk * block_n

                    for pp in T.Parallel(block_p):
                        p_idx = p0 + pp
                        dt_val = T.if_then_else(
                            p_idx < P,
                            dt[b, h, p_idx],
                            T.float32(0.0),
                        )
                        for nn in T.serial(block_n):
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

                            # State update: new_s = exp(dt*A) * old_s + dt * x[p] * B[n]
                            dA_val = T.exp(dt_val * A_val)
                            new_s = dA_val * old_s + dt_val * x_tile[pp] * B_val
                            if valid:
                                state[b, h, p_idx, n_idx] = new_s

                            # Accumulate y: y[p] += new_s * C[n]
                            y_acc[pp] += new_s * C_val

                # --------------------------------------------------------
                # 4. Write y_out.
                # --------------------------------------------------------
                for pp in T.Parallel(block_p):
                    p_idx = p0 + pp
                    if p_idx < P:
                        y_out[b, h, p_idx] = y_acc[pp]

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

    @property
    def default_config(self) -> dict:
        return {
            "block_p": 64,
            "block_n": 32,
            "threads": 64,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_p": bp, "block_n": bn, "threads": bp}
            for bp in [32, 64]
            for bn in [16, 32]
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
