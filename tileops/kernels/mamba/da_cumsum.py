"""
Mamba-2 dA_cumsum forward kernel.

Inputs:
  dt:       (batch, seq_len, n_heads)                  -- raw per-position dt (float32)
  A:        (n_heads,)                                  -- State Space Model (SSM) decay parameter (float32)
  dt_bias:  (n_heads,)                                  -- optional per-head dt bias (float32)

Outputs:
  dt_out:    (batch, n_heads, num_chunks, chunk_len)   -- float32, processed dt after bias/softplus/clamp
  dA_cumsum: (batch, n_heads, num_chunks, chunk_len)   -- float32, inclusive prefix sum of dA = dt_out * A

For each (b, h, c, l), the kernel computes:

  dt_val            = dt[b, c*Q + l, h]
  if has_dt_bias:   dt_val += dt_bias[h]
  if dt_softplus:   dt_val = softplus(dt_val)   # with bypass for dt_val > 20
                    dt_val = clamp(dt_val, dt_min, dt_max)
  dt_out[b,h,c,l]  = dt_val
  dA_cumsum[b,h,c,l] = sum_{i=0}^{l} dt_out[b,h,c,i] * A[h]

This matches _chunk_cumsum_fwd_kernel in the Mamba-2 Triton reference
(mamba_ssm/ops/triton/ssd_chunk_state.py).

Alignment with Mamba-2 paper:
  In ssd_minimal_discrete, A already absorbs dt (A = dt * A_log), so A_cumsum = cumsum(A).
  Here dt and A are kept separate; dA = dt * A achieves the same result.
  Since A <= 0 in Mamba-2, dA_cumsum is monotonically non-increasing within each chunk,
  and exp(dA_cumsum[l] - dA_cumsum[s]) is a decaying factor in (0, 1] for s <= l.

Notation:
  B = batch, S = seq_len = C * Q, H = n_heads, C = num_chunks, Q = chunk_len
"""

import functools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["DaCumsumFwdKernel"]


@functools.lru_cache(maxsize=32)
def _da_cumsum_fwd_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    dt_softplus: bool = False,
    has_dt_bias: bool = False,
    dt_min: float = 0.0,
    dt_max: float = float("inf"),
) -> Callable:
    """Build a TileLang parallel dA_cumsum kernel.

    Grid layout: (batch, num_chunks, ceil(n_heads / block_h)).
    Each block loads a (block_h, chunk_len) tile of dt values, applies
    bias/softplus/clamp per element in parallel, multiplies by A to get dA,
    then calls T.cumsum along the chunk dimension.  This eliminates the serial
    Q-step scan of the previous kernel and matches mamba_ssm's tl.cumsum approach.

    Args:
        block_h: Number of heads processed per CUDA block (passed to kernel_func).
                 Must be a power of 2 and satisfy block_h * chunk_len <= 1024.
    """
    accum_dtype = "float"

    B = batch
    C = num_chunks
    Q = chunk_len
    H = n_heads
    S = seq_len

    @tilelang.jit(out_idx=[-2, -1])
    def kernel_func(block_h: int, threads: int):
        @T.prim_func
        def main(
            dt:        T.Tensor((B, S, H), accum_dtype),          # type: ignore  # raw dt input
            A:         T.Tensor((H,),       accum_dtype),           # type: ignore
            dt_bias:   T.Tensor((H,),       accum_dtype),           # type: ignore
            dt_out:    T.Tensor((B, H, C, Q), accum_dtype),        # type: ignore
            dA_cumsum: T.Tensor((B, H, C, Q), accum_dtype),        # type: ignore
        ):
            # Grid: one block per (batch, chunk, head-tile).
            # block_h heads are processed together in parallel within each block.
            with T.Kernel(B, C, T.ceildiv(H, block_h), threads=threads) as (bb, bc, bh_tile):
                # Shared tiles for (block_h, Q) dt and dA values.
                # Two tiles: one for dt_out, one for dA (will be overwritten by cumsum).
                dt_shared = T.alloc_shared((block_h, Q), accum_dtype)
                dA_shared = T.alloc_shared((block_h, Q), accum_dtype)

                # ── Step 1: load raw dt, apply transforms, compute dA ─────────
                # All (block_h × Q) elements are processed in parallel.
                for i, j in T.Parallel(block_h, Q):
                    bh      = bh_tile * block_h + i
                    seq_idx = bc * Q + j
                    in_b    = T.And(bh < H, seq_idx < S)

                    # Clamp indices to valid range before memory reads.
                    # T.if_then_else lowers to a select instruction (not a branch),
                    # so both arms are evaluated — out-of-bounds indices must be
                    # clamped to prevent illegal memory access on padding threads.
                    safe_bh      = T.min(bh, H - 1)
                    safe_seq_idx = T.min(seq_idx, S - 1)

                    # Load dt; zero-pad out-of-bounds positions.
                    val = T.if_then_else(in_b, dt[bb, safe_seq_idx, safe_bh], T.float32(0.0))

                    # Optional bias addition.
                    if has_dt_bias:
                        bias = T.if_then_else(bh < H, dt_bias[safe_bh], T.float32(0.0))
                        val  = val + bias

                    # Optional softplus (log(1+exp(x))) with large-value bypass.
                    if dt_softplus:
                        val = T.if_then_else(
                            val <= T.float32(20.0),
                            T.log(T.float32(1.0) + T.exp(val)),
                            val,
                        )

                    # Clamp to [dt_min, dt_max].
                    val = T.min(T.max(val, T.float32(dt_min)), T.float32(dt_max))

                    # Re-apply out-of-bounds zero mask after nonlinearities.
                    val = T.if_then_else(in_b, val, T.float32(0.0))

                    # Compute A[h] * dt_val for the cumsum input.
                    a_val = T.if_then_else(bh < H, A[safe_bh], T.float32(0.0))

                    dt_shared[i, j] = val
                    dA_shared[i, j] = val * a_val

                T.sync_threads()
                # ── Step 2: parallel prefix sum along Q dimension ────────────
                # T.cumsum operates in-place on the shared tile, replacing each
                # element with the inclusive prefix sum up to that position.
                T.cumsum(dA_shared, dim=1)
                T.sync_threads()

                # ── Step 3: write outputs ─────────────────────────────────────
                for i, j in T.Parallel(block_h, Q):
                    bh = bh_tile * block_h + i
                    with T.If(bh < H), T.Then():
                        dt_out[bb, bh, bc, j]    = dt_shared[i, j]
                        dA_cumsum[bb, bh, bc, j] = dA_shared[i, j]

        return main

    return kernel_func


@torch.library.custom_op("top::da_cumsum_fwd", mutates_args=())
def _da_cumsum_fwd_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    threads: int,
    dt_softplus: bool,
    has_dt_bias: bool,
    dt_min: float,
    dt_max: float,
    block_h: int,
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _da_cumsum_fwd_kernel(
        batch, num_chunks, chunk_len, n_heads, seq_len,
        dt_softplus, has_dt_bias, dt_min, dt_max,
    )(block_h, threads)(dt, A, dt_bias)


@_da_cumsum_fwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    threads: int,
    dt_softplus: bool,
    has_dt_bias: bool,
    dt_min: float,
    dt_max: float,
    block_h: int,
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt_out    = dt.new_empty((batch, n_heads, num_chunks, chunk_len), dtype=torch.float32)
    dA_cumsum = dt.new_empty((batch, n_heads, num_chunks, chunk_len), dtype=torch.float32)
    return dt_out, dA_cumsum


class DaCumsumFwdKernel(Kernel):
    """Mamba-2 dA_cumsum forward kernel.

    Applies optional per-head bias, optional softplus activation, and clamping to
    raw dt values, then computes the chunk-local inclusive prefix sum of dA = dt * A.

    Uses a parallel tile approach: each CUDA block processes block_h heads × chunk_len
    positions simultaneously, with T.cumsum for the prefix scan — matching the
    parallelism of mamba_ssm's tl.cumsum Triton kernel.

    Inputs:
        dt      (batch, seq_len, n_heads) float32 — raw dt values.
        A       (n_heads,) float32 — State Space Model (SSM) decay parameters.
        dt_bias (n_heads,) float32 — per-head dt bias; required when has_dt_bias=True.

    Outputs:
        dt_out    (batch, n_heads, num_chunks, chunk_len) float32 — processed dt.
        dA_cumsum (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        seq_len: int,
        dt_softplus: bool = False,
        has_dt_bias: bool = False,
        dt_min: float = 0.0,
        dt_max: float = float("inf"),
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.dt_softplus = dt_softplus
        self.has_dt_bias = has_dt_bias
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dtype = torch.float32
        self.kernel = _da_cumsum_fwd_kernel(
            batch, num_chunks, chunk_len, n_heads, seq_len,
            dt_softplus, has_dt_bias, dt_min, dt_max,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # block_h=4 processes 4 heads per block with threads=min(4*chunk_len, 1024).
        # For chunk_len=256 this gives threads=1024, matching the warp occupancy of
        # mamba_ssm's _chunk_cumsum_fwd_kernel (BLOCK_SIZE_H × BLOCK_SIZE_CHUNK).
        # block_h must evenly divide n_heads; if not, padding rows are guard-checked.
        return {"block_h": 4, "threads": min(4 * self.chunk_len, 1024)}

    @property
    def autotune_configs(self) -> list[dict]:
        # Sweep block_h ∈ {1, 2, 4, 8, 16} subject to:
        #   - block_h * chunk_len <= 1024  (threads budget)
        #   - block_h <= n_heads           (no more tile rows than heads)
        valid = []
        for bh in [1, 2, 4, 8, 16]:
            if bh > self.n_heads:
                break
            if bh * self.chunk_len > 1024:
                break
            threads = bh * self.chunk_len
            valid.append({"block_h": bh, "threads": threads})
        return valid

    def forward(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the dA_cumsum forward pass.

        Args:
            dt: (batch, seq_len, n_heads) float32 — raw dt values.
            A:  (n_heads,) float32 — SSM decay parameters.
            dt_bias: (n_heads,) float32, optional — per-head dt bias.
                Required when the kernel was constructed with has_dt_bias=True.

        Returns:
            dt_out: (batch, n_heads, num_chunks, chunk_len) float32 — processed dt.
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum.
        """
        dt = dt.contiguous()
        A  = A.contiguous()
        if self.has_dt_bias and dt_bias is None:
            raise ValueError("dt_bias is required when has_dt_bias=True")
        # Dummy zero bias keeps the kernel signature stable when has_dt_bias=False.
        dt_bias = dt.new_zeros(self.n_heads) if dt_bias is None else dt_bias.contiguous()

        return _da_cumsum_fwd_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.seq_len,
            self.config["threads"],
            self.dt_softplus, self.has_dt_bias, self.dt_min, self.dt_max,
            self.config["block_h"],
            dt, A, dt_bias,
        )
