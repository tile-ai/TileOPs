"""Mamba-2 end-to-end SSD forward operator.

Chains the four sub-ops in order:
  1. DaCumsumFwdOp       — dt preprocessing + dA cumulative sum
  2. SSDChunkStateFwdOp  — per-chunk SSM state computation
  3. SSDStatePassingFwdOp — inter-chunk recurrent state scan
  4. SSDChunkScanFwdOp   — final output scan

The interface mirrors mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined.

Design notes
------------
* SSDChunkStateFwdOp output is float32 with shape (B, C, H, P, N).
  SSDStatePassingFwdOp expects (B, C, H, d_state) in its construction dtype.
  We reshape chunk_states to (B, C, H, P*N) and build the state-passing op
  with d_state = d_head * d_state_ssm (float32) so the TileOPs kernel is used
  instead of a Python for-loop.

* CB (causal C@B coupling matrix per group) has shape (B, C, G, Q, Q).
  It is the intra-chunk outer product C[l] @ B[s] (group-owned).
  SSDChunkScanFwdKernel then multiplies cb by exp(dA[l] - dA[s]) * dt[s]
  internally, so cb must contain only the C@B term — not the decay.
  We compute cb via a batched matmul: C_chunked @ B_chunked^T, masked causal.

* All intermediate tensors remain on-device; no host syncs between sub-ops.
"""

import functools
from typing import Optional, Tuple

import torch

from .da_cumsum import DaCumsumFwdOp
from .ssd_chunk_scan import SSDChunkScanFwdOp
from .ssd_chunk_state import SSDChunkStateFwdOp
from .ssd_state_passing import SSDStatePassingFwdOp


@functools.lru_cache(maxsize=32)
def _get_cb_fn(dtype: torch.dtype):
    """Return a torch.compile-fused CB-matrix builder.

    Computes the causal C@B coupling matrix for one group:
      cb[b, c, g, l, s] = C[b, c*Q+l, g, :] @ B[b, c*Q+s, g, :]^T,  for s <= l
                         = 0                                           for s > l

    This is a lower-triangular batched outer product in state-space dimension N,
    matching _bmm_chunk_fwd from mamba_ssm. The kernel then multiplies cb by
    exp(dA[l] - dA[s]) * dt[s] — so cb must not include any decay factor.
    """
    def _build_cb(C_g: torch.Tensor, B_g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # C_g, B_g: (B, C, G, Q, N)  in storage dtype
        # mask:     (1, 1, 1, Q, Q)  lower-triangular bool
        cb = torch.einsum("bcgln,bcgsn->bcgls", C_g.float(), B_g.float())  # (B, C, G, Q, Q)
        return torch.where(mask, cb, 0.0).to(dtype)

    return torch.compile(_build_cb, fullgraph=True)


__all__ = ["Mamba2FwdOp"]


class Mamba2FwdOp:
    """Mamba-2 State-Space Dual (SSD) full forward pass operator.

    Combines DaCumsum → SSDChunkState → SSDStatePassing → SSDChunkScan into
    a single callable whose interface matches mamba_chunk_scan_combined from
    the official mamba_ssm library.

    Args:
        batch:              Batch size.
        seqlen:             Total sequence length (must be divisible by chunk_size).
        n_heads:            Number of State Space Model (SSM) heads.
        d_head:             Head dimension.
        d_state:            SSM state dimension.
        n_groups:           Number of B/C groups (n_heads must be divisible by n_groups).
        dtype:              Data type for x, B, C inputs (float16 or bfloat16).
        chunk_size:         Tokens per chunk (default 256).
        dt_softplus:        Apply softplus to (dt + dt_bias) before use.
        has_initial_states: Whether initial_states tensor will be provided at forward time.
        tune:               Whether to autotune tile configs on construction.
    """

    def __init__(
        self,
        batch: int,
        seqlen: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        chunk_size: int = 256,
        dt_softplus: bool = True,
        has_initial_states: bool = False,
        tune: bool = False,
    ):
        if seqlen % chunk_size != 0:
            raise ValueError(f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})")
        if n_heads % n_groups != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_groups ({n_groups})")

        self.batch = batch
        self.seqlen = seqlen
        self.chunk_size = chunk_size
        self.num_chunks = seqlen // chunk_size
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.dt_softplus = dt_softplus
        self.has_initial_states = has_initial_states
        self._heads_per_group = n_heads // n_groups

        num_chunks = self.num_chunks

        self._da_cumsum_op = DaCumsumFwdOp(
            batch=batch,
            num_chunks=num_chunks,
            chunk_len=chunk_size,
            n_heads=n_heads,
            seq_len=seqlen,
            dt_softplus=dt_softplus,
            has_dt_bias=True,   # always pass dt_bias; caller passes zeros if unused
            tune=tune,
        )

        self._chunk_state_op = SSDChunkStateFwdOp(
            batch=batch,
            num_chunks=num_chunks,
            chunk_len=chunk_size,
            n_heads=n_heads,
            d_head=d_head,
            d_state=d_state,
            n_groups=n_groups,
            dtype=dtype,
            has_seq_idx=False,
            tune=tune,
        )

        # chunk_states output is float32 (B, C, H, P, N).
        # Flatten P*N into a single state dim so SSDStatePassingFwdOp is used
        # instead of a Python loop, keeping everything on the GPU.
        self._state_passing_op = SSDStatePassingFwdOp(
            batch=batch,
            num_chunks=num_chunks,
            n_heads=n_heads,
            d_state=d_head * d_state,   # flat state: P*N
            dtype=torch.float32,
            has_initial_states=has_initial_states,
            tune=tune,
        )

        self._chunk_scan_op = SSDChunkScanFwdOp(
            batch=batch,
            num_chunks=num_chunks,
            chunk_len=chunk_size,
            n_heads=n_heads,
            d_head=d_head,
            d_state=d_state,
            n_groups=n_groups,
            dtype=dtype,
            tune=tune,
        )

        # Causal mask for CB construction — lower-triangular (Q, Q), broadcast
        # over (B, C, G, Q, Q).  Allocated once; moved to GPU on first forward.
        mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool).tril()
        self._cb_mask_cpu = mask
        self._cb_fn = _get_cb_fn(dtype)  # compiled fused C@B kernel

        # Pre-allocated zero tensors — avoids FillFunctor kernel launches on the
        # hot path for optional inputs that are commonly omitted.
        # Allocated on CPU here; moved to the right CUDA device on first forward.
        self._zero_dt_bias_cpu = torch.zeros(n_heads, dtype=torch.float32)
        self._zero_init_flat_cpu = torch.zeros(
            batch, n_heads, d_head * d_state, dtype=torch.float32,
        )
        # seq_idx placeholder for SSDChunkStateFwdOp (has_seq_idx=False path still
        # passes the tensor; pre-allocating avoids a FillFunctor<int> each call).
        self._zero_seq_idx_cpu = torch.zeros(batch, seqlen, dtype=torch.int32)

        self._zero_dt_bias: Optional[torch.Tensor]   = None
        self._zero_init_flat: Optional[torch.Tensor] = None
        self._zero_seq_idx: Optional[torch.Tensor]   = None
        self._cb_mask: Optional[torch.Tensor]        = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
        return_final_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the full Mamba-2 SSD forward pass.

        Args:
            x:               (batch, seqlen, n_heads, d_head)          dtype
            dt:              (batch, seqlen, n_heads)                   float32
            A:               (n_heads,)                                 float32  (log-space, ≤ 0)
            B:               (batch, seqlen, n_groups, d_state)         dtype
            C:               (batch, seqlen, n_groups, d_state)         dtype
            dt_bias:         (n_heads,) float32, optional
            initial_states:  (batch, n_heads, d_head, d_state) float32, optional
            return_final_states: Whether to include final chunk states in output.

        Returns:
            y:            (batch, seqlen, n_heads, d_head)   float32
            final_states: (batch, n_heads, d_head, d_state)  float32, or None
        """
        if initial_states is not None and not self.has_initial_states:
            raise ValueError(
                "initial_states was provided but this op was constructed with "
                "has_initial_states=False — the kernel ignores it, which would "
                "silently produce wrong results.  Reconstruct with has_initial_states=True."
            )
        batch, seqlen, n_heads, d_head = x.shape
        chunk_size   = self.chunk_size
        num_chunks   = seqlen // chunk_size
        d_state      = self.d_state
        dev          = x.device

        # Ensure pre-allocated tensors live on the right device.
        if self._zero_dt_bias is None or self._zero_dt_bias.device != dev:
            self._zero_dt_bias    = self._zero_dt_bias_cpu.to(dev)
            self._zero_init_flat  = self._zero_init_flat_cpu.to(dev)
            self._zero_seq_idx    = self._zero_seq_idx_cpu.to(dev)
            self._cb_mask         = self._cb_mask_cpu.to(dev).view(1, 1, 1, chunk_size, chunk_size)
        # ── 1. DaCumsum ──────────────────────────────────────────────────────
        if dt_bias is None:
            dt_bias = self._zero_dt_bias

        dt_out, dA_cumsum = self._da_cumsum_op.forward(dt, A, dt_bias)
        # dt_out:    (B, H, C, Q)  float32
        # dA_cumsum: (B, H, C, Q)  float32

        # ── 2. CB matrix ─────────────────────────────────────────────────────
        # cb[b,c,g,l,s] = C[b,c*Q+l,g,:] @ B[b,c*Q+s,g,:]^T  for s <= l, else 0.
        # Reshape seqlen-fused B/C into (B, C, G, Q, N) for the batched matmul.
        n_groups  = self.n_groups
        C_chunked = C.reshape(batch, num_chunks, chunk_size, n_groups, d_state).permute(0, 1, 3, 2, 4)  # (B, C, G, Q, N)
        B_chunked = B.reshape(batch, num_chunks, chunk_size, n_groups, d_state).permute(0, 1, 3, 2, 4)  # (B, C, G, Q, N)
        cb = self._cb_fn(C_chunked, B_chunked, self._cb_mask)  # (B, C, G, Q, Q)  dtype

        # ── 3. SSDChunkState ─────────────────────────────────────────────────
        # Pass pre-allocated seq_idx zeros; avoids a FillFunctor<int> per call.
        # SSDChunkStateFwdOp is constructed with has_seq_idx=False, so the kernel
        # ignores this tensor entirely — no shape check or data access occurs.
        chunk_states = self._chunk_state_op.forward(x, B, dt_out, dA_cumsum, self._zero_seq_idx)
        # chunk_states: (B, C, H, P, N)  float32

        # ── 4. SSDStatePassing ───────────────────────────────────────────────
        chunk_states_flat = chunk_states.reshape(batch, num_chunks, n_heads, d_head * d_state)
        # Standard slice + clone: contiguous tensor, portable across all PyTorch versions.
        dA_chunk_cumsum = dA_cumsum[..., chunk_size - 1].clone()  # (B, H, C)

        if initial_states is None:
            init_flat = self._zero_init_flat
        else:
            init_flat = initial_states.reshape(batch, n_heads, d_head * d_state).float()

        prev_states_flat, final_states_flat = self._state_passing_op.forward(
            chunk_states_flat, dA_chunk_cumsum, init_flat,
        )

        # Unflatten to (B, C, H, P, N) in float32 (accum_dtype) for chunk_scan.
        prev_states  = prev_states_flat.reshape(batch, num_chunks, n_heads, d_head, d_state)
        dt_out_typed = dt_out.to(self.dtype)

        # ── 5. SSDChunkScan ──────────────────────────────────────────────────
        y = self._chunk_scan_op.forward(x, cb, dA_cumsum, C, prev_states, dt_out_typed)
        # y: (B, S, H, P)  float32

        if return_final_states:
            return y, final_states_flat.reshape(batch, n_heads, d_head, d_state)
        return y, None
