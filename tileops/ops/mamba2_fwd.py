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

from typing import Optional, Tuple

import torch

from .cb_producer import CBProducerOp
from .da_cumsum import DaCumsumFwdOp
from .ssd_chunk_scan import SSDChunkScanFwdOp
from .ssd_chunk_state import SSDChunkStateFwdOp
from .ssd_state_passing import SSDStatePassingFwdOp

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
        chunk_size: int = 256,
        dt_softplus: bool = True,
        has_initial_states: bool = False,
        tune: bool = False,
    ):
        self.batch = None
        self.seqlen = None
        self.chunk_size = chunk_size
        self.num_chunks = None
        self.n_heads = None
        self.d_head = None
        self.d_state = None
        self.n_groups = None
        self.dtype = None
        self.dt_softplus = dt_softplus
        self.has_initial_states = has_initial_states
        self._heads_per_group = None
        self.tune = tune

        self._da_cumsum_ops: dict[torch.dtype, DaCumsumFwdOp] = {}
        self._chunk_state_op = SSDChunkStateFwdOp(
            has_seq_idx=False,
            tune=tune,
        )

        # chunk_states output is float32 (B, C, H, P, N).
        # Flatten P*N into a single state dim so SSDStatePassingFwdOp is used
        # instead of a Python loop, keeping everything on the GPU.
        self._state_passing_op = SSDStatePassingFwdOp(
            has_initial_states=has_initial_states,
            tune=tune,
        )

        self._chunk_scan_op = SSDChunkScanFwdOp(tune=tune)
        self._cb_producer_ops: dict[tuple, CBProducerOp] = {}

        # Pre-allocated zero tensors — avoids FillFunctor kernel launches on the
        # hot path for optional inputs that are commonly omitted.
        self._zero_dt_bias: Optional[torch.Tensor]   = None
        self._zero_init_flat: Optional[torch.Tensor] = None
        self._zero_seq_idx: Optional[torch.Tensor]   = None

    def _get_da_cumsum_op(self, dtype: torch.dtype) -> DaCumsumFwdOp:
        if dtype not in self._da_cumsum_ops:
            self._da_cumsum_ops[dtype] = DaCumsumFwdOp(
                chunk_len=self.chunk_size,
                dtype=dtype,
                dt_softplus=self.dt_softplus,
                has_dt_bias=True,
                tune=self.tune,
            )
        return self._da_cumsum_ops[dtype]

    def _get_cb_producer_op(
        self,
        batch: int,
        num_chunks: int,
        n_groups: int,
        d_state: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> CBProducerOp:
        key = (batch, num_chunks, n_groups, self.chunk_size, d_state, dtype, device_index, self.tune)
        if key not in self._cb_producer_ops:
            self._cb_producer_ops[key] = CBProducerOp(
                batch=batch,
                num_chunks=num_chunks,
                n_groups=n_groups,
                chunk_len=self.chunk_size,
                d_state=d_state,
                dtype=dtype,
                tune=self.tune,
            )
        return self._cb_producer_ops[key]

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
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, seqlen, n_heads, d_head]")
        batch, seqlen, n_heads, d_head = x.shape
        chunk_size = self.chunk_size
        if seqlen % chunk_size != 0:
            raise ValueError(f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})")
        num_chunks = seqlen // chunk_size
        if B.ndim != 4 or B.shape[0] != batch or B.shape[1] != seqlen:
            raise ValueError("B must have shape [batch, seqlen, n_groups, d_state]")
        n_groups, d_state = B.shape[2], B.shape[3]
        if n_heads % n_groups != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_groups ({n_groups})")
        if C.shape != (batch, seqlen, n_groups, d_state):
            raise ValueError("C must have shape [batch, seqlen, n_groups, d_state]")
        if dt.shape != (batch, seqlen, n_heads):
            raise ValueError("dt must have shape [batch, seqlen, n_heads]")
        if A.shape != (n_heads,):
            raise ValueError("A must have shape [n_heads]")
        if dt_bias is not None and dt_bias.shape != (n_heads,):
            raise ValueError("dt_bias must have shape [n_heads]")
        if initial_states is not None and initial_states.shape != (batch, n_heads, d_head, d_state):
            raise ValueError("initial_states must have shape [batch, n_heads, d_head, d_state]")
        if B.dtype != x.dtype:
            raise ValueError(f"B.dtype must be {x.dtype}, got {B.dtype}")
        if C.dtype != x.dtype:
            raise ValueError(f"C.dtype must be {x.dtype}, got {C.dtype}")
        dev = x.device

        self.batch = batch
        self.seqlen = seqlen
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = x.dtype
        self._heads_per_group = n_heads // n_groups

        if self._zero_dt_bias is None or self._zero_dt_bias.shape != (n_heads,) or self._zero_dt_bias.device != dev:
            self._zero_dt_bias = torch.zeros(n_heads, dtype=torch.float32, device=dev)
        init_shape = (batch, n_heads, d_head * d_state)
        if (
            self._zero_init_flat is None
            or self._zero_init_flat.shape != init_shape
            or self._zero_init_flat.device != dev
        ):
            self._zero_init_flat = torch.zeros(init_shape, dtype=torch.float32, device=dev)
        seq_idx_shape = (batch, seqlen)
        if (
            self._zero_seq_idx is None
            or self._zero_seq_idx.shape != seq_idx_shape
            or self._zero_seq_idx.device != dev
        ):
            self._zero_seq_idx = torch.zeros(seq_idx_shape, dtype=torch.int32, device=dev)
        # ── 1. DaCumsum ──────────────────────────────────────────────────────
        if dt_bias is None:
            dt_bias = self._zero_dt_bias

        dt_out, dA_cumsum = self._get_da_cumsum_op(x.dtype).forward(dt, A, dt_bias)
        # dt_out:    (B, H, C, Q)  dtype
        # dA_cumsum: (B, H, C, Q)  float32

        # ── 2. CB matrix ─────────────────────────────────────────────────────
        # cb[b,c,g,l,s] = C[b,c*Q+l,g,:] @ B[b,c*Q+s,g,:]^T  for s <= l, else 0.
        # Pass contiguous C and B directly to avoid reshape/permute/contiguous overhead
        cb_producer_op = self._get_cb_producer_op(
            batch, num_chunks, n_groups, d_state, x.dtype, x.device.index)
        cb = cb_producer_op.forward(C, B)  # (B, C, G, Q, Q)  dtype (direct output, no cast needed)

        # ── 3. SSDChunkState ─────────────────────────────────────────────────
        # Pass pre-allocated seq_idx zeros; avoids a FillFunctor<int> per call.
        # SSDChunkStateFwdOp is constructed with has_seq_idx=False, so the kernel
        # ignores this tensor entirely — no shape check or data access occurs.
        chunk_states = self._chunk_state_op.forward(x, B, dt_out, dA_cumsum, self._zero_seq_idx)
        # chunk_states: (B, C, H, P, N)  float32

        # ── 4. SSDStatePassing ───────────────────────────────────────────────
        chunk_states_flat = chunk_states.reshape(batch, num_chunks, n_heads, d_head * d_state)
        # Extract last dA value per chunk - use contiguous() to ensure a contiguous layout
        # Note: since this is a slice of a 4D tensor, it is non-contiguous and will always copy
        dA_chunk_cumsum = dA_cumsum[..., chunk_size - 1].contiguous()  # (B, H, C)

        if initial_states is None:
            init_flat = self._zero_init_flat
        else:
            init_flat = initial_states.reshape(batch, n_heads, d_head * d_state).float()

        prev_states_flat, final_states_flat = self._state_passing_op.forward(
            chunk_states_flat, dA_chunk_cumsum, init_flat,
        )

        # Unflatten to (B, C, H, P, N) in float32 (accum_dtype) for chunk_scan.
        prev_states  = prev_states_flat.reshape(batch, num_chunks, n_heads, d_head, d_state)
        # dt_out is now in dtype (no cast needed) - DaCumsum outputs typed dt directly

        # ── 5. SSDChunkScan ──────────────────────────────────────────────────
        y = self._chunk_scan_op.forward(x, cb, dA_cumsum, C, prev_states, dt_out)
        # y: (B, S, H, P)  float32

        if return_final_states:
            return y, final_states_flat.reshape(batch, n_heads, d_head, d_state)
        return y, None
