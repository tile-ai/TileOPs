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

from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from .cb_producer import CBProducerFwdOp
from .da_cumsum import DaCumsumFwdOp
from .ssd_chunk_scan import SSDChunkScanFwdOp
from .ssd_chunk_state import SSDChunkStateFwdOp
from .ssd_state_passing import SSDStatePassingFwdOp

__all__ = ["Mamba2FwdOp"]


class Mamba2FwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) full forward pass operator.

    Combines DaCumsum → SSDChunkState → SSDStatePassing → SSDChunkScan into
    a single callable whose interface matches mamba_chunk_scan_combined from
    the official mamba_ssm library.

    """

    def __init__(
        self,
        chunk_size: int = 256,
        dt_softplus: bool = True,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            chunk_size:         Tokens per chunk (default 256).
            dt_softplus:        Apply softplus to (dt + dt_bias) before use.
            tune:               Whether to autotune tile configs on construction.
        """
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
        self.tune = tune
        # This composite owns no kernel; the override reaches the sub-ops that do.
        self.dispatch_kernel(kernel_map)
        self._kernel_map_override = kernel_map

        self._da_cumsum_ops: dict[torch.dtype, DaCumsumFwdOp] = {}
        self._chunk_state_op = SSDChunkStateFwdOp(tune=tune, kernel_map=kernel_map)

        # chunk_states output is float32 (B, C, H, P, N).
        # Flatten P*N into a single state dim so SSDStatePassingFwdOp is used
        # instead of a Python loop, keeping everything on the GPU.
        self._state_passing_op = SSDStatePassingFwdOp(tune=tune, kernel_map=kernel_map)

        self._chunk_scan_op = SSDChunkScanFwdOp(tune=tune, kernel_map=kernel_map)
        self._cb_producer_ops: dict[tuple, CBProducerFwdOp] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # A composite registers no kernel of its own: the five it drives belong
        # to the sub-ops it builds, and each of those is a replacement point.
        return {}

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import mamba2_fwd_roofline

        return mamba2_fwd_roofline(self)

    def _validate_dtypes(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Manifest ``dtype``: x is half, B and C follow it, the rest are float32."""
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"x.dtype must be float16 or bfloat16, got {x.dtype}")
        for name, tensor in (("B", B), ("C", C)):
            if tensor.dtype != x.dtype:
                raise ValueError(f"{name}.dtype must be {x.dtype}, got {tensor.dtype}")
        for name, tensor in (
            ("dt", dt),
            ("A", A),
            ("dt_bias", dt_bias),
            ("initial_states", initial_states),
        ):
            if tensor is not None and tensor.dtype != torch.float32:
                raise ValueError(f"{name}.dtype must be float32, got {tensor.dtype}")
        self.dtype = x.dtype

    def _get_da_cumsum_op(self, dtype: torch.dtype) -> DaCumsumFwdOp:
        if dtype not in self._da_cumsum_ops:
            self._da_cumsum_ops[dtype] = DaCumsumFwdOp(
                chunk_len=self.chunk_size,
                dtype=dtype,
                dt_softplus=self.dt_softplus,
                tune=self.tune,
                kernel_map=self._kernel_map_override,
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
    ) -> CBProducerFwdOp:
        key = (
            batch,
            num_chunks,
            n_groups,
            self.chunk_size,
            d_state,
            dtype,
            device_index,
            self.tune,
        )
        if key not in self._cb_producer_ops:
            self._cb_producer_ops[key] = CBProducerFwdOp(
                batch=batch,
                num_chunks=num_chunks,
                n_groups=n_groups,
                chunk_len=self.chunk_size,
                d_state=d_state,
                tune=self.tune,
                kernel_map=self._kernel_map_override,
            )
        return self._cb_producer_ops[key]

    # Forward

    def _infer_output_shapes(
        self,
        x_shape: tuple[int, ...],
        dt_shape: tuple[int, ...],
        A_shape: tuple[int, ...],
        B_shape: tuple[int, ...],
        C_shape: tuple[int, ...],
        dt_bias_shape: tuple[int, ...],
        initial_states_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: ``y`` follows *x*, the states carry the state size of *B*."""
        b, s, h, p = x_shape
        return {"y": (b, s, h, p), "final_states": (b, h, p, B_shape[-1])}

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
        initial_states: Optional[torch.Tensor] = None,
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

        Returns:
            y:            (batch, seqlen, n_heads, d_head)   float32
            final_states: (batch, n_heads, d_head, d_state)  float32, or None
        """
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
        self._validate_dtypes(x, dt, A, B, C, dt_bias, initial_states)

        self.batch = batch
        self.seqlen = seqlen
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = x.dtype
        self.dt_bias_shape = None if dt_bias is None else tuple(dt_bias.shape)
        self.initial_states_shape = None if initial_states is None else tuple(initial_states.shape)

        # ── 1. DaCumsum ──────────────────────────────────────────────────────
        dt_out, dA_cumsum = self._get_da_cumsum_op(x.dtype).forward(dt, A, dt_bias)
        # dt_out:    (B, H, C, Q)  dtype
        # dA_cumsum: (B, H, C, Q)  float32

        # ── 2. CB matrix ─────────────────────────────────────────────────────
        # cb[b,c,g,l,s] = C[b,c*Q+l,g,:] @ B[b,c*Q+s,g,:]^T  for s <= l, else 0.
        # Pass contiguous C and B directly to avoid reshape/permute/contiguous overhead
        cb_producer_op = self._get_cb_producer_op(
            batch, num_chunks, n_groups, d_state, x.dtype, x.device.index
        )
        cb = cb_producer_op.forward(C, B)  # (B, C, G, Q, Q)  dtype (direct output, no cast needed)

        # ── 3. SSDChunkState ─────────────────────────────────────────────────
        # No seq_idx: this composite does not segment a chunk, so the kernel is
        # built without that branch.
        chunk_states = self._chunk_state_op.forward(x, B, dt_out, dA_cumsum)
        # chunk_states: (B, C, H, P, N)  float32

        # ── 4. SSDStatePassing ───────────────────────────────────────────────
        chunk_states_flat = chunk_states.reshape(batch, num_chunks, n_heads, d_head * d_state)
        # Extract last dA value per chunk - use contiguous() to ensure a contiguous layout
        # Note: since this is a slice of a 4D tensor, it is non-contiguous and will always copy
        dA_chunk_cumsum = dA_cumsum[..., chunk_size - 1].contiguous()  # (B, H, C)

        init_flat = (
            None
            if initial_states is None
            else initial_states.reshape(batch, n_heads, d_head * d_state).float()
        )

        prev_states_flat, final_states_flat = self._state_passing_op.forward(
            chunk_states_flat,
            dA_chunk_cumsum,
            init_flat,
        )

        # Unflatten to (B, C, H, P, N) in float32 (accum_dtype) for chunk_scan.
        prev_states = prev_states_flat.reshape(batch, num_chunks, n_heads, d_head, d_state)
        # dt_out is now in dtype (no cast needed) - DaCumsum outputs typed dt directly

        # ── 5. SSDChunkScan ──────────────────────────────────────────────────
        y = self._chunk_scan_op.forward(x, cb, dA_cumsum, C, prev_states, dt_out)
        # y: (B, S, H, P)  float32

        return y, final_states_flat.reshape(batch, n_heads, d_head, d_state)
