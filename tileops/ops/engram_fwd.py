from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.engram import EngramGateConvFwdKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["EngramGateConvFwdOp"]

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class EngramGateConvFwdOp(Op):
    """Engram GateConv forward operator (post-projection fusion).

    Assumes k = E @ W_K and v = E @ W_V have been computed externally
    via standard GEMM. This op fuses the remaining memory-bound stages:

        RMSNorm gating -> causal DWConv1D -> SiLU + residual

    Returns Y plus saved intermediates for backward (strategy B):
        vhat, alpha, rrms_h, rrms_k, rrms_v

    Args:
        M: Batch size.
        seq_len: Sequence length.
        d: Model hidden dimension.
        dtype: Data type (float16 or bfloat16).
        eps: RMSNorm epsilon (default 1e-6).
    """

    def __init__(
        self,
        M: int,
        seq_len: int,
        d: int,
        dtype: torch.dtype,
        eps: float = 1e-6,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps
        self.d_padded = _align_up(d, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["engram_gate_conv_fwd"](
            M, seq_len, d, eps, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"engram_gate_conv_fwd": EngramGateConvFwdKernel}

    def forward(
        self,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            H: (M, seq_len, d) — hidden states from preceding layers.
            k: (M, seq_len, d) — key projection output (E @ W_K).
            v: (M, seq_len, d) — value projection output (E @ W_V).
            rms_w_h: (d,) — RMSNorm weight for H and k.
            rms_w_v: (d,) — RMSNorm weight for gated value (conv input).
            conv_w: (4, d) — depthwise causal conv1d weights.

        Returns:
            List of [Y, vhat, alpha, rrms_h, rrms_k, rrms_v]:
              Y:      (M, seq_len, d) — output to add as residual to H.
              vhat:   (M, seq_len, d) — saved for backward.
              alpha:  (M, seq_len) — scalar gate, saved for backward.
              rrms_h: (M, seq_len) — RMSNorm reciprocal rms of H.
              rrms_k: (M, seq_len) — RMSNorm reciprocal rms of k.
              rrms_v: (M, seq_len) — RMSNorm reciprocal rms of v_hat.
        """
        if not H.is_cuda:
            raise ValueError("H must be a CUDA tensor")
        if H.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {H.dtype}")
        if H.shape[-1] != self.d:
            raise ValueError(
                f"Expected hidden dim {self.d}, got {H.shape[-1]}"
            )
        if conv_w.shape[0] != CONV_KERNEL_SIZE:
            raise ValueError(
                f"Expected conv kernel size {CONV_KERNEL_SIZE}, got {conv_w.shape[0]}"
            )

        H = H.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Pad hidden dim to alignment if needed
        if self.d_padded != self.d:
            H = F.pad(H, (0, self.d_padded - self.d))
            k = F.pad(k, (0, self.d_padded - self.d))
            v = F.pad(v, (0, self.d_padded - self.d))
            rms_w_h = F.pad(rms_w_h, (0, self.d_padded - self.d))
            rms_w_v = F.pad(rms_w_v, (0, self.d_padded - self.d))
            conv_w = F.pad(conv_w, (0, self.d_padded - self.d))

        results = self.kernel(H, k, v, rms_w_h, rms_w_v, conv_w)
        # results: [Y, vhat, alpha, rrms_h, rrms_k, rrms_v]

        # Trim padding on d-dim tensors
        if self.d_padded != self.d:
            results[0] = results[0][:, :, :self.d]   # Y
            results[1] = results[1][:, :, :self.d]   # vhat

        return results
