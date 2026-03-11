from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.engram import EngramGateConvBwdKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["EngramGateConvBwdOp"]

ALIGNMENT = 256
CONV_KERNEL_SIZE = 4


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class EngramGateConvBwdOp(Op):
    """Engram GateConv backward operator.

    Given dY and saved intermediates from forward (vhat, alpha, rrms_*),
    computes gradients for H, k, v, rms_w_h, rms_w_v, conv_w.

    The caller is responsible for the projection backward (GEMM):
        dE  = dk @ W_K^T + dv @ W_V^T
        dW_K = E^T @ dk
        dW_V = E^T @ dv

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
        self.kernel = self.kernel_map["engram_gate_conv_bwd"](
            M, seq_len, d, eps, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"engram_gate_conv_bwd": EngramGateConvBwdKernel}

    def forward(
        self,
        dY: torch.Tensor,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
        vhat: torch.Tensor,
        alpha: torch.Tensor,
        rrms_h: torch.Tensor,
        rrms_k: torch.Tensor,
        rrms_v: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            dY:     (M, T, d) — gradient of output Y.
            H:      (M, T, d) — hidden states (forward input).
            k:      (M, T, d) — key projection (forward input, or recomputed).
            v:      (M, T, d) — value projection (forward input).
            rms_w_h: (d,) — RMSNorm weight for H and k.
            rms_w_v: (d,) — RMSNorm weight for v_hat.
            conv_w:  (4, d) — conv weights.
            vhat:   (M, T, d) — saved from forward.
            alpha:  (M, T) — saved from forward.
            rrms_h: (M, T) — saved from forward.
            rrms_k: (M, T) — saved from forward.
            rrms_v: (M, T) — saved from forward.

        Returns:
            List of [dH, dk, dv, drms_w_h, drms_w_v, dconv_w]:
              dH:       (M, T, d)
              dk:       (M, T, d)
              dv:       (M, T, d)
              drms_w_h: (d,) — fp32
              drms_w_v: (d,) — fp32
              dconv_w:  (4, d) — fp32
        """
        if not dY.is_cuda:
            raise ValueError("dY must be a CUDA tensor")
        if dY.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {dY.dtype}")

        dY = dY.contiguous()
        H = H.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        vhat = vhat.contiguous()

        # Pad hidden dim if needed
        if self.d_padded != self.d:
            dY = F.pad(dY, (0, self.d_padded - self.d))
            H = F.pad(H, (0, self.d_padded - self.d))
            k = F.pad(k, (0, self.d_padded - self.d))
            v = F.pad(v, (0, self.d_padded - self.d))
            vhat = F.pad(vhat, (0, self.d_padded - self.d))
            rms_w_h = F.pad(rms_w_h, (0, self.d_padded - self.d))
            rms_w_v = F.pad(rms_w_v, (0, self.d_padded - self.d))
            conv_w = F.pad(conv_w, (0, self.d_padded - self.d))

        results = self.kernel(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v,
        )
        # results: [dH, dk, dv, drms_w_h, drms_w_v, dconv_w, dvhat_buf]

        dH, dk, dv = results[0], results[1], results[2]
        drms_w_h, drms_w_v, dconv_w = results[3], results[4], results[5]

        # Trim padding
        if self.d_padded != self.d:
            dH = dH[:, :, :self.d]
            dk = dk[:, :, :self.d]
            dv = dv[:, :, :self.d]
            drms_w_h = drms_w_h[:self.d]
            drms_w_v = drms_w_v[:self.d]
            dconv_w = dconv_w[:, :self.d]

        return [dH, dk, dv, drms_w_h, drms_w_v, dconv_w]
