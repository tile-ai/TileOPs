from typing import Dict, List, Optional

import torch

from tileops.kernels.engram import EngramGateConvBwdKernel, EngramGateConvFwdKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["EngramGateConvBwdOp", "EngramGateConvFwdOp"]

CONV_KERNEL_SIZE = 4


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
        eps: RMSNorm epsilon (default 1e-6).
    """

    def __init__(
        self,
        M: int,
        seq_len: int,
        d: int,
        eps: float = 1e-6,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.eps = eps
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[torch.dtype, Kernel] = {}

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        if dtype not in self._kernel_cache:
            self._kernel_cache[dtype] = self.kernel_map["engram_gate_conv_fwd"](
                self.M, self.seq_len, self.d, self.eps, dtype, tune=self.tune,
            )
        return self._kernel_cache[dtype]

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
        self._validate_dtypes(H, k, v, rms_w_h, rms_w_v, conv_w)
        self.dtype = H.dtype
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

        return self._get_kernel(H.dtype)(H, k, v, rms_w_h, rms_w_v, conv_w)


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
        eps: RMSNorm epsilon (default 1e-6).
    """

    def __init__(
        self,
        M: int,
        seq_len: int,
        d: int,
        eps: float = 1e-6,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.eps = eps
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[torch.dtype, Kernel] = {}

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        if dtype not in self._kernel_cache:
            self._kernel_cache[dtype] = self.kernel_map["engram_gate_conv_bwd"](
                self.M, self.seq_len, self.d, self.eps, dtype, tune=self.tune,
            )
        return self._kernel_cache[dtype]

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
        self._validate_dtypes(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v,
        )
        self.dtype = dY.dtype

        dY = dY.contiguous()
        H = H.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        vhat = vhat.contiguous()

        return self._get_kernel(dY.dtype)(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w,
            vhat, alpha, rrms_h, rrms_k, rrms_v,
        )
