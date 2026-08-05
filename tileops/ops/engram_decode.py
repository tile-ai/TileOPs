from typing import Dict, List, Optional

import torch

from tileops.kernels.engram import EngramDecodeKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["EngramDecodeOp"]


class EngramDecodeOp(Op):
    """Engram fused decode operator — single-token inference.

    Fuses GEMV projection + gating + dilated conv + SiLU into one kernel call.

    Like MHA decode compiles with max_seqlen_kv and accepts real_seqlen_kv
    at runtime, this op compiles with max_conv_len and accepts the actual
    conv_state length at runtime without recompilation.

    Args:
        batch: Batch size.
        d_mem: Memory embedding dimension.
        d: Model hidden dimension.
        max_conv_len: Max conv cache capacity (compile-time).
        conv_kernel_size: Number of conv taps w (model param, e.g. 4).
        dilation: Dilation factor δ (model param, e.g. max N-gram order).
        eps: RMSNorm epsilon (default 1e-6).
    """

    def __init__(
        self,
        batch: int,
        d_mem: int,
        d: int,
        max_conv_len: int,
        conv_kernel_size: int,
        dilation: int,
        eps: float = 1e-6,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = batch
        self.d_mem = d_mem
        self.d = d
        self.max_conv_len = max_conv_len
        self.conv_kernel_size = conv_kernel_size
        self.dilation = dilation
        self.eps = eps
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[torch.dtype, Kernel] = {}

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        if dtype not in self._kernel_cache:
            self._kernel_cache[dtype] = self.kernel_map["engram_decode"](
                self.batch, self.d_mem, self.d, self.max_conv_len,
                self.conv_kernel_size, self.dilation, self.eps, dtype,
                tune=self.tune,
            )
        return self._kernel_cache[dtype]

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"engram_decode": EngramDecodeKernel}

    def forward(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        conv_state: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            e_t:        (B, d_mem) — gathered N-gram embedding for current token.
            h_t:        (B, d) — hidden state for current token.
            conv_state: (B, L, d) — conv history, L <= max_conv_len.
                        Left-padded to max_conv_len internally if L < max_conv_len.
            W_K:        (d_mem, d) — key projection weight.
            W_V:        (d_mem, d) — value projection weight.
            rms_w_h:    (d,) — RMSNorm weight for h and k.
            rms_w_v:    (d,) — RMSNorm weight for v_hat.
            conv_w:     (w, d) — depthwise conv weights (w = conv_kernel_size).

        Returns:
            [y_t, new_conv_state]:
              y_t:            (B, d) — output to add as residual.
              new_conv_state: (B, max_conv_len, d) — updated state for next step.
        """
        if not e_t.is_cuda:
            raise ValueError("e_t must be a CUDA tensor")
        self._validate_dtypes(
            e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
        )
        self.dtype = e_t.dtype

        e_t = e_t.contiguous()
        h_t = h_t.contiguous()
        conv_state = conv_state.contiguous()

        return self._get_kernel(e_t.dtype)(
            e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
        )
