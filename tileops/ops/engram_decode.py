from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.engram import EngramDecodeKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["EngramDecodeOp"]

ALIGNMENT = 256


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


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
        dtype: Data type (float16 or bfloat16).
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
        dtype: torch.dtype,
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
        self.dtype = dtype
        self.eps = eps
        self.d_padded = _align_up(d, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["engram_decode"](
            batch, d_mem, d, max_conv_len, conv_kernel_size, dilation,
            eps, dtype, tune=tune,
        )

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
        if e_t.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {e_t.dtype}")

        e_t = e_t.contiguous()
        h_t = h_t.contiguous()
        conv_state = conv_state.contiguous()

        # Track real conv history length before padding
        real_conv_len = conv_state.shape[1]

        # Left-pad conv_state to max_conv_len if needed
        if real_conv_len < self.max_conv_len:
            conv_state = F.pad(
                conv_state,
                (0, 0, self.max_conv_len - real_conv_len, 0),
                mode='constant', value=0,
            )

        # Pad d dimension to alignment
        if self.d_padded != self.d:
            h_t = F.pad(h_t, (0, self.d_padded - self.d))
            conv_state = F.pad(conv_state, (0, self.d_padded - self.d))
            W_K = F.pad(W_K, (0, self.d_padded - self.d))
            W_V = F.pad(W_V, (0, self.d_padded - self.d))
            rms_w_h = F.pad(rms_w_h, (0, self.d_padded - self.d))
            rms_w_v = F.pad(rms_w_v, (0, self.d_padded - self.d))
            conv_w = F.pad(conv_w, (0, self.d_padded - self.d))

        results = self.kernel(
            e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
            real_conv_len,
        )

        if self.d_padded != self.d:
            results[0] = results[0][:, :self.d]
            results[1] = results[1][:, :, :self.d]

        return results
