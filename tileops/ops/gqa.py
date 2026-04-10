from typing import Dict, Optional

import torch

from tileops.kernels.flash_attn import (
    FlashAttnBwdPostprocessKernel,
    FlashAttnBwdPreprocessKernel,
    GqaBwdKernel,
    GqaBwdWgmmaPipelinedKernel,
    GqaFwdKernel,
    GqaFwdWgmmaPipelinedKernel,
    GqaFwdWsKernel,
    GqaFwdWsPersistentKernel,
)
from tileops.kernels.kernel import Kernel
from tileops.utils import is_hopper

from .op import Op

__all__ = ['GqaFwdOp', 'GqaBwdOp']


class GqaFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_fwd_kernel"](
            batch, heads, heads_kv, seq_len, dim, is_causal, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        # Hopper + dim=128: use the warp-specialized FA3-aligned kernels.
        # GqaFwdWsPersistentKernel adds persistent CTA + causal tile pairing
        # on top of GqaFwdWsKernel, but is causal-only and requires an even
        # M_blocks count.  Falls back to GqaFwdWsKernel for non-causal or
        # odd-M_blocks shapes, and to GqaFwdWgmmaPipelinedKernel for
        # non-dim=128.  Non-Hopper falls back to the generic kernel.
        #
        # M_blocks is computed via ceil division to match the JIT-time
        # formula in _gqa_fwd_ws_persistent_func; using floor division
        # would dispatch to the persistent kernel for seq_len values
        # like 257..383 where the ceil-div M_blocks is odd, causing a
        # ValueError at JIT compile time.
        #
        # NOTE: ``default_block_m=128`` is hardcoded here because dispatch
        # runs in __init__ before ``init_config`` (where the actual
        # block_m is set, possibly by autotune).  This is consistent with
        # GqaFwdWsPersistentKernel.autotune_configs which only offers
        # block_m=[128].  If anyone adds smaller block_m values to the
        # persistent kernel's autotune sweep, this gate must move to a
        # post-config-resolution check (or both must agree on a single
        # canonical M_blocks formula).
        if is_hopper() and self.dim == 128:
            default_block_m = 128
            m_blocks = (self.seq_len + default_block_m - 1) // default_block_m
            if self.is_causal and m_blocks > 0 and m_blocks % 2 == 0:
                return {"gqa_fwd_kernel": GqaFwdWsPersistentKernel}
            return {"gqa_fwd_kernel": GqaFwdWsKernel}
        if is_hopper():
            return {"gqa_fwd_kernel": GqaFwdWgmmaPipelinedKernel}
        return {"gqa_fwd_kernel": GqaFwdKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, k, v)


class GqaBwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.prep_kernel = self.kernel_map["gqa_bwd_preprocess_kernel"](batch, heads, seq_len, dim,
                                                                        self.dtype)
        self.kernel = self.kernel_map["gqa_bwd_kernel"](
            batch, heads, heads_kv, seq_len, dim, is_causal, self.dtype, tune=tune)
        if not is_hopper():
            self.post_kernel = self.kernel_map["gqa_bwd_postprocess_kernel"](batch, heads, seq_len,
                                                                             dim, self.dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_bwd_preprocess_kernel":
                FlashAttnBwdPreprocessKernel,
            "gqa_bwd_kernel":
                GqaBwdWgmmaPipelinedKernel if is_hopper() else GqaBwdKernel,
            "gqa_bwd_postprocess_kernel":
                FlashAttnBwdPostprocessKernel if not is_hopper() else None,
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                do: torch.Tensor,
                lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        do = do.contiguous()
        delta = self.prep_kernel(o, do)
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        self.kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = dq.to(self.dtype) if is_hopper() else self.post_kernel(dq)
        dk, dv = dk.to(self.dtype), dv.to(self.dtype)
        return dq, dk, dv
