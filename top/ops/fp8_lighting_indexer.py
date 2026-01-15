from typing import Dict, Optional, Tuple

import torch

from top.kernels.deepseek_mla import fp8_lighting_indexer_kernel
from top.kernels.kernel import Kernel

from .op import Op

__all__ = ["Fp8LightingIndexerOp"]


class Fp8LightingIndexerOp(Op):

    def __init__(self,
                 seq_len,
                 heads,
                 index_dim,
                 seq_len_kv,
                 clean_logits=True,
                 config: Optional[dict] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False) -> None:
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.clean_logits = clean_logits
        self.config = config

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["fp8_lighting_indexer_kernel"](
            seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fp8_lighting_indexer_kernel": fp8_lighting_indexer_kernel}

    def forward(self, index_q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor,
                cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> torch.Tensor:
        index_q = index_q.to(torch.float8_e4m3fn)
        index_k, index_k_scale = self.per_custom_dims_cast_to_fp8(index_k, (0,), False)
        return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    def per_custom_dims_cast_to_fp8(self, x: torch.Tensor, dims: Tuple[int],
                                    use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
        x_amax = x.to(torch.float32).abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
        sf = x_amax / 448.0
        if use_ue8m0:
            assert sf.view(-1).amax().item() > 0
            sf = torch.pow(2.0, torch.ceil(torch.log2(x.abs())))
        x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled, sf.squeeze()
