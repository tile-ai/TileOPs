from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.deepseek_mla import Fp8LightingIndexerKernel
from tileops.kernels.kernel import Kernel

from .op import Op

__all__ = ["Fp8LightingIndexerOp"]


class Fp8LightingIndexerOp(Op):

    def __init__(self,
                 batch,
                 seq_len,
                 heads,
                 index_dim,
                 seq_len_kv,
                 kv_group,
                 clean_logits=True,
                 config: Optional[dict] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.clean_logits = clean_logits

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["fp8_lighting_indexer_kernel"](
            batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits, config, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fp8_lighting_indexer_kernel": Fp8LightingIndexerKernel}

    def torch_quant_forward(self, index_q: torch.Tensor, index_k: torch.Tensor,
                            weights: torch.Tensor, cu_seqlen_ks: torch.Tensor,
                            cu_seqlen_ke: torch.Tensor) -> torch.Tensor:
        index_q = index_q.to(torch.float8_e4m3fn)
        index_k, index_k_scale = self.per_custom_dims_cast_to_fp8(index_k, (0,), False)

        return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    def tl_quant_forward(self, index_q: torch.Tensor, index_k: torch.Tensor,
                         index_k_scale: torch.Tensor, weights: torch.Tensor,
                         cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor):
        return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    def forward(self, *args) -> torch.Tensor:
        assert len(args) == 5 or len(
            args
        ) == 6, f"Invalid number of arguments passed to forward method. Expected 5 or 6, got {len(args)}."
        if len(args) == 5:
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke = args
            return self.torch_quant_forward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke)
        elif len(args) == 6:
            index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke = args
            return self.tl_quant_forward(index_q, index_k, index_k_scale, weights, cu_seqlen_ks,
                                         cu_seqlen_ke)

    def per_custom_dims_cast_to_fp8(self, x: torch.Tensor, dims: Tuple[int],
                                    use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        x_absmax = x.to(torch.float32).abs().amax(dim=-1, keepdim=True).clamp(1e-4)
        sf = x_absmax / 448.0
        if use_ue8m0:
            assert sf.view(-1).amax().item() > 0
            sf = torch.pow(2.0, torch.ceil(torch.log2(x_absmax)))
        x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled, sf.squeeze(-1)
