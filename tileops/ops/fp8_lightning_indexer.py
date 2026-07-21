from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.fp8_lightning_indexer import FP8LightningIndexerKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["FP8LightningIndexerOp"]


class FP8LightningIndexerOp(Op):

    def __init__(self,
                 clean_logits=True,
                 config: Optional[dict] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False) -> None:
        self.batch = None
        self.seq_len = None
        self.heads = None
        self.index_dim = None
        self.seq_len_kv = None
        self.kv_group = None
        self.clean_logits = clean_logits
        self.config = config
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fp8_lightning_indexer_kernel": FP8LightningIndexerKernel}

    @property
    def _config_cache_key(self) -> tuple:
        if not self.config:
            return ()
        return tuple(sorted((key, repr(value)) for key, value in self.config.items()))

    def _get_kernel(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        index_dim: int,
        seq_len_kv: int,
        kv_group: int,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            seq_len,
            heads,
            index_dim,
            seq_len_kv,
            kv_group,
            self.clean_logits,
            self._config_cache_key,
            device_index,
            self.tune,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["fp8_lightning_indexer_kernel"](
                batch,
                seq_len,
                heads,
                index_dim,
                seq_len_kv,
                kv_group,
                self.clean_logits,
                self.config,
                tune=self.tune)
        return self._kernel_cache[key]

    def _resolve_and_bind(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        index_k_scale: Optional[torch.Tensor],
    ) -> None:
        if not index_q.is_cuda or not index_k.is_cuda:
            raise ValueError("FP8LightningIndexerOp expects CUDA inputs")
        if index_q.ndim != 4 or index_k.ndim != 4:
            raise ValueError("FP8LightningIndexerOp expects index_q/index_k to be 4D tensors")
        batch, seq_len, heads, index_dim = index_q.shape
        k_batch, seq_len_kv, kv_group, k_dim = index_k.shape
        if k_batch != batch or k_dim != index_dim:
            raise ValueError("index_q and index_k must agree on batch and index_dim")
        if heads % kv_group != 0:
            raise ValueError("heads must be divisible by kv_group")
        if weights.shape != (seq_len, heads):
            raise ValueError("weights must have shape [seq_len, heads]")
        if cu_seqlen_ks.shape != (seq_len,) or cu_seqlen_ke.shape != (seq_len,):
            raise ValueError("cu_seqlen_ks/cu_seqlen_ke must have shape [seq_len]")
        if index_k_scale is not None and index_k_scale.shape != (batch, seq_len_kv, kv_group):
            raise ValueError("index_k_scale must have shape [batch, seq_len_kv, kv_group]")

        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.kernel = self._get_kernel(
            batch, seq_len, heads, index_dim, seq_len_kv, kv_group, index_q.device.index)

    def torch_quant_forward(self, index_q: torch.Tensor, index_k: torch.Tensor,
                            weights: torch.Tensor, cu_seqlen_ks: torch.Tensor,
                            cu_seqlen_ke: torch.Tensor) -> torch.Tensor:
        index_q = index_q.to(torch.float8_e4m3fn)
        index_k, index_k_scale = self.per_custom_dims_cast_to_fp8(index_k, (0,), False)

        return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    def tl_quant_forward(self, index_q: torch.Tensor, index_k: torch.Tensor,
                         index_k_scale: torch.Tensor, weights: torch.Tensor,
                         cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor) -> torch.Tensor:
        return self.kernel(index_q, index_k, index_k_scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    def forward(self,
                index_q: torch.Tensor,
                index_k: torch.Tensor,
                weights: torch.Tensor,
                cu_seqlen_ks: torch.Tensor,
                cu_seqlen_ke: torch.Tensor,
                index_k_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._resolve_and_bind(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, index_k_scale)
        if index_k_scale is None:
            return self.torch_quant_forward(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke)
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
