from typing import Dict, Optional, Tuple

import torch

from top.kernels.deepseek_nsa.mean_pooling_fwd import MeanPoolingFwdKernel
from top.kernels.deepseek_nsa.nsa_fwd import NSAFwdVarlenKernel
from top.kernels.deepseek_nsa.nsa_topk import NSATopkVarlenKernel
from top.kernels.deepseek_nsa.nsa_cmp_fwd import NSACmpFwdVarlenKernel
from top.kernels.kernel import Kernel
from top.ops.op import Op

__all__ = ["MeanPoolingForwardOp", "NSAFwdVarlenOp", "NSATopkVarlenOp", "NSACmpFwdVarlenOp"]


class MeanPoolingForwardOp(Op):

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        heads: int,
        dim: int,
        chunk_size: int,
        chunks_per_bacth: int,
        seq_num: int,
        use_offsets: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ('self', 'kernel_map')}
        for key, value in params.items():
            setattr(self, key, value)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mean_pooling_fwd_kernel"](**params)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mean_pooling_fwd_kernel": MeanPoolingFwdKernel}

    def forward(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.kernel(x, offsets, indices=indices)


class NSATopkVarlenOp(Op):

    def __init__(
        self,
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim: int,
        chunk_num: int,
        group: int,
        scale: float,
        selected_block_num: int,
        bc: int,
        bs: int,
        bk: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ('self', 'kernel_map')}
        for key, value in params.items():
            setattr(self, key, value)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["nsa_topk_varlen_kernel"](**params)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_topk_varlen_kernel": NSATopkVarlenKernel}

    def forward(self, q: torch.Tensor, k_cmp: torch.Tensor, lse_in: torch.Tensor,
                offsets: torch.Tensor, chunk_offsets: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, k_cmp, lse_in, offsets, chunk_offsets, token_indices)


class NSAFwdVarlenOp(Op):

    def __init__(
        self,
        batch: int,
        heads: int,
        c_seq_len: int,
        dim: int,
        is_causal: bool,
        scale: float,
        block_size: int,
        groups: int,
        selected_blocks: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ('self', 'kernel_map')}
        for key, value in params.items():
            setattr(self, key, value)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["nsa_fwd_varlen_kernel"](**params)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_fwd_varlen_kernel": NSAFwdVarlenKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                block_indices: torch.Tensor, block_counts: torch.Tensor, offsets: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, k, v, block_indices, block_counts, offsets, token_indices)


class NSACmpFwdVarlenOp(Op):

    def __init__(
        self,
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_num: int,
        group: int,
        scale: float,
        bc: int,
        bs: int,
        bk: int,
        bv: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ('self', 'kernel_map')}
        for key, value in params.items():
            setattr(self, key, value)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["nsa_cmp_fwd_varlen_kernel"](**params)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_cmp_fwd_varlen_kernel": NSACmpFwdVarlenKernel}

    def forward(self, q: torch.Tensor, k_cmp: torch.Tensor, v_cmp: torch.Tensor,
                offsets: torch.Tensor, chunk_offsets: torch.Tensor,
                token_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kernel(q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices)
