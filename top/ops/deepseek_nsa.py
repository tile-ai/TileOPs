from typing import Dict, Optional

import torch

from top.kernels.deepseek_nsa.mean_pooling_fwd import MeanPoolingFwdKernel
from top.kernels.deepseek_nsa.nsa_fwd import NSAFwdVarlenKernel
from top.kernels.deepseek_nsa.nsa_topk import NSATopkVarlenKernel
from top.kernels.kernel import Kernel
from top.ops.op import Op

__all__ = ["MeanPoolingForwardOp", "NSAFwdVarlenOp", "NSATopkVarlenOp"]


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
        params = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "heads": heads,
            "dim": dim,
            "chunk_size": chunk_size,
            "chunks_per_bacth": chunks_per_bacth,
            "seq_num": seq_num,
            "use_offsets": use_offsets,
            "dtype": dtype,
            "accum_dtype": accum_dtype,
            "tune": tune,
        }
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
        params = {
            "seq_num": seq_num,
            "c_seq_len": c_seq_len,
            "heads": heads,
            "dim": dim,
            "chunk_num": chunk_num,
            "group": group,
            "scale": scale,
            "selected_block_num": selected_block_num,
            "bc": bc,
            "bs": bs,
            "bk": bk,
            "dtype": dtype,
            "accum_dtype": accum_dtype,
            "tune": tune,
        }
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
        params = {
            "batch": batch,
            "heads": heads,
            "c_seq_len": c_seq_len,
            "dim": dim,
            "is_causal": is_causal,
            "scale": scale,
            "block_size": block_size,
            "groups": groups,
            "selected_blocks": selected_blocks,
            "dtype": dtype,
            "accum_dtype": accum_dtype,
            "tune": tune,
        }
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
