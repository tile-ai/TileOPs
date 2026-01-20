from typing import Dict, Optional

import torch

from top.kernels.deepseek_nsa.mean_pooling_fwd import MeanPoolingFwdKernel
from top.kernels.deepseek_nsa.nsa_fwd import NSAFwdVarlenKernel
from top.kernels.kernel import Kernel
from top.ops.op import Op

__all__ = ["MeanPoolingForwardOp", "NSAFwdVarlenOp"]


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
