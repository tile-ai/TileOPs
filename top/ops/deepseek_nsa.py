from typing import Dict, Optional, Tuple

import torch

from top.kernels.deepseek_nsa.mean_pooling_fwd import MeanPoolingFwdKernel
from top.kernels.deepseek_nsa.nsa_fwd import NSAFwdVarlenKernel
from top.kernels.deepseek_nsa.nsa_topk import NSATopkVarlenKernel
from top.kernels.deepseek_nsa.nsa_cmp_fwd import NSACmpFwdVarlenKernel
from top.kernels.deepseek_nsa.gqa_window_sliding import GQAWindowSlidingKernel
from top.kernels.kernel import Kernel
from top.ops.op import Op

__all__ = [
    "MeanPoolingForwardOp",
    "NSAFwdVarlenOp",
    "NSATopkVarlenOp",
    "NSACmpFwdVarlenOp",
    "GQAWindowSlidingOp",
]


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
        params = {
            "seq_num": seq_num,
            "c_seq_len": c_seq_len,
            "heads": heads,
            "dim_k": dim_k,
            "dim_v": dim_v,
            "chunk_num": chunk_num,
            "group": group,
            "scale": scale,
            "bc": bc,
            "bs": bs,
            "bk": bk,
            "bv": bv,
            "dtype": dtype,
            "accum_dtype": accum_dtype,
            "tune": tune,
        }
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


class GQAWindowSlidingOp(Op):

    def __init__(
        self,
        batch_size: int,
        groups: int,
        uq: int,
        ukv: int,
        heads: int,
        dim: int,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ('self', 'kernel_map')}
        for key, value in params.items():
            setattr(self, key, value)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_window_sliding_kernel"](**params)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_window_sliding_kernel": GQAWindowSlidingKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor,
                cu_seqlens_k: torch.Tensor, max_seqlen_q: int) -> torch.Tensor:
        # Security validation: prevent OOB writes by validating input tensors
        # Using explicit if statements instead of assert to ensure validation
        # is always performed, even with Python -O optimization flag

        # 1. Check tensor shapes
        if cu_seqlens_q.shape[0] != self.batch_size + 1:
            raise ValueError(
                f"cu_seqlens_q.shape[0] ({cu_seqlens_q.shape[0]}) must equal batch_size + 1 ({self.batch_size + 1})"
            )
        if cu_seqlens_k.shape[0] != self.batch_size + 1:
            raise ValueError(
                f"cu_seqlens_k.shape[0] ({cu_seqlens_k.shape[0]}) must equal batch_size + 1 ({self.batch_size + 1})"
            )

        # 2. Check that values are non-decreasing
        cu_seqlens_q_diff = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        if not torch.all(cu_seqlens_q_diff >= 0):
            raise ValueError("cu_seqlens_q must be non-decreasing")
        cu_seqlens_k_diff = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if not torch.all(cu_seqlens_k_diff >= 0):
            raise ValueError("cu_seqlens_k must be non-decreasing")

        # 3. Check that maximum values don't exceed tensor dimensions
        max_q_idx = cu_seqlens_q[-1].item()
        if max_q_idx > self.uq:
            raise ValueError(f"cu_seqlens_q[-1] ({max_q_idx}) must not exceed uq ({self.uq})")
        max_kv_idx = cu_seqlens_k[-1].item()
        if max_kv_idx > self.ukv:
            raise ValueError(f"cu_seqlens_k[-1] ({max_kv_idx}) must not exceed ukv ({self.ukv})")

        # 4. Check that max_seqlen_q is consistent with actual maximum sequence length
        actual_max_seqlen_q = cu_seqlens_q_diff.max().item()
        if max_seqlen_q < actual_max_seqlen_q:
            raise ValueError(
                f"max_seqlen_q ({max_seqlen_q}) must be >= actual max sequence length ({actual_max_seqlen_q})"
            )

        # 5. Additional safety: ensure cu_seqlens_q starts at 0
        if cu_seqlens_q[0].item() != 0:
            raise ValueError(f"cu_seqlens_q[0] must be 0, got {cu_seqlens_q[0].item()}")
        if cu_seqlens_k[0].item() != 0:
            raise ValueError(f"cu_seqlens_k[0] must be 0, got {cu_seqlens_k[0].item()}")

        return self.kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
