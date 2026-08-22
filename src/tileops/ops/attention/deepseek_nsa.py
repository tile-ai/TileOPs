from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.attention import (
    NSACmpFwdVarlenKernel,
    NSAFwdVarlenKernel,
    NSATopkVarlenKernel,
)
from tileops.kernels.kernel_base import Kernel

from ..op_base import UnmanifestedOp

__all__ = [
    "NSACmpFwdVarlenOp",
    "NSAFwdVarlenOp",
    "NSATopkVarlenOp",
]


class NSATopkVarlenOp(UnmanifestedOp):
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
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ("self", "kernel_map")}
        for key, value in params.items():
            setattr(self, key, value)

        self._kernel_params = params
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "nsa_topk_varlen_kernel",
            inputs,
            key=dtype,
            build=lambda: self.kernel_map["nsa_topk_varlen_kernel"](
                **self._kernel_params,
                dtype=dtype,
            ),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_topk_varlen_kernel": NSATopkVarlenKernel}

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        lse_in: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        self.dtype = q.dtype
        tensors = (q, k_cmp, lse_in, offsets, chunk_offsets, token_indices)
        return self._get_kernel(tensors, q.dtype)(*tensors)


class NSAFwdVarlenOp(UnmanifestedOp):
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
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ("self", "kernel_map")}
        for key, value in params.items():
            setattr(self, key, value)

        self._kernel_params = params
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "nsa_fwd_varlen_kernel",
            inputs,
            key=dtype,
            build=lambda: self.kernel_map["nsa_fwd_varlen_kernel"](
                **self._kernel_params,
                dtype=dtype,
            ),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_fwd_varlen_kernel": NSAFwdVarlenKernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        self.dtype = q.dtype
        tensors = (q, k, v, block_indices, block_counts, offsets, token_indices)
        return self._get_kernel(tensors, q.dtype)(*tensors)


class NSACmpFwdVarlenOp(UnmanifestedOp):
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
            "accum_dtype": accum_dtype,
            "tune": tune,
        }
        for key, value in params.items():
            setattr(self, key, value)

        self._kernel_params = params
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "nsa_cmp_fwd_varlen_kernel",
            inputs,
            key=dtype,
            build=lambda: self.kernel_map["nsa_cmp_fwd_varlen_kernel"](
                **self._kernel_params,
                dtype=dtype,
            ),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_cmp_fwd_varlen_kernel": NSACmpFwdVarlenKernel}

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.dtype = q.dtype
        tensors = (q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices)
        return self._get_kernel(tensors, q.dtype)(*tensors)
