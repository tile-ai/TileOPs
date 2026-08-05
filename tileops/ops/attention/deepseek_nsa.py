from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.attention import (
    NSACmpFwdVarlenKernel,
    NSAFwdVarlenKernel,
    NSATopkVarlenKernel,
)
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op

__all__ = [
    "NSACmpFwdVarlenOp",
    "NSAFwdVarlenOp",
    "NSATopkVarlenOp",
]


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
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ('self', 'kernel_map')}
        for key, value in params.items():
            setattr(self, key, value)

        self._kernel_params = params
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[torch.dtype, Kernel] = {}

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        if dtype not in self._kernel_cache:
            self._kernel_cache[dtype] = self.kernel_map["nsa_topk_varlen_kernel"](
                **self._kernel_params, dtype=dtype,
            )
        return self._kernel_cache[dtype]

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_topk_varlen_kernel": NSATopkVarlenKernel}

    def forward(self, q: torch.Tensor, k_cmp: torch.Tensor, lse_in: torch.Tensor,
                offsets: torch.Tensor, chunk_offsets: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(q, k_cmp, lse_in, offsets, chunk_offsets, token_indices)


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
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ('self', 'kernel_map')}
        for key, value in params.items():
            setattr(self, key, value)

        self._kernel_params = params
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[torch.dtype, Kernel] = {}

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        if dtype not in self._kernel_cache:
            self._kernel_cache[dtype] = self.kernel_map["nsa_fwd_varlen_kernel"](
                **self._kernel_params, dtype=dtype,
            )
        return self._kernel_cache[dtype]

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_fwd_varlen_kernel": NSAFwdVarlenKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                block_indices: torch.Tensor, block_counts: torch.Tensor, offsets: torch.Tensor,
                token_indices: torch.Tensor) -> torch.Tensor:
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(q, k, v, block_indices, block_counts, offsets, token_indices)


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
        self._kernel_cache: Dict[torch.dtype, Kernel] = {}

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        if dtype not in self._kernel_cache:
            self._kernel_cache[dtype] = self.kernel_map["nsa_cmp_fwd_varlen_kernel"](
                **self._kernel_params, dtype=dtype,
            )
        return self._kernel_cache[dtype]

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_cmp_fwd_varlen_kernel": NSACmpFwdVarlenKernel}

    def forward(self, q: torch.Tensor, k_cmp: torch.Tensor, v_cmp: torch.Tensor,
                offsets: torch.Tensor, chunk_offsets: torch.Tensor,
                token_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices)
