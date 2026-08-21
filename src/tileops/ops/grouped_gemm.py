from typing import Dict, Optional

import torch

from tileops.kernels.grouped_gemm import (
    GroupedGemmCall,
    GroupedGemmKernel,
    GroupedGemmPersistent3WGKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

from .op_base import Op

__all__ = ["GroupedGemmFwdOp"]


class GroupedGemmFwdOp(Op):
    """
    Grouped GEMM with configurable transpose modes.

    Supports four layouts via (transpose_a, transpose_b):
      - (False, True)  NT: A @ B^T -> C
      - (False, False)  NN: A @ B   -> C
      - (True,  False) TN: A^T @ B  -> C
      - (True,  True)  TT: A^T @ B^T -> C
    """

    def __init__(
        self,
        transpose_a: bool = False,
        transpose_b: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.batch_sum = None
        self.batch_count = None
        self.N = None
        self.K = None
        self.dtype = None
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    #: Selection asks both, and takes the specialised one where it applies.
    _KERNEL_KEYS = ("grouped_gemm_persistent_3wg_kernel", "grouped_gemm_kernel")

    @property
    def default_kernel_map(self) -> Dict:
        return {
            "grouped_gemm_kernel": GroupedGemmKernel,
            "grouped_gemm_persistent_3wg_kernel": GroupedGemmPersistent3WGKernel,
        }

    def _resolve_spec(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: torch.Tensor,
    ) -> tuple[int, int, int, int, torch.dtype, int | None]:
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("a and b must be CUDA tensors")
        if a.dtype != b.dtype:
            raise ValueError(f"a and b must have the same dtype, got {a.dtype} and {b.dtype}")
        if a.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"a.dtype must be float16 or bfloat16, got {a.dtype}")
        if batch_sizes.ndim != 1 or batch_offsets.ndim != 1 or batch_padded_offsets.ndim != 1:
            raise ValueError("batch metadata tensors must be 1D")
        batch_count = batch_sizes.shape[0]
        if batch_offsets.shape[0] != batch_count or batch_padded_offsets.shape[0] != batch_count:
            raise ValueError("batch metadata tensors must have matching lengths")
        if (
            batch_sizes.dtype != torch.int32
            or batch_offsets.dtype != torch.int32
            or batch_padded_offsets.dtype != torch.int32
        ):
            raise ValueError("batch metadata tensors must use int32 dtype")

        if not self.transpose_a:
            if a.ndim != 2 or b.ndim != 3:
                raise ValueError("GroupedGemmFwdOp expects 2D a and 3D b when transpose_a=False")
            batch_sum, k = a.shape
            if b.shape[0] != batch_count:
                raise ValueError(
                    f"b.shape[0] must match batch_count={batch_count}, got {b.shape[0]}"
                )
            if self.transpose_b:
                n, b_k = b.shape[1], b.shape[2]
            else:
                b_k, n = b.shape[1], b.shape[2]
            if b_k != k:
                raise ValueError(f"GroupedGemmFwdOp expected K={k}, got b K dimension {b_k}")
        else:
            if a.ndim != 2 or b.ndim != 2:
                raise ValueError("GroupedGemmFwdOp expects 2D a and b when transpose_a=True")
            batch_sum, n = a.shape
            if self.transpose_b:
                k, b_batch_sum = b.shape
            else:
                b_batch_sum, k = b.shape
            if b_batch_sum != batch_sum:
                raise ValueError(
                    f"GroupedGemmFwdOp expected b batch_sum dimension {batch_sum}, got {b_batch_sum}"
                )
        return batch_sum, batch_count, n, k, a.dtype, a.device.index

    def _get_kernel(
        self,
        batch_sum: int,
        batch_count: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> tuple[str, Kernel]:
        call = GroupedGemmCall(
            arch=get_sm_version(device_index),
            numel=batch_sum,
            num_experts=batch_count,
            n=n,
            k=k,
            dtype=dtype,
            transpose_a=self.transpose_a,
            transpose_b=self.transpose_b,
        )
        key_name = self.select_kernel_key(self._KERNEL_KEYS, call)
        kernel_cls = self.kernel_map[key_name]
        kwargs: Dict[str, object] = {"dtype": dtype, "tune": self.tune}
        if key_name == "grouped_gemm_kernel":
            kwargs["transpose_a"] = self.transpose_a
            kwargs["transpose_b"] = self.transpose_b
        key = (
            batch_sum,
            batch_count,
            n,
            k,
            dtype,
            device_index,
            self.transpose_a,
            self.transpose_b,
            self.tune,
            key_name,
        )

        return key_name, self.get_or_build_kernel(
            key_name,
            key=key,
            build=lambda: kernel_cls(batch_sum, batch_count, n, k, **kwargs),
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: torch.Tensor,
    ) -> torch.Tensor:
        batch_sum, batch_count, n, k, dtype, device_index = self._resolve_spec(
            a,
            b,
            batch_sizes,
            batch_offsets,
            batch_padded_offsets,
        )
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = n
        self.K = k
        self.dtype = dtype
        key_name, self.kernel = self._get_kernel(batch_sum, batch_count, n, k, dtype, device_index)
        if key_name == "grouped_gemm_persistent_3wg_kernel":
            # It reads the tight layout straight off batch_sizes / batch_offsets
            # and has no use for the padded ones.
            return self.kernel(a, b, batch_sizes, batch_offsets)
        return self.kernel(a, b, batch_sizes, batch_offsets, batch_padded_offsets)
