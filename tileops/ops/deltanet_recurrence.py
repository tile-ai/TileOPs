from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.deltanet_recurrence import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

from .op_base import Op

__all__ = ["DeltaNetDecodeOp"]


class DeltaNetDecodeOp(Op):
    """DeltaNet decode (single-step recurrence, ungated).

    Computes one step of the delta rule (no gate):
        v_new = beta * (v - S @ k)
        o     = S @ q + (q . k) * v_new
        S_new = S + outer(k, v_new)

    Layout: BHD (batch, head, dim).
    Supports float32, float16, and bfloat16 with fp32 accumulation.

    For fp32 dtype, dispatches to a dedicated FP32 kernel that uses
    element-wise matvec instead of T.gemm to avoid TF32 mantissa truncation.
    """

    @staticmethod
    def _raw_cuda_decode_arch_supported() -> bool:
        try:
            sm_version = get_sm_version()
        except Exception:
            return False
        return sm_version in DeltaNetDecodeRawCudaFlaStyleKernel.supported_archs

    @staticmethod
    def _should_use_raw_cuda_decode(
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        tune: bool,
    ) -> bool:
        if tune or dtype not in (torch.float16, torch.bfloat16) or dim_k != 128 or dim_v != 128:
            return False
        return DeltaNetDecodeOp._raw_cuda_decode_arch_supported()

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)

        # Dispatch:
        #   fp32 -> FP32 kernel (no TF32)
        #   fp16/bf16 DK=DV=128 on Hopper -> raw CUDA warp-per-Vtile kernel
        #   other fp16/bf16 shapes -> default TileLang kernel
        if dtype == torch.float32:
            kernel_cls = self.kernel_map["DeltaNetDecodeFP32Kernel"]
        elif self._should_use_raw_cuda_decode(dim_k, dim_v, dtype, tune):
            kernel_cls = self.kernel_map["DeltaNetDecodeRawCudaFlaStyleKernel"]
        else:
            kernel_cls = self.kernel_map["DeltaNetDecodeKernel"]
        kernel_dtype = Kernel.dtype_to_str(dtype)
        self.kernel = kernel_cls(
            batch, heads, dim_k, dim_v,
            dtype=kernel_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernels = {
            "DeltaNetDecodeKernel": DeltaNetDecodeKernel,
            "DeltaNetDecodeFP32Kernel": DeltaNetDecodeFP32Kernel,
        }
        if self._raw_cuda_decode_arch_supported():
            kernels["DeltaNetDecodeRawCudaFlaStyleKernel"] = (
                DeltaNetDecodeRawCudaFlaStyleKernel
            )
        return kernels

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, beta_shape
        return {
            "o": (q_shape[0], q_shape[1], v_shape[-1]),
            "new_state": state_shape,
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        for name, tensor in (("q", q), ("k", k), ("v", v), ("beta", beta), ("state", state)):
            if tensor.dtype != self.dtype:
                raise ValueError(f"{name}.dtype must be {self.dtype}, got {tensor.dtype}")

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        q_shape = (self.batch, self.heads, self.dim_k)
        v_shape = (self.batch, self.heads, self.dim_v)
        beta_shape = (self.batch, self.heads)
        state_shape = (self.batch, self.heads, self.dim_k, self.dim_v)
        expected_shapes = (
            ("q", q, q_shape),
            ("k", k, q_shape),
            ("v", v, v_shape),
            ("beta", beta, beta_shape),
            ("state", state, state_shape),
        )
        for name, tensor, expected in expected_shapes:
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
                )
        if not all(tensor.is_cuda for tensor in (q, k, v, beta, state)):
            raise ValueError("q, k, v, beta, and state must be CUDA tensors")

    def _validate_output_shapes(
        self,
        o: torch.Tensor,
        new_state: torch.Tensor,
    ) -> None:
        o_shape = (self.batch, self.heads, self.dim_v)
        state_shape = (self.batch, self.heads, self.dim_k, self.dim_v)
        if tuple(o.shape) != o_shape:
            raise ValueError(f"o must have shape {o_shape}, got {tuple(o.shape)}")
        if tuple(new_state.shape) != state_shape:
            raise ValueError(
                f"new_state must have shape {state_shape}, got {tuple(new_state.shape)}"
            )

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import deltanet_decode_roofline

        return deltanet_decode_roofline(self)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_dtypes(q, k, v, beta, state)
        self._validate_shapes(q, k, v, beta, state)
        o, new_state = self.kernel(q, k, v, beta, state)
        self._validate_output_shapes(o, new_state)
        return o, new_state
