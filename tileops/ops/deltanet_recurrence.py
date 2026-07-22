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
    ) -> bool:
        if dtype not in (torch.float16, torch.bfloat16) or dim_k != 128 or dim_v != 128:
            return False
        return DeltaNetDecodeOp._raw_cuda_decode_arch_supported()

    def __init__(
        self,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = None
        self.heads = None
        self.dim_k = None
        self.dim_v = None
        self.dtype = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self._active_sig: Optional[tuple] = None
        self.kernel = None

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

    def _get_kernel(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, heads, dim_k, dim_v, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            use_raw_cuda_decode = self._should_use_raw_cuda_decode(dim_k, dim_v, dtype)
            if dtype == torch.float32:
                kernel_cls = self.kernel_map["DeltaNetDecodeFP32Kernel"]
            elif use_raw_cuda_decode:
                kernel_cls = self.kernel_map["DeltaNetDecodeRawCudaFlaStyleKernel"]
            else:
                kernel_cls = self.kernel_map["DeltaNetDecodeKernel"]
            self._kernel_cache[key] = kernel_cls(
                batch,
                heads,
                dim_k,
                dim_v,
                dtype=Kernel.dtype_to_str(dtype),
                tune=self.tune,
            )
        return self._kernel_cache[key]

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
        dtype = q.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {dtype}")
        for name, tensor in (("q", q), ("k", k), ("v", v), ("beta", beta), ("state", state)):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> None:
        if q.ndim != 3:
            raise ValueError("q must have shape [batch, heads, dim_k]")
        batch, heads, dim_k = q.shape
        if v.ndim != 3 or v.shape[:2] != (batch, heads):
            raise ValueError("v must have shape [batch, heads, dim_v]")
        dim_v = v.shape[2]
        q_shape = (batch, heads, dim_k)
        v_shape = (batch, heads, dim_v)
        beta_shape = (batch, heads)
        state_shape = (batch, heads, dim_k, dim_v)
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
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = q.dtype
        self.kernel = self._get_kernel(batch, heads, dim_k, dim_v, q.dtype, q.device.index)

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
        sig = (
            q.shape,
            k.shape,
            v.shape,
            beta.shape,
            state.shape,
            q.dtype,
            k.dtype,
            v.dtype,
            beta.dtype,
            state.dtype,
            q.device,
            k.device,
            v.device,
            beta.device,
            state.device,
            getattr(self, "tune", None),
        )
        if sig != getattr(self, "_active_sig", None):
            self._validate_dtypes(q, k, v, beta, state)
            self._validate_shapes(q, k, v, beta, state)
            self._active_sig = sig
        o, new_state = self.kernel(q, k, v, beta, state)
        self._validate_output_shapes(o, new_state)
        return o, new_state
