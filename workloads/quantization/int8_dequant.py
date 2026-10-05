from typing import Any

import torch

from workloads.device import run_device
from workloads.workload_base import WorkloadBase

_BLOCK_K = 128


def int8_dequant_per_tensor(
    q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """``INT8DequantPerTensorFwdOp``'s reference: one scale for the tensor."""
    _m, _k = q.shape
    return (q.float() * scale).to(out_dtype)


def int8_dequant_per_channel(
    q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """``INT8DequantPerChannelFwdOp``'s reference: one scale per row."""
    return (q.float() * scale[:, None]).to(out_dtype)


def int8_dequant_per_block(
    q: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """``INT8DequantPerBlockFwdOp``'s reference: one scale per 128 elements of a row.

    The last block of a row may be partial.
    """
    _m, k = q.shape
    return (q.float() * scale.repeat_interleave(_BLOCK_K, dim=1)[:, :k]).to(out_dtype)


class _INT8DequantWorkload(WorkloadBase):
    """An ``[m, k]`` INT8 matrix in the symmetric range [-127, 127] and its positive scales."""

    def __init__(self, m: int, k: int, out_dtype: torch.dtype):
        self.m = m
        self.k = k
        self.out_dtype = out_dtype

    @classmethod
    def from_call(cls, call: Any) -> "_INT8DequantWorkload":
        """The workload of one manifest call of the op this class is named for."""
        return cls(call.ix["M"], call.ix["K"], getattr(torch, call.params["out_dtype"]))

    def scale_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = torch.randint(-127, 128, (self.m, self.k), dtype=torch.int8, device=run_device())
        scale = torch.rand(self.scale_shape(), dtype=torch.float32, device=run_device())
        return q, scale * 0.01 + 1e-4

    def ref_program(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return self.reference(q, scale, self.out_dtype)


class INT8DequantPerTensorWorkload(_INT8DequantWorkload):
    def scale_shape(self) -> tuple[int, ...]:
        return (1,)

    reference = staticmethod(int8_dequant_per_tensor)


class INT8DequantPerChannelWorkload(_INT8DequantWorkload):
    def scale_shape(self) -> tuple[int, ...]:
        return (self.m,)

    reference = staticmethod(int8_dequant_per_channel)


class INT8DequantPerBlockWorkload(_INT8DequantWorkload):
    def scale_shape(self) -> tuple[int, ...]:
        return (self.m, -(-self.k // _BLOCK_K))

    reference = staticmethod(int8_dequant_per_block)
