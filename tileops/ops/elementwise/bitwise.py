"""Element-wise bitwise ops."""

import torch

from tileops.kernels.elementwise import (
    BitwiseAndBoolStorageFwdKernel,
    BitwiseAndFwdKernel,
    BitwiseNotFwdKernel,
    BitwiseOrBoolStorageFwdKernel,
    BitwiseOrFwdKernel,
    BitwiseXorBoolStorageFwdKernel,
    BitwiseXorFwdKernel,
)

from ._base import BinaryOp, KernelEntry, UnaryOp, resolve_output_dtype


class _BoolStorageBitwiseBinaryOp(BinaryOp):
    """Binary bitwise op with a uint8-backed fast path for bool tensors."""

    bool_storage_kernel_cls = None

    @property
    def default_kernel_map(self):
        kernel_map = {self._op_name: self.kernel_cls}
        if self.bool_storage_kernel_cls is not None:
            kernel_map[f"{self._op_name}_bool_storage"] = self.bool_storage_kernel_cls
        return kernel_map

    def _build_entry(self, dtype: torch.dtype) -> KernelEntry:
        """bool operands run on a uint8 kernel; the mode travels with the entry."""
        bool_storage = dtype == torch.bool and self.bool_storage_kernel_cls is not None
        if not bool_storage:
            return super()._build_entry(dtype)
        return KernelEntry(
            kernel=self.kernel_map[f"{self._op_name}_bool_storage"](
                self.a_shape, self.b_shape, torch.uint8, tune=self.tune,
            ),
            compute_dtype=torch.uint8,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
            bool_storage=True,
        )

    def _eager_forward(
        self,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        self.dtype = input.dtype
        entry = self._entry(input.dtype)
        if entry.bool_storage:
            result = entry.kernel(
                input.contiguous().view(-1).view(torch.uint8),
                other.contiguous().view(-1).view(torch.uint8),
            )
            return result.view(torch.bool).reshape(self.out_shape)
        return super()._eager_forward(input, other)


class BitwiseAndFwdOp(_BoolStorageBitwiseBinaryOp):
    """Element-wise bitwise AND with broadcast: y = a & b."""

    _op_name = "bitwise_and"
    kernel_cls = BitwiseAndFwdKernel
    bool_storage_kernel_cls = BitwiseAndBoolStorageFwdKernel


class BitwiseOrFwdOp(_BoolStorageBitwiseBinaryOp):
    """Element-wise bitwise OR with broadcast: y = a | b."""

    _op_name = "bitwise_or"
    kernel_cls = BitwiseOrFwdKernel
    bool_storage_kernel_cls = BitwiseOrBoolStorageFwdKernel


class BitwiseXorFwdOp(_BoolStorageBitwiseBinaryOp):
    """Element-wise bitwise XOR with broadcast: y = a ^ b."""

    _op_name = "bitwise_xor"
    kernel_cls = BitwiseXorFwdKernel
    bool_storage_kernel_cls = BitwiseXorBoolStorageFwdKernel


class BitwiseNotFwdOp(UnaryOp):
    """Element-wise bitwise NOT (~x) for bool/integer inputs."""

    _op_name = "bitwise_not"
    kernel_cls = BitwiseNotFwdKernel
