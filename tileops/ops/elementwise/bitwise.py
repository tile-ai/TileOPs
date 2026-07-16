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

from ._base import BinaryOp, UnaryOp


class _BoolStorageBitwiseBinaryOp(BinaryOp):
    """Binary bitwise op with a uint8-backed fast path for bool tensors."""

    _bool_storage = False
    bool_storage_kernel_cls = None

    @property
    def default_kernel_map(self):
        kernel_map = {self._op_name: self.kernel_cls}
        if self.bool_storage_kernel_cls is not None:
            kernel_map[f"{self._op_name}_bool_storage"] = self.bool_storage_kernel_cls
        return kernel_map

    def _build_kernel_instance(
        self, coalesced_shape, a_strides, b_strides, strategy, tune,
    ):
        self._bool_storage = (
            self.dtype == torch.bool and self.bool_storage_kernel_cls is not None
        )
        if self._bool_storage:
            return self.kernel_map[f"{self._op_name}_bool_storage"](
                self.N_total, torch.uint8, coalesced_shape, a_strides, b_strides,
                self.a_numel, self.b_numel, strategy=strategy, tune=tune,
            )
        return super()._build_kernel_instance(
            coalesced_shape, a_strides, b_strides, strategy, tune,
        )

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002
        other: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(self, "_bool_storage", False):
            result = self.kernel(
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
