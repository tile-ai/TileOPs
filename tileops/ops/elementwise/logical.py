"""Element-wise logical ops (output bool)."""

import torch

from tileops.kernels.elementwise import (
    LogicalAndBoolStorageFwdKernel,
    LogicalAndFwdKernel,
    LogicalNotBoolStorageFwdKernel,
    LogicalNotFwdKernel,
    LogicalOrBoolStorageFwdKernel,
    LogicalOrFwdKernel,
)

from ._base import UnaryOp, _BoolOutputBinaryOp


class LogicalAndFwdOp(_BoolOutputBinaryOp):
    """Element-wise logical AND with broadcast using non-zero truthiness."""

    _op_name = "logical_and"
    kernel_cls = LogicalAndFwdKernel
    bool_storage_kernel_cls = LogicalAndBoolStorageFwdKernel


class LogicalOrFwdOp(_BoolOutputBinaryOp):
    """Element-wise logical OR with broadcast using non-zero truthiness."""

    _op_name = "logical_or"
    kernel_cls = LogicalOrFwdKernel
    bool_storage_kernel_cls = LogicalOrBoolStorageFwdKernel


class LogicalNotFwdOp(UnaryOp):
    """Element-wise logical NOT with bool output."""

    _op_name = "logical_not"
    kernel_cls = LogicalNotFwdKernel
    bool_storage_kernel_cls = LogicalNotBoolStorageFwdKernel

    @property
    def default_kernel_map(self):
        return {
            self._op_name: self.kernel_cls,
            f"{self._op_name}_bool_storage": self.bool_storage_kernel_cls,
        }

    def _build_kernel_instance(
        self,
        *,
        N_total: int,
        dtype: torch.dtype,
        strategy,
        tune: bool = False,
    ):
        self._bool_storage = dtype == torch.bool
        if self._bool_storage:
            return self.kernel_map[f"{self._op_name}_bool_storage"](
                N_total, torch.uint8, strategy=strategy, tune=tune,
            )
        return super()._build_kernel_instance(
            N_total=N_total, dtype=dtype, strategy=strategy, tune=tune,
        )

    def _resolve_output_dtype(self):
        if self._bool_storage:
            return torch.bool
        return super()._resolve_output_dtype()

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if self._bool_storage:
            orig_shape = input.shape
            flat = input.contiguous().view(-1).view(torch.uint8)
            return self.kernel(flat).view(torch.bool).reshape(orig_shape)
        return super()._eager_forward(input)
