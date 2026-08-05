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

from ._base import KernelEntry, UnaryOp, _BoolOutputBinaryOp


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

    def _build_entry(self, dtype: torch.dtype) -> KernelEntry:
        """bool runs on a uint8 kernel and returns bool."""
        if dtype != torch.bool:
            return super()._build_entry(dtype)
        return KernelEntry(
            kernel=self.kernel_map[f"{self._op_name}_bool_storage"](
                self.N_total, torch.uint8, tune=self.tune,
            ),
            compute_dtype=torch.uint8,
            output_dtype=torch.bool,
        )
