"""Element-wise logical ops (output bool)."""

from typing import ClassVar, Mapping

from tileops.kernels.elementwise import (
    LogicalAndBoolStorageFwdKernel,
    LogicalAndFwdKernel,
    LogicalNotBoolStorageFwdKernel,
    LogicalNotFwdKernel,
    LogicalOrBoolStorageFwdKernel,
    LogicalOrFwdKernel,
)
from tileops.kernels.elementwise.call_spec import (
    BinaryPredicateFwdInterface,
    UnaryPredicateFwdInterface,
)
from tileops.kernels.kernel_base import KernelInterface
from tileops.ops.elementwise._base import ELEMENTWISE, BinaryOp, UnaryOp


class LogicalAndFwdOp(BinaryOp):
    """Element-wise logical AND with broadcast using non-zero truthiness."""

    kernel_types = {
        "logical_and": LogicalAndFwdKernel,
        "logical_and_bool": LogicalAndBoolStorageFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BinaryPredicateFwdInterface
    }


class LogicalOrFwdOp(BinaryOp):
    """Element-wise logical OR with broadcast using non-zero truthiness."""

    kernel_types = {
        "logical_or": LogicalOrFwdKernel,
        "logical_or_bool": LogicalOrBoolStorageFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BinaryPredicateFwdInterface
    }


class LogicalNotFwdOp(UnaryOp):
    """Element-wise logical NOT with bool output."""

    kernel_types = {
        "logical_not": LogicalNotFwdKernel,
        "logical_not_bool": LogicalNotBoolStorageFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: UnaryPredicateFwdInterface
    }
