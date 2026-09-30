"""Element-wise comparison ops (output bool)."""

from typing import ClassVar, Mapping

from tileops.kernels.elementwise import (
    AlwaysFalseFwdKernel,
    AlwaysTrueFwdKernel,
    EqBoolStorageFwdKernel,
    EqFwdKernel,
    GeBoolStorageFwdKernel,
    GeFwdKernel,
    GtBoolStorageFwdKernel,
    GtFwdKernel,
    IsfiniteFwdKernel,
    IsinfFwdKernel,
    IsnanFwdKernel,
    LeBoolStorageFwdKernel,
    LeFwdKernel,
    LtBoolStorageFwdKernel,
    LtFwdKernel,
    NeBoolStorageFwdKernel,
    NeFwdKernel,
)
from tileops.kernels.elementwise.call_spec import (
    BinaryPredicateFwdInterface,
    UnaryPredicateFwdInterface,
)
from tileops.kernels.kernel_base import KernelInterface
from tileops.ops.elementwise._base import ELEMENTWISE, BinaryOp, UnaryOp


class EqFwdOp(BinaryOp):
    """Element-wise equality with broadcast: y = (a == b)."""

    kernel_types = {"eq": EqFwdKernel, "eq_bool": EqBoolStorageFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BinaryPredicateFwdInterface
    }


class NeFwdOp(BinaryOp):
    """Element-wise not-equal with broadcast: y = (a != b)."""

    kernel_types = {"ne": NeFwdKernel, "ne_bool": NeBoolStorageFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BinaryPredicateFwdInterface
    }


class GtFwdOp(BinaryOp):
    """Element-wise greater-than with broadcast: y = (a > b)."""

    kernel_types = {"gt": GtFwdKernel, "gt_bool": GtBoolStorageFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BinaryPredicateFwdInterface
    }


class LtFwdOp(BinaryOp):
    """Element-wise less-than with broadcast: y = (a < b)."""

    kernel_types = {"lt": LtFwdKernel, "lt_bool": LtBoolStorageFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BinaryPredicateFwdInterface
    }


class GeFwdOp(BinaryOp):
    """Element-wise greater-equal with broadcast: y = (a >= b)."""

    kernel_types = {"ge": GeFwdKernel, "ge_bool": GeBoolStorageFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BinaryPredicateFwdInterface
    }


class LeFwdOp(BinaryOp):
    """Element-wise less-equal with broadcast: y = (a <= b)."""

    kernel_types = {"le": LeFwdKernel, "le_bool": LeBoolStorageFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: BinaryPredicateFwdInterface
    }


class IsnanFwdOp(UnaryOp):
    """Element-wise isnan with bool output.

    Always False on integer / bool input (no NaN representation in those
    dtypes).
    """

    kernel_types = {"isnan": IsnanFwdKernel, "isnan_exact": AlwaysFalseFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: UnaryPredicateFwdInterface
    }


class IsinfFwdOp(UnaryOp):
    """Element-wise isinf with bool output.

    Always False on integer / bool input (no Inf representation in those
    dtypes).
    """

    kernel_types = {"isinf": IsinfFwdKernel, "isinf_exact": AlwaysFalseFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: UnaryPredicateFwdInterface
    }


class IsfiniteFwdOp(UnaryOp):
    """Element-wise isfinite with bool output.

    Always True on integer / bool input (every value in those dtypes is
    finite).
    """

    kernel_types = {"isfinite": IsfiniteFwdKernel, "isfinite_exact": AlwaysTrueFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        ELEMENTWISE: UnaryPredicateFwdInterface
    }
