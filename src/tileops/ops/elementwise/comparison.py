"""Element-wise comparison ops (output bool)."""

import torch

from tileops.kernels.elementwise import (
    EqFwdKernel,
    GeFwdKernel,
    GtFwdKernel,
    IsfiniteFwdKernel,
    IsinfFwdKernel,
    IsnanFwdKernel,
    LeFwdKernel,
    LtFwdKernel,
    NeFwdKernel,
)

from ._base import (
    _PREDICATE_FALLBACK_DTYPES,
    BinaryOp,
    _IntIdentityUnaryOp,
)


class EqFwdOp(BinaryOp):
    """Element-wise equality with broadcast: y = (a == b)."""

    kernel_types = {"eq": EqFwdKernel}


class NeFwdOp(BinaryOp):
    """Element-wise not-equal with broadcast: y = (a != b)."""

    kernel_types = {"ne": NeFwdKernel}


class GtFwdOp(BinaryOp):
    """Element-wise greater-than with broadcast: y = (a > b)."""

    kernel_types = {"gt": GtFwdKernel}


class LtFwdOp(BinaryOp):
    """Element-wise less-than with broadcast: y = (a < b)."""

    kernel_types = {"lt": LtFwdKernel}


class GeFwdOp(BinaryOp):
    """Element-wise greater-equal with broadcast: y = (a >= b)."""

    kernel_types = {"ge": GeFwdKernel}


class LeFwdOp(BinaryOp):
    """Element-wise less-equal with broadcast: y = (a <= b)."""

    kernel_types = {"le": LeFwdKernel}


class IsnanFwdOp(_IntIdentityUnaryOp):
    """Element-wise isnan with bool output.

    Always False on integer / bool input (no NaN representation in those
    dtypes).
    """

    kernel_types = {"isnan": IsnanFwdKernel}
    _fallback_dtypes = _PREDICATE_FALLBACK_DTYPES

    @staticmethod
    def _int_handler(input: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input.shape, dtype=torch.bool, device=input.device)


class IsinfFwdOp(_IntIdentityUnaryOp):
    """Element-wise isinf with bool output.

    Always False on integer / bool input (no Inf representation in those
    dtypes).
    """

    kernel_types = {"isinf": IsinfFwdKernel}
    _fallback_dtypes = _PREDICATE_FALLBACK_DTYPES

    @staticmethod
    def _int_handler(input: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input.shape, dtype=torch.bool, device=input.device)


class IsfiniteFwdOp(_IntIdentityUnaryOp):
    """Element-wise isfinite with bool output.

    Always True on integer / bool input (every value in those dtypes is
    finite).
    """

    kernel_types = {"isfinite": IsfiniteFwdKernel}
    _fallback_dtypes = _PREDICATE_FALLBACK_DTYPES

    @staticmethod
    def _int_handler(input: torch.Tensor) -> torch.Tensor:
        return torch.ones(input.shape, dtype=torch.bool, device=input.device)
