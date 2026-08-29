"""The facts of one GEMM call, as the op knows them after inferring ``(m, n, k)``."""

import dataclasses
from typing import Literal, Optional

import torch

from ..call_spec import CallSpec

__all__ = ["GemmCall"]


@dataclasses.dataclass(frozen=True)
class GemmCall(CallSpec):
    """One matmul, as the op knows it after inferring ``(m, n, k)``."""

    m: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
    trans_a: bool = False
    trans_b: bool = False

    @property
    def gemv_mode(self) -> Optional[Literal["lhs_row", "rhs_col"]]:
        """Which operand is the vector: ``"lhs_row"``, ``"rhs_col"``, or neither.

        ``a`` is a single row with ``b`` transposed, or ``b`` is a single column
        with neither transposed; the other two layouts have no GEMV form here.
        Two readers need this one fact -- ``GemvKernel.applies`` to claim the
        call, and the op to reshape the operand the kernel takes flat.
        """
        if self.trans_a:
            return None
        if self.m == 1 and self.trans_b:
            return "lhs_row"
        if self.n == 1 and not self.trans_b:
            return "rhs_col"
        return None

    @property
    def gemv_n(self) -> int:
        """Output elements the GEMV kernel produces, its ``n``.

        Raises:
            ValueError: The call has no GEMV form, so there is no such count.
        """
        if self.gemv_mode is None:
            raise ValueError(f"call has no GEMV form: {self}")
        return self.n if self.gemv_mode == "lhs_row" else self.m
