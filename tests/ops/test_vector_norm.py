"""Correctness tests for vector norm ops (l1_norm, l2_norm, inf_norm).

Covers: L1NormFwdOp, L2NormFwdOp, InfNormFwdOp.
All norms reduce along a configurable dim and return the same dtype as input.
Uses torch.linalg.vector_norm as the reference implementation.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, allclose_compare
from workloads.vector_norm import L1NormTest as _L1NormWorkload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class VectorNormBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# TestBase helpers — inherit gen_inputs() from workload classes
# ---------------------------------------------------------------------------


# Map op_kind to the ord parameter for torch.linalg.vector_norm
_ORD_MAP = {"l1": 1, "l2": 2, "inf": float("inf")}


class VectorNormTest(_L1NormWorkload, TestBase):
    """Parameterized test helper for vector norm ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for reference, then cast back to input dtype
        ord_val = _ORD_MAP[self.op_kind]
        ref = torch.linalg.vector_norm(x.float(), ord=ord_val, dim=-1)
        return ref.to(self.dtype)


def _get_tolerances(dtype: torch.dtype):
    """Return (atol, rtol) for the given dtype."""
    if dtype == torch.float32:
        return 1e-5, 1e-5
    # fp16/bf16 have larger rounding errors
    return 1e-2, 1e-2


def _norm_compare(output: torch.Tensor, output_ref: torch.Tensor, atol: float, rtol: float):
    """Comparison with configurable tolerance."""
    allclose_compare(output, output_ref, atol=atol, rtol=rtol)


def _make_op(dtype: torch.dtype, op_kind: str, dim: int = -1, keepdim: bool = False):
    """Create the appropriate Op for the given op_kind."""
    from tileops.ops.reduction.inf_norm import InfNormFwdOp
    from tileops.ops.reduction.l1_norm import L1NormFwdOp
    from tileops.ops.reduction.l2_norm import L2NormFwdOp

    op_map = {
        "l1": L1NormFwdOp,
        "l2": L2NormFwdOp,
        "inf": InfNormFwdOp,
    }
    cls = op_map[op_kind]
    return cls(dtype=dtype, dim=dim, keepdim=keepdim)


# ---------------------------------------------------------------------------
# L1NormFwdOp tests
# ---------------------------------------------------------------------------


@VectorNormBasicFixture
def test_l1_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l1")
    op = _make_op(dtype, "l1")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@VectorNormBasicFixture
def test_l2_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "l2")
    op = _make_op(dtype, "l2")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@VectorNormBasicFixture
def test_inf_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = VectorNormTest(m, n, dtype, "inf")
    op = _make_op(dtype, "inf")
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
