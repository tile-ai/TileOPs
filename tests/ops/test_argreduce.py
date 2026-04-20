"""Correctness tests for argreduce ops (argmax, argmin).

Covers: ArgmaxFwdOp, ArgminFwdOp.
Each op reduces along a configurable dim and returns int64 indices.
Uses exact match (torch.equal) instead of allclose.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from workloads.argreduce import ArgmaxTest as _ArgmaxWorkload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ArgreduceBasicFixture(FixtureBase):
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


class ArgreduceTest(_ArgmaxWorkload, TestBase):
    """Parameterized test helper for argreduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_kind == "argmax":
            return x.argmax(dim=-1)
        elif self.op_kind == "argmin":
            return x.argmin(dim=-1)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


def _exact_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact match comparison using torch.equal."""
    assert output.dtype == torch.int64, f"Expected int64, got {output.dtype}"
    assert output_ref.dtype == torch.int64, f"Expected ref int64, got {output_ref.dtype}"
    assert torch.equal(output, output_ref), (
        f"Indices mismatch.\n"
        f"  output:     {output[:10]}...\n"
        f"  output_ref: {output_ref[:10]}...\n"
        f"  mismatches: {(output != output_ref).sum().item()} / {output.numel()}"
    )


# ---------------------------------------------------------------------------
# ArgmaxFwdOp tests
# ---------------------------------------------------------------------------


@ArgreduceBasicFixture
def test_argmax_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxFwdOp

    test = ArgreduceTest(m, n, dtype, "argmax")
    op = ArgmaxFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


# ---------------------------------------------------------------------------
# ArgminFwdOp tests
# ---------------------------------------------------------------------------


@ArgreduceBasicFixture
def test_argmin_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminFwdOp

    test = ArgreduceTest(m, n, dtype, "argmin")
    op = ArgminFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
