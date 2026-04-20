"""Correctness tests for logical reduce ops (any, all, count_nonzero).

Covers: AnyFwdOp, AllFwdOp, CountNonzeroFwdOp.
any/all reduce along the configured dim and return bool dtype.
count_nonzero reduces along the configured dim and returns int64 dtype.
Uses exact match (torch.equal) for comparison.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from workloads.logical_reduce import AnyTest as _AnyWorkload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class LogicalReduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bool, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.int32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.int64, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.complex64, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.complex128, marks=pytest.mark.smoke),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# TestBase helpers — inherit gen_inputs() from workload classes
# ---------------------------------------------------------------------------


class LogicalReduceTest(_AnyWorkload, TestBase):
    """Parameterized test helper for logical reduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_kind == "any":
            return x.bool().any(dim=-1)
        elif self.op_kind == "all":
            return x.bool().all(dim=-1)
        elif self.op_kind == "count_nonzero":
            return torch.count_nonzero(x, dim=-1).to(torch.int64)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


def _exact_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact match comparison using torch.equal."""
    assert output.dtype == torch.bool, f"Expected bool dtype, got {output.dtype}"
    assert output_ref.dtype == torch.bool, f"Expected ref bool dtype, got {output_ref.dtype}"
    assert torch.equal(output, output_ref), (
        f"Bool mismatch.\n"
        f"  output:     {output[:10]}...\n"
        f"  output_ref: {output_ref[:10]}...\n"
        f"  mismatches: {(output != output_ref).sum().item()} / {output.numel()}"
    )


def _exact_compare_int64(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact match comparison for int64 count_nonzero outputs."""
    assert output.dtype == torch.int64, f"Expected int64 dtype, got {output.dtype}"
    assert output_ref.dtype == torch.int64, f"Expected ref int64 dtype, got {output_ref.dtype}"
    assert torch.equal(output, output_ref), (
        f"Int64 mismatch.\n"
        f"  output:     {output[:10]}...\n"
        f"  output_ref: {output_ref[:10]}...\n"
        f"  mismatches: {(output != output_ref).sum().item()} / {output.numel()}"
    )


def _make_random_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Create random inputs across all logical-reduce supported dtype families."""
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-3, 4, shape, dtype=dtype, device="cuda")
    if dtype in (torch.complex64, torch.complex128):
        real = torch.randn(shape, dtype=torch.float32, device="cuda")
        imag = torch.randn(shape, dtype=torch.float32, device="cuda")
        return torch.complex(real, imag).to(dtype)
    return torch.randn(shape, dtype=dtype, device="cuda")


def _make_noncontig_input(m: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a non-contiguous 2D tensor of shape (m, n*2) for slicing tests."""
    return _make_random_input((m, n * 2), dtype)


def _make_1d_input(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a 1D tensor of shape (n,) for 1D tests."""
    return _make_random_input((n,), dtype)


def _make_nd_input(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Create an N-D tensor for dim/keepdim tests."""
    return _make_random_input(shape, dtype)


# ---------------------------------------------------------------------------
# AnyFwdOp tests
# ---------------------------------------------------------------------------


@LogicalReduceBasicFixture
def test_any_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyFwdOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


# ---------------------------------------------------------------------------
# AllFwdOp tests
# ---------------------------------------------------------------------------


@LogicalReduceBasicFixture
def test_all_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllFwdOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


# ---------------------------------------------------------------------------
# CountNonzeroFwdOp tests
# ---------------------------------------------------------------------------


@LogicalReduceBasicFixture
def test_count_nonzero_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    test = LogicalReduceTest(m, n, dtype, "count_nonzero")
    op = CountNonzeroFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare_int64)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
