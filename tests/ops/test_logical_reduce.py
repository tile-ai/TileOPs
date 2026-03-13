"""Correctness tests for logical reduce ops (any, all).

Covers: AnyOp, AllOp.
Each op reduces along dim=-1 and returns bool dtype.
Uses exact match (torch.equal) for comparison.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class LogicalReduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bool, marks=pytest.mark.full),
                pytest.param(128, 512, torch.complex64, marks=pytest.mark.full),
                pytest.param(128, 512, torch.complex128, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-pow2 last dim
                pytest.param(128, 300, torch.float32, marks=pytest.mark.full),
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bool, marks=pytest.mark.full),
                pytest.param(128, 300, torch.complex64, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduceNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bool, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduce3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduce4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class LogicalReduce1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.float32, marks=pytest.mark.full),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(512, torch.bool, marks=pytest.mark.full),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# TestBase helpers
# ---------------------------------------------------------------------------


class LogicalReduceTest(TestBase):
    """Parameterized test helper for logical reduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        # Mix of zeros and non-zeros for meaningful logical testing
        if self.dtype == torch.bool:
            x = torch.randint(0, 2, (self.m, self.n), dtype=torch.bool, device="cuda")
            # Force some rows to be all-False for meaningful "any" tests
            if self.m > 4:
                x[0] = False
            # Force some rows to be all-True for "all" tests
            if self.m > 4:
                x[1] = True
        elif self.dtype in (torch.complex64, torch.complex128):
            real = torch.randn(self.m, self.n, dtype=torch.float32, device="cuda")
            imag = torch.randn(self.m, self.n, dtype=torch.float32, device="cuda")
            x = torch.complex(real, imag).to(self.dtype)
            # Force some rows to be all-zero (complex zero)
            if self.m > 4:
                x[0] = 0 + 0j
            # Force some rows to have all non-zero
            if self.m > 4:
                x[1] = 1 + 1j
        else:
            x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
            # Force some rows to be all-zero for meaningful "any" tests
            if self.m > 4:
                x[0] = 0.0
            # Force some rows to have all non-zero for "all" tests
            if self.m > 4:
                x[1] = 1.0
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_kind == "any":
            return x.bool().any(dim=-1)
        elif self.op_kind == "all":
            return x.bool().all(dim=-1)
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


# ---------------------------------------------------------------------------
# AnyOp tests
# ---------------------------------------------------------------------------


@LogicalReduceBasicFixture
def test_any_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyOp

    test = LogicalReduceTest(m, n, dtype, "any")
    op = AnyOp(M=m, N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@LogicalReduceNonContigFixture
def test_any_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyOp

    if dtype == torch.bool:
        x_full = torch.randint(0, 2, (m, n * 2), dtype=torch.bool, device="cuda")
    elif dtype in (torch.complex64, torch.complex128):
        real = torch.randn(m, n * 2, dtype=torch.float32, device="cuda")
        imag = torch.randn(m, n * 2, dtype=torch.float32, device="cuda")
        x_full = torch.complex(real, imag).to(dtype)
    else:
        x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = AnyOp(M=m, N=n, dtype=dtype)
    ref = x.contiguous().bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"non-contig any mismatch: {(y != ref).sum().item()}"


@LogicalReduce3DFixture
def test_any_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = AnyOp(M=M, N=hidden, dtype=dtype)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"3D any mismatch: {(y != ref).sum().item()}"


@LogicalReduce4DFixture
def test_any_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    M = b0 * b1 * b2
    op = AnyOp(M=M, N=n, dtype=dtype)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"4D any mismatch: {(y != ref).sum().item()}"


@LogicalReduce1DFixture
def test_any_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.any_op import AnyOp

    if dtype == torch.bool:
        x = torch.randint(0, 2, (n,), dtype=torch.bool, device="cuda")
    elif dtype in (torch.complex64, torch.complex128):
        real = torch.randn(n, dtype=torch.float32, device="cuda")
        imag = torch.randn(n, dtype=torch.float32, device="cuda")
        x = torch.complex(real, imag).to(dtype)
    else:
        x = torch.randn(n, dtype=dtype, device="cuda")
    op = AnyOp(M=1, N=n, dtype=dtype)
    ref = x.bool().any(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y.view_as(ref), ref), "1D any mismatch"


# ---------------------------------------------------------------------------
# AllOp tests
# ---------------------------------------------------------------------------


@LogicalReduceBasicFixture
def test_all_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllOp

    test = LogicalReduceTest(m, n, dtype, "all")
    op = AllOp(M=m, N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@LogicalReduceNonContigFixture
def test_all_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllOp

    if dtype == torch.bool:
        x_full = torch.randint(0, 2, (m, n * 2), dtype=torch.bool, device="cuda")
    elif dtype in (torch.complex64, torch.complex128):
        real = torch.randn(m, n * 2, dtype=torch.float32, device="cuda")
        imag = torch.randn(m, n * 2, dtype=torch.float32, device="cuda")
        x_full = torch.complex(real, imag).to(dtype)
    else:
        x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = AllOp(M=m, N=n, dtype=dtype)
    ref = x.contiguous().bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"non-contig all mismatch: {(y != ref).sum().item()}"


@LogicalReduce3DFixture
def test_all_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = AllOp(M=M, N=hidden, dtype=dtype)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"3D all mismatch: {(y != ref).sum().item()}"


@LogicalReduce4DFixture
def test_all_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    M = b0 * b1 * b2
    op = AllOp(M=M, N=n, dtype=dtype)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y, ref), f"4D all mismatch: {(y != ref).sum().item()}"


@LogicalReduce1DFixture
def test_all_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.all_op import AllOp

    if dtype == torch.bool:
        x = torch.randint(0, 2, (n,), dtype=torch.bool, device="cuda")
    elif dtype in (torch.complex64, torch.complex128):
        real = torch.randn(n, dtype=torch.float32, device="cuda")
        imag = torch.randn(n, dtype=torch.float32, device="cuda")
        x = torch.complex(real, imag).to(dtype)
    else:
        x = torch.randn(n, dtype=dtype, device="cuda")
    op = AllOp(M=1, N=n, dtype=dtype)
    ref = x.bool().all(dim=-1)
    y = op(x)
    assert y.dtype == torch.bool
    assert torch.equal(y.view_as(ref), ref), "1D all mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
