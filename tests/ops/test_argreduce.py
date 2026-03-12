"""Correctness tests for argreduce ops (argmax, argmin).

Covers: ArgmaxOp, ArgminOp.
Each op reduces along dim=-1 and returns int64 indices.
Uses exact match (torch.equal) instead of allclose.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ArgreduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned N (non-pow2 last dim)
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class ArgreduceNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class Argreduce3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class Argreduce4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class Argreduce1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.float32, marks=pytest.mark.full),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# TestBase helpers
# ---------------------------------------------------------------------------


class ArgreduceTest(TestBase):
    """Parameterized test helper for argreduce ops."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

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
# ArgmaxOp tests
# ---------------------------------------------------------------------------


@ArgreduceBasicFixture
def test_argmax_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxOp

    test = ArgreduceTest(m, n, dtype, "argmax")
    op = ArgmaxOp(M=m, N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@ArgreduceNonContigFixture
def test_argmax_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = ArgmaxOp(M=m, N=n, dtype=dtype)
    ref = x.contiguous().argmax(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig argmax mismatch: {(y != ref).sum().item()}"


@Argreduce3DFixture
def test_argmax_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = ArgmaxOp(M=M, N=hidden, dtype=dtype)
    ref = x.argmax(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D argmax mismatch: {(y != ref).sum().item()}"


@Argreduce4DFixture
def test_argmax_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    M = b0 * b1 * b2
    op = ArgmaxOp(M=M, N=n, dtype=dtype)
    ref = x.argmax(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D argmax mismatch: {(y != ref).sum().item()}"


@Argreduce1DFixture
def test_argmax_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmax import ArgmaxOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = ArgmaxOp(M=1, N=n, dtype=dtype)
    ref = x.argmax(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D argmax mismatch"


# ---------------------------------------------------------------------------
# ArgminOp tests
# ---------------------------------------------------------------------------


@ArgreduceBasicFixture
def test_argmin_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminOp

    test = ArgreduceTest(m, n, dtype, "argmin")
    op = ArgminOp(M=m, N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=_exact_compare)


@ArgreduceNonContigFixture
def test_argmin_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = ArgminOp(M=m, N=n, dtype=dtype)
    ref = x.contiguous().argmin(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"non-contig argmin mismatch: {(y != ref).sum().item()}"


@Argreduce3DFixture
def test_argmin_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = ArgminOp(M=M, N=hidden, dtype=dtype)
    ref = x.argmin(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"3D argmin mismatch: {(y != ref).sum().item()}"


@Argreduce4DFixture
def test_argmin_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    M = b0 * b1 * b2
    op = ArgminOp(M=M, N=n, dtype=dtype)
    ref = x.argmin(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y, ref), f"4D argmin mismatch: {(y != ref).sum().item()}"


@Argreduce1DFixture
def test_argmin_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.argmin import ArgminOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = ArgminOp(M=1, N=n, dtype=dtype)
    ref = x.argmin(dim=-1)
    y = op(x)
    assert y.dtype == torch.int64
    assert torch.equal(y.view_as(ref), ref), "1D argmin mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
