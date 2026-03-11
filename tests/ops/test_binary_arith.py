"""Tests for binary arithmetic elementwise ops (add) with broadcast.

Covers L1 smoke correctness, L3 broadcast patterns, and coalesce_broadcast_dims unit tests.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.elementwise import AddOp, coalesce_broadcast_dims

# ---------------------------------------------------------------------------
# coalesce_broadcast_dims unit tests
# ---------------------------------------------------------------------------


class CoalesceFixture(FixtureBase):
    PARAMS = [
        ("a_shape, b_shape, expected_ndim", [
            # same-shape: coalesces to 1D
            pytest.param((1024, 1024), (1024, 1024), 1, marks=pytest.mark.smoke),
            # bias-add: (B,S,D) + (1,1,D) -> 2 groups
            pytest.param((2, 512, 768), (1, 1, 768), 2, marks=pytest.mark.full),
            # row broadcast: (B,S,D) + (B,S,1) -> 2 groups
            pytest.param((2, 512, 768), (2, 512, 1), 2, marks=pytest.mark.full),
            # scalar: (M,N) + (1,1) -> 2 groups (M*N collapsed, 1 broadcast)
            pytest.param((1024, 1024), (1, 1), 1, marks=pytest.mark.full),
            # interleaved: (A,1,C) + (1,B,1) -> 3 groups
            pytest.param((4, 1, 8), (1, 8, 1), 3, marks=pytest.mark.full),
            # outer product: (M,1) + (1,N) -> 2 groups
            pytest.param((64, 1), (1, 128), 2, marks=pytest.mark.full),
        ]),
    ]


@CoalesceFixture
def test_coalesce_broadcast_dims(a_shape, b_shape, expected_ndim) -> None:
    """Verify coalesce output shape count matches expected coalesced ndim."""
    out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
        a_shape, b_shape,
    )
    # Verify output shape matches torch broadcast
    assert out_shape == torch.broadcast_shapes(a_shape, b_shape)
    # Verify coalesced ndim
    assert len(coalesced_shape) == expected_ndim, (
        f"Expected {expected_ndim} coalesced dims, got {len(coalesced_shape)}: "
        f"{coalesced_shape}"
    )
    # Verify strides have correct length
    assert len(a_strides) == len(coalesced_shape)
    assert len(b_strides) == len(coalesced_shape)


# ---------------------------------------------------------------------------
# Add op correctness tests
# ---------------------------------------------------------------------------


class AddSameShapeFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            # Smoke: fp16, 1M
            pytest.param(1_000_000, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_000_000, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(1_000_000, torch.float32, marks=pytest.mark.full),
            pytest.param(4_000_000, torch.float16, marks=pytest.mark.full),
        ]),
    ]


class AddSameShapeTest(TestBase):

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() + b.float()).to(a.dtype)


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@AddSameShapeFixture
def test_add_same_shape(n_total: int, dtype: torch.dtype) -> None:
    test = AddSameShapeTest(n_total, dtype)
    shape = (n_total,)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Broadcast pattern tests (L3)
# ---------------------------------------------------------------------------


class AddBroadcastFixture(FixtureBase):
    PARAMS = [
        ("a_shape, b_shape, dtype", [
            # bias-add: (B,S,D) + (1,1,D)
            pytest.param(
                (2, 512, 768), (1, 1, 768), torch.float16, marks=pytest.mark.smoke,
            ),
            # row broadcast: (B,S,D) + (B,S,1)
            pytest.param(
                (2, 512, 768), (2, 512, 1), torch.float16, marks=pytest.mark.full,
            ),
            # scalar broadcast: (M,N) + (1,1)
            pytest.param(
                (1024, 1024), (1, 1), torch.float16, marks=pytest.mark.full,
            ),
            # interleaved: (A,1,C) + (1,B,1)
            pytest.param(
                (4, 1, 8), (1, 8, 1), torch.float16, marks=pytest.mark.full,
            ),
        ]),
    ]


class AddBroadcastTest(TestBase):

    def __init__(self, a_shape: tuple, b_shape: tuple, dtype: torch.dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.a_shape, dtype=self.dtype, device="cuda")
        b = torch.randn(self.b_shape, dtype=self.dtype, device="cuda")
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() + b.float()).to(a.dtype)


@AddBroadcastFixture
def test_add_broadcast(a_shape, b_shape, dtype: torch.dtype) -> None:
    test = AddBroadcastTest(a_shape, b_shape, dtype)
    op = AddOp(a_shape=a_shape, b_shape=b_shape, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class AddStrategyFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype, strategy", [
            pytest.param(1_000_000, torch.float16, "direct", marks=pytest.mark.smoke),
            pytest.param(1_000_000, torch.float16, "explicit_parallel", marks=pytest.mark.full),
        ]),
    ]


@AddStrategyFixture
def test_add_strategies(n_total: int, dtype: torch.dtype, strategy: str) -> None:
    """Verify both binary strategies produce correct results."""
    test = AddSameShapeTest(n_total, dtype)
    shape = (n_total,)
    op = AddOp(a_shape=shape, b_shape=shape, dtype=dtype, strategy=strategy)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
