"""Tests for DropoutFwdOp.

Covers:
- Deterministic replay (same seed = same output)
- Statistical drop rate within 3 sigma for p in {0.1, 0.3, 0.5}
- Scale factor correctness: non-dropped elements x (1/(1-p))
- Edge cases: p=0 (identity), p=1 (all zeros), training=False (identity)
- Multi-dtype coverage
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from workloads.device import run_device
from workloads.elementwise import ElementwiseWorkload


class DropoutStatFixture(FixtureBase):
    """Fixture for statistical drop-rate tests.

    Uses 4M elements so that sigma is small enough for the 3-sigma bound
    to be robust (sigma ~ 2.3e-4 for p=0.5 at N=4M).
    """

    PARAMS = [
        (
            "n_total, dtype, p",
            [
                # Smoke: basic dropout
                pytest.param(4_000_000, torch.float16, 0.5, marks=pytest.mark.smoke),
                pytest.param(4_000_000, torch.bfloat16, 0.5, marks=pytest.mark.smoke),
                pytest.param(4_000_000, torch.float32, 0.5, marks=pytest.mark.smoke),
                # Full: required p values and additional dtypes
                pytest.param(4_000_000, torch.float16, 0.1, marks=pytest.mark.full),
                pytest.param(4_000_000, torch.float16, 0.3, marks=pytest.mark.full),
            ],
        ),
    ]


class DropoutScaleFixture(FixtureBase):
    """Fixture for scale-factor tests (does not need large N)."""

    PARAMS = [
        (
            "n_total, dtype, p",
            [
                pytest.param(1_000_000, torch.float16, 0.5, marks=pytest.mark.smoke),
                pytest.param(1_000_000, torch.bfloat16, 0.5, marks=pytest.mark.smoke),
                pytest.param(1_000_000, torch.float32, 0.5, marks=pytest.mark.smoke),
                pytest.param(1_000_000, torch.float16, 0.1, marks=pytest.mark.full),
                pytest.param(1_000_000, torch.float16, 0.3, marks=pytest.mark.full),
            ],
        ),
    ]


class DropoutDeterminismFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype, p",
            [
                pytest.param(1_000_000, torch.float16, 0.5, marks=pytest.mark.smoke),
                pytest.param(1_000_000, torch.float32, 0.3, marks=pytest.mark.smoke),
            ],
        ),
    ]


class DropoutEdgeCaseFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_000_000, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1_000_000, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@DropoutStatFixture
def test_dropout_statistical_rate(n_total: int, dtype: torch.dtype, p: float) -> None:
    """Verify the shared dropout scaling and sampling-rate contract."""
    from tileops.ops.elementwise.dropout import DropoutFwdOp

    x = torch.ones(n_total, dtype=dtype, device=run_device())
    op = DropoutFwdOp(p=p, seed=42)
    workload = ElementwiseWorkload("DropoutFwdOp", (x,), p=p)
    TestBase.check(workload, op, x)


@DropoutScaleFixture
def test_dropout_scale_factor(n_total: int, dtype: torch.dtype, p: float) -> None:
    """Verify non-dropped elements are scaled by 1/(1-p)."""
    from tileops.ops.elementwise.dropout import DropoutFwdOp

    x = torch.ones(n_total, dtype=dtype, device=run_device())
    op = DropoutFwdOp(p=p, seed=123)
    workload = ElementwiseWorkload("DropoutFwdOp", (x,), p=p)
    TestBase.check(workload, op, x)


@DropoutDeterminismFixture
def test_dropout_deterministic_replay(n_total: int, dtype: torch.dtype, p: float) -> None:
    """Same seed must produce identical output."""
    from tileops.ops.elementwise.dropout import DropoutFwdOp

    x = torch.randn(n_total, dtype=dtype, device=run_device())
    op1 = DropoutFwdOp(p=p, seed=777)
    op2 = DropoutFwdOp(p=p, seed=777)
    y1 = op1(x)
    y2 = op2(x)
    assert torch.equal(y1, y2), "Deterministic replay failed: same seed produced different outputs"


@DropoutDeterminismFixture
def test_dropout_different_seeds(n_total: int, dtype: torch.dtype, p: float) -> None:
    """Different seeds must produce different outputs (with overwhelming probability)."""
    from tileops.ops.elementwise.dropout import DropoutFwdOp

    x = torch.ones(n_total, dtype=dtype, device=run_device())
    op1 = DropoutFwdOp(p=p, seed=42)
    op2 = DropoutFwdOp(p=p, seed=99)
    y1 = op1(x)
    y2 = op2(x)
    assert not torch.equal(y1, y2), "Different seeds produced identical outputs"


@DropoutEdgeCaseFixture
def test_dropout_p0_identity(n_total: int, dtype: torch.dtype) -> None:
    """p=0 means no dropout: output equals input."""
    from tileops.ops.elementwise.dropout import DropoutFwdOp

    x = torch.randn(n_total, dtype=dtype, device=run_device())
    op = DropoutFwdOp(p=0.0, seed=42)
    workload = ElementwiseWorkload("DropoutFwdOp", (x,), p=0.0)
    TestBase.check(workload, op, x)


@DropoutEdgeCaseFixture
def test_dropout_p1_all_zeros(n_total: int, dtype: torch.dtype) -> None:
    """p=1 means all elements dropped: output is all zeros."""
    from tileops.ops.elementwise.dropout import DropoutFwdOp

    x = torch.randn(n_total, dtype=dtype, device=run_device())
    op = DropoutFwdOp(p=1.0, seed=42)
    workload = ElementwiseWorkload("DropoutFwdOp", (x,), p=1.0)
    TestBase.check(workload, op, x)


@DropoutEdgeCaseFixture
def test_dropout_training_false(n_total: int, dtype: torch.dtype) -> None:
    """training=False means identity pass-through regardless of p."""
    from tileops.ops.elementwise.dropout import DropoutFwdOp

    x = torch.randn(n_total, dtype=dtype, device=run_device())
    op = DropoutFwdOp(p=0.5, seed=42, training=False)
    workload = ElementwiseWorkload("DropoutFwdOp", (x,), p=0.5, training=False)
    TestBase.check(workload, op, x)


@DropoutEdgeCaseFixture
def test_dropout_preserves_shape(n_total: int, dtype: torch.dtype) -> None:
    """Output shape and dtype must match input."""
    from tileops.ops.elementwise.dropout import DropoutFwdOp

    shape = (100, n_total // 100)
    x = torch.randn(shape, dtype=dtype, device=run_device())
    op = DropoutFwdOp(p=0.3, seed=42)
    y = op(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert y.dtype == x.dtype, f"Dtype mismatch: {y.dtype} vs {x.dtype}"


# Regression: non-default kernel config
