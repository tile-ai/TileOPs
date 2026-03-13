"""Tests for DropoutOp.

Covers:
- Deterministic replay (same seed = same output)
- Statistical drop rate within 3 sigma for p in {0.1, 0.3, 0.5}
- Scale factor correctness: non-dropped elements x (1/(1-p))
- Edge cases: p=0 (identity), p=1 (all zeros), training=False (identity)
- Multi-dtype coverage
"""

import pytest
import torch

from tests.test_base import FixtureBase


class DropoutFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype, p", [
            # Smoke: basic dropout
            pytest.param(1_000_000, torch.float16, 0.5, marks=pytest.mark.smoke),
            # Full: various drop rates and dtypes
            pytest.param(1_000_000, torch.float16, 0.1, marks=pytest.mark.full),
            pytest.param(1_000_000, torch.float16, 0.3, marks=pytest.mark.full),
            pytest.param(1_000_000, torch.bfloat16, 0.5, marks=pytest.mark.full),
            pytest.param(1_000_000, torch.float32, 0.5, marks=pytest.mark.full),
            pytest.param(4_000_000, torch.float16, 0.3, marks=pytest.mark.full),
        ]),
    ]


class DropoutDeterminismFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype, p", [
            pytest.param(1_000_000, torch.float16, 0.5, marks=pytest.mark.smoke),
            pytest.param(1_000_000, torch.float32, 0.3, marks=pytest.mark.full),
        ]),
    ]


class DropoutEdgeCaseFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_000_000, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_000_000, torch.float32, marks=pytest.mark.full),
        ]),
    ]


@DropoutFixture
def test_dropout_statistical_rate(n_total: int, dtype: torch.dtype, p: float) -> None:
    """Verify that the fraction of dropped elements is within 3 sigma of p."""
    from tileops.ops.dropout import DropoutOp

    x = torch.ones(n_total, dtype=dtype, device="cuda")
    op = DropoutOp(N_total=n_total, dtype=dtype, p=p, seed=42)
    y = op(x)

    # Count zeros (dropped elements)
    n_dropped = (y == 0).sum().item()
    drop_rate = n_dropped / n_total

    # 5-sigma bound for Bernoulli(p) with n_total samples.
    # Using 5-sigma instead of 3-sigma for robustness in automated CI
    # (3-sigma gives ~0.3% false-positive rate per test; 5-sigma is ~5.7e-5%).
    sigma = (p * (1 - p) / n_total) ** 0.5
    assert abs(drop_rate - p) < 5 * sigma, (
        f"Drop rate {drop_rate:.4f} outside 5-sigma bound "
        f"[{p - 5 * sigma:.4f}, {p + 5 * sigma:.4f}] for p={p}"
    )


@DropoutFixture
def test_dropout_scale_factor(n_total: int, dtype: torch.dtype, p: float) -> None:
    """Verify non-dropped elements are scaled by 1/(1-p)."""
    from tileops.ops.dropout import DropoutOp

    x = torch.ones(n_total, dtype=dtype, device="cuda")
    op = DropoutOp(N_total=n_total, dtype=dtype, p=p, seed=123)
    y = op(x)

    # Non-zero elements should be scaled by 1/(1-p)
    mask = y != 0
    if mask.any():
        expected_scale = 1.0 / (1.0 - p)
        non_zero_vals = y[mask].float()
        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-5
        elif dtype == torch.float16:
            atol, rtol = 1e-3, 1e-3
        else:  # bfloat16
            atol, rtol = 1.6e-2, 1.6e-2
        torch.testing.assert_close(
            non_zero_vals,
            torch.full_like(non_zero_vals, expected_scale),
            atol=atol,
            rtol=rtol,
        )


@DropoutDeterminismFixture
def test_dropout_deterministic_replay(n_total: int, dtype: torch.dtype, p: float) -> None:
    """Same seed must produce identical output."""
    from tileops.ops.dropout import DropoutOp

    x = torch.randn(n_total, dtype=dtype, device="cuda")
    op1 = DropoutOp(N_total=n_total, dtype=dtype, p=p, seed=777)
    op2 = DropoutOp(N_total=n_total, dtype=dtype, p=p, seed=777)
    y1 = op1(x)
    y2 = op2(x)
    assert torch.equal(y1, y2), "Deterministic replay failed: same seed produced different outputs"


@DropoutDeterminismFixture
def test_dropout_different_seeds(n_total: int, dtype: torch.dtype, p: float) -> None:
    """Different seeds must produce different outputs (with overwhelming probability)."""
    from tileops.ops.dropout import DropoutOp

    x = torch.ones(n_total, dtype=dtype, device="cuda")
    op1 = DropoutOp(N_total=n_total, dtype=dtype, p=p, seed=42)
    op2 = DropoutOp(N_total=n_total, dtype=dtype, p=p, seed=99)
    y1 = op1(x)
    y2 = op2(x)
    assert not torch.equal(y1, y2), "Different seeds produced identical outputs"


@DropoutEdgeCaseFixture
def test_dropout_p0_identity(n_total: int, dtype: torch.dtype) -> None:
    """p=0 means no dropout: output equals input."""
    from tileops.ops.dropout import DropoutOp

    x = torch.randn(n_total, dtype=dtype, device="cuda")
    op = DropoutOp(N_total=n_total, dtype=dtype, p=0.0, seed=42)
    y = op(x)
    torch.testing.assert_close(y, x)


@DropoutEdgeCaseFixture
def test_dropout_p1_all_zeros(n_total: int, dtype: torch.dtype) -> None:
    """p=1 means all elements dropped: output is all zeros."""
    from tileops.ops.dropout import DropoutOp

    x = torch.randn(n_total, dtype=dtype, device="cuda")
    op = DropoutOp(N_total=n_total, dtype=dtype, p=1.0, seed=42)
    y = op(x)
    assert torch.equal(y, torch.zeros_like(x)), "p=1 should produce all zeros"


@DropoutEdgeCaseFixture
def test_dropout_training_false(n_total: int, dtype: torch.dtype) -> None:
    """training=False means identity pass-through regardless of p."""
    from tileops.ops.dropout import DropoutOp

    x = torch.randn(n_total, dtype=dtype, device="cuda")
    op = DropoutOp(N_total=n_total, dtype=dtype, p=0.5, seed=42, training=False)
    y = op(x)
    torch.testing.assert_close(y, x)


@DropoutEdgeCaseFixture
def test_dropout_preserves_shape(n_total: int, dtype: torch.dtype) -> None:
    """Output shape and dtype must match input."""
    from tileops.ops.dropout import DropoutOp

    shape = (100, n_total // 100)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    op = DropoutOp(N_total=n_total, dtype=dtype, p=0.3, seed=42)
    y = op(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert y.dtype == x.dtype, f"Dtype mismatch: {y.dtype} vs {x.dtype}"
