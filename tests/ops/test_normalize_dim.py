"""CPU-only unit tests for ``normalize_dim``.

These tests pin the empty-dim ``[] / ()`` semantics at the helper-function
layer, so the behavioral contract is verifiable without invoking the GPU
kernel. The kernel-level integration tests in ``test_vector_norm.py``
continue to gate end-to-end correctness on supported architectures
(sm80/86/89/90).

Why a separate CPU-only layer:

- ``normalize_dim`` is the single point that translates ``dim=[] / dim=()``
  into "reduce over all dims". L1/L2/Inf vector-norm ops route their
  ``dim`` through this helper before reaching the kernel.
- Asserting equivalence with ``dim=None`` here proves the contract that
  upstream ``torch.linalg.vector_norm`` requires, regardless of which arch
  the kernel ultimately runs on.
"""

from __future__ import annotations

import pytest

from tileops.ops.reduction._multidim import normalize_dim

# ---------------------------------------------------------------------------
# Empty-dim semantics
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("empty_dim", [[], ()], ids=["list", "tuple"])
@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
def test_empty_dim_equivalent_to_none(empty_dim, ndim: int) -> None:
    """``dim=[]`` and ``dim=()`` reduce over all dims, matching ``dim=None``."""
    assert normalize_dim(empty_dim, ndim) == normalize_dim(None, ndim)
    assert normalize_dim(empty_dim, ndim) == list(range(ndim))


# ---------------------------------------------------------------------------
# Existing paths must remain intact (no regressions)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_none_returns_all_dims() -> None:
    assert normalize_dim(None, 4) == [0, 1, 2, 3]


@pytest.mark.smoke
def test_int_dim() -> None:
    assert normalize_dim(0, 3) == [0]
    assert normalize_dim(2, 3) == [2]


@pytest.mark.smoke
def test_negative_int_dim_normalized() -> None:
    assert normalize_dim(-1, 3) == [2]
    assert normalize_dim(-3, 3) == [0]


@pytest.mark.smoke
def test_list_of_ints_sorted() -> None:
    assert normalize_dim([2, 0], 3) == [0, 2]


@pytest.mark.smoke
def test_tuple_of_ints_sorted() -> None:
    assert normalize_dim((2, 0), 3) == [0, 2]


@pytest.mark.smoke
def test_negative_dims_in_list() -> None:
    assert normalize_dim([-1, 0], 3) == [0, 2]


@pytest.mark.smoke
def test_duplicate_dims_raise() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_dim([0, 0], 3)


@pytest.mark.smoke
def test_duplicate_via_negative_raise() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_dim([0, -3], 3)


@pytest.mark.smoke
def test_out_of_range_raises() -> None:
    with pytest.raises(IndexError):
        normalize_dim([3], 3)
    with pytest.raises(IndexError):
        normalize_dim([-4], 3)
