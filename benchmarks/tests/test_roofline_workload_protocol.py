"""Unit tests for the ``RooflineWorkload`` structural protocol.

These tests verify that ``roofline_vars`` accepts any object with ``shape``
and ``dtype`` (duck-typed via ``typing.Protocol``), not just ``WorkloadBase``
subclasses.
"""

import pytest
import torch

from benchmarks.benchmark import RooflineWorkload, roofline_vars


class _DuckWorkload:
    """Object with shape and dtype but NOT a WorkloadBase subclass."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


class _MissingDtype:
    """Object with shape only -- should NOT satisfy RooflineWorkload."""

    def __init__(self, shape: tuple):
        self.shape = shape


class _MissingShape:
    """Object with dtype only -- should NOT satisfy RooflineWorkload."""

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype


@pytest.mark.smoke
def test_roofline_vars_accepts_duck_typed_workload():
    """roofline_vars should work with any object that has shape and dtype."""
    w = _DuckWorkload((4, 8, 1024), torch.float16)
    result = roofline_vars(w)
    assert result["M"] == 4 * 8
    assert result["N"] == 1024
    assert result["elem_bytes"] == 2


@pytest.mark.smoke
def test_roofline_vars_with_1d_shape():
    """Single-dimension shape: M should be 1, N should be that dimension."""
    w = _DuckWorkload((512,), torch.bfloat16)
    result = roofline_vars(w)
    assert result["M"] == 1
    assert result["N"] == 512
    assert result["elem_bytes"] == 2


@pytest.mark.smoke
def test_roofline_workload_protocol_is_runtime_checkable():
    """RooflineWorkload should be runtime-checkable for isinstance() use."""
    good = _DuckWorkload((4, 8), torch.float32)
    bad_no_dtype = _MissingDtype((4, 8))
    bad_no_shape = _MissingShape(torch.float32)

    assert isinstance(good, RooflineWorkload)
    assert not isinstance(bad_no_dtype, RooflineWorkload)
    assert not isinstance(bad_no_shape, RooflineWorkload)
