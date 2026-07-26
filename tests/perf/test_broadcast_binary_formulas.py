"""Spec pins for the broadcast-binary roofline helpers (no CUDA build)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from tileops.perf import formulas


@dataclass
class _StubBinaryOp:
    a_numel: int
    b_numel: int
    N_total: int
    dtype: torch.dtype


@pytest.mark.smoke
def test_broadcast_binary_helper_no_broadcast():
    """When inputs share the output shape, a_numel == b_numel == N_total."""
    op = _StubBinaryOp(a_numel=1024, b_numel=1024, N_total=1024, dtype=torch.float32)
    flops, nbytes = formulas.add_fwd_roofline(op)
    assert flops == 2 * 1024
    # 2 reads (4 bytes each) + 1 write (4 bytes) per element
    assert nbytes == (1024 + 1024 + 1024) * 4


@pytest.mark.smoke
def test_broadcast_binary_helper_bool_output_byte_accounting():
    """Comparison ops emit a 1-byte output regardless of input dtype."""
    op = _StubBinaryOp(a_numel=1024, b_numel=1024, N_total=1024, dtype=torch.float32)
    flops, nbytes = formulas.eq_fwd_roofline(op)
    assert flops == 1024
    # 2 fp32 reads + 1 bool write
    assert nbytes == (1024 + 1024) * 4 + 1024
