"""Unit tests for broadcast-binary roofline helpers in tileops.perf.formulas.

These exercise the (flops, bytes) accounting for the 21 broadcast-binary
manifest entries that switched from inline mode to ``roofline.func``. The
tests use a lightweight stub that mirrors the ``BinaryOp`` attribute
surface (``a_numel``, ``b_numel``, ``N_total``, ``dtype``) so the helpers
can be exercised without a CUDA build.
"""

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


def _expected(
    a_numel: int,
    b_numel: int,
    n_total: int,
    elem_bytes: int,
    flops_per_elem: int,
    *,
    bool_output: bool,
) -> tuple[int, int]:
    out_elem_bytes = 1 if bool_output else elem_bytes
    flops = flops_per_elem * n_total
    nbytes = (a_numel + b_numel) * elem_bytes + n_total * out_elem_bytes
    return flops, nbytes


# (helper, flops_per_elem, bool_output)
_ARITHMETIC_CASES = [
    (formulas.add_fwd_roofline, 2, False),
    (formulas.sub_fwd_roofline, 2, False),
    (formulas.mul_fwd_roofline, 1, False),
    (formulas.div_fwd_roofline, 1, False),
    (formulas.remainder_fwd_roofline, 4, False),
    (formulas.pow_fwd_roofline, 3, False),
    (formulas.floor_divide_fwd_roofline, 2, False),
    (formulas.lerp_fwd_roofline, 3, False),
    (formulas.maximum_fwd_roofline, 1, False),
    (formulas.minimum_fwd_roofline, 1, False),
    (formulas.bitwise_and_fwd_roofline, 1, False),
    (formulas.bitwise_or_fwd_roofline, 1, False),
    (formulas.bitwise_xor_fwd_roofline, 1, False),
]

_BOOL_CASES = [
    (formulas.eq_fwd_roofline, 1, True),
    (formulas.ne_fwd_roofline, 1, True),
    (formulas.gt_fwd_roofline, 1, True),
    (formulas.lt_fwd_roofline, 1, True),
    (formulas.ge_fwd_roofline, 1, True),
    (formulas.le_fwd_roofline, 1, True),
    (formulas.logical_and_fwd_roofline, 3, True),
    (formulas.logical_or_fwd_roofline, 3, True),
]


@pytest.mark.smoke
@pytest.mark.parametrize(("helper", "flops_per_elem", "bool_output"),
                         _ARITHMETIC_CASES + _BOOL_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_broadcast_binary_helper_matches_formula(helper, flops_per_elem, bool_output,
                                                 dtype):
    # broadcast (4096, 1) with (1, 4096) -> (4096, 4096)
    a_numel = 4096
    b_numel = 4096
    n_total = 4096 * 4096
    op = _StubBinaryOp(a_numel=a_numel, b_numel=b_numel, N_total=n_total, dtype=dtype)
    flops, nbytes = helper(op)
    expected_flops, expected_bytes = _expected(
        a_numel, b_numel, n_total, dtype.itemsize, flops_per_elem,
        bool_output=bool_output,
    )
    assert flops == expected_flops
    assert nbytes == expected_bytes
    assert isinstance(flops, int)
    assert isinstance(nbytes, int)


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
