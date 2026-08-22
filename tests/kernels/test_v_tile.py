"""Narrow V-tiles must be rejected at kernel-build time.

The pinned tilelang compiles a ``block_v = 8`` h-recurrence silently; only
tilelang main rejects the resulting sub-16-column WGMMA B operand. Without
this test, removing the guard passes CI on the pinned image unnoticed.
"""

import pytest

from tileops.kernels.linear_attention.deltanet.deltanet_fwd import (
    _h_recurrence_tl as deltanet_h_recurrence,
)
from tileops.kernels.linear_attention.gated_deltanet.gated_deltanet_fwd import (
    _h_recurrence_tl as gated_h_recurrence,
)
from tileops.kernels.linear_attention.v_tile import GEMM_MIN_N

pytestmark = pytest.mark.smoke

#: Both delta-rule recurrences tile V the same way, so both owe the same guard.
_H_RECURRENCE_BUILDERS = [
    pytest.param(deltanet_h_recurrence, id="deltanet"),
    pytest.param(gated_h_recurrence, id="gated"),
]


@pytest.mark.parametrize("build", _H_RECURRENCE_BUILDERS)
def test_h_recurrence_rejects_previously_defaulted_narrow_v_tile(build) -> None:
    # The shape whose default config formerly selected block_v = 8.
    with pytest.raises(ValueError, match=str(GEMM_MIN_N)):
        build(1, 4, 128, 64, 64, 64, "float16", block_v=8)


@pytest.mark.parametrize("build", _H_RECURRENCE_BUILDERS)
def test_h_recurrence_rejects_a_v_tile_that_does_not_divide_dim_v(build) -> None:
    """A width that leaves a partial tile would silently drop the trailing columns.

    ``dim_v=48`` with ``block_v=32`` yields ``48 // 32 == 1`` tile, so the
    recurrence would write 32 of 48 columns and leave the rest uninitialized
    for the output projection to consume.
    """
    with pytest.raises(ValueError, match="divisible"):
        build(1, 4, 128, 64, 64, 48, "float16", block_v=32)
