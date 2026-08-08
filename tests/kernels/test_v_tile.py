"""Narrow V-tiles must be rejected at kernel-build time.

The pinned tilelang compiles a ``block_v = 8`` h-recurrence silently; only
tilelang main rejects the resulting sub-16-column WGMMA B operand. Without
this test, removing the guard passes CI on the pinned image unnoticed.
"""

import pytest

from tileops.kernels.gated_deltanet.gated_deltanet_fwd import _h_recurrence_tl
from tileops.kernels.v_tile import GEMM_MIN_N

pytestmark = pytest.mark.smoke


def test_h_recurrence_rejects_previously_defaulted_narrow_v_tile() -> None:
    # The shape whose default config formerly selected block_v = 8.
    with pytest.raises(ValueError, match=str(GEMM_MIN_N)):
        _h_recurrence_tl(1, 4, 128, 64, 64, 64, "float16", block_v=8)
