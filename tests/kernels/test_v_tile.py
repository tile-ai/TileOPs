"""V-tile width validation: no T.gemm B operand narrower than GEMM_MIN_N columns."""

import pytest

from tileops.kernels.gated_deltanet.gated_deltanet_fwd import _h_recurrence_tl
from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
    GatedDeltaNetPrefillFwdKernel,
    _prefill_h_recurrence_bthd_tl,
)
from tileops.kernels.gla.gla_fwd import _gla_fwd_h_kernel
from tileops.kernels.v_tile import GEMM_MIN_N, resolve_block_v

pytestmark = pytest.mark.smoke


def test_resolve_block_v_derivation() -> None:
    assert resolve_block_v(64, 0) == 64  # block_v <= 0 means no tiling
    assert resolve_block_v(64, 32) == 32


@pytest.mark.parametrize(
    "dim_v,block_v",
    [
        (64, 8),  # tile below the WGMMA minimum
        (8, 0),   # no tiling, but dim_v itself is below the minimum
        (64, 24),  # not a divisor of dim_v
    ],
)
def test_resolve_block_v_rejects_invalid(dim_v: int, block_v: int) -> None:
    with pytest.raises(ValueError):
        resolve_block_v(dim_v, block_v)


def test_h_recurrence_rejects_narrow_v_tile() -> None:
    with pytest.raises(ValueError, match=str(GEMM_MIN_N)):
        _h_recurrence_tl(1, 4, 128, 64, 64, 64, "float16", block_v=8)


def test_prefill_h_recurrence_bthd_rejects_narrow_v_tile() -> None:
    with pytest.raises(ValueError, match=str(GEMM_MIN_N)):
        _prefill_h_recurrence_bthd_tl(1, 4, 128, 64, 64, 64, "float16", block_v=8)


def test_gla_fwd_h_rejects_narrow_v_partition() -> None:
    with pytest.raises(ValueError, match=str(GEMM_MIN_N)):
        _gla_fwd_h_kernel(1, 128, 4, 64, 32, 64, "float16", num_v_partitions=4)


@pytest.mark.parametrize(
    "seq_len,chunk_size",
    [
        (128, 64),   # streams <= 4 branch: previously selected h_block_v = 8
        (256, 128),  # 1 < streams <= 8 branch: previously selected h_block_v = 8
    ],
)
def test_prefill_bhtd_default_config_never_selects_narrow_v_tile(
    seq_len: int, chunk_size: int
) -> None:
    kernel = GatedDeltaNetPrefillFwdKernel(
        1, 4, seq_len, chunk_size, 64, 64, layout="bhtd", dtype="float16"
    )
    block_v = kernel.config["h_block_v"]
    assert block_v <= 0 or block_v >= GEMM_MIN_N
