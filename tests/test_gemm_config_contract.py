"""Host-side validation of cooperative GEMM epilogue chunk dimensions."""

import pytest

from tileops.kernels.gemm.dense import _gemm_coop2_kernel

pytestmark = pytest.mark.smoke


def test_dense_coop2_validates_epilogue_chunks_before_compilation() -> None:
    """Whole aligned chunks are legal; chunks that drop tile columns are rejected."""
    columns = 176
    factory = _gemm_coop2_kernel(
        256,
        2 * columns,
        128,
        False,
        True,
        "bfloat16",
        sm_count=132,
        consumer_bar_wg0=1,
        consumer_bar_wg1=2,
    )
    for chunk in range(-1, columns + 2):
        config = dict(block_n=columns, block_k=64, num_stages=4, group_size_m=4, stage_n=chunk)
        valid = chunk <= 0 or (columns % chunk == 0 and chunk % 8 == 0)
        if valid:
            assert factory.get_tir(**config) is not None, config
        else:
            with pytest.raises(ValueError, match="stage_n must divide block_n"):
                factory.get_tir(**config)
