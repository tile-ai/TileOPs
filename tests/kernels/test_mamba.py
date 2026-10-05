"""Mamba kernels validate the arguments they are called with."""

import pytest
import torch


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_ssd_chunk_cumsum_fwd_missing_bias_raises():
    """SSDChunkCumsumFwdKernel must raise when has_dt_bias=True but dt_bias is None."""
    from tileops.kernels.mamba import SSDChunkCumsumFwdKernel

    kernel = SSDChunkCumsumFwdKernel(
        batch=1,
        num_chunks=2,
        chunk_len=64,
        n_heads=4,
        seq_len=128,
        has_dt_bias=True,
    )
    dt = torch.randn(1, 128, 4, dtype=torch.float32, device="cuda")
    A = -torch.rand(4, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="dt_bias is required"):
        kernel(dt, A, dt_bias=None)
