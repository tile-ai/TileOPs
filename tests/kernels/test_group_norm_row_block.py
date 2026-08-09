"""Correctness tests for the group-norm kernels under a multi-row block.

Verifies both kernels match a PyTorch oracle when ``block_m > 1`` puts several
normalization rows in one block and the last block runs past M. ``block_m`` is
an autotune candidate but ``select_row_config`` pins the default to 1, so an
op-level test cannot reach these configs deterministically.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from tileops.kernels.norm import GroupNormKernel, GroupNormNoAffineKernel

pytestmark = pytest.mark.smoke

_ATOL = _RTOL = 1e-3  # fp16, matching the norm op tests


@pytest.mark.parametrize("n, c, spatial, g, block_m", [
    # M = 9 rows, D = 256: the tail row block runs past M with aligned columns.
    (3, 24, (4, 8), 3, 4),
    # M = 9 rows, D = 200: the tail row block and the column padding together.
    (3, 24, (5, 5), 3, 4),
])
def test_affine_multi_row_block(n: int, c: int, spatial: tuple, g: int,
                                block_m: int) -> None:
    """Per-channel affine stays correct when a row block runs past M."""
    dtype = torch.float16
    cpg = c // g
    m, d = n * g, cpg * math.prod(spatial)
    x = torch.randn((n, c, *spatial), dtype=dtype, device="cuda")
    weight = torch.randn(c, dtype=dtype, device="cuda")
    bias = torch.randn(c, dtype=dtype, device="cuda")

    kernel = GroupNormKernel(
        m, d, 1e-5, dtype, g, cpg,
        config={"block_m": block_m, "threads": 128},
    )
    y = kernel(x.reshape(m, d), weight, bias).reshape(x.shape)

    y_ref = F.group_norm(
        x.float(), g, weight=weight.float(), bias=bias.float(), eps=1e-5,
    ).to(dtype)
    assert torch.allclose(y, y_ref, atol=_ATOL, rtol=_RTOL), \
        f"max err: {(y - y_ref).abs().max()}"


@pytest.mark.parametrize("m, d, block_m", [
    (9, 256, 4),   # tail block, aligned columns
    (9, 200, 4),   # tail block and column padding together
    (3, 256, 4),   # M < block_m: the only block is a partial one
])
def test_no_affine_multi_row_block(m: int, d: int, block_m: int) -> None:
    """No-affine rows stay correct when a row block runs past M."""
    dtype = torch.float16
    x = torch.randn((m, d), dtype=dtype, device="cuda")

    kernel = GroupNormNoAffineKernel(
        m, d, 1e-5, dtype, config={"block_m": block_m, "threads": 128},
    )
    y = kernel(x)

    y_ref = F.group_norm(
        x.float().reshape(m, 1, d), 1, weight=None, bias=None, eps=1e-5,
    ).reshape(m, d).to(dtype)
    assert y.shape == x.shape, f"expected {tuple(x.shape)}, got {tuple(y.shape)}"
    assert torch.allclose(y, y_ref, atol=_ATOL, rtol=_RTOL), \
        f"max err: {(y - y_ref).abs().max()}"
