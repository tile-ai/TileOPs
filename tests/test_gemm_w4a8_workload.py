"""The W4A8 workload's reference weight order, checked nibble by nibble.

The layout is the one ``repack_w4a8_weight``'s docstring states. Tensors stay on the host
and no kernel runs.
"""

import pytest
import torch

from workloads.gemm import (
    repack_w4a8_weight,
    unpack_signed_int4,
    unrepack_w4a8_weight,
)


@pytest.mark.smoke
def test_repack_w4a8_weight_places_each_nibble() -> None:
    """Every nibble lands where the layout formula puts it, encoded, and comes back."""
    n, k = 112, 384  # one full 64-row block and three 16-row units past it
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    q = unpack_signed_int4(packed)

    prepacked = repack_w4a8_weight(packed)

    row = torch.arange(n)[:, None].expand(n, k)
    col = torch.arange(k)[None, :].expand(n, k)
    block, units = row // 64, torch.where(row // 64 < n // 64, 4, (n % 64) // 16)
    group, half, step, slot, t, i = (
        col // 128,
        col // 64 % 2,
        col // 32 % 2,
        col // 16 % 2,
        col // 4 % 4,
        col % 4,
    )
    offset = (
        block * 64 * k // 2
        + (group * 2 + half) * units * 512
        + (row % 64 // 16) * 512
        + (4 * (row % 8) + t) * 16
        + (2 * step + row % 16 // 8) * 4
        + (4 * slot + i) // 2
    )
    nibble = (prepacked.reshape(-1)[offset].to(torch.int16) >> (4 * (i % 2))) & 0xF
    code = torch.where(q > 0, q - 1, torch.where(q == 0, 7, 8 | (-q - 1)))
    assert torch.equal(nibble, code)
    assert torch.equal(unrepack_w4a8_weight(prepacked), packed)
