"""Check that a pooled op neither reads nor writes a state for a negative slot."""

import torch

from tests.workload_test_base import _by_signature

# Rows of random states on each side of the pool; slot -1 addresses the last row before it.
_GUARD_ROWS = 2


def check_padded_slots_skipped(op, inputs, pool_at: int, slots_at: int, padded: list) -> None:
    """Run *op* on *inputs* with its pool between two guard bands, once as given and once with
    ``slots[i]`` set to -1 for each index ``i`` in *padded*.

    The padded run must leave the guard bands and the slots only the padded rows name as they
    were, and every other slot as the unpadded run leaves it.
    """
    inputs = list(inputs)
    call = _by_signature(op, op)
    pool, slots = inputs[pool_at], inputs[slots_at]
    held = torch.randn((pool.shape[0] + 2 * _GUARD_ROWS, *pool.shape[1:]), device=pool.device).to(
        pool.dtype
    )
    held[_GUARD_ROWS:-_GUARD_ROWS] = pool

    def run(slots: torch.Tensor) -> torch.Tensor:
        band = held.clone()
        inputs[pool_at], inputs[slots_at] = band[_GUARD_ROWS:-_GUARD_ROWS], slots
        call(*inputs)
        return band

    full = run(slots)
    mask = torch.zeros_like(slots, dtype=torch.bool)
    for index in padded:
        mask[index] = True
    expected = full.clone()
    kept = _GUARD_ROWS + slots[mask].long()
    expected[kept] = held[kept]
    torch.testing.assert_close(run(slots.masked_fill(mask, -1)), expected, rtol=0, atol=0)
