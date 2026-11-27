"""Checks of the state-pool contract shared by the pooled linear-attention ops."""

import torch

from tests.workload_test_base import _by_signature
from workloads.workload_base import manifest_call

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


def pooled_call(op_name: str, test_cls: type, dtype: torch.dtype, *, some=(), **row):
    """*test_cls* over a manifest call of *op_name* whose states live in a pool: sequence ``n``
    starts from slot ``state_indices[n]`` and leaves its final state there, every other slot
    untouched."""
    call = manifest_call(
        op_name,
        {"T": str(dtype).removeprefix("torch.")},
        some=["initial_state", "state_indices", *some],
        **row,
    )
    return test_cls(call)


def check_strided_pool_in_place(test, op) -> None:
    """Run *op* on *test*'s inputs with the pool, input 5, as a strided view: the call must
    advance that view and return it."""
    q, k, v, g, beta, pool, *rest = test.gen_inputs()
    strided = torch.empty((2 * pool.shape[0], *pool.shape[1:]), device=pool.device)[::2]
    strided.copy_(pool)

    def run(*args, **keywords):
        # The compared state is the caller's view, so a missed copy-back fails the check.
        o, final_state = op(*args, **keywords)
        assert final_state is args[5]
        return o, final_state

    test.check(op, q, k, v, g, beta, strided, *rest, runs=run)
