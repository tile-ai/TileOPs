"""Input values and recurrence bookkeeping shared by the linear-attention workloads.

A manifest call fixes shapes, dtypes and presence; the helpers here condition the values
the call does not determine, and resolve the sequences and state slots a reference reads.
"""

from collections.abc import Callable

import torch

from workloads.numerics import Custom, assert_close, assert_normalized_error, assert_rounded


def step_sizes(like: torch.Tensor) -> torch.Tensor:
    """Delta-rule step sizes in ``[0, 0.5)``, with *like*'s shape and dtype."""
    return torch.rand(like.shape, device=like.device).to(like.dtype) * 0.5


def log_gates(like: torch.Tensor) -> torch.Tensor:
    """Log-space forget gates in ``(-1, 0]``, with *like*'s shape and dtype."""
    return -torch.rand(like.shape, device=like.device).to(like.dtype)


def small(t: torch.Tensor | None, scale: float = 0.1) -> torch.Tensor | None:
    """*t* times *scale*; ``None`` stays ``None``."""
    return None if t is None else t * scale


def recurrence_verification(
    tol: float, state_tol: float, *, single_step: float | None = None, normalized: bool = False
) -> Custom:
    """``(o, final_state)`` within atol *tol* and *state_tol*, rtol *tol*; *normalized* adds
    the normalized-error bound, which rejects a cleared low-amplitude output.

    With *single_step*, one step keeps its state in FP32: the output may take one adjacent
    stored value after two FP32 reduction orders, then both meet *single_step*.
    """
    if single_step is not None:

        def validate_step(got, expected):
            assert_rounded(got[0], expected[0], atol=single_step)
            assert_close(got[1], expected[1], atol=single_step, rtol=single_step)

        return Custom(validate_step, "single-step FP32 state and one-rounding-unit output")

    def validate(got, expected):
        assert_close(got[0], expected[0], atol=tol, rtol=tol)
        assert_close(got[1], expected[1], atol=state_tol, rtol=tol)
        if normalized:
            assert_normalized_error(got, expected)

    return Custom(validate, "output and accumulated FP32 state")


def sequences(
    q: torch.Tensor, cu_seqlens: torch.Tensor | None, cu_seqlens_cpu: torch.Tensor | None
) -> tuple[list[int], list[int], list[int]]:
    """The batch row, first token and length of every sequence of a ``[B, T, ...]`` call.

    Without offsets each batch row is one sequence; with them, row 0 packs one sequence per
    offset span.
    """
    if cu_seqlens is None:
        return list(range(q.shape[0])), [0] * q.shape[0], [q.shape[1]] * q.shape[0]
    bounds = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
    lengths = [end - start for start, end in zip(bounds[:-1], bounds[1:], strict=True)]
    return [0] * len(lengths), bounds[:-1], lengths


def longest_first(
    q: torch.Tensor,
    v: torch.Tensor,
    rows: list[int],
    starts: list[int],
    lengths: list[int],
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    dtype: torch.dtype,
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The call's sequences longest first, so those still running at any step are a prefix.

    Returns the order and, in that order, each sequence's batch row, first token and length,
    and its ``[V heads, K, V]`` starting state in *dtype*: zero when *initial_state* is
    ``None``, transposed from ``[V, K]`` with *state_v_first*.
    """
    order = sorted(range(len(lengths)), key=lambda sequence: -lengths[sequence])
    row, start, length = (
        torch.tensor([values[s] for s in order], dtype=torch.long, device=q.device)
        for values in (rows, starts, lengths)
    )
    if initial_state is None:
        state = torch.zeros(
            len(order), v.shape[2], q.shape[-1], v.shape[-1], dtype=dtype, device=q.device
        )
    else:
        state = initial_state[order].to(dtype)
        state = state.transpose(-1, -2).contiguous() if state_v_first else state
    return order, row, start, length, state


def in_call_order(state: torch.Tensor, order: list[int], state_v_first: bool) -> torch.Tensor:
    """The float32 states of :func:`longest_first`'s *order* back in the call's order and
    layout."""
    final = torch.empty_like(state)
    final[order] = state
    return (final.transpose(-1, -2) if state_v_first else final).float()


def read_states(pool: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """The states *slots* name in *pool*; a slot of -1 names none and reads zeros."""
    states = pool.new_zeros((slots.numel(), *pool.shape[1:]))
    kept = slots >= 0
    states[kept] = pool[slots[kept]]
    return states


def write_states(pool: torch.Tensor, slots: torch.Tensor, states: torch.Tensor) -> None:
    """Store ``states[i]`` in ``pool[slots[i]]``; a slot of -1 is never written."""
    kept = slots >= 0
    pool[slots[kept]] = states[kept]


def on_state_pool(
    reference: Callable[[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]],
    initial_state: torch.Tensor | None,
    state_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(o, final_state)`` of *reference* run from *initial_state*, or, with *state_indices*,
    from the pool slots they name.

    The pool is updated in place: each sequence's slot takes its final state, and every other
    slot keeps the state it held. A slot of -1 is never written.
    """
    if state_indices is None:
        return reference(initial_state)
    slots = state_indices.long()
    o, final_state = reference(read_states(initial_state, slots))
    pool = initial_state.clone()
    write_states(pool, slots, final_state)
    return o, pool
