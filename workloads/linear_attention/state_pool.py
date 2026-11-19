"""Reading and writing a pool of recurrent states by slot, where a slot of -1 is padding."""

import torch


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
