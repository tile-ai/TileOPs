"""Inputs and the guarded-output check shared by the grouped GEMM op tests."""

import math

import torch

from workloads.numerics import compare_outputs

# An output value no valid row produces.
SENTINEL = -12288.0
# Partial last tiles, an empty expert, a one-row expert, and runs over several tiles.
SIZES = [5, 0, 77, 1, 300, 64, 129]


def contiguous_layout(packing: str, metadata_kind: str, alignment: int = 1) -> dict:
    return {
        "contiguous": {"packing": packing, "metadata_kind": metadata_kind, "alignment": alignment}
    }


def with_counts(workload, counts: list) -> tuple:
    """The workload's inputs with *counts* valid rows per expert in place of the generated ones."""
    *operands, metadata = workload.gen_inputs()
    return (*operands, torch.tensor(counts, dtype=torch.int32, device=metadata.device))


def run_into_guarded_out(op, workload, *inputs) -> torch.Tensor:
    """Check output reuse, guard integrity and correctness on valid rows."""
    a, b, metadata = inputs[0], inputs[2], inputs[-1]
    shape = (*a.shape[:-1], b.shape[1])
    guard = 4096
    flat = torch.full((math.prod(shape) + guard,), SENTINEL, dtype=torch.bfloat16, device=a.device)
    out = flat[:-guard].view(shape)
    got = op(*inputs, out=out)
    assert got is out
    assert bool((flat[-guard:] == SENTINEL).all()), "a store ran past the output"
    reference_inputs = (*inputs[:-1], metadata.clamp(0, a.shape[1])) if a.ndim == 3 else inputs
    compare_outputs(
        got, workload.ref_program(*reference_inputs), workload.verification(*reference_inputs)
    )
    return got
