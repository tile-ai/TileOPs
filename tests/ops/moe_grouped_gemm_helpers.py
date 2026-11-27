"""Inputs and the guarded-output check shared by the grouped GEMM op tests."""

import math

import torch

from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

# An output value no valid row produces.
SENTINEL = -12288.0
# Partial last tiles, an empty expert, a one-row expert, and runs over several tiles.
SIZES = [5, 0, 77, 1, 300, 64, 129]


def contiguous_layout(packing: str, metadata_kind: str, alignment: int = 1) -> dict:
    return {
        "contiguous": {"packing": packing, "metadata_kind": metadata_kind, "alignment": alignment}
    }


def masked_call(op: str, max_m: int, dtype_case: dict | None = None, **dims):
    """*op*'s manifest call on slabs of *max_m* rows."""
    # T only feeds the generated counts; half the slabs' capacity keeps every count in range.
    dims.setdefault("T", dims["E"] * max_m // 2)
    return manifest_call(op, dtype_case, layout={"masked": {"max_m": max_m}}, **dims)


def contiguous_call(op: str, layout: dict, sizes: list, dtype_case: dict | None = None, **dims):
    """*op*'s manifest call on *layout*'s runs of *sizes* rows, one run an expert."""
    spec = layout["contiguous"]
    if spec["metadata_kind"] == "physical_psum":
        align = spec["alignment"]
        dims["P"] = sum(-(-size // align) * align for size in sizes)
    return manifest_call(op, dtype_case, layout=layout, E=len(sizes), sizes=sizes, **dims)


def check_against_reference(out: torch.Tensor, workload, *inputs) -> None:
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))


def check_manifest_row(op_type, workload_type, row: dict, dtype_case: dict | None = None) -> None:
    """A manifest row, with its generated valid counts, gives the reference in bfloat16."""
    fields = {key: value for key, value in row.items() if key not in ("label", "dtype_cases")}
    workload = workload_type(manifest_call(op_type.__name__, dtype_case, **fields))
    inputs = workload.gen_inputs()
    out = op_type(**workload.call.arguments())(*inputs)
    assert out.dtype is torch.bfloat16
    check_against_reference(out, workload, *inputs)


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
    check_against_reference(got, workload, *reference_inputs)
    return got
