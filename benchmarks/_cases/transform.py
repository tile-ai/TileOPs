"""Case factories of the transform family."""

import torch

from benchmarks._cases import Entry
from workloads.transform import HadamardTransformWorkload


def _hadamard(call) -> HadamardTransformWorkload:
    shape, dtype = call.tensors["x"]
    return HadamardTransformWorkload(
        shape[:-1], shape[-1], getattr(torch, dtype), **call.arguments({})
    )


ENTRIES = {
    "HadamardTransformFwdOp": Entry(_hadamard),
}
