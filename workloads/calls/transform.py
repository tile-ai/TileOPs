"""Call workloads of the transform family."""

import torch

from workloads.transform import HadamardTransformWorkload


def _hadamard(call) -> HadamardTransformWorkload:
    shape, dtype = call.tensors["x"]
    return HadamardTransformWorkload(
        shape[:-1], shape[-1], getattr(torch, dtype), **call.arguments({})
    )


WORKLOADS = {
    "HadamardTransformFwdOp": _hadamard,
}
