"""The facts of one GLAInferenceFwdOp call that its in-tree kernels select and build on."""

import dataclasses
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Entry

__all__ = ["GLAInferenceCallSpec", "dense_entry", "serves_dense"]


@dataclasses.dataclass(frozen=True)
class GLAInferenceCallSpec(CallSpec):
    """One inference call, as the op knows it after reading its inputs."""

    batch: int = 0
    seq_len: int = 0
    heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    dtype: Optional[torch.dtype] = None
    scale: float = 0.0
    varlen: bool = False


def serves_dense(call: GLAInferenceCallSpec) -> bool:
    """Whether *call* is a dense (not packed) call with K = V in {64, 128} in fp16 or bf16."""
    return (
        not call.varlen
        and call.dim_k == call.dim_v
        and call.dim_k in (64, 128)
        and call.dtype in (torch.float16, torch.bfloat16)
    )


def dense_entry(cls: type, call: GLAInferenceCallSpec, **extents: int) -> Entry:
    """Build *cls* from the call's scale, dtype and device plus the *extents* it compiles."""
    device_index = call.device.index if call.device is not None else None
    arguments = dict(extents, scale=call.scale, dtype=call.dtype, device_index=device_index)
    return tuple(sorted(arguments.items(), key=lambda item: item[0])), lambda: cls(**arguments)
