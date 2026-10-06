"""Case factories of the engram family."""

import torch

from benchmarks._cases import Entry
from workloads.sequence_modeling.engram import (
    EngramDecodeWorkload,
    EngramGateConvBwdWorkload,
    EngramGateConvFwdWorkload,
)


def _dtype(call, tensor: str) -> torch.dtype:
    return getattr(torch, call.tensors[tensor][1])


def _gate_conv_fwd(call) -> EngramGateConvFwdWorkload:
    return EngramGateConvFwdWorkload(**call.arguments({}), dtype=_dtype(call, "H"))


def _gate_conv_bwd(call) -> EngramGateConvBwdWorkload:
    return EngramGateConvBwdWorkload(**call.arguments({}), dtype=_dtype(call, "dY"))


def _decode(call) -> EngramDecodeWorkload:
    return EngramDecodeWorkload(
        **call.arguments({}), dtype=_dtype(call, "e_t"), conv_len=call.ix["L"]
    )


ENTRIES = {
    "EngramGateConvFwdOp": Entry(_gate_conv_fwd),
    "EngramGateConvBwdOp": Entry(_gate_conv_bwd),
    "EngramDecodeFwdOp": Entry(_decode),
}
