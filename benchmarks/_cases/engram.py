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


def _extents(call, *names: str) -> dict:
    return {name: call.indices[name] for name in names}


def _gate_conv_fwd(call) -> EngramGateConvFwdWorkload:
    return EngramGateConvFwdWorkload(
        **_extents(call, "M", "seq_len", "d"), **call.arguments({}), dtype=_dtype(call, "H")
    )


def _gate_conv_bwd(call) -> EngramGateConvBwdWorkload:
    return EngramGateConvBwdWorkload(
        **_extents(call, "M", "seq_len", "d"), **call.arguments({}), dtype=_dtype(call, "dY")
    )


def _decode(call) -> EngramDecodeWorkload:
    return EngramDecodeWorkload(
        **_extents(call, "batch", "d_mem", "d"),
        **call.arguments({}),
        dtype=_dtype(call, "e_t"),
        conv_len=call.indices["L"],
    )


ENTRIES = {
    "EngramGateConvFwdOp": Entry(_gate_conv_fwd),
    "EngramGateConvBwdOp": Entry(_gate_conv_bwd),
    "EngramDecodeFwdOp": Entry(_decode),
}
