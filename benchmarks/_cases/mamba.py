"""Case factories of the mamba family."""

from typing import Any

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.mamba import (
    CausalConv1dFwdCall,
    CausalConv1dUpdateFwdCall,
    Mamba2FwdCall,
    SelectiveScanFwdCall,
    SSDChunkCouplingFwdCall,
    SSDChunkCumsumFwdCall,
    SSDChunkScanFwdCall,
    SSDChunkStateFwdCall,
    SSDDecodeFwdCall,
    SSDStatePassingFwdCall,
    conv_state_result,
    ssd_decode_result,
)


def _ssd_decode_binder(op: Any, case: Any) -> Implementation:
    """The op on a private copy of the recurrent state it updates, returning output and state."""
    return private_inputs(lambda *args: ssd_decode_result(op, *args), case.inputs, 5)


def _conv_state_binder(op: Any, case: Any) -> Implementation:
    """The op on a private copy of the convolution state it overwrites, returning output and
    state."""
    return private_inputs(lambda *args: conv_state_result(op, *args), case.inputs, 2)


ENTRIES = {
    "CausalConv1dFwdOp": Entry(CausalConv1dFwdCall, binder=_conv_state_binder),
    "CausalConv1dUpdateFwdOp": Entry(CausalConv1dUpdateFwdCall, binder=_conv_state_binder),
    "SSDChunkCouplingFwdOp": Entry(SSDChunkCouplingFwdCall),
    "SSDChunkCumsumFwdOp": Entry(SSDChunkCumsumFwdCall),
    "SSDChunkScanFwdOp": Entry(SSDChunkScanFwdCall),
    "SSDChunkStateFwdOp": Entry(SSDChunkStateFwdCall),
    "SSDStatePassingFwdOp": Entry(SSDStatePassingFwdCall),
    "SSDRecurrentFwdOp": Entry(SSDDecodeFwdCall, binder=_ssd_decode_binder),
    "Mamba2FwdOp": Entry(Mamba2FwdCall),
    "SelectiveScanFwdOp": Entry(SelectiveScanFwdCall),
}
