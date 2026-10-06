"""Case factories of the mamba family."""

from typing import Any

from benchmarks._cases import Entry
from benchmarks.api import Implementation
from benchmarks.baselines import private_inputs
from workloads.mamba import (
    Mamba2FwdCall,
    SSDChunkCouplingFwdCall,
    SSDChunkCumsumFwdCall,
    SSDChunkScanFwdCall,
    SSDChunkStateFwdCall,
    SSDDecodeFwdCall,
    SSDStatePassingFwdCall,
    ssd_decode_result,
)


def _ssd_decode_binder(op: Any, case: Any) -> Implementation:
    """The op on a private copy of the recurrent state it updates, returning output and state."""
    return private_inputs(lambda *args: ssd_decode_result(op, *args), case.inputs, 5)


ENTRIES = {
    "SSDChunkCouplingFwdOp": Entry(SSDChunkCouplingFwdCall),
    "SSDChunkCumsumFwdOp": Entry(SSDChunkCumsumFwdCall),
    "SSDChunkScanFwdOp": Entry(SSDChunkScanFwdCall),
    "SSDChunkStateFwdOp": Entry(SSDChunkStateFwdCall),
    "SSDStatePassingFwdOp": Entry(SSDStatePassingFwdCall),
    "SSDRecurrentFwdOp": Entry(SSDDecodeFwdCall, binder=_ssd_decode_binder),
    "Mamba2FwdOp": Entry(Mamba2FwdCall),
}
