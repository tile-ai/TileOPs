"""Call workloads of the hyper-connection family."""

from workloads.sequence_modeling.hyper_connection import (
    HyperConnectionHeadWorkload,
    HyperConnectionPostWorkload,
    HyperConnectionPreCall,
)

WORKLOADS = {
    "HyperConnectionHeadFwdOp": HyperConnectionHeadWorkload.from_call,
    "HyperConnectionPostFwdOp": HyperConnectionPostWorkload.from_call,
    "HyperConnectionPreFwdOp": HyperConnectionPreCall,
}
