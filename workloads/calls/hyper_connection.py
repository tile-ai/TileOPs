"""Call workloads of the hyper-connection family."""

from workloads.sequence_modeling.hyper_connection import (
    HyperConnectionPostWorkload,
    HyperConnectionPreCall,
)

WORKLOADS = {
    "HyperConnectionPostFwdOp": HyperConnectionPostWorkload.from_call,
    "HyperConnectionPreFwdOp": HyperConnectionPreCall,
}
