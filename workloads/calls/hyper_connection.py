"""Call workloads of the hyper-connection family."""

from workloads.sequence_modeling.hyper_connection import HyperConnectionPreCall

WORKLOADS = {
    "HyperConnectionPreFwdOp": HyperConnectionPreCall,
}
