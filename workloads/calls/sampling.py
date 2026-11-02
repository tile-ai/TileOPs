"""Call workloads of the sampling family."""

from workloads.sampling import (
    ChainSpeculativeSamplingWorkload,
    MinPMaskWorkload,
    SamplingFromProbsWorkload,
    TopKMaskWorkload,
    TopKTopPMaskWorkload,
    TopPMaskWorkload,
)

WORKLOADS = {
    "ChainSpeculativeSamplingFwdOp": ChainSpeculativeSamplingWorkload,
    "MinPMaskFwdOp": MinPMaskWorkload,
    "SamplingFromProbsFwdOp": SamplingFromProbsWorkload,
    "TopKMaskFwdOp": TopKMaskWorkload,
    "TopKTopPMaskFwdOp": TopKTopPMaskWorkload,
    "TopPMaskFwdOp": TopPMaskWorkload,
}
