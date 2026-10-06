"""Case factories of the sampling family."""

from benchmarks._cases import Entry
from workloads.sampling import (
    ChainSpeculativeSamplingWorkload,
    MinPMaskWorkload,
    SamplingFromProbsWorkload,
    TopKMaskWorkload,
    TopKTopPMaskWorkload,
    TopPMaskWorkload,
)

ENTRIES = {
    "ChainSpeculativeSamplingFwdOp": Entry(ChainSpeculativeSamplingWorkload),
    "MinPMaskFwdOp": Entry(MinPMaskWorkload),
    "SamplingFromProbsFwdOp": Entry(SamplingFromProbsWorkload),
    "TopKMaskFwdOp": Entry(TopKMaskWorkload),
    "TopKTopPMaskFwdOp": Entry(TopKTopPMaskWorkload),
    "TopPMaskFwdOp": Entry(TopPMaskWorkload),
}
