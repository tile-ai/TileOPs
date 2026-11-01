"""Call workloads of the moe family."""

from workloads.moe import (
    FusedMoESharedExpertWorkload,
    FusedMoEWorkload,
    FusedTopKWorkload,
    IndexedExpertMLPWorkload,
    MoEExpertMLPWorkload,
    MoEExpertsWorkload,
    MoEGroupedGEMMFP8Workload,
    MoEGroupedGEMMWorkload,
    MoEPermuteAlignWorkload,
    MoEPostPermuteWorkload,
    MoEPrePermuteWorkload,
    SharedExpertMLPWorkload,
)

WORKLOADS = {
    "FusedTopKFwdOp": FusedTopKWorkload,
    "MoEPermuteAlignFwdOp": MoEPermuteAlignWorkload,
    "MoEPrePermuteFwdOp": MoEPrePermuteWorkload,
    "MoEPostPermuteFwdOp": MoEPostPermuteWorkload,
    "MoEGroupedGEMMFwdOp": MoEGroupedGEMMWorkload,
    "MoEGroupedGEMMFP8FwdOp": MoEGroupedGEMMFP8Workload,
    "MoEExpertMLPFwdOp": MoEExpertMLPWorkload,
    "FusedMoEExpertsFwdOp": MoEExpertsWorkload,
    "IndexedExpertMLPFwdOp": IndexedExpertMLPWorkload,
    "FusedMoEFwdOp": FusedMoEWorkload,
    "FusedMoESharedExpertFwdOp": FusedMoESharedExpertWorkload,
    "SharedExpertMLPFwdOp": SharedExpertMLPWorkload,
}
