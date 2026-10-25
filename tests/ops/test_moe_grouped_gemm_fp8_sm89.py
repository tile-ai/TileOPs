"""SM89 masked FP8 grouped GEMM configuration and storage contracts."""

import pytest

from tests.ops.test_moe_grouped_gemm_fp8 import (
    _OP,
    _run_into_guarded_out,
    _with_counts,
    _workload,
)
from tileops._kernels.moe import MoEGroupedGEMMFP8MMAKernel
from tileops._ops.moe import MaskedLayoutSpec, MoEGroupedGEMMFP8FwdOp
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type


def _forced_mma_op(config):
    class ForcedMMA(MoEGroupedGEMMFP8MMAKernel):
        preferred_over = frozenset(MoEGroupedGEMMFP8FwdOp.kernel_types)

        @property
        def default_config(self):
            return dict(config)

    state = registry.snapshot()
    try:
        register_kernel_type(_OP, "grouped_gemm_fp8_mma_forced", ForcedMMA)
        return MoEGroupedGEMMFP8FwdOp(MaskedLayoutSpec(max_m=65))
    finally:
        registry.restore(state)


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("block_m,block_n", [(32, 64), (32, 128), (64, 64), (64, 128)])
def test_mma_explicit_configs(block_m, block_n):
    workload = _workload(65, E=3, N=256, K=256)
    op = _forced_mma_op({"block_m": block_m, "block_n": block_n})
    _run_into_guarded_out(op, workload, *_with_counts(workload, [0, 33, 65]))
    entries = op._entries_built("grouped_gemm_fp8").values()
    assert entries
    for entry in entries:
        assert isinstance(entry, MoEGroupedGEMMFP8MMAKernel)
        assert entry.config == {"block_m": block_m, "block_n": block_n}


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize(
    "change",
    [
        {"threads": 128},
        {"num_stages": 2},
        {"block_m": 16},
        {"block_n": 32},
        {"block_n": 64.0},
        {"block_m": True},
    ],
)
def test_mma_rejects_invalid_config(change):
    workload = _workload(65, E=3, N=256, K=256)
    inputs = _with_counts(workload, [0, 33, 65])
    op = MoEGroupedGEMMFP8FwdOp(MaskedLayoutSpec(max_m=65))
    op(*inputs)
    entries = op._entries_built("grouped_gemm_fp8").values()
    assert entries
    for entry in entries:
        assert isinstance(entry, MoEGroupedGEMMFP8MMAKernel)
        with pytest.raises(ValueError, match="FP8 MMA"):
            entry.init_config(change)
