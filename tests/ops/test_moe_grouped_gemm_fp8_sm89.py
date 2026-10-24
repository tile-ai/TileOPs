"""SM89 masked FP8 grouped GEMM configuration and storage contracts."""

import pytest
import torch

from tests.ops.test_moe_grouped_gemm_fp8 import (
    _OP,
    _SENTINEL,
    _run_into_guarded_out,
    _with_counts,
    _workload,
)
from tileops.backend import register_kernel_type, registry
from tileops.kernels.moe import MoEGroupedGemmFP8MMAKernel
from tileops.ops.moe import MaskedLayoutSpec, MoEGroupedGemmFP8FwdOp
from workloads.numerics import compare_outputs


def _forced_mma_op(config):
    class ForcedMMA(MoEGroupedGemmFP8MMAKernel):
        preferred_over = frozenset(MoEGroupedGemmFP8FwdOp.kernel_types)

        @property
        def default_config(self):
            return dict(config)

    state = registry.snapshot()
    try:
        register_kernel_type(_OP, "grouped_gemm_fp8_mma_forced", ForcedMMA)
        return MoEGroupedGemmFP8FwdOp(MaskedLayoutSpec(max_m=65))
    finally:
        registry.restore(state)


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("block_m,block_n", [(32, 64), (32, 128), (64, 64), (64, 128)])
def test_mma_explicit_configs(block_m, block_n):
    workload = _workload(65, E=3, N=256, K=256)
    op = _forced_mma_op({"block_m": block_m, "block_n": block_n})
    _run_into_guarded_out(op, workload, *_with_counts(workload, [0, 33, 65]))
    entries = op.built_entries("grouped_gemm_fp8").values()
    assert entries
    for entry in entries:
        assert isinstance(entry, MoEGroupedGemmFP8MMAKernel)
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
    op = MoEGroupedGemmFP8FwdOp(MaskedLayoutSpec(max_m=65))
    op(*inputs)
    entries = op.built_entries("grouped_gemm_fp8").values()
    assert entries
    for entry in entries:
        assert isinstance(entry, MoEGroupedGemmFP8MMAKernel)
        with pytest.raises(ValueError, match="FP8 MMA"):
            entry.init_config(change)


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("offset_inputs", [False, True])
def test_mma_unaligned_storage(offset_inputs):
    workload = _workload(65, E=3, N=256, K=256)
    inputs = list(_with_counts(workload, [0, 33, 65]))
    if offset_inputs:
        for index in (0, 2):
            original = inputs[index]
            shifted = torch.empty(
                original.numel() + 1, dtype=original.dtype, device=original.device
            )[1:].view(original.shape)
            shifted.copy_(original)
            inputs[index] = shifted
            assert shifted.data_ptr() % 16
    before = [x.clone() for x in inputs]
    storage = torch.full(
        (3 * 65 * 256 + 2,), _SENTINEL, dtype=torch.bfloat16, device=inputs[0].device
    )
    out = storage[1:-1].view(3, 65, 256)
    assert out.data_ptr() % 16
    op = MoEGroupedGemmFP8FwdOp(MaskedLayoutSpec(max_m=65))
    assert op(*inputs, out=out) is out
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))
    assert bool((out[0] == _SENTINEL).all())
    assert storage[0].item() == _SENTINEL and storage[-1].item() == _SENTINEL
    for old, new in zip(before, inputs, strict=True):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
