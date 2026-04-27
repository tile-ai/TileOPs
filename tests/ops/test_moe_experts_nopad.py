"""Tests for MoEExpertsNopad/Padded and supporting ABCs."""
import pytest
import torch
from tileops.ops.moe.abc import (
    PrepareResult, WeightedReduce, WeightedReduceNoOp,
    MoEPrepareAndFinalize, MoEExperts, MoEExpertsModular,
)


@pytest.mark.smoke
def test_abc_imports():
    """ABCs and data structures can be imported."""
    assert issubclass(WeightedReduceNoOp, WeightedReduce)


@pytest.mark.smoke
def test_weighted_reduce_noop():
    """WeightedReduceNoOp copies expert_out to output."""
    T, H = 4, 8
    expert_out = torch.randn(T, H)
    output = torch.zeros(T, H)
    reduce = WeightedReduceNoOp()
    reduce.apply(output, expert_out,
                 topk_weights=torch.ones(T, 2),
                 topk_ids=torch.zeros(T, 2, dtype=torch.int32))
    assert torch.allclose(output, expert_out)


@pytest.mark.smoke
def test_weighted_reduce_noop_same_tensor():
    """WeightedReduceNoOp is a no-op when output is expert_out."""
    T, H = 4, 8
    t = torch.randn(T, H)
    original = t.clone()
    WeightedReduceNoOp().apply(t, t,
                               topk_weights=torch.ones(T, 2),
                               topk_ids=torch.zeros(T, 2, dtype=torch.int32))
    assert torch.allclose(t, original)


# ---------------------------------------------------------------------------
# MoEPrepareAndFinalizeNoDPEP
# ---------------------------------------------------------------------------

from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP


class TestMoEPrepareAndFinalizeNoDPEP:

    @pytest.mark.smoke
    def test_prepare_passthrough(self):
        T, H, K = 8, 64, 2
        hidden = torch.randn(T, H, dtype=torch.bfloat16)
        weights = torch.rand(T, K, dtype=torch.float32)
        ids = torch.randint(0, 4, (T, K), dtype=torch.int32)
        pf = MoEPrepareAndFinalizeNoDPEP()
        r = pf.prepare(hidden, weights, ids, num_experts=4, expert_map=None)
        assert r.hidden_q is hidden
        assert r.scale is None
        assert r.topk_weights is weights
        assert r.topk_ids is ids

    @pytest.mark.smoke
    def test_finalize_noop_reduce(self):
        T, H, K = 8, 64, 2
        expert_out = torch.randn(T, H, dtype=torch.bfloat16)
        output = torch.zeros(T, H, dtype=torch.bfloat16)
        weights = torch.rand(T, K, dtype=torch.float32)
        ids = torch.randint(0, 4, (T, K), dtype=torch.int32)
        pf = MoEPrepareAndFinalizeNoDPEP()
        pf.finalize(output, expert_out, weights, ids, WeightedReduceNoOp())
        assert torch.allclose(output, expert_out)

