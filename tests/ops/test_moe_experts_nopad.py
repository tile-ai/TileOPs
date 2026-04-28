"""Tests for MoEExpertsNopadFwdOp/PaddedFwdOp and supporting ABCs."""
import pytest
import torch

from tileops.ops.moe.abc import (
    WeightedReduce,
    WeightedReduceNoOp,
)
from tileops.ops.moe.experts.nopad import MoEExpertsNopadFwdOp
from tileops.ops.moe.experts.padded import MoEExpertsPaddedFwdOp
from tileops.ops.moe.fused_moe_experts import FusedMoeExpertsFwdOp, FusedMoeExpertsPaddedFwdOp
from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP


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


# ---------------------------------------------------------------------------
# MoEExpertsNopadFwdOp
# ---------------------------------------------------------------------------

@pytest.fixture
def moe_tensors():
    T, H, F, E, K = 16, 64, 32, 4, 2
    dtype = torch.bfloat16
    hidden = torch.randn(T, H, dtype=dtype, device="cuda")
    w1 = torch.randn(E, 2 * F, H, dtype=dtype, device="cuda")
    w2 = torch.randn(E, H, F, dtype=dtype, device="cuda")
    weights = torch.softmax(torch.randn(T, K, dtype=torch.float32, device="cuda"), dim=-1)
    ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
    return dict(T=T, H=H, F=F, E=E, K=K, dtype=dtype,
                hidden=hidden, w1=w1, w2=w2, weights=weights, ids=ids)


class TestMoEExpertsNopadFwdOp:

    @pytest.mark.smoke
    def test_workspace_shapes(self, moe_tensors):
        d = moe_tensors
        experts = MoEExpertsNopadFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
        )
        ws1, ws2 = experts.workspace_shapes(d["T"], d["F"], d["H"], d["K"], d["E"])
        assert ws1 == (0,) and ws2 == (0,)

    @pytest.mark.smoke
    def test_output_shape(self, moe_tensors):
        d = moe_tensors
        experts = MoEExpertsNopadFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
        )
        assert experts.output_shape(d["T"], d["H"]) == (d["T"], d["H"])

    @pytest.mark.smoke
    def test_make_weighted_reduce_is_noop(self, moe_tensors):
        d = moe_tensors
        experts = MoEExpertsNopadFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
        )
        assert isinstance(experts.make_weighted_reduce(), WeightedReduceNoOp)

    @pytest.mark.smoke
    def test_apply_matches_fused_moe_experts(self, moe_tensors):
        """MoEExpertsNopadFwdOp.apply() must match FusedMoeExpertsFwdOp.forward()."""
        d = moe_tensors
        kwargs = dict(num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
                      hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"])
        ref = FusedMoeExpertsFwdOp(**kwargs)
        new = MoEExpertsNopadFwdOp(**kwargs)

        ref_out = ref.forward(d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"])

        output = torch.empty(d["T"], d["H"], dtype=d["dtype"], device="cuda")
        ws1 = torch.empty(0, device="cuda")
        ws2 = torch.empty(0, device="cuda")
        new.apply(output, d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"],
                  d["E"], None, ws1, ws2)

        assert torch.allclose(output.float(), ref_out.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# MoEExpertsPaddedFwdOp
# ---------------------------------------------------------------------------

class TestMoEExpertsPaddedFwdOp:

    @pytest.mark.smoke
    def test_apply_matches_fused_moe_experts_padded(self, moe_tensors):
        """MoEExpertsPaddedFwdOp.apply() must match FusedMoeExpertsPaddedFwdOp.forward()."""
        d = moe_tensors
        kwargs = dict(num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
                      hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"])
        ref = FusedMoeExpertsPaddedFwdOp(**kwargs)
        new = MoEExpertsPaddedFwdOp(**kwargs)

        ref_out = ref.forward(d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"])

        output = torch.empty(d["T"], d["H"], dtype=d["dtype"], device="cuda")
        ws1 = torch.empty(0, device="cuda")
        ws2 = torch.empty(0, device="cuda")
        new.apply(output, d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"],
                  d["E"], None, ws1, ws2)

        assert torch.allclose(output.float(), ref_out.float(), atol=1e-2, rtol=1e-2)
