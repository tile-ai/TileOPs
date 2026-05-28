"""Tests for FusedMoEExpertsNopadPersistent3WGFwdOp/PaddedFwdOp and supporting ABCs."""
import logging

import pytest
import torch
import torch.nn.functional as F

from tileops.ops.moe._activation import build_activation_op
from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP
from tileops.ops.moe.routed_expert.abc import (
    WeightedReduce,
    WeightedReduceNoOp,
)
from tileops.ops.moe.routed_expert.fused_routed_expert import (
    FusedMoEExpertsNopadPersistent3WGFwdOp,
    FusedMoEExpertsPaddedFwdOp,
)


def _torch_ref_moe(hidden, w1, w2, topk_weights, topk_ids):
    """Per-expert PyTorch reference: ground-truth MoE FFN."""
    T, H = hidden.shape
    E, twoF, _ = w1.shape
    F_dim = twoF // 2
    output = torch.zeros(T, H, dtype=torch.float32, device=hidden.device)
    ids_i64 = topk_ids.to(torch.int64)
    for e in range(E):
        mask = (ids_i64 == e)
        if not mask.any():
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)
        h = hidden[t_idx].float()
        gate_up = h @ w1[e].float().t()
        act = F.silu(gate_up[:, :F_dim]) * gate_up[:, F_dim:]
        down = act @ w2[e].float().t()
        output.index_add_(0, t_idx, down * topk_weights[t_idx, k_idx].float().unsqueeze(-1))
    return output.to(hidden.dtype)


def _torch_ref_moe_activation(hidden, w1, w2, topk_weights, topk_ids, activation="silu_and_mul"):
    """Per-expert PyTorch reference supporting silu_and_mul and gelu_and_mul."""
    T, H = hidden.shape
    E, twoF, _ = w1.shape
    F_dim = twoF // 2
    output = torch.zeros(T, H, dtype=torch.float32, device=hidden.device)
    ids_i64 = topk_ids.to(torch.int64)
    for e in range(E):
        mask = (ids_i64 == e)
        if not mask.any():
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)
        h = hidden[t_idx].float()
        gate_up = h @ w1[e].float().t()
        gate, up = gate_up[:, :F_dim], gate_up[:, F_dim:]
        # gelu_and_mul uses exact erf GELU which matches GeluAndMulFwdKernel
        act = F.silu(gate) * up if activation == "silu_and_mul" else F.gelu(gate) * up
        down = act @ w2[e].float().t()
        output.index_add_(0, t_idx, down * topk_weights[t_idx, k_idx].float().unsqueeze(-1))
    return output.to(hidden.dtype)


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
# FusedMoEExpertsNopadPersistent3WGFwdOp
# ---------------------------------------------------------------------------

@pytest.fixture
def moe_meta():
    T, H, F_dim, E, K = 128, 256, 128, 4, 2
    return dict(T=T, H=H, F=F_dim, E=E, K=K, dtype=torch.bfloat16)


@pytest.fixture(params=[torch.bfloat16, torch.float16], ids=["bfloat16", "float16"])
def moe_tensors(request):
    T, H, F_dim, E, K = 128, 256, 128, 4, 2
    dtype = request.param
    hidden = torch.randn(T, H, dtype=dtype, device="cuda") * 0.1
    w1 = torch.randn(E, 2 * F_dim, H, dtype=dtype, device="cuda") * 0.02
    w2 = torch.randn(E, H, F_dim, dtype=dtype, device="cuda") * 0.02
    weights = torch.softmax(torch.randn(T, K, dtype=torch.float32, device="cuda"), dim=-1)
    ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
    return dict(T=T, H=H, F=F_dim, E=E, K=K, dtype=dtype,
                hidden=hidden, w1=w1, w2=w2, weights=weights, ids=ids)


class TestFusedMoEExpertsNopadPersistent3WGFwdOp:

    @pytest.mark.smoke
    def test_workspace_shapes(self, moe_meta):
        d = moe_meta
        experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
        )
        ws1, ws2 = experts.workspace_shapes(d["T"], d["F"], d["H"], d["K"], d["E"])
        assert ws1 == (0,) and ws2 == (0,)

    @pytest.mark.smoke
    def test_output_shape(self, moe_meta):
        d = moe_meta
        experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
        )
        assert experts.output_shape(d["T"], d["H"]) == (d["T"], d["H"])

    @pytest.mark.smoke
    def test_make_weighted_reduce_is_noop(self, moe_meta):
        d = moe_meta
        experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
        )
        assert isinstance(experts.make_weighted_reduce(), WeightedReduceNoOp)

    @pytest.mark.smoke
    def test_forward_matches_torch_ref(self, moe_tensors):
        """forward() output must match a per-expert PyTorch reference."""
        d = moe_tensors
        experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
        )

        ref_out = _torch_ref_moe(d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"])

        output = torch.empty(d["T"], d["H"], dtype=d["dtype"], device="cuda")
        ws1 = torch.empty(0, dtype=d["dtype"], device="cuda")
        ws2 = torch.empty(0, dtype=d["dtype"], device="cuda")
        experts.forward(
            output, d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"],
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=d["E"],
        )

        assert torch.allclose(output.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.smoke
    def test_forward_fallback_path_unaligned_dims(self, caplog):
        """Unaligned dims must trigger the MoeGroupedGemmNopadKernel fallback
        and still produce correct output.

        H=128, F=96: gate_up_n=192 is not divisible by 3WG block_n=256, so the
        op selects MoeGroupedGemmNopadKernel instead of the 3WG persistent
        kernel.
        """
        T, H, F_dim, E, K = 64, 128, 96, 4, 2
        dtype = torch.bfloat16
        hidden = torch.randn(T, H, dtype=dtype, device="cuda") * 0.1
        w1 = torch.randn(E, 2 * F_dim, H, dtype=dtype, device="cuda") * 0.02
        w2 = torch.randn(E, H, F_dim, dtype=dtype, device="cuda") * 0.02
        weights = torch.softmax(torch.randn(T, K, dtype=torch.float32, device="cuda"), dim=-1)
        ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")

        with caplog.at_level(
            logging.WARNING, logger="tileops.ops.moe.routed_expert.fused_routed_expert"
        ):
            experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
                num_tokens=T, num_experts=E, top_k=K,
                hidden_size=H, ffn_size=F_dim, dtype=dtype,
            )
        assert any(
            "falling back to MoeGroupedGemmNopadKernel" in rec.message
            for rec in caplog.records
        ), f"expected fallback warning, got: {[rec.message for rec in caplog.records]}"

        ref_out = _torch_ref_moe(hidden, w1, w2, weights, ids)
        output = torch.empty(T, H, dtype=dtype, device="cuda")
        ws1 = torch.empty(0, dtype=dtype, device="cuda")
        ws2 = torch.empty(0, dtype=dtype, device="cuda")
        experts.forward(
            output, hidden, w1, w2, weights, ids,
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=E,
        )
        assert torch.allclose(output.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.smoke
    def test_forward_with_expert_map_runs(self):
        """expert_map (EP mode) must construct + forward without raising.

        This is a smoke check that the EP path is wired up — the Nopad+3WG
        implementation supports expert_map (unlike Padded which raises). The
        per-expert numerical correctness check under expert_map filtering is
        non-trivial to write a torch reference for and is left to a follow-up
        end-to-end test against vLLM.
        """
        T, H, F_dim, E_global, E_local, K = 64, 128, 64, 8, 4, 2
        dtype = torch.bfloat16
        # Map first E_local experts to local ids 0..E_local-1; rest to -1.
        expert_map = torch.full((E_global,), -1, dtype=torch.int32, device="cuda")
        expert_map[:E_local] = torch.arange(E_local, dtype=torch.int32, device="cuda")

        hidden = torch.randn(T, H, dtype=dtype, device="cuda") * 0.1
        # Weights sized to local experts only, since nopad's _gemm_gate_up
        # is constructed with num_experts_local.
        w1 = torch.randn(E_local, 2 * F_dim, H, dtype=dtype, device="cuda") * 0.02
        w2 = torch.randn(E_local, H, F_dim, dtype=dtype, device="cuda") * 0.02
        weights = torch.softmax(torch.randn(T, K, dtype=torch.float32, device="cuda"), dim=-1)
        # Mix local + non-local expert ids to exercise the -1 fwd_idx path.
        ids = torch.randint(0, E_global, (T, K), dtype=torch.int32, device="cuda")

        experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=T, num_experts=E_global, top_k=K,
            hidden_size=H, ffn_size=F_dim, dtype=dtype,
            expert_map=expert_map,
        )
        output = torch.empty(T, H, dtype=dtype, device="cuda")
        ws1 = torch.empty(0, dtype=dtype, device="cuda")
        ws2 = torch.empty(0, dtype=dtype, device="cuda")
        experts.forward(
            output, hidden, w1, w2, weights, ids,
            expert_map=expert_map, workspace1=ws1, workspace2=ws2,
            num_experts=E_global,
        )
        # Output must be finite (no NaN/Inf from the -1 fwd_idx path).
        assert torch.isfinite(output.float()).all()

    @pytest.mark.smoke
    @pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
    def test_forward_matches_torch_ref_activation(self, moe_tensors, activation):
        """forward() output matches PyTorch reference for each activation."""
        d = moe_tensors
        experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
            activation=activation,
        )
        assert experts.activation == activation
        ref_out = _torch_ref_moe_activation(
            d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"], activation=activation,
        )
        output = torch.empty(d["T"], d["H"], dtype=d["dtype"], device="cuda")
        ws1 = torch.empty(0, dtype=d["dtype"], device="cuda")
        ws2 = torch.empty(0, dtype=d["dtype"], device="cuda")
        experts.forward(
            output, d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"],
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=d["E"],
        )
        assert torch.allclose(output.float(), ref_out.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# FusedMoEExpertsPaddedFwdOp
# ---------------------------------------------------------------------------

class TestFusedMoEExpertsPaddedFwdOp:

    @pytest.mark.smoke
    def test_forward_matches_torch_ref(self, moe_tensors):
        """forward() output must match a per-expert PyTorch reference."""
        d = moe_tensors
        experts = FusedMoEExpertsPaddedFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
        )

        ref_out = _torch_ref_moe(d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"])

        output = torch.empty(d["T"], d["H"], dtype=d["dtype"], device="cuda")
        ws1 = torch.empty(0, device="cuda")
        ws2 = torch.empty(0, device="cuda")
        experts.forward(
            output, d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"],
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=d["E"],
        )

        assert torch.allclose(output.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.smoke
    @pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
    def test_forward_matches_torch_ref_activation(self, moe_tensors, activation):
        d = moe_tensors
        experts = FusedMoEExpertsPaddedFwdOp(
            num_tokens=d["T"], num_experts=d["E"], top_k=d["K"],
            hidden_size=d["H"], ffn_size=d["F"], dtype=d["dtype"],
            activation=activation,
        )
        assert experts.activation == activation
        ref_out = _torch_ref_moe_activation(
            d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"], activation=activation,
        )
        output = torch.empty(d["T"], d["H"], dtype=d["dtype"], device="cuda")
        ws1, ws2 = torch.empty(0, device="cuda"), torch.empty(0, device="cuda")
        experts.forward(
            output, d["hidden"], d["w1"], d["w2"], d["weights"], d["ids"],
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=d["E"],
        )
        assert torch.allclose(output.float(), ref_out.float(), atol=1e-2, rtol=1e-2)


class TestBuildActivationOp:

    @pytest.mark.smoke
    def test_silu_and_mul_returns_correct_type(self):
        from tileops.ops.elementwise import SiluAndMulFwdOp
        op = build_activation_op("silu_and_mul", M=16, N=32, dtype=torch.bfloat16)
        assert isinstance(op, SiluAndMulFwdOp)

    @pytest.mark.smoke
    def test_gelu_and_mul_returns_correct_type(self):
        from tileops.ops.elementwise import GeluAndMulFwdOp
        op = build_activation_op("gelu_and_mul", M=16, N=32, dtype=torch.bfloat16)
        assert isinstance(op, GeluAndMulFwdOp)

    @pytest.mark.smoke
    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="activation must be one of"):
            build_activation_op("unknown_act", M=16, N=32, dtype=torch.bfloat16)


class TestFusedMoeActivationInjection:

    def _make_experts(self):
        return FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=128, num_experts=4, top_k=2,
            hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
        )

    @pytest.mark.smoke
    def test_injection_with_explicit_activation_raises(self):
        """Passing experts= and an explicit activation= must raise ValueError."""
        from tileops.ops.moe.fused_moe import FusedMoe
        experts = self._make_experts()
        with pytest.raises(ValueError, match="activation must not be set"):
            FusedMoe(
                num_tokens=128, num_experts=4, top_k=2,
                hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
                experts=experts, activation="gelu_and_mul",
            )

    @pytest.mark.smoke
    def test_injection_with_explicit_default_activation_raises(self):
        """Even activation='silu_and_mul' + experts= must raise (sentinel catches this)."""
        from tileops.ops.moe.fused_moe import FusedMoe
        experts = self._make_experts()
        with pytest.raises(ValueError, match="activation must not be set"):
            FusedMoe(
                num_tokens=128, num_experts=4, top_k=2,
                hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
                experts=experts, activation="silu_and_mul",
            )

    @pytest.mark.smoke
    def test_injection_without_activation_works(self):
        """experts= without activation= should succeed."""
        from tileops.ops.moe.fused_moe import FusedMoe
        experts = self._make_experts()
        moe = FusedMoe(
            num_tokens=128, num_experts=4, top_k=2,
            hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
            experts=experts,
        )
        assert moe.activation == "silu_and_mul"

    @pytest.mark.smoke
    def test_default_path_activation_forwarded(self):
        """FusedMoe(activation='gelu_and_mul') creates experts with gelu_and_mul."""
        from tileops.ops.moe.fused_moe import FusedMoe
        moe = FusedMoe(
            num_tokens=128, num_experts=4, top_k=2,
            hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
            activation="gelu_and_mul",
        )
        assert moe.activation == "gelu_and_mul"
        assert moe._experts.activation == "gelu_and_mul"


class TestSharedFusedMoeActivation:

    @pytest.mark.smoke
    def test_activation_forwarded_to_routed_experts(self):
        """SharedFusedMoE(activation='gelu_and_mul') reaches the routed-experts path."""
        from tileops.ops.moe.shared_fused_moe import SharedFusedMoE
        moe = SharedFusedMoE(
            num_tokens=128, num_experts=4, top_k=2,
            hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
            activation="gelu_and_mul",
        )
        assert moe.activation == "gelu_and_mul"
        assert moe._experts.activation == "gelu_and_mul"
