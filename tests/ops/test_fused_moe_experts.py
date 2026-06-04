"""Tests for FusedMoEExpertsNopadPersistent3WGFwdOp and supporting ABCs."""
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
)
from tileops.ops.moe.routed_expert.moe_grouped_gemm_nopad_fused_act import (
    MoeGroupedGemmNopad3WGFusedActFwdOp,
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
    # gelu_and_mul: PyTorch's F.gelu(x, approximate="none") is exact erf GELU,
    # which matches GeluAndMulFwdKernel's `x * 0.5 * (1 + erf(x/sqrt(2)))`. We
    # pass approximate="none" explicitly so this is locked at the test level —
    # PyTorch's default happens to be "none", but a different default in some
    # future version would silently switch the reference to tanh approximation
    # (which is GeluTanhAndMulFwdKernel, a separate registry entry).
    # Resolve once outside the per-expert loop so an unsupported activation
    # raises immediately rather than silently falling back to a wrong
    # reference value when this helper is extended.
    _ACT_FNS = {
        "silu_and_mul": lambda gate, up: F.silu(gate) * up,  # noqa: E731
        "gelu_and_mul": lambda gate, up: F.gelu(gate, approximate="none") * up,  # noqa: E731
    }
    if activation not in _ACT_FNS:
        raise ValueError(
            f"_torch_ref_moe_activation has no reference for activation={activation!r}; "
            "extend this helper before adding the activation to the registry."
        )
    gated = _ACT_FNS[activation]
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
        act = gated(gate, up)
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
        implementation supports expert_map. The
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


@pytest.mark.parametrize(
    "activation",
    [
        pytest.param("silu_and_mul", marks=pytest.mark.smoke),
        pytest.param("gelu_and_mul", marks=pytest.mark.nightly),
    ],
)
def test_use_fused_activation_parity(activation):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires SM90")
    torch.manual_seed(0)
    T_count, E, top_k, H, Fdim = 256, 8, 2, 256, 768
    hidden = torch.randn(T_count, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w_gate_up = torch.randn(E, 2 * Fdim, H, dtype=torch.bfloat16, device="cuda") * 0.02
    w_down = torch.randn(E, H, Fdim, dtype=torch.bfloat16, device="cuda") * 0.02
    topk_w = torch.rand(T_count, top_k, dtype=torch.float32, device="cuda")
    topk_ids = torch.randint(0, E, (T_count, top_k), dtype=torch.int32, device="cuda")

    def run(use_fused):
        op = FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=T_count, num_experts=E, top_k=top_k, hidden_size=H,
            ffn_size=Fdim, dtype=torch.bfloat16, activation=activation,
            use_fused_activation=use_fused)
        out = torch.empty(T_count, H, dtype=torch.bfloat16, device="cuda")
        ws = torch.empty(0, dtype=torch.bfloat16, device="cuda")
        op.forward(out, hidden, w_gate_up, w_down, topk_w, topk_ids, None, ws, ws, E)
        return out

    fused_out = run(True)
    assert fused_out.shape == (T_count, H)
    # bf16 fused-vs-unfused accumulation differs slightly; tolerance covers the
    # fused-epilogue vs separate-activation-kernel rounding gap.
    torch.testing.assert_close(fused_out, run(False), rtol=3e-2, atol=3e-2)


@pytest.mark.smoke
def test_use_fused_activation_disabled_on_gemm_override():
    """A moe_grouped_gemm_kernel override must disable fusion.

    The fused gate_up wrapper cannot honor a moe_grouped_gemm_kernel override
    (it keys off moe_grouped_gemm_fused_act_kernel), so enabling fusion would
    apply the override only to the down GEMM, leaving a fused 3WG gate_up — an
    inconsistent pipeline. Eligibility must fall back to the unfused path.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires SM90")
    from tileops.kernels.moe.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel
    experts = FusedMoEExpertsNopadPersistent3WGFwdOp(
        num_tokens=256, num_experts=8, top_k=2, hidden_size=256, ffn_size=768,
        dtype=torch.bfloat16, activation="silu_and_mul", use_fused_activation=True,
        kernel_map={"moe_grouped_gemm_kernel": MoeGroupedGemmNopadKernel},
    )
    assert experts.use_fused_activation is False
    assert experts._activation_op is not None


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

    def _make_experts(self, activation="silu_and_mul"):
        return FusedMoEExpertsNopadPersistent3WGFwdOp(
            num_tokens=128, num_experts=4, top_k=2,
            hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
            activation=activation,
        )

    @pytest.mark.smoke
    def test_injection_with_conflicting_activation_raises(self):
        """experts= + activation= that disagree must raise ValueError."""
        from tileops.ops.moe.fused_moe import FusedMoe
        experts = self._make_experts(activation="silu_and_mul")
        with pytest.raises(ValueError, match="activation conflicts"):
            FusedMoe(
                num_tokens=128, num_experts=4, top_k=2,
                hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
                experts=experts, activation="gelu_and_mul",
            )

    @pytest.mark.smoke
    def test_injection_with_matching_activation_works(self):
        """experts= + activation= that match the injected experts is accepted."""
        from tileops.ops.moe.fused_moe import FusedMoe
        experts = self._make_experts(activation="gelu_and_mul")
        moe = FusedMoe(
            num_tokens=128, num_experts=4, top_k=2,
            hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
            experts=experts, activation="gelu_and_mul",
        )
        assert moe.activation == "gelu_and_mul"

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

    @pytest.mark.smoke
    def test_injection_without_activation_attribute_raises(self):
        """A third-party experts instance missing .activation must raise.

        Catches the silent-fallback footgun: without .activation, the conflict
        guard would default to comparing against 'silu_and_mul' and could
        silently accept a non-matching activation argument.
        """
        from tileops.ops.moe.fused_moe import FusedMoe
        from tileops.ops.moe.routed_expert.abc import FusedMoEExpertsModular

        class ExpertsWithoutActivation(FusedMoEExpertsModular):
            """Stand-in for a third-party experts impl that forgot .activation."""

            def __init__(self):
                pass

            @property
            def default_kernel_map(self):
                return {}

            def workspace_shapes(self, M, N, K, topk, num_experts):
                return ((0,), (0,))

            def output_shape(self, T_prime, H):
                return (T_prime, H)

            def forward(self, output, hidden_states, w_gate_up, w_down,
                        topk_weights, topk_ids, expert_map, workspace1,
                        workspace2, num_experts):
                pass

            def make_weighted_reduce(self):
                from tileops.ops.moe.routed_expert.abc import WeightedReduceNoOp
                return WeightedReduceNoOp()

        with pytest.raises(ValueError, match="missing the required `.activation`"):
            FusedMoe(
                num_tokens=128, num_experts=4, top_k=2,
                hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
                experts=ExpertsWithoutActivation(),
            )


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

    @pytest.mark.smoke
    def test_shared_expert_with_non_default_activation_raises(self):
        """shared_ffn_size + non-silu activation must raise NotImplementedError.

        SharedExpertMLPKernel hardcodes silu_and_mul; allowing a different
        activation here would silently produce mixed outputs (routed=gelu,
        shared=silu).
        """
        from tileops.ops.moe.shared_fused_moe import SharedFusedMoE
        with pytest.raises(NotImplementedError, match="shared-expert path only supports"):
            SharedFusedMoE(
                num_tokens=128, num_experts=4, top_k=2,
                hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
                shared_ffn_size=128,
                activation="gelu_and_mul",
            )

    @pytest.mark.smoke
    def test_shared_expert_with_default_activation_works(self):
        """shared_ffn_size + silu_and_mul (default) is fine."""
        from tileops.ops.moe.shared_fused_moe import SharedFusedMoE
        moe = SharedFusedMoE(
            num_tokens=128, num_experts=4, top_k=2,
            hidden_size=256, ffn_size=128, dtype=torch.bfloat16,
            shared_ffn_size=128,
        )
        assert moe.activation == "silu_and_mul"


@pytest.mark.smoke
def test_fused_act_fwd_op_shape_and_values():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires SM90")
    T_count, E, top_k, ffn, K = 256, 8, 2, 768, 128
    numel = T_count * top_k
    sizes = torch.full((E,), numel // E, dtype=torch.int32, device="cuda")
    sizes[:numel % E] += 1  # spread remainder; safe when numel < E
    offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    A = torch.randn(numel, K, dtype=torch.bfloat16, device="cuda") * 0.02
    B = torch.randn(E, 2 * ffn, K, dtype=torch.bfloat16, device="cuda") * 0.02
    op = MoeGroupedGemmNopad3WGFusedActFwdOp(
        numel=numel, num_experts=E, ffn=ffn, k=K, dtype=torch.bfloat16,
        activation="silu_and_mul")
    out = op(A, B, sizes, offsets)
    assert out.shape == (numel, ffn)
    exp = torch.zeros(numel, ffn, dtype=torch.bfloat16, device="cuda")
    for e in range(E):
        n, o = int(sizes[e]), int(offsets[e])
        gu = A[o:o+n].float() @ B[e].float().t()
        exp[o:o+n] = (F.silu(gu[:, :ffn]) * gu[:, ffn:]).to(torch.bfloat16)
    torch.testing.assert_close(out, exp, rtol=2e-2, atol=2e-2)
