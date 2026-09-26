"""Tests for FusedMoEExpertsFwdOp, IndexedExpertMLPFwdOp and supporting ABCs."""

import pytest
import torch

from tileops.ops.moe.abc import FusedMoEExpertsModular, WeightedReduce, WeightedReduceNoOp
from tileops.ops.moe.fused_moe import FusedMoeFwdOp
from tileops.ops.moe.fused_moe_shared_expert import FusedMoeSharedExpertFwdOp
from tileops.ops.moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEP
from tileops.ops.moe.routed_expert import FusedMoEExpertsFwdOp, IndexedExpertMLPFwdOp
from tileops.utils import get_sm_version
from workloads.moe import MoeExpertsWorkload, moe_call, ref_routed_experts


def _experts_case(dtype=torch.bfloat16, activation="silu_and_mul", **dims):
    """A manifest call of the expert MLP at *dims*, its inputs and the op built from it."""
    call = moe_call(
        "FusedMoEExpertsFwdOp",
        {"D": str(dtype).removeprefix("torch.")},
        activation=activation,
        **dims,
    )
    workload = MoeExpertsWorkload(call)
    return FusedMoEExpertsFwdOp(**call.arguments({})), workload, workload.gen_inputs()


def _small_route_case(ids, dtype=torch.bfloat16):
    """An indexed-path call routed by *ids*: few routes per expert, H = 128, F = 256."""
    T, K = len(ids), len(ids[0])
    E = max(8, max(max(row) for row in ids) + 1)
    hidden = torch.randn(T, 128, dtype=dtype, device="cuda") * 0.1
    w1 = torch.randn(E, 512, 128, dtype=dtype, device="cuda") * 0.02
    w2 = torch.randn(E, 128, 256, dtype=dtype, device="cuda") * 0.02
    weights = torch.softmax(torch.randn(T, K, dtype=torch.float32, device="cuda"), -1)
    topk_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    output = torch.empty(T, 128, dtype=dtype, device="cuda")
    return FusedMoEExpertsFwdOp(), (output, hidden, w1, w2, weights, topk_ids)


def _reference(args) -> torch.Tensor:
    return ref_routed_experts(*args[1:])


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
    reduce.apply(
        output,
        expert_out,
        topk_weights=torch.ones(T, 2),
        topk_ids=torch.zeros(T, 2, dtype=torch.int32),
    )
    assert torch.allclose(output, expert_out)


@pytest.mark.smoke
def test_weighted_reduce_noop_same_tensor():
    """WeightedReduceNoOp is a no-op when output is expert_out."""
    T, H = 4, 8
    t = torch.randn(T, H)
    original = t.clone()
    WeightedReduceNoOp().apply(
        t, t, topk_weights=torch.ones(T, 2), topk_ids=torch.zeros(T, 2, dtype=torch.int32)
    )
    assert torch.allclose(t, original)


# MoEPrepareAndFinalizeNoDPEP


class TestMoEPrepareAndFinalizeNoDPEP:
    @pytest.mark.smoke
    def test_prepare_passthrough(self):
        T, H, K = 8, 64, 2
        hidden = torch.randn(T, H, dtype=torch.bfloat16)
        weights = torch.rand(T, K, dtype=torch.float32)
        ids = torch.randint(0, 4, (T, K), dtype=torch.int32)
        pf = MoEPrepareAndFinalizeNoDPEP()
        r = pf.prepare(hidden, weights, ids, num_experts=4)
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


# FusedMoEExpertsFwdOp


class TestFusedMoEExpertsFwdOp:
    @pytest.mark.smoke
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
    def test_the_staged_pipeline_matches_the_reference(self, dtype, activation):
        experts, workload, inputs = _experts_case(dtype, activation, T=128, E=4, K=2, H=256, F=128)
        experts(*inputs)
        torch.testing.assert_close(
            inputs[0].float(), workload.ref_program(*inputs).float(), rtol=2e-2, atol=2e-2
        )

    @pytest.mark.smoke
    def test_dims_off_the_tile_grid(self):
        experts, workload, inputs = _experts_case(T=64, E=4, K=2, H=128, F=96)
        experts(*inputs)
        torch.testing.assert_close(
            inputs[0].float(), workload.ref_program(*inputs).float(), rtol=2e-2, atol=2e-2
        )

    @pytest.mark.smoke
    def test_one_instance_serves_two_expert_counts(self):
        """The pre-permute stage takes the call's expert count, so each count holds its own."""
        experts = FusedMoEExpertsFwdOp(activation="gelu_and_mul")
        for e in (4, 6):
            _, workload, inputs = _experts_case(
                activation="gelu_and_mul", T=32, E=e, K=2, H=128, F=128
            )
            experts(*inputs)
            torch.testing.assert_close(
                inputs[0].float(), workload.ref_program(*inputs).float(), rtol=2e-2, atol=2e-2
            )
        assert [op.num_local_experts for op in experts.kernel_delegates()[:2]] == [4, 6]

    @pytest.mark.smoke
    def test_a_call_prices_the_experts_its_routing_reads(self):
        experts, args = _small_route_case([[0, 3], [3, 7], [0, 7], [3, 0]])
        experts(*args)
        T, K, H, F = 4, 2, 128, 256
        elem = args[1].element_size()
        active = 3  # experts 0, 3 and 7
        expected = active * 3 * F * H * elem + 2 * T * H * elem + T * K * (4 + 4)
        assert experts.eval_roofline()[1] == expected
        assert experts.roofline_inputs() == {"active_experts": active}

    @pytest.mark.smoke
    @pytest.mark.parametrize("tokens,indexed", [(64, True), (65, False)])
    def test_the_indexed_path_ends_at_two_routes_per_expert(self, tokens, indexed):
        """The choice is made per call: at most two routes per expert takes the indexed op."""
        experts = FusedMoEExpertsFwdOp()
        E, K, H, F = 256, 8, 256, 256
        args = (
            torch.empty(tokens, H, dtype=torch.bfloat16, device="cuda"),
            torch.randn(tokens, H, dtype=torch.bfloat16, device="cuda") * 0.1,
            torch.randn(E, 2 * F, H, dtype=torch.bfloat16, device="cuda") * 0.02,
            torch.randn(E, H, F, dtype=torch.bfloat16, device="cuda") * 0.02,
            torch.rand(tokens, K, dtype=torch.float32, device="cuda"),
            torch.randint(0, E, (tokens, K), dtype=torch.int32, device="cuda"),
        )
        experts(*args)
        assert bool(experts.last_call.stages["indexed_small_route"]) is indexed

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "ids",
        [
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            [[0, 1], [0, 1], [0, 2], [0, 2]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, expert] for expert in range(1, 18)],
        ],
        ids=["dispersed", "reused", "duplicate-fallback", "second-route-group"],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_small_route_branch_matches_reference(self, ids, dtype):
        experts, args = _small_route_case(ids, dtype)
        experts(*args)
        torch.testing.assert_close(args[0].float(), _reference(args).float(), rtol=2e-2, atol=1e-1)
        if get_sm_version() == 90:
            indexed = experts._indexed_mlp
            built = {r for r in indexed.kernel_types if indexed.built_kernels(r)}
            assert built == set(indexed.kernel_types), built

    @pytest.mark.smoke
    def test_small_route_dispatch_replays_in_cuda_graph(self):
        experts, args = _small_route_case([[0, 1], [2, 3], [4, 5], [6, 7]])
        experts.forward(*args)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            experts.forward(*args)

        args[5].copy_(
            torch.tensor([[0, 1], [0, 1], [0, 2], [0, 2]], dtype=torch.int32, device="cuda")
        )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(args[0].float(), _reference(args).float(), rtol=2e-2, atol=1e-1)

    @pytest.mark.smoke
    def test_the_indexed_op_scales_its_output(self):
        _, args = _small_route_case([[0, 1], [2, 3]])
        IndexedExpertMLPFwdOp(routed_scaling_factor=2.5)(*args)
        expected = ref_routed_experts(*args[1:], scale=2.5)
        torch.testing.assert_close(args[0].float(), expected.float(), rtol=2e-2, atol=1e-1)

    @pytest.mark.smoke
    def test_output_shape_and_weighted_reduce(self):
        experts = FusedMoEExpertsFwdOp()
        assert experts.output_shape(128, 256) == (128, 256)
        assert isinstance(experts.make_weighted_reduce(), WeightedReduceNoOp)


class TestFusedMoeActivationInjection:
    @pytest.mark.smoke
    def test_injection_with_conflicting_activation_raises(self):
        with pytest.raises(ValueError, match="activation conflicts"):
            FusedMoeFwdOp(2, experts=FusedMoEExpertsFwdOp(), activation="gelu_and_mul")

    @pytest.mark.smoke
    def test_injection_takes_the_experts_activation(self):
        experts = FusedMoEExpertsFwdOp(activation="gelu_and_mul")
        assert FusedMoeFwdOp(2, experts=experts, activation="gelu_and_mul").activation == (
            "gelu_and_mul"
        )
        assert FusedMoeFwdOp(2, experts=experts).activation == "gelu_and_mul"

    @pytest.mark.smoke
    def test_default_path_activation_forwarded(self):
        moe = FusedMoeFwdOp(2, activation="gelu_and_mul")
        assert moe._experts.activation == "gelu_and_mul"
        shared = FusedMoeSharedExpertFwdOp(2, activation="gelu_and_mul")
        assert shared._experts.activation == "gelu_and_mul"

    @pytest.mark.smoke
    def test_injection_without_activation_attribute_raises(self):
        """A third-party experts instance missing ``.activation`` is refused, so a
        non-matching ``activation`` cannot pass silently."""

        class ExpertsWithoutActivation(FusedMoEExpertsModular):
            def __init__(self):
                pass

            def _infer_output_shapes(self, *args, **kwargs):
                raise NotImplementedError

            def _validate_dtypes(self, *args, **kwargs):
                raise NotImplementedError

            def eval_roofline(self):
                raise NotImplementedError

            def output_shape(self, T_prime, H):
                return (T_prime, H)

            def forward(self, output, hidden_states, w_gate_up, w_down, topk_weights, topk_ids):
                pass

            def make_weighted_reduce(self):
                return WeightedReduceNoOp()

        with pytest.raises(ValueError, match="missing the required `.activation`"):
            FusedMoeFwdOp(2, experts=ExpertsWithoutActivation())


@pytest.mark.smoke
def test_the_shared_expert_refuses_a_non_silu_activation():
    """The shared-expert kernel applies silu_and_mul; with gelu routed experts the two halves
    would disagree, so a call passing the shared weights is refused."""
    T, E, H, F, S = 4, 4, 128, 128, 128
    op = FusedMoeSharedExpertFwdOp(2, activation="gelu_and_mul")
    args = (
        torch.randn(T, H, dtype=torch.bfloat16, device="cuda"),
        torch.randn(T, E, device="cuda"),
        torch.randn(E, 2 * F, H, dtype=torch.bfloat16, device="cuda"),
        torch.randn(E, H, F, dtype=torch.bfloat16, device="cuda"),
        None,
        torch.randn(2 * S, H, dtype=torch.bfloat16, device="cuda"),
        torch.randn(H, S, dtype=torch.bfloat16, device="cuda"),
    )
    with pytest.raises(ValueError, match="silu_and_mul"):
        op(*args)
