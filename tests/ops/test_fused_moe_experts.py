"""Tests for FusedMoEExpertsFwdOp and IndexedExpertMLPFwdOp."""

import pytest
import torch

from tests.cuda_graph_contract import CapturedGraph
from tileops._ops.moe.fused_moe import FusedMoEFwdOp
from tileops._ops.moe.fused_moe_shared_expert import FusedMoESharedExpertFwdOp
from tileops._ops.moe.routed_expert import FusedMoEExpertsFwdOp, IndexedExpertMLPFwdOp
from workloads.device import run_device
from workloads.moe import MoEExpertsWorkload, moe_verification, ref_routed_experts
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


def _experts_case(dtype=torch.bfloat16, activation="silu_and_mul", some=(), **dims):
    """A manifest call of the expert MLP at *dims* passing the optional inputs *some*, its
    inputs and the op built from it."""
    call = manifest_call(
        "FusedMoEExpertsFwdOp",
        {"D": str(dtype).removeprefix("torch.")},
        activation=activation,
        **({"some": list(some)} if some else {}),
        **dims,
    )
    workload = MoEExpertsWorkload(call)
    return FusedMoEExpertsFwdOp(**call.arguments()), workload, workload.gen_inputs()


def _small_route_case(ids, dtype=torch.bfloat16):
    """An indexed-path call routed by *ids*: few routes per expert, H = 128, F = 256."""
    T, K = len(ids), len(ids[0])
    E = max(8, max(max(row) for row in ids) + 1)
    hidden = torch.randn(T, 128, dtype=dtype, device=run_device()) * 0.1
    w1 = torch.randn(E, 512, 128, dtype=dtype, device=run_device()) * 0.02
    w2 = torch.randn(E, 128, 256, dtype=dtype, device=run_device()) * 0.02
    weights = torch.softmax(torch.randn(T, K, dtype=torch.float32, device=run_device()), -1)
    topk_ids = torch.tensor(ids, dtype=torch.int32, device=run_device())
    return FusedMoEExpertsFwdOp(), (hidden, w1, w2, weights, topk_ids)


class TestFusedMoEExpertsFwdOp:
    @pytest.mark.parametrize(
        "dtype, activation",
        [
            pytest.param(torch.bfloat16, "silu_and_mul", marks=pytest.mark.smoke),
            pytest.param(torch.float16, "silu_and_mul", marks=pytest.mark.smoke),
            pytest.param(torch.bfloat16, "gelu_and_mul", marks=pytest.mark.full),
            pytest.param(torch.float16, "gelu_and_mul", marks=pytest.mark.full),
        ],
    )
    def test_the_staged_pipeline_matches_the_reference(self, dtype, activation):
        experts, workload, inputs = _experts_case(dtype, activation, T=128, E=4, K=2, H=256, F=128)
        out = experts(*inputs)
        compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))

    @pytest.mark.parametrize(
        ("activation", "dtype", "some", "dims"),
        [
            # F off the tile grid, both biases.
            pytest.param(
                "swigluoai",
                torch.bfloat16,
                ("b_gate_up", "b_down"),
                dict(T=64, E=4, K=2, H=128, F=96),
                marks=pytest.mark.smoke,
                id="swigluoai-off-grid",
            ),
            # Few routes per expert (T * K <= 2 * E), no bias.
            pytest.param(
                "swigluoai",
                torch.bfloat16,
                (),
                dict(T=4, E=8, K=2, H=128, F=256),
                marks=pytest.mark.full,
                id="swigluoai-few-routes",
            ),
            # Few routes per expert on silu_and_mul, with each bias alone.
            pytest.param(
                "silu_and_mul",
                torch.bfloat16,
                ("b_gate_up",),
                dict(T=4, E=8, K=2, H=128, F=256),
                marks=pytest.mark.full,
                id="silu-few-routes-gate-up-bias",
            ),
            pytest.param(
                "silu_and_mul",
                torch.bfloat16,
                ("b_down",),
                dict(T=4, E=8, K=2, H=128, F=256),
                marks=pytest.mark.full,
                id="silu-few-routes-down-bias",
            ),
        ],
    )
    def test_expert_biases_match_the_reference(self, activation, dtype, some, dims):
        """``b_gate_up`` is added before the activation, ``b_down`` before the routing weight."""
        experts, workload, inputs = _experts_case(dtype, activation, some, **dims)
        out = experts(*inputs)
        compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))

    @pytest.mark.in_tree_kernels
    @pytest.mark.smoke
    def test_one_instance_serves_two_expert_counts(self):
        """The pre-permute stage takes the call's expert count, so each count holds its own."""
        experts = FusedMoEExpertsFwdOp(activation="gelu_and_mul")
        for e in (4, 6):
            _, workload, inputs = _experts_case(
                activation="gelu_and_mul", T=32, E=e, K=2, H=128, F=128
            )
            out = experts(*inputs)
            compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))

    @pytest.mark.smoke
    def test_a_call_prices_the_experts_its_routing_reads(self):
        experts, args = _small_route_case([[0, 3], [3, 7], [0, 7], [3, 0]])
        experts(*args)
        T, K, H, F = 4, 2, 128, 256
        elem = args[0].element_size()
        active = 3  # experts 0, 3 and 7
        expected = active * 3 * F * H * elem + 2 * T * H * elem + T * K * (4 + 4)
        assert experts.eval_roofline()[1] == expected
        assert experts.roofline_data_terms() == {"active_experts": active}

    @pytest.mark.in_tree_kernels
    @pytest.mark.parametrize(
        "ids, dtype",
        [
            pytest.param(
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                torch.bfloat16,
                marks=pytest.mark.smoke,
                id="dispersed-bf16",
            ),
            pytest.param(
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                torch.float16,
                marks=pytest.mark.smoke,
                id="dispersed-fp16",
            ),
            pytest.param(
                [[0, 1], [0, 1], [0, 2], [0, 2]],
                torch.bfloat16,
                marks=pytest.mark.full,
                id="reused",
            ),
            pytest.param(
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                torch.bfloat16,
                marks=pytest.mark.full,
                id="duplicate-fallback",
            ),
            pytest.param(
                [[0, expert] for expert in range(1, 18)],
                torch.bfloat16,
                marks=pytest.mark.full,
                id="second-route-group",
            ),
        ],
    )
    def test_small_route_branch_matches_reference(self, ids, dtype):
        experts, args = _small_route_case(ids, dtype)
        out = experts(*args)
        compare_outputs(out, ref_routed_experts(*args), moe_verification(2))

    @pytest.mark.sm90
    @pytest.mark.cuda_graph(op="FusedMoEExpertsFwdOp")
    @pytest.mark.smoke
    def test_small_route_dispatch_replays_in_cuda_graph(self):
        experts, args = _small_route_case([[0, 1], [2, 3], [4, 5], [6, 7]])
        out = torch.empty_like(args[0])
        graph = CapturedGraph((lambda *inputs: experts(*inputs, out=out), args))
        assert graph.outputs[0] is out
        graph.renew(_small_route_case([[0, 1], [0, 1], [0, 2], [0, 2]])[1])
        graph.replay()
        compare_outputs(out, ref_routed_experts(*args), moe_verification(2))

    @pytest.mark.parametrize(
        "ids",
        [
            pytest.param([[0, 1], [2, 3], [4, 5], [6, 7]], marks=pytest.mark.smoke, id="indexed"),
            pytest.param(
                [[e % 8, (e + 1) % 8] for e in range(64)], marks=pytest.mark.full, id="tight"
            ),
        ],
    )
    def test_out_may_be_the_hidden_states_but_not_the_routing_weights(self, ids):
        """The experts read the hidden states before the reduction writes ``out``, which reads
        the routing weights while it writes."""
        experts, args = _small_route_case(ids)
        expected = ref_routed_experts(*args)
        hidden = args[0]
        assert experts(*args, out=hidden) is hidden
        compare_outputs(hidden, expected, moe_verification(2))
        tokens, top_k = args[3].shape
        buffer = torch.zeros_like(hidden).view(-1)
        weights = buffer[: 2 * tokens * top_k].view(torch.float32).view(tokens, top_k)
        weights.copy_(args[3])
        before = buffer.clone()
        with pytest.raises(ValueError, match="out must not overlap topk_weights"):
            experts(hidden, *args[1:3], weights, args[4], out=buffer.view_as(hidden))
        assert torch.equal(buffer, before)

    @pytest.mark.smoke
    def test_the_indexed_op_scales_its_output(self):
        _, args = _small_route_case([[0, 1], [2, 3]])
        out = IndexedExpertMLPFwdOp(routed_scaling_factor=2.5)(*args)
        expected = ref_routed_experts(*args, scale=2.5)
        compare_outputs(out, expected, moe_verification(2))


class TestFusedMoEActivationInjection:
    @pytest.mark.smoke
    def test_injection_with_conflicting_activation_raises(self):
        with pytest.raises(ValueError, match="activation conflicts"):
            FusedMoEFwdOp(2, experts=FusedMoEExpertsFwdOp(), activation="gelu_and_mul")

    @pytest.mark.smoke
    def test_injection_takes_the_experts_activation(self):
        experts = FusedMoEExpertsFwdOp(activation="gelu_and_mul")
        assert FusedMoEFwdOp(2, experts=experts, activation="gelu_and_mul").activation == (
            "gelu_and_mul"
        )
        assert FusedMoEFwdOp(2, experts=experts).activation == "gelu_and_mul"

    @pytest.mark.smoke
    def test_default_path_activation_forwarded(self):
        moe = FusedMoEFwdOp(2, activation="gelu_and_mul")
        assert moe._experts().activation == "gelu_and_mul"
        shared = FusedMoESharedExpertFwdOp(2, activation="gelu_and_mul")
        assert shared._experts().activation == "gelu_and_mul"
