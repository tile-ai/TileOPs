"""Behavioral contract tests for the staged MoE public boundary."""

import dataclasses

import pytest
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import MGroupedGemmCall, PostPermuteCall, PrePermuteCall
from tileops.ops.moe import (
    AlignmentPolicy,
    ComputeFamilyKey,
    ContiguousLayoutSpec,
    Fp8OneDOneDComputeSpec,
    InversePermuteContext,
    MaskedLayoutSpec,
    MaskedMetadata,
    MaterializationPolicy,
    MaterializedExpertLayout,
    MetadataKind,
    MoeExpertMLPFwdOp,
    MoeGroupedGemmFwdOp,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
    NoScaleComputeSpec,
    PaddingPolicy,
    PerRowExpertMetadata,
    PhysicalPsumMetadata,
    PrePermuteOutput,
    RoutingEpilogueSpec,
    TileBoundaryPolicy,
    routing_epilogue_reference,
)

pytestmark = pytest.mark.smoke


def _tight(kind: MetadataKind = MetadataKind.PHYSICAL_PSUM):
    return ContiguousLayoutSpec(metadata_kind=kind).resolve(arch=90)


def test_tight_and_aligned_layout_policies_resolve_unambiguously() -> None:
    assert _tight().resolved_alignment == 1
    aligned = ContiguousLayoutSpec(
        metadata_kind=MetadataKind.PER_ROW_EXPERT_IDS,
        materialization=MaterializationPolicy.ALIGNED,
        alignment_policy=AlignmentPolicy.SM90_128,
        padding_policy=PaddingPolicy.EXPLICIT_NEGATIVE_ONE,
        tile_boundary_policy=TileBoundaryPolicy.ALIGNED,
    )
    assert aligned.resolve(arch=90).resolved_alignment == 128
    with pytest.raises(ValueError, match="architecture 90"):
        aligned.resolve(arch=100)
    sm100 = dataclasses.replace(
        aligned,
        alignment_policy=AlignmentPolicy.SM100_EXPECTED_M,
        expected_m=64,
    )
    with pytest.raises(ValueError, match="no SM100 policy resolver"):
        sm100.resolve(arch=100)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alignment_policy": AlignmentPolicy.SM90_128},
        {"padding_policy": PaddingPolicy.EXPLICIT_NEGATIVE_ONE},
        {"tile_boundary_policy": TileBoundaryPolicy.ALIGNED},
    ],
)
def test_tight_layout_rejects_aligned_semantics(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ContiguousLayoutSpec(metadata_kind=MetadataKind.PHYSICAL_PSUM, **kwargs)


def test_materialized_layout_rejects_incompatible_metadata() -> None:
    with pytest.raises(TypeError, match="PhysicalPsumMetadata"):
        MaterializedExpertLayout(
            resolved_layout=_tight(),
            metadata=PerRowExpertMetadata(torch.empty(4, dtype=torch.int32)),
            num_experts=2,
            materialized_rows=4,
        )
    masked = MaskedLayoutSpec(max_m=4).resolve(arch=90)
    with pytest.raises(TypeError, match="MaskedMetadata"):
        MaterializedExpertLayout(
            resolved_layout=masked,
            metadata=PhysicalPsumMetadata(torch.empty(2, dtype=torch.int32)),
            num_experts=2,
            materialized_rows=8,
        )


def test_device_value_guards_cover_empty_experts_padding_and_ranges() -> None:
    psum = PhysicalPsumMetadata(torch.tensor([0, 0, 3], dtype=torch.int32))
    assert psum.device_value_guard(_tight(), materialized_rows=3).item()
    bad_psum = PhysicalPsumMetadata(torch.tensor([2, 1], dtype=torch.int32))
    assert not bad_psum.device_value_guard(_tight(), materialized_rows=1).item()

    aligned_ids = torch.full((256,), -1, dtype=torch.int32)
    aligned_ids[0] = 0
    aligned_ids[128] = 1
    ids = PerRowExpertMetadata(aligned_ids)
    assert ids.device_value_guard(
        num_experts=2,
        layout=ContiguousLayoutSpec(
            metadata_kind=MetadataKind.PER_ROW_EXPERT_IDS,
            materialization=MaterializationPolicy.ALIGNED,
            alignment_policy=AlignmentPolicy.SM90_128,
            padding_policy=PaddingPolicy.EXPLICIT_NEGATIVE_ONE,
            tile_boundary_policy=TileBoundaryPolicy.ALIGNED,
        ).resolve(arch=90),
    ).item()
    tight_with_gap = PerRowExpertMetadata(torch.tensor([0, -1, 1], dtype=torch.int32))
    assert not tight_with_gap.device_value_guard(
        num_experts=2, layout=_tight(MetadataKind.PER_ROW_EXPERT_IDS)
    ).item()

    masked = MaskedMetadata(torch.tensor([0, 4, 5], dtype=torch.int32))
    assert not masked.device_value_guard(max_m=4).item()


@pytest.mark.parametrize(
    ("ends", "materialized_rows", "expected"),
    [
        pytest.param([], 0, True, id="zero-experts"),
        pytest.param([0, 0, 0], 0, True, id="all-empty"),
        pytest.param([0, 0, 3], 3, True, id="consecutive-empty"),
        pytest.param([0, 2, 1], 1, False, id="decreasing-end"),
        pytest.param([0, 2], 3, False, id="shape-not-authoritative-end"),
    ],
)
def test_physical_psum_guard_covers_empty_and_capacity_edges(
    ends: list[int], materialized_rows: int, expected: bool
) -> None:
    metadata = PhysicalPsumMetadata(torch.tensor(ends, dtype=torch.int32))
    assert (
        metadata.device_value_guard(_tight(), materialized_rows=materialized_rows).item()
        is expected
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="guard test requires CUDA")
def test_device_value_validation_returns_a_device_guard_without_host_readback() -> None:
    metadata = PhysicalPsumMetadata(torch.tensor([0, 2], dtype=torch.int32, device="cuda"))

    guard = metadata.device_value_guard(_tight(), materialized_rows=2)

    assert guard.device.type == "cuda"
    assert guard.dtype is torch.bool
    assert guard.shape == ()


def test_pre_output_binds_layout_and_inverse_context_to_one_materialization() -> None:
    layout = MaterializedExpertLayout(
        resolved_layout=_tight(),
        metadata=PhysicalPsumMetadata(torch.tensor([1, 2], dtype=torch.int32)),
        num_experts=2,
        materialized_rows=2,
    )
    context = InversePermuteContext(
        inverse_indices=torch.tensor([0, 1], dtype=torch.int32),
        resolved_layout=layout.resolved_layout,
        materialization_token=layout.materialization_token,
        num_tokens=2,
        top_k=1,
        materialized_rows=2,
    )
    output = PrePermuteOutput(torch.empty(2, 8), layout, context)
    assert output.expert_layout is layout


def test_pre_output_rejects_context_from_another_materialization() -> None:
    layout = MaterializedExpertLayout(
        resolved_layout=_tight(),
        metadata=PhysicalPsumMetadata(torch.tensor([1], dtype=torch.int32)),
        num_experts=1,
        materialized_rows=1,
    )
    context = InversePermuteContext(
        inverse_indices=torch.tensor([0], dtype=torch.int32),
        resolved_layout=layout.resolved_layout,
        materialization_token=object(),
        num_tokens=1,
        top_k=1,
        materialized_rows=1,
    )
    with pytest.raises(ValueError, match="different materialization"):
        PrePermuteOutput(torch.empty(1, 8), layout, context)


def test_inverse_context_device_guard_accepts_only_local_rows_or_minus_one() -> None:
    context = InversePermuteContext(
        inverse_indices=torch.tensor([0, -1, 2], dtype=torch.int32),
        resolved_layout=_tight(),
        materialization_token=object(),
        num_tokens=3,
        top_k=1,
        materialized_rows=3,
    )
    assert context.device_value_guard().item()
    invalid = dataclasses.replace(
        context, inverse_indices=torch.tensor([0, -2, 3], dtype=torch.int32)
    )
    assert not invalid.device_value_guard().item()


def test_routing_epilogue_reference_assigns_each_operation_once() -> None:
    layout = _tight()
    context = InversePermuteContext(
        inverse_indices=torch.tensor([2, 0, 1, -1], dtype=torch.int32),
        resolved_layout=layout,
        materialization_token=object(),
        num_tokens=2,
        top_k=2,
        materialized_rows=3,
    )
    expert_output = torch.tensor([[1.0], [4.0], [8.0]], dtype=torch.bfloat16)
    weights = torch.tensor([[0.25, 0.5], [0.75, 100.0]], dtype=torch.float32)
    epilogue = RoutingEpilogueSpec(routed_scaling_factor=2.0)

    actual = routing_epilogue_reference(expert_output, context, weights, epilogue)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual.float(), torch.tensor([[5.0], [6.0]]))


def test_compute_and_epilogue_specs_reject_implicit_semantic_changes() -> None:
    with pytest.raises(ValueError, match="torch.float32"):
        NoScaleComputeSpec(accumulation_dtype=torch.float16)
    with pytest.raises(ValueError, match="routing weights"):
        RoutingEpilogueSpec(apply_routing_weight=False)
    with pytest.raises(ValueError, match="reduce top-k"):
        RoutingEpilogueSpec(reduce_topk=False)
    with pytest.raises(TypeError, match="only NoScaleComputeSpec"):
        MoeGroupedGemmFwdOp(Fp8OneDOneDComputeSpec())


def test_family_call_specs_are_frozen_and_keep_selection_axes_separate() -> None:
    pre = PrePermuteCall(arch=90, layout=ContiguousLayoutSpec(MetadataKind.PHYSICAL_PSUM))
    gemm = MGroupedGemmCall(arch=90, compute=NoScaleComputeSpec(), layout=_tight())
    post = PostPermuteCall(arch=90, layout=_tight(), epilogue=RoutingEpilogueSpec())
    with pytest.raises(dataclasses.FrozenInstanceError):
        pre.arch = 100
    assert gemm.compute != post.epilogue


class _PhysicalPsumCandidate(Kernel):
    supported_archs = [90]

    @classmethod
    def applies(cls, call: object) -> bool:
        layout = getattr(call, "layout", None)
        return (
            getattr(getattr(layout, "spec", None), "metadata_kind", None)
            is MetadataKind.PHYSICAL_PSUM
        )

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _PerRowCandidate(Kernel):
    supported_archs = [90]

    @classmethod
    def applies(cls, call: object) -> bool:
        layout = getattr(call, "layout", None)
        return (
            getattr(getattr(layout, "spec", None), "metadata_kind", None)
            is MetadataKind.PER_ROW_EXPERT_IDS
        )

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _GeneralCandidate(Kernel):
    general = True
    supported_archs = [90]

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _NeverCandidate(Kernel):
    @classmethod
    def applies(cls, call: object) -> bool:
        return False

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _ExecutableGroupedCandidate(Kernel):
    builds = 0

    def __init__(self, call: MGroupedGemmCall) -> None:
        super().__init__()
        type(self).builds += 1
        self.call = call

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        expert_layout: MaterializedExpertLayout,
        *,
        scales: object | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = a.new_zeros((*a.shape[:-1], b.shape[1]))
        if out is not None:
            out.copy_(result)
            return out
        return result


def _grouped_op_with_declared_candidates(**candidates: type[Kernel]) -> MoeGroupedGemmFwdOp:
    class DeclaringGroupedGemmOp(MoeGroupedGemmFwdOp):
        @property
        def default_kernel_map(self) -> dict[str, Kernel]:
            return dict(candidates)

    return DeclaringGroupedGemmOp(NoScaleComputeSpec())


def test_grouped_gemm_selection_behavior_table() -> None:
    physical_call = MGroupedGemmCall(arch=90, compute=NoScaleComputeSpec(), layout=_tight())
    per_row_call = MGroupedGemmCall(
        arch=90,
        compute=NoScaleComputeSpec(),
        layout=_tight(MetadataKind.PER_ROW_EXPERT_IDS),
    )
    op = _grouped_op_with_declared_candidates(
        physical=_PhysicalPsumCandidate,
        per_row=_PerRowCandidate,
        general=_GeneralCandidate,
    )

    assert op.select_kernel_key(("physical", "per_row", "general"), physical_call) == "physical"
    assert op.select_kernel_key(("physical", "per_row", "general"), per_row_call) == "per_row"

    ancient = dataclasses.replace(physical_call, arch=80)
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op.select_kernel_key(("physical", "per_row", "general"), ancient)


def test_grouped_gemm_ambiguous_and_incompatible_override_fail_explicitly() -> None:
    call = MGroupedGemmCall(arch=90, compute=NoScaleComputeSpec(), layout=_tight())
    ambiguous = _grouped_op_with_declared_candidates(
        first=_PhysicalPsumCandidate,
        second=_PhysicalPsumCandidate,
    )
    with pytest.raises(ValueError, match="dispatch is ambiguous"):
        ambiguous.select_kernel_key(("first", "second"), call)

    class OverrideableOp(MoeGroupedGemmFwdOp):
        @property
        def default_kernel_map(self) -> dict[str, Kernel]:
            return {"special": _PhysicalPsumCandidate, "general": _GeneralCandidate}

    overridden = OverrideableOp(NoScaleComputeSpec(), kernel_map={"special": _NeverCandidate})
    with pytest.raises(ValueError, match="does not fall back"):
        overridden.select_kernel_key(("special", "general"), call)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CallSpec records CUDA architecture")
def test_public_ops_build_complete_calls_before_selection() -> None:
    hidden = torch.empty(2, 8, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    pre_call = MoePrePermuteFwdOp(ContiguousLayoutSpec(MetadataKind.PHYSICAL_PSUM)).make_call(
        hidden, topk_ids
    )
    assert (pre_call.num_tokens, pre_call.hidden_size, pre_call.top_k) == (2, 8, 1)

    layout = MaterializedExpertLayout(
        resolved_layout=_tight(),
        metadata=PhysicalPsumMetadata(torch.tensor([1, 2], dtype=torch.int32, device="cuda")),
        num_experts=2,
        materialized_rows=2,
    )
    a = torch.empty(2, 8, dtype=torch.bfloat16, device="cuda")
    b = torch.empty(2, 4, 8, dtype=torch.bfloat16, device="cuda")
    gemm_call = MoeGroupedGemmFwdOp(NoScaleComputeSpec()).make_call(a, b, layout)
    assert (gemm_call.materialized_rows, gemm_call.num_experts, gemm_call.n, gemm_call.k) == (
        2,
        2,
        4,
        8,
    )
    assert gemm_call.compute_family is ComputeFamilyKey.SM90_BF16_NO_SCALE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CallSpec records CUDA architecture")
def test_staged_wiring_builds_all_family_calls_without_an_executable_candidate() -> None:
    device = torch.device("cuda")
    resolved = _tight()
    layout = MaterializedExpertLayout(
        resolved_layout=resolved,
        metadata=PhysicalPsumMetadata(torch.tensor([1, 2], dtype=torch.int32, device=device)),
        num_experts=2,
        materialized_rows=2,
    )
    expert_input = torch.empty(2, 8, dtype=torch.bfloat16, device=device)
    w_gate_up = torch.empty(2, 12, 8, dtype=torch.bfloat16, device=device)
    w_down = torch.empty(2, 8, 6, dtype=torch.bfloat16, device=device)
    mlp = MoeExpertMLPFwdOp()

    gate_call, down_call = mlp.make_calls(expert_input, w_gate_up, w_down, layout)

    assert gate_call.layout is down_call.layout is resolved
    assert (gate_call.k, gate_call.n, down_call.k, down_call.n) == (8, 12, 6, 8)
    assert tuple(mlp.kernel_delegates()) == (mlp.gate_up, mlp.activation_op, mlp.down)

    context = InversePermuteContext(
        inverse_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        resolved_layout=resolved,
        materialization_token=layout.materialization_token,
        num_tokens=2,
        top_k=1,
        materialized_rows=2,
    )
    expert_output = torch.empty(2, 8, dtype=torch.bfloat16, device=device)
    weights = torch.ones(2, 1, dtype=torch.float32, device=device)
    post_call = MoePostPermuteFwdOp(RoutingEpilogueSpec()).make_call(
        expert_output, context, weights
    )
    assert post_call.layout is resolved
    assert (post_call.num_tokens, post_call.top_k, post_call.hidden_size) == (2, 1, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="candidate test uses CUDA calls")
def test_injected_candidate_uses_common_selection_and_call_spec_cache() -> None:
    device = torch.device("cuda")
    layout = MaterializedExpertLayout(
        resolved_layout=_tight(),
        metadata=PhysicalPsumMetadata(torch.tensor([1], dtype=torch.int32, device=device)),
        num_experts=1,
        materialized_rows=1,
    )
    a = torch.ones(1, 4, dtype=torch.bfloat16, device=device)
    b = torch.ones(1, 2, 4, dtype=torch.bfloat16, device=device)
    _ExecutableGroupedCandidate.builds = 0
    op = MoeGroupedGemmFwdOp(
        NoScaleComputeSpec(), kernel_map={"grouped": _ExecutableGroupedCandidate}
    )

    first = op(a, b, layout)
    second = op(a, b, layout)

    assert first.shape == second.shape == (1, 2)
    assert _ExecutableGroupedCandidate.builds == 1
    assert len(op.built_kernels("grouped")) == 1


def test_expert_mlp_forwards_only_caller_replacements_to_matching_delegates() -> None:
    mlp = MoeExpertMLPFwdOp(
        kernel_map={
            "grouped": _ExecutableGroupedCandidate,
            "silu_and_mul": _NeverCandidate,
        }
    )

    assert mlp.gate_up.forwarded_overrides() == {"grouped": _ExecutableGroupedCandidate}
    assert mlp.down.forwarded_overrides() == {"grouped": _ExecutableGroupedCandidate}
    assert mlp.activation_op.forwarded_overrides() == {"silu_and_mul": _NeverCandidate}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="candidate test uses CUDA calls")
def test_public_op_without_candidates_fails_explicitly() -> None:
    hidden = torch.empty(1, 4, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    op = MoePrePermuteFwdOp(ContiguousLayoutSpec(MetadataKind.PHYSICAL_PSUM))

    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(hidden, topk_ids)
